"""
Módulo de Reasignación Automática de Canales mediante Árboles de Decisión.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


class MotorReasignacionCanal:

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth, random_state=42
        )
        self.mapa_liq = {
            'Baja Liquidez': 1,
            'Liquidez Media': 2,
            'Alta Liquidez': 3,
        }

    def _preparar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae y transforma las variables predictoras del vehículo."""
        X = pd.DataFrame()
        #X['Dias_en_Inventario'] = df['Dias_en_Inventario']
        X['Liquidez_Num'] = (
            df['Demanda_Mercado_RUNT'].map(self.mapa_liq).fillna(2)
        )
        X['Costo_Adquisicion_COP'] = df['Costo_Adquisicion_COP']
        X['Modelo_Anio'] = df['Modelo']
        X['Kilometraje'] = df['Kilometraje']
        return X

    def generar_target_financiero(self, df: pd.DataFrame) -> pd.Series:
        """Combina el histórico de ventas exitosas con la optimización
        financiera para el inventario no adjudicado o estancado.
        """
        def definir_canal(row):
        
            if (
                row.get('Estado_Subasta') == 'Vendido'
                and pd.notnull(row.get('Canal_Venta'))
            ):
                return row['Canal_Venta']

            if (
                row['Dias_en_Inventario'] > 45
                and row.get('Demanda_Mercado_RUNT') == 'Baja Liquidez'
            ):
                return 'Concesionario Aliado (Liquidación)'
            elif (
                row['Dias_en_Inventario'] > 40
                #and row.get('Demanda_Mercado_RUNT') == 'Baja Liquidez'
            ):
                return 'Venta Directa Flotas'
            else:
                return 'Subasta Virtual'

        return df.apply(definir_canal, axis=1)

    def fit(self, df_inventario: pd.DataFrame):
        """Entrena el árbol de decisión sobre el inventario evaluado."""
        X = self._preparar_features(df_inventario)
        y = self.generar_target_financiero(df_inventario)
        self.model.fit(X, y)
        return self

    def predict(self, df_inventario: pd.DataFrame) -> pd.Series:
        """Predice el canal óptimo para nuevos vehículos."""
        X = self._preparar_features(df_inventario)
        return pd.Series(self.model.predict(X), index=df_inventario.index)

    def obtener_reglas_texto(self) -> str:
        """Devuelve la estructura de reglas del árbol en texto interpretable."""
        feature_names = [
            #'Dias_en_Inventario',
            'Liquidez_Num',
            'Costo_Adquisicion_COP',
            'Modelo_Anio',
            'Kilometraje'
        ]
        return export_text(self.model, feature_names=feature_names)