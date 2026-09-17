import pandas as pd
import numpy as np

def cargar_y_procesar_datos(filepath: str) -> pd.DataFrame:
    """
    Carga y procesa el inventario de subastas calculando métricas de rendimiento,
    costo de tenencia y clasificación Fasecolda.
    """
    df = pd.read_csv(filepath)
    
    # 1. Tratamiento de Nulos
    df['ID_Comprador'] = df['ID_Comprador'].fillna('Sin Adjudicar')
    df['Tipo_Comprador'] = df['Tipo_Comprador'].fillna('Sin Adjudicar')
    
    # 2. Métricas Módulo A: Win Rate, Porcentaje de Recuperación y Spread de Puja
    df['Win_Rate_Flag'] = (df['Estado_Subasta'] == 'Vendido').astype(int)
    df['Porcentaje_Recuperacion'] = np.where(
        df['Estado_Subasta'] == 'Vendido',
        (df['Precio_Final_Venta_COP'] / df['Costo_Adquisicion_COP']) * 100,
        np.nan
    )
    df['Spread_Puja_COP'] = np.where(
        df['Estado_Subasta'] == 'Vendido',
        df['Precio_Final_Venta_COP'] - df['Precio_Reserva_COP'],
        np.nan
    )
    
    # 3. Métricas Módulo B: Categoria Fasecolda, Tramo de Aging y Holding Cost
    df['Ratio_Costo_Fasecolda'] = df['Costo_Adquisicion_COP'] / df['Valor_Fasecolda_COP']
    df['Ratio_Reserva_Fasecolda'] = df['Precio_Reserva_COP'] / df['Valor_Fasecolda_COP']

    conditions = [
        (df['Ratio_Reserva_Fasecolda'] > 0.95) | (df['Ratio_Costo_Fasecolda'] > 0.85),
        (df['Ratio_Reserva_Fasecolda'] < 0.85) & (df['Ratio_Costo_Fasecolda'] < 0.75)
    ]
    choices = ['Overpriced', 'Underpriced']
    df['Categoria_Fasecolda'] = np.select(conditions, choices, default='Fair Value')
    
    bins_aging = [-1, 30, 60, 90, 180]
    labels_aging = ['<=30 días', '31-60 días', '61-90 días', '>90 días']
    df['Tramo_Aging'] = pd.cut(df['Dias_en_Inventario'], bins=bins_aging, labels=labels_aging)
    
    # Holding Cost: Tasa del 1.5% mensual (compuesto sobre días)
    df['Holding_Cost_COP'] = df['Costo_Adquisicion_COP'] * (((1 + 0.015)**(df['Dias_en_Inventario']/30)) - 1)
    
    return df

def clasificar_fasecolda_dual(row: pd.Series) -> str:
    ratio_reserva = row['Precio_Reserva_COP'] / row['Valor_Fasecolda_COP']
    ratio_costo = row['Costo_Adquisicion_COP'] / row['Valor_Fasecolda_COP']
    
    # 1. Overpriced: El precio de reserva está desfasado (>95%) O el costo de adquisición fue tan alto (>85%) 
    # que comprime el margen EBITDA esperado.
    if ratio_reserva > 0.95 or ratio_costo > 0.85:
        return 'Overpriced'
    
    # 2. Underpriced: Activo adquirido a excelente precio (<75%) y con reserva agresiva (<85%), 
    # representativo de una clara oportunidad de captura de margen.
    elif ratio_reserva < 0.85 and ratio_costo < 0.75:
        return 'Underpriced'
    
    # 3. Fair Value: Alineado con los márgenes de operación habituales (75%-85% costo y 85%-95% reserva).
    else:
        return 'Fair Value'

def resumen_modulo_a(df: pd.DataFrame) -> dict:
    return {
        'win_rate_global_%': round(df['Win_Rate_Flag'].mean() * 100, 2),
        'win_rate_por_canal': (df.groupby('Canal_Asignado')['Win_Rate_Flag'].mean() * 100).to_dict(),
        'recuperacion_promedio_%': round(df['Porcentaje_Recuperacion'].mean(), 2),
        'spread_promedio_COP': round(df['Spread_Puja_COP'].mean(), 2)
    }

def resumen_modulo_b(df: pd.DataFrame) -> dict:
    return {
        'distribucion_fasecolda': df['Categoria_Fasecolda'].value_counts().to_dict(),
        'holding_cost_total_COP': round(df['Holding_Cost_COP'].sum(), 2),
        'holding_cost_por_tramo_COP': df.groupby('Tramo_Aging', observed=False)['Holding_Cost_COP'].sum().to_dict()
    }

def enriquecer_con_fuentes_externas(df_interno: pd.DataFrame, filepath_externo: str) -> pd.DataFrame:
    """
    Realiza el matcheo entre el inventario propio y la referencia oficial
    de mercado (RUNT y Guía Fasecolda).
    """
    df_externo = pd.read_csv(filepath_externo)
    
    # Merge por Marca, Línea y Modelo
    df_consolidado = pd.merge(
        df_interno, 
        df_externo, 
        on=['Marca', 'Linea', 'Modelo'], 
        how='left'
    )
    
    return df_consolidado

if __name__ == '__main__':
    data = cargar_y_procesar_datos('data/raw/inventario_subastas_simon.csv')
    print("Módulo A:", resumen_modulo_a(data))
    print("Módulo B:", resumen_modulo_b(data))