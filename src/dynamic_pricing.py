"""
Módulo D: Motor de Pricing Dinámico, Elasticidad y Reasignación de Canales.
"""

import numpy as np
import pandas as pd


def calcular_elasticidad_descuento(df: pd.DataFrame) -> pd.DataFrame:
  """Evalúa la relación entre el porcentaje de descuento otorgado respecto a la guía Fasecolda
  y la tasa de conversión de subasta (Win Rate).
  Retorna un DataFrame agrupado por tramos de descuento con sus respectivas
  métricas de elasticidad.
  """
  df_temp = df.copy()

  # Porcentaje de Descuento sobre la referencia oficial de Fasecolda
  df_temp['Pct_Descuento_Fasecolda'] = (
      (df_temp['Valor_Fasecolda_COP'] - df_temp['Precio_Reserva_COP'])
      / df_temp['Valor_Fasecolda_COP']
  ).clip(lower=0)

  df_temp['Vendido_Flag'] = (df_temp['Estado_Subasta'] == 'Vendido').astype(int)

  # Segmentación por cuartiles de descuento
  df_temp['Rango_Descuento'] = pd.qcut(
      df_temp['Pct_Descuento_Fasecolda'],
      q=4,
      labels=[
          'Bajo (0-8%)',
          'Medio-Bajo (8-14%)',
          'Medio-Alto (14-18%)',
          'Alto (>18%)',
      ],
  )

  elasticidad = (
      df_temp.groupby('Rango_Descuento', observed=False)
      .agg(
          Descuento_Promedio=('Pct_Descuento_Fasecolda', 'mean'),
          Win_Rate=('Vendido_Flag', 'mean'),
          Total_Vehiculos=('ID_Vehiculo', 'count'),
      )
      .reset_index()
  )

  # Cálculo de variación porcentual inter-tramo
  elasticidad['Delta_Win_Rate'] = elasticidad['Win_Rate'].pct_change()
  elasticidad['Delta_Descuento'] = elasticidad[
      'Descuento_Promedio'
  ].pct_change()

  # Elasticidad (sensibilidad de conversión ante variaciones de descuento)
  elasticidad['Elasticidad_Demanda'] = (
      (elasticidad['Delta_Win_Rate'] / elasticidad['Delta_Descuento'])
      .fillna(0)
      .round(2)
  )

  return elasticidad


def sugerir_precio_reserva_optimo(
    row: pd.Series
) -> float:
  """Calcula el precio de reserva sugerido aplicando un descuento dinámico en función
  de los días estancado en inventario y el indicador de demanda del RUNT.
  """
  # Parámetros Financieros
  TASA_WACC_ANUAL = 0.14
  COSTO_PARQUEADERO_DIA = 12000

  dias = row['Dias_en_Inventario']
  liquidez = row['Demanda_Mercado_RUNT']
  precio_reserva = row['Precio_Reserva_COP']
  costo_adq = row['Costo_Adquisicion_COP']
    
  # Si el vehículo ya se vendió, mantiene su precio original
  if row.get('Estado_Subasta') == 'Vendido':
    return precio_reserva

  # Costo de tenencia acumulado
  costo_tenencia_diario = (costo_adq * (TASA_WACC_ANUAL / 365)) + COSTO_PARQUEADERO_DIA
  costo_tenencia_acum = costo_tenencia_diario * dias
    
  # Multiplicador por demanda RUNT
  mult_liquidez = {'Alta Liquidez': 0.5, #Alta demanda, mitad de descuento 
                  'Liquidez Media': 1.0, #Baseline neutral, sin ajuste
                  'Baja Liquidez': 1.3}.get(liquidez, 1.0) #Baja demanda, aumento del 30%
    
  # Factor continuo acotado al 10% máx.
  pct_desgaste = (costo_tenencia_acum / costo_adq) * mult_liquidez
  factor_descuento = min(0.10, pct_desgaste * 0.8) # 80% del costo acumulado por tenencia (factor de traspaso de costo a descuento)
    
  return round(precio_reserva * (1 - factor_descuento), -5)

def evaluar_reasignacion_canal(row: pd.Series) -> str:
  """Determina la estrategia óptima de canal comercial por vehículo con base en
  el tiempo en inventario y la liquidez de mercado obtenida del RUNT.
  """
  dias = row['Dias_en_Inventario']
  liquidez = row.get('Demanda_Mercado_RUNT', 'Liquidez Media')

  if dias > 45:
    if liquidez == 'Baja Liquidez':
      return 'Reasignar a Concesionario Aliado (Liquidación)'
    else:
      return 'Subasta Virtual con Ajuste Agresivo (-10%)'
  elif dias > 30:
    if liquidez == 'Baja Liquidez':
      return 'Subasta Virtual con Ajuste Moderado (-5%)'
    else:
      return 'Mantener Subasta Virtual (Precio Objetivo)'
  else:
    return 'Canal Óptimo - Prioridad Alta'


def ejecutar_motor_pricing_dinamico(
    df_inventario: pd.DataFrame, df_ref_mercado: pd.DataFrame
) -> pd.DataFrame:
  """Función principal del Módulo D:

  Recibe el inventario enriquecido con costo de tenencia y lo cruza con las
  referencias de mercado para aplicar la matriz de pricing y reasignación.
  """
  # 1. Cruce con referencia de mercado RUNT / Fasecolda (si aún no está cruzado)
  if 'Demanda_Mercado_RUNT' not in df_inventario.columns:
    df_consolidado = pd.merge(
        df_inventario, df_ref_mercado, on=['Marca', 'Linea', 'Modelo'], how='left'
    )
  else:
    df_consolidado = df_inventario.copy()

  # Completar valores nulos por defecto en caso de no haber match exacto
  df_consolidado['Demanda_Mercado_RUNT'] = df_consolidado[
      'Demanda_Mercado_RUNT'
  ].fillna('Liquidez Media')

  # 2. Aplicar matriz de decisiones
  df_consolidado['Precio_Reserva_Sugerido_COP'] = df_consolidado.apply(
      sugerir_precio_reserva_optimo, axis=1
  )
  df_consolidado['Estrategia_Canal_Recomendada'] = df_consolidado.apply(
      evaluar_reasignacion_canal, axis=1
  )

  # 3. Calcular impacto potencial en recupero de capital
  df_consolidado['Diferencia_Precio_Sugerido_COP'] = (
      df_consolidado['Precio_Reserva_COP']
      - df_consolidado['Precio_Reserva_Sugerido_COP']
  )

  return df_consolidado


if __name__ == '__main__':
  # Ejemplo de ejecución
  df_inv = pd.read_csv('data/data/raw/inventario_subastas_simon.csv')
  ref_mercado = pd.read_csv('data/data/external/ref_mercado_runt_fasecolda.csv')

  df_resultado = ejecutar_motor_pricing_dinamico(df_inv, ref_mercado)
  print(df_resultado['Estrategia_Canal_Recomendada'].value_counts())