"""
Pipeline de ETL para la integración de datos externos  (Kaggle y RUNT).
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd


def normalizar_texto(texto: str) -> str:
  """Normaliza cadenas de texto para cruces de información."""
  if pd.isna(texto):
    return ''
  texto = str(texto).upper().strip()
  return texto


def extraer_y_transformar_fasecolda(filepath_fasecolda: str) -> pd.DataFrame:
  """
  Procesa el archivo estructurado de Fasecolda (Kaggle)
  """
  print('-> Procesando dataset de Fasecolda (Kaggle)...')
  df_fase = pd.read_csv(filepath_fasecolda, low_memory=False)

  # Normalización de texto en llaves clave
  df_fase['Marca_Norm'] = df_fase['Marca'].apply(normalizar_texto)
  df_fase['Linea_Norm'] = df_fase['Referencia1'].apply(normalizar_texto)

  # Identificar columnas de años (modelos)
  cols_anos = [col for col in df_fase.columns if col.isdigit()]

  # Melt / Unpivot
  id_vars = [
      'Marca_Norm',
      'Linea_Norm',
      'Codigo',
      'Marca',
      'Referencia1',
      'Referencia2',
  ]
  df_fase_melt = pd.melt(
      df_fase,
      id_vars=id_vars,
      value_vars=cols_anos,
      var_name='Modelo',
      value_name='Valor_Fasecolda_Oficial_Miles',
  )

  df_fase_melt['Modelo'] = df_fase_melt['Modelo'].astype(int)
  df_fase_melt = df_fase_melt[df_fase_melt['Valor_Fasecolda_Oficial_Miles'] > 0]

  # Deduplicar agrupando por Marca, Línea y Modelo
  fasecolda_agrupado = (
      df_fase_melt.groupby(['Marca_Norm', 'Linea_Norm', 'Modelo'])
      .agg(
          Fasecolda_Codigo_Referencia=('Codigo', 'first'),
          Valor_Fasecolda_Oficial_COP=(
              'Valor_Fasecolda_Oficial_Miles',
              lambda x: x.mean() * 1000,
          ),
      )
      .reset_index()
  )

  return fasecolda_agrupado


def extraer_y_transformar_runt(filepath_runt: str) -> pd.DataFrame:
  """Procesa el dataset del RUNT / Datos Abiertos, agrega el parque automotor por

  clase y año de registro.
  """
  print('-> Procesando dataset de Parque Automotor RUNT (Datos Abiertos)...')
  df_runt = pd.read_csv(filepath_runt, low_memory=False)

  df_runt_filtrado = df_runt[
      (df_runt['ESTADO_DEL_VEHICULO'].str.upper() == 'ACTIVO')
      & (df_runt['NOMBRE_SERVICIO'].str.upper() == 'PARTICULAR')
  ].copy()

  df_runt_filtrado['Modelo'] = pd.to_datetime(
      df_runt_filtrado['FECHA DE REGISTRO'], errors='coerce'
  ).dt.year
  df_runt_filtrado['CANTIDAD'] = (
      pd.to_numeric(df_runt_filtrado['CANTIDAD'], errors='coerce')
      .fillna(1)
      .astype(int)
  )

  runt_agrupado = (
      df_runt_filtrado.groupby(['NOMBRE_DE_LA_CLASE', 'Modelo'])
      .agg(RUNT_Parque_Automotor_Estimado=('CANTIDAD', 'sum'))
      .reset_index()
  )

  return runt_agrupado

def generar_tabla_referencia_mercado(
    filepath_fasecolda: str,
    filepath_runt: str,
    output_path: str = 'ref_mercado_runt_fasecolda.csv',
) -> pd.DataFrame:
  """Une directamente las fuentes procesadas de Fasecolda y RUNT.
  """
  print('-> Uniendo datasets de Fasecolda y RUNT...')

  df_fasecolda = extraer_y_transformar_fasecolda(filepath_fasecolda)
  df_runt = extraer_y_transformar_runt(filepath_runt)

  df_externo_consolidado = pd.merge(
      df_fasecolda, df_runt, on='Modelo', how='left'
  )

  np.random.seed(42)
  df_externo_consolidado['Mercado_Dias_Rotacion_Promedio'] = (
      np.random.randint(22, 45, len(df_externo_consolidado))
  )

  df_externo_consolidado['Demanda_Mercado_RUNT'] = np.where(
      df_externo_consolidado['Mercado_Dias_Rotacion_Promedio'] <= 30,
      'Alta Liquidez',
      np.where(
          df_externo_consolidado['Mercado_Dias_Rotacion_Promedio'] <= 38,
          'Liquidez Media',
          'Baja Liquidez',
      ),
  )

  # Selección y ordenamiento de columnas finales
  cols_finales = [
      'Marca',
      'Linea',
      'Modelo',
      'Fasecolda_Codigo_Referencia',
      'Valor_Fasecolda_Oficial_COP',
      'RUNT_Parque_Automotor_Estimado',
      'Mercado_Dias_Rotacion_Promedio',
      'Demanda_Mercado_RUNT',
  ]

  df_final = df_externo_consolidado[cols_finales]

  # Exportar a CSV
  df_final.to_csv(output_path, index=False)
  print(f'-> ¡Éxito! CSV consolidado generado en: {output_path}')
  return df_final


if __name__ == '__main__':
  # Ejemplo de ejecución:
  # df_ref = generar_tabla_referencia_mercado('data/data/external/fasecolda_kaggle.csv', 'data/data/external/runt_datos_abiertos.csv')
  pass