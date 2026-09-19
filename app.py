import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.channel_allocation import MotorReasignacionCanal
from src.data_processing import (
    cargar_y_procesar_datos,
    enriquecer_con_fuentes_externas,
)
from src.dynamic_pricing import (
    calcular_elasticidad_descuento,
    ejecutar_motor_pricing_dinamico,
)

# Configuración de la página en Streamlit
st.set_page_config(
    page_title='Dashboard - Optimización Estratégica de Subastas',
    page_icon='📈',
    layout='wide',
)


# 2. Carga de datos optimizada con caché
@st.cache_data
def cargar_datos_completos():
    path_inv = '/workspaces/Caso-de-Uso-Analista-de-Datos/data/data/raw/inventario_subastas_simon.csv'
    path_ref = '/workspaces/Caso-de-Uso-Analista-de-Datos/data/data/external/ref_mercado_runt_fasecolda.csv'

    df_raw = cargar_y_procesar_datos(path_inv)
    df_env = enriquecer_con_fuentes_externas(df_raw, path_ref)
    return df_env


df_inventario = cargar_datos_completos()

# 3. Sidebar - Filtros Globales
st.sidebar.title('🚗 Filtros Estratégicos')

marcas_opt = ['Todas'] + sorted(
    df_inventario['Marca'].dropna().unique().tolist()
)
marca_sel = st.sidebar.selectbox('Seleccionar Marca:', marcas_opt)

if marca_sel != 'Todas':
    df_filtrado = df_inventario[df_inventario['Marca'] == marca_sel].copy()
else:
    df_filtrado = df_inventario.copy()

# Header Principal
st.title('📊 Motor de Optimización de Subastas, Pricing & Canales')
st.markdown(
    'Herramienta de inteligencia de negocio para maximizar la recuperación de capital y reducir la fricción en ventas.'
)

# 4. Navegación por las 4 Pestañas Reestructuradas
tab_a, tab_b, tab_c, tab_d = st.tabs([
    '🔨 Módulo A: Subastas & Recuperación',
    '📊 Módulo B: Benchmarking vs. Mercado',
    '👥 Módulo C: Perfilación de Compradores',
    '🎯 Módulo D: Estrategia Prescriptiva & ML',
])


