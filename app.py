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
from src.customer_profiling import analizar_compradores
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

# =========================================================
# MÓDULO A: ANALÍTICA DE SUBASTAS & RECUPERACIÓN DE CAPITAL
# =========================================================
with tab_a:
    st.header(
        'Analítica de Subastas, Eficiencia de Puja y Recuperación de Capital'
    )

    # 1. KPIs Principales usando las métricas de data_processing.py
    total_subastados = len(df_filtrado)
    win_rate_global = df_filtrado['Win_Rate_Flag'].mean() * 100

    # Promedios excluyendo NaN (los NaN corresponden a vehículos no vendidos/sin adjudicar)
    recuperacion_promedio = df_filtrado['Porcentaje_Recuperacion'].mean()
    spread_promedio = df_filtrado['Spread_Puja_COP'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Vehículos', f'{total_subastados:,}')
    c2.metric('Win Rate Global', f'{win_rate_global:.1f}%')
    c3.metric(
        '% Recuperación Prom.',
        (
            f'{recuperacion_promedio:.1f}%'
            if pd.notnull(recuperacion_promedio)
            else '0.0%'
        ),
    )
    c4.metric(
        'Spread de Puja Prom.',
        (
            f'${spread_promedio:,.0f} COP'
            if pd.notnull(spread_promedio)
            else '$0 COP'
        ),
    )

    st.markdown('---')

    # 2. Análisis Visual de Desempeño
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.subheader('Win Rate y Recuperación por Canal de Asignación')

        # Agrupación por canal basada en Win_Rate_Flag y Porcentaje_Recuperacion
        df_canal = (
            df_filtrado.groupby('Canal_Asignado')
            .agg(
                Win_Rate=('Win_Rate_Flag', lambda x: x.mean() * 100),
                Pct_Recuperacion=('Porcentaje_Recuperacion', 'mean'),
                Total_Vehiculos=('Win_Rate_Flag', 'count'),
            )
            .reset_index()
        )

        fig_canal = px.bar(
            df_canal,
            x='Canal_Asignado',
            y='Win_Rate',
            text_auto='.1f',
            color='Pct_Recuperacion',
            color_continuous_scale='Greens',
            labels={
                'Win_Rate': 'Win Rate (%)',
                'Canal_Asignado': 'Canal de Asignación',
                'Pct_Recuperacion': '% Recuperación Promedio',
            },
            title='Conversión (Win Rate) y Retorno por Canal',
        )
        fig_canal.update_traces(
            textposition='outside', texttemplate='%{y:.1f}%'
        )
        st.plotly_chart(fig_canal, width='stretch')

    with col_a2:
        st.subheader('Evaluación de Fricción: Spread de Puja vs. Reserva')

        # Filtramos solo los vendidos para analizar la puja efectiva
        df_vendidos = df_filtrado[
            df_filtrado['Estado_Subasta'] == 'Vendido'
        ].copy()

        if not df_vendidos.empty:
            fig_spread = px.scatter(
                df_vendidos,
                x='Precio_Reserva_COP',
                y='Spread_Puja_COP',
                color='Categoria_Fasecolda',
                size='Costo_Adquisicion_COP',
                hover_data=[
                    'Marca',
                    'Linea',
                    'Modelo',
                    'Precio_Final_Venta_COP',
                ],
                labels={
                    'Precio_Reserva_COP': 'Precio de Reserva (COP)',
                    'Spread_Puja_COP': 'Spread de Puja (Precio Venta - Reserva)',
                    'Categoria_Fasecolda': 'Clasificación',
                },
                title='Excedente de Oferta (Spread) por Precio de Reserva',
            )
            # Línea de referencia horizontal en Spread = 0
            fig_spread.add_hline(
                y=0,
                line_dash='dash',
                line_color='red',
                annotation_text='Punto de Cierre (Venta = Reserva)',
            )
            st.plotly_chart(fig_spread, width='stretch')
        else:
            st.info(
                'No hay registros de vehículos vendidos bajo el filtro seleccionado para calcular el Spread.'
            )

    # 3. Diagnóstico de Pérdida de Margen por Categoria Fasecolda
    st.subheader('Factores de Pérdida de Margen por Clasificación de Precio')

    df_fasecolda_analysis = (
        df_filtrado.groupby('Categoria_Fasecolda')
        .agg(
            Total=('Win_Rate_Flag', 'count'),
            Win_Rate=('Win_Rate_Flag', lambda x: x.mean() * 100),
            Recuperacion=('Porcentaje_Recuperacion', 'mean'),
            Spread=('Spread_Puja_COP', 'mean'),
        )
        .reset_index()
    )

    st.dataframe(
        df_fasecolda_analysis.style.format({
            'Win_Rate': '{:.2f}%',
            'Recuperacion': '{:.2f}%',
            'Spread': '${:,.0f} COP',
        }),
        width='stretch',
    )

# =========================================================
# MÓDULO B: BENCHMARKING DE INVENTARIO VS. MERCADO
# =========================================================
with tab_b:
    st.header(
        'Benchmarking de Mercado, Categorización Fasecolda & Holding Cost'
    )

    # 1. Métricas Agregadas del Módulo B
    holding_total = df_filtrado['Holding_Cost_COP'].sum()
    ratio_costo_prom = df_filtrado['Ratio_Costo_Fasecolda'].mean() * 100
    ratio_reserva_prom = df_filtrado['Ratio_Reserva_Fasecolda'].mean() * 100

    pct_overpriced = (
        (df_filtrado['Categoria_Fasecolda'] == 'Overpriced').mean() * 100
    )

    mb1, mb2, mb3 = st.columns(3)
    mb1.metric('Holding Cost Total', f'${holding_total:,.0f} COP')
    mb2.metric(
        'Ratio Costo / Fasecolda',
        f'{ratio_costo_prom:.1f}%',
        help='Costo de Adquisición vs. Valor Comercial Fasecolda',
    )
    mb3.metric(
        'Ratio Reserva / Fasecolda',
        f'{ratio_reserva_prom:.1f}%',
        help='Precio de Reserva vs. Valor Comercial Fasecolda',
    )

    st.markdown('---')

    # 2. Análisis por Categoria_Fasecolda
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.subheader('Estructura de Precios por Categoria Fasecolda')

        # Comparación de Promedios Monetarios
        df_cat_summary = (
            df_filtrado.groupby('Categoria_Fasecolda')
            .agg(
                Valor_Fasecolda=('Valor_Fasecolda_COP', 'mean'),
                Costo_Adquisicion=('Costo_Adquisicion_COP', 'mean'),
                Precio_Reserva=('Precio_Reserva_COP', 'mean'),
                Total_Vehiculos=('Categoria_Fasecolda', 'count'),
            )
            .reset_index()
        )

        fig_bench = px.bar(
            df_cat_summary,
            x='Categoria_Fasecolda',
            y=['Valor_Fasecolda', 'Costo_Adquisicion', 'Precio_Reserva'],
            barmode='group',
            labels={
                'value': 'COP Promedio',
                'variable': 'Métrica',
                'Categoria_Fasecolda': 'Categoría Fasecolda',
            },
            title='Fasecolda vs. Costo Adquisición vs. Precio Reserva',
            color_discrete_map={
                'Valor_Fasecolda': '#2b5c8f',
                'Costo_Adquisicion': '#d95f02',
                'Precio_Reserva': '#7570b3',
            },
        )
        fig_bench.update_layout(legend_title_text='Métrica')
        st.plotly_chart(fig_bench, width='stretch')

    with col_b2:
        st.subheader('Distribución del Inventario por Categoría')

        fig_pie_cat = px.pie(
            df_cat_summary,
            names='Categoria_Fasecolda',
            values='Total_Vehiculos',
            hole=0.4,
            title='Proporción de Stock por Tipo de Precio',
            color='Categoria_Fasecolda',
            color_discrete_map={
                'Overpriced': '#ef553b',
                'Fair Value': '#636efa',
                'Underpriced': '#00cc96',
            },
        )
        st.plotly_chart(fig_pie_cat, width='stretch')

    st.markdown('---')

    # 3. Análisis de Antigüedad (Aging) & Holding Cost
    col_b3, col_b4 = st.columns(2)

    with col_b3:
        st.subheader('Holding Cost Acumulado por Tramo de Aging')

        # Agrupación por Tramo_Aging utilizando Holding_Cost_COP
        df_aging = (
            df_filtrado.groupby('Tramo_Aging', observed=False)
            .agg(
                Holding_Cost_Total=('Holding_Cost_COP', 'sum'),
                Vehiculos=('Holding_Cost_COP', 'count'),
                Holding_Promedio=('Holding_Cost_COP', 'mean'),
            )
            .reset_index()
        )

        fig_aging_cost = px.bar(
            df_aging,
            x='Tramo_Aging',
            y='Holding_Cost_Total',
            text_auto=',.0f',
            #color='Holding_Cost_Total',
            #color_continuous_scale='Reds',
            labels={
                'Tramo_Aging': 'Tramo de Días en Inventario',
                'Holding_Cost_Total': 'Holding Cost Total (COP)',
            },
            title='Erosión Financiera Acumulada por Tramo de Tiempo',
        )
        fig_aging_cost.update_traces(textposition='outside')
        st.plotly_chart(fig_aging_cost, width='stretch')

    with col_b4:
        st.subheader('Impacto de Aging en la Categoria Fasecolda')

        # Composición de Categorías según el Tramo de Aging
        fig_aging_cat = px.histogram(
            df_filtrado,
            x='Tramo_Aging',
            color='Categoria_Fasecolda',
            barmode='stack',
            title='Evolución del Stock Overpriced/Underpriced por Días',
            labels={
                'Tramo_Aging': 'Tramo de Aging',
                'count': 'Cantidad de Vehículos',
            },
            color_discrete_map={
                'Overpriced': '#ef553b',
                'Fair Value': '#636efa',
                'Underpriced': '#00cc96',
            },
        )
        st.plotly_chart(fig_aging_cat, width='stretch')

# =========================================================
# MÓDULO C: PERFILACIÓN & COMPORTAMIENTO DE COMPRADORES
# =========================================================
with tab_c:
    st.header(
        'Perfilación, Concentración & Arquetipos de Compradores'
    )

    # Validar que existan ventas en el dataset filtrado
    df_vendidos_filtrados = df_filtrado[
        df_filtrado['Estado_Subasta'] == 'Vendido'
    ]

    if not df_vendidos_filtrados.empty:
        # 1. Ejecutar el análisis centralizado de src/customer_profiling.py
        resumen_c = analizar_compradores(df_filtrado)

        # KPIs Principales extraídos de analizar_compradores()
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(
            'Compradores Únicos', f"{resumen_c['total_compradores_unicos']:,}"
        )
        mc2.metric(
            'Tasa de Recompra', f"{resumen_c['tasa_recompra_pct']:.2f}%"
        )
        mc3.metric(
            'Volumen Top 20% Clientes',
            f"{resumen_c['pareto_top20_pct_volumen']:.2f}%",
            help='% del volumen monetario adjudicado al Top 20% de compradores',
        )
        mc4.metric(
            'Margen Top 20% Clientes',
            f"{resumen_c['pareto_top20_pct_margen']:.2f}%",
            help='% del margen acumulado generado por el Top 20% de compradores',
        )

        st.markdown('---')

        # 2. Convertir la segmentación en DataFrame para gráficos de Streamlit/Plotly
        df_segmentacion = pd.DataFrame(resumen_c['segmentacion'])

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.subheader('Aporte por Tipo de Comprador (% Volumen vs % Margen)')

            fig_seg = px.bar(
                df_segmentacion,
                x='Tipo_Comprador',
                y=['%_Volumen', '%_Margen'],
                barmode='group',
                title='Participación en Volumen monetario y Margen Total',
                labels={
                    'value': 'Porcentaje (%)',
                    'variable': 'Métrica',
                    'Tipo_Comprador': 'Tipo de Comprador',
                },
                color_discrete_sequence=['#1f77b4', '#2ca02c'],
            )
            fig_seg.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            st.plotly_chart(fig_seg, width='stretch')

        with col_c2:
            st.subheader('Distribución de Flota Adjudicada por Arquetipo')

            fig_pie_arquetipo = px.pie(
                df_segmentacion,
                names='Tipo_Comprador',
                values='Vehiculos',
                title='Proporción de Vehículos Comprados por Segmento',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_pie_arquetipo, width='stretch')

        st.markdown('---')

        # 3. Detalle de Arquetipos y Concentración
        st.subheader('Tabla Comparativa por Arquetipo de Comprador')

        # Calculamos el margen promedio por vehículo dentro del segmento
        df_segmentacion['Margen_Promedio_COP'] = (
            df_segmentacion['Margen_Acumulado_COP'] / df_segmentacion['Vehiculos']
        )
        df_segmentacion['Ticket_Promedio_COP'] = (
            df_segmentacion['Volumen_Ventas_COP'] / df_segmentacion['Vehiculos']
        )

        st.dataframe(
            df_segmentacion.style.format({
                'Vehiculos': '{:,}',
                'Volumen_Ventas_COP': '${:,.0f} COP',
                'Margen_Acumulado_COP': '${:,.0f} COP',
                '%_Volumen': '{:.2f}%',
                '%_Margen': '{:.2f}%',
                'Margen_Promedio_COP': '${:,.0f} COP',
                'Ticket_Promedio_COP': '${:,.0f} COP',
            }),
            width='stretch',
        )

    else:
        st.warning(
            'No se encontraron vehículos vendidos/adjudicados para calcular las métricas de compradores bajo este filtro.'
        )