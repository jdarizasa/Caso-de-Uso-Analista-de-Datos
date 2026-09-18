import pandas as pd
import numpy as np

# 1. Cargar datasets
df_inv = pd.read_csv('data/data/raw/inventario_subastas_simon.csv')
ref_mercado = pd.read_csv('data/data/external/ref_mercado_runt_fasecolda.csv')

# 2. Cruce de datos con liquidez RUNT
df = pd.merge(df_inv, ref_mercado, on=['Marca', 'Linea', 'Modelo'], how='left')
df['Demanda_Mercado_RUNT'] = df['Demanda_Mercado_RUNT'].fillna('Liquidez Media')

# Parámetros Financieros
TASA_WACC_ANUAL = 0.14
COSTO_PARQUEADERO_DIA = 12000

# Filtrar unidades estancadas / no vendidas (81 No Adjudicados + 16 Retirados)
df_no_vendidos = df[df['Estado_Subasta'].isin(['No Adjudicado', 'Retirado'])].copy()

# 3. ALGORITMO 1: Hardcoded (Reglas Fijas)
def algoritmo_hardcoded(row):
    dias = row['Dias_en_Inventario']
    liquidez = row['Demanda_Mercado_RUNT']
    precio_reserva = row['Precio_Reserva_COP']
    
    if dias > 45:
        factor = 0.10 if liquidez == 'Baja Liquidez' else 0.07
    elif dias > 30:
        factor = 0.05 if liquidez != 'Alta Liquidez' else 0.02
    else:
        factor = 0.0
        
    return round(precio_reserva * (1 - factor), -5)

# 4. ALGORITMO 2: Modelo Financiero Continuo Optimizado
def algoritmo_financiero_continuo(row):
    dias = row['Dias_en_Inventario']
    liquidez = row['Demanda_Mercado_RUNT']
    precio_reserva = row['Precio_Reserva_COP']
    costo_adq = row['Costo_Adquisicion_COP']
    
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

# 5. Ejecución de Modelos
df_no_vendidos['P_Reserva_Hardcoded'] = df_no_vendidos.apply(algoritmo_hardcoded, axis=1)
df_no_vendidos['P_Reserva_Continuo'] = df_no_vendidos.apply(algoritmo_financiero_continuo, axis=1)

# Porcentajes de descuento aplicados
df_no_vendidos['Desc_Pct_Hardcoded'] = (df_no_vendidos['Precio_Reserva_COP'] - df_no_vendidos['P_Reserva_Hardcoded']) / df_no_vendidos['Precio_Reserva_COP']
df_no_vendidos['Desc_Pct_Continuo'] = (df_no_vendidos['Precio_Reserva_COP'] - df_no_vendidos['P_Reserva_Continuo']) / df_no_vendidos['Precio_Reserva_COP']

# Modelo de Elasticidad Precio de la Demanda: Probabilidad de Venta (Win Rate Elasticidad)
# 2.2 es la elasticidad asumida de la subasta, puede ser calculada a través de un modelo de regresión logística con datos históricos de subastas.
df_no_vendidos['Prob_Venta_Hardcoded'] = np.clip(0.50 + 2.2 * df_no_vendidos['Desc_Pct_Hardcoded'], 0, 0.95)
df_no_vendidos['Prob_Venta_Continuo'] = np.clip(0.50 + 2.2 * df_no_vendidos['Desc_Pct_Continuo'], 0, 0.95)

# Costo de Tenencia Acumulado por Vehículo
df_no_vendidos['Costo_Tenencia'] = ((df_no_vendidos['Costo_Adquisicion_COP'] * (TASA_WACC_ANUAL / 365)) + COSTO_PARQUEADERO_DIA) * df_no_vendidos['Dias_en_Inventario']

# Margen Operativo Esperado (Ingresos Esperados - Costo Adquisición Esperado)
df_no_vendidos['Margen_Op_Hardcoded'] = (df_no_vendidos['Prob_Venta_Hardcoded'] * df_no_vendidos['P_Reserva_Hardcoded']) - (df_no_vendidos['Prob_Venta_Hardcoded'] * df_no_vendidos['Costo_Adquisicion_COP'])
df_no_vendidos['Margen_Op_Continuo'] = (df_no_vendidos['Prob_Venta_Continuo'] * df_no_vendidos['P_Reserva_Continuo']) - (df_no_vendidos['Prob_Venta_Continuo'] * df_no_vendidos['Costo_Adquisicion_COP'])

# EBITDA Neto (Margen Operativo Bruto - Costo de Tenencia Total de la Flota)
ebitda_hardcoded = df_no_vendidos['Margen_Op_Hardcoded'].sum() - df_no_vendidos['Costo_Tenencia'].sum()
ebitda_continuo = df_no_vendidos['Margen_Op_Continuo'].sum() - df_no_vendidos['Costo_Tenencia'].sum()

# 6. Tabla Comparativa Resumen
resumen = pd.DataFrame({
    'Métrica de Negocio': [
        'Vehículos No Vendidos (Stock Evaluado)',
        'Descuento Promedio Aplicado (%)',
        'Conversión Proyectada (Win Rate)',
        'Unidades Vendidas Estimadas',
        'Margen Operativo Bruto Rescatado',
        'Costo de Tenencia Acumulado',
        'EBITDA Neto Proyectado del Stock'
    ],
    'Algoritmo Hardcoded (Reglas Fijas)': [
        len(df_no_vendidos),
        f"{df_no_vendidos['Desc_Pct_Hardcoded'].mean()*100:.2f}%",
        f"{df_no_vendidos['Prob_Venta_Hardcoded'].mean()*100:.2f}%",
        f"{df_no_vendidos['Prob_Venta_Hardcoded'].sum():.1f}",
        f"${df_no_vendidos['Margen_Op_Hardcoded'].sum():,.0f} COP",
        f"${df_no_vendidos['Costo_Tenencia'].sum():,.0f} COP",
        f"${ebitda_hardcoded:,.0f} COP"
    ],
    'Modelo Financiero Continuo': [
        len(df_no_vendidos),
        f"{df_no_vendidos['Desc_Pct_Continuo'].mean()*100:.2f}%",
        f"{df_no_vendidos['Prob_Venta_Continuo'].mean()*100:.2f}%",
        f"{df_no_vendidos['Prob_Venta_Continuo'].sum():.1f}",
        f"${df_no_vendidos['Margen_Op_Continuo'].sum():,.0f} COP",
        f"${df_no_vendidos['Costo_Tenencia'].sum():,.0f} COP",
        f"${ebitda_continuo:,.0f} COP"
    ]
})

print(resumen.to_string(index=False))