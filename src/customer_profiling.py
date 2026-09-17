import pandas as pd
import numpy as np

def analizar_compradores(df: pd.DataFrame) -> dict:
    """
    Realiza la segmentación por tipo de comprador, cálculo de Pareto (Top 20%)
    y tasa de recompra sobre la flota adjudicada.
    """
    df_vendidos = df[df['Estado_Subasta'] == 'Vendido'].copy()
    
    # 1. Segmentación por Tipo
    df_vendidos['Margen_COP'] = df_vendidos['Precio_Final_Venta_COP'] - df_vendidos['Costo_Adquisicion_COP']
    
    segmentacion = df_vendidos.groupby('Tipo_Comprador').agg(
        Vehiculos=('ID_Vehiculo', 'count'),
        Volumen_Ventas_COP=('Precio_Final_Venta_COP', 'sum'),
        Margen_Acumulado_COP=('Margen_COP', 'sum')
    ).reset_index()
    
    segmentacion['%_Volumen'] = (segmentacion['Volumen_Ventas_COP'] / segmentacion['Volumen_Ventas_COP'].sum()) * 100
    segmentacion['%_Margen'] = (segmentacion['Margen_Acumulado_COP'] / segmentacion['Margen_Acumulado_COP'].sum()) * 100
    
    # 2. Pareto Top 20%
    pareto_df = df_vendidos.groupby('ID_Comprador').agg(
        Tipo_Comprador=('Tipo_Comprador', 'first'),
        Vehiculos_Comprados=('ID_Vehiculo', 'count'),
        Volumen_Ventas_COP=('Precio_Final_Venta_COP', 'sum'),
        Margen_Total_COP=('Margen_COP', 'sum')
    ).sort_values(by='Volumen_Ventas_COP', ascending=False).reset_index()
    
    total_compradores = len(pareto_df)
    top_20_count = int(np.ceil(0.20 * total_compradores))
    top_20 = pareto_df.head(top_20_count)
    
    pct_volumen_top20 = (top_20['Volumen_Ventas_COP'].sum() / pareto_df['Volumen_Ventas_COP'].sum()) * 100
    pct_margen_top20 = (top_20['Margen_Total_COP'].sum() / pareto_df['Margen_Total_COP'].sum()) * 100
    
    # 3. Tasa de Recompra
    recompradores = pareto_df[pareto_df['Vehiculos_Comprados'] > 1]
    tasa_recompra = (len(recompradores) / total_compradores) * 100
    
    return {
        'segmentacion': segmentacion.to_dict(orient='records'),
        'total_compradores_unicos': total_compradores,
        'pareto_top20_pct_volumen': round(pct_volumen_top20, 2),
        'pareto_top20_pct_margen': round(pct_margen_top20, 2),
        'tasa_recompra_pct': round(tasa_recompra, 2)
    }

if __name__ == '__main__':
    from data_processing import cargar_y_procesar_datos
    df = cargar_y_procesar_datos('data/data/raw/inventario_subastas_simon.csv')
    resumen = analizar_compradores(df)
    print("Resumen Módulo C:", resumen)