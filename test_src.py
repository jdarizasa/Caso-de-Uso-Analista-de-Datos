import os
import pytest
import pandas as pd
from src.data_processing import cargar_y_procesar_datos
from src.customer_profiling import analizar_compradores
from src.dynamic_pricing import ejecutar_motor_pricing_dinamico

@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Create minimal test data files if they don't exist"""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/external", exist_ok=True)
    
    # Create minimal test CSV for inventario_subastas_simon.csv
    if not os.path.exists("data/raw/inventario_subastas_simon.csv"):
        test_df = pd.DataFrame({
            'ID_Vehiculo': ['1001', '1002'],
            'Marca': ['Toyota', 'Hyundai'],
            'Linea': ['Hilux', 'Tucson'],
            'Modelo': ['2022', '2024'],
            'Kilometraje': [72036, 43054],
            'Estado_Vehiculo': ['Excelente', 'Excelente'],
            'Costo_Adquisicion_COP': [154800000.0, 93000000.0],
            'Valor_Fasecolda_COP': [190000000.0, 105800000.0],
            'Dias_en_Inventario': [30, 45],
            'Canal_Asignado': ['Subasta Virtual', 'Subasta Virtual'],
            'Precio_Reserva_COP': [153700000.0, 99700000.0],
            'Precio_Final_Venta_COP': [198900000.0, 98800000.0],
            'Estado_Subasta': ['Vendido', 'Vendido'],
            'ID_Comprador': ['153', '116'],
            'Tipo_Comprador': ['Partner', 'Reventa']
        })
        test_df.to_csv("data/data/raw/inventario_subastas_simon.csv", index=False)
    
    # Create minimal test CSV for ref_mercado_runt_fasecolda.csv
    if not os.path.exists("data/external/ref_mercado_runt_fasecolda.csv"):
        ref_df = pd.DataFrame({
            'Marca': ['Chevrolet', 'Chevrolet'],
            'Linea': ['Onix', 'Onix'],
            'Modelo': ['2018', '2019'],
            'RUNT_Parque_Automotor_Estimado': [248654, 240795],
            'Mercado_Dias_Rotacion_Promedio': [32, 30],
            'Demanda_Mercado_RUNT': ['Liquidez Media', 'Alta Liquidez'],
            'Fasecolda_Codigo_Referencia': ['FAS-10000', 'FAS-10001']
        })
        ref_df.to_csv("data/data/external/ref_mercado_runt_fasecolda.csv", index=False)
        
def test_cargar_y_procesar_datos():
    df = cargar_y_procesar_datos("data/data/raw/inventario_subastas_simon.csv")
    assert not df.empty, "DataFrame should not be empty after loading data"
    assert "Marca" in df.columns, "DataFrame should contain 'Marca' column"
    assert "Valor_Fasecolda_COP" in df.columns, "DataFrame should contain 'Valor_Fasecolda_COP' column"

def test_analizar_compradores():
    df = cargar_y_procesar_datos("data/data/raw/inventario_subastas_simon.csv")
    compradores_perfilados = analizar_compradores(df)
    assert "segmentacion" in compradores_perfilados.keys(), "Result should contain 'segmentacion'"

def test_ejecutar_motor_pricing_dinamico():
    df = cargar_y_procesar_datos("data/data/raw/inventario_subastas_simon.csv")
    ref_mercado = pd.read_csv("data/data/external/ref_mercado_runt_fasecolda.csv")
    precios_dinamicos = ejecutar_motor_pricing_dinamico(df, ref_mercado)
    assert not precios_dinamicos.empty, "Precios dinámicos should not be empty"
