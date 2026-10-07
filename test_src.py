# Make a test script for the src module
import pytest
import pandas as pd
from src.data_processing import cargar_y_procesar_datos
from src.customer_profiling import analizar_compradores
from src.dynamic_pricing import ejecutar_motor_pricing_dinamico

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