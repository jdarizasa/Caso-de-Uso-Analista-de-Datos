# Caso-de-Uso-Analista-de-Datos
Este es el repositorio de la Prueba Técnica y Caso de Uso: Analista de Datos para Finanzauto

## Paso 1
* Creación de devcontainer (.devcontainer)
* Creación de entorno virtual
```
python -m venv ~/.venv
echo "source ~/.venv/bin/activate" >> ~/.bashrc
source ~/.bashrc
```
* Creación de requirements 
```
pip install --upgrade pip && pip install -r requirements.txt
```
* Carga de datos internos y externos en las carpetas de data (data)

## Paso 2
* ETL para datos externos (scripts/etl_fasecolda_runt_pipeline.py)
* Limpieza y creación de indices (src/data_processing.py)
* EDA (? - copilot)

## Paso 3
* Perfilamiento de los clientes (src/customer_profiling.py)
* Modelamiento Pricing (src/dynamic_pricing.py)

    * Algoritmo ajuste precio de reserva:
        * Versión 1: Hardcoded
        ```    
        if dias > 45:       # Compromete margen EBITDA
            factor_descuento = 0.10 if liquidez == 'Baja Liquidez' else 0.07
        elif dias > 30:     # Costo igual o superior a 1.5%
            factor_descuento = 0.05 if liquidez == 'Baja Liquidez' else 0.02
        else:               # Costo menor al 1.5%
            factor_descuento = 0.0
        # Promedio de ganancia es 17%, todos los descuentos propuestos respetan margen de ganancia bruto
        ```
        * Versión 2: Modelo dinámico
        * Benchmark entre V1 y V2 (scripts/benchmark_pricing.py)

    * Reasignación de canales:
        * Versión 1: Hardcoded
        ```
        if dias > 45:
            if liquidez == 'Baja Liquidez':
                return 'Reasignar a Concesionario Aliado (Liquidación)'
            else:
                return 'Subasta Virtual con Ajuste Agresivo Dinámico'
        elif dias > 30:
            if liquidez == 'Baja Liquidez':
                return 'Subasta Virtual con Ajuste Moderado Dinámico'
            else:
                return 'Mantener Subasta Virtual (Precio Objetivo)'
        else:
            return 'Canal Óptimo - Prioridad Alta'
        ```
        * Versión 2: Clasificación por árbol de decisión (src/channel_allocation.py)

## Paso ?
* Implementar integración continua (CI)