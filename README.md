# Caso-de-Uso-Analista-de-Datos
Este es el repositorio de la Prueba Técnica y Caso de Uso: Analista de Datos para Finanzauto

## Paso 1
* Creación de devcontainer
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
* Carga de datos internos y externos en las carpetas de data

## Paso 2
* ETL para datos externos (scripts/)
* Limpieza y creación de indices (src/)
* EDA (? - copilot)

## Paso 3
* Perfilamiento de los clientes (src/)
* Modelamiento Pricing (src/)

## Paso ?
* Implementar integración continua (CI)