# Trabajo 2: Técnicas Avanzadas de la IA

Este repositorio contiene el código para el desarrollo de un modelo de detección de letras por lenguaje de señas (Sign Language MNIST). El objetivo principal del proyecto es utilizar algoritmos de optimización por poblaciones para buscar y definir la mejor arquitectura posible de una Red Neuronal Convolucional (CNN).

## Estructura del Repositorio

```text
Trabajo_2_Tecnicas_Ia/
│
├── dataset/                # Carpeta local para los CSV e imágenes (NO se sube al repo)
├── notebooks/              # Análisis exploratorio inicial y construcción del informe final
│   └── Trabajo2.ipynb
│
├── src/                    # Carpeta con el código fuente (Módulos)
│   ├── __init__.py         # Archivo vacío para que Python reconozca el paquete
│   ├── data.py             # Persona 1: Pipeline de datos, Dataset y DataLoaders
│   ├── modelo.py           # Persona 2: Arquitectura de la CNN dinámica y bucle de entrenamiento
│   └── optimizador.py      # Persona 3: Lógica del algoritmo poblacional (NAS)
│
├── main.py                 # Script principal que importa e integra los tres módulos
├── requirements.txt        # Dependencias del proyecto (torch, pandas, kagglehub, etc.)
└── .gitignore              # Exclusiones de Git (dataset/, venv/, __pycache__/)