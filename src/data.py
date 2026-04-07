# src/data.py
import kagglehub
import pandas as pd

def descargar_y_cargar_datos():
    # Código extraído del notebook original
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")
    print("Datos descargados en:", path)

    train = pd.read_csv(f"{path}/sign_mnist_train.csv")
    return train