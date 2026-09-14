# PARA EL USUARIO

from src.io import Dataset

def train():

    # Importacion y preprocesamiento datos
    data = Dataset(seed=43)

    X, y = data.load_xy()
    print(X)


    # Entrenamiento y métricas

    return

if __name__ == "__main__":
    train()