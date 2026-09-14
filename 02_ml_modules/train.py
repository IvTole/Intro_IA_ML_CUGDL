# PARA EL USUARIO

from src.io import Dataset
from src.evaluation import ModelEvaluation

# Modelos
from src.preprocessor import build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


def train():

    # Importacion y preprocesamiento datos
    data = Dataset(seed=43, num_samples=None)
    X_train, y_train = data.load_xy()

    # Model pipeline (preprocessing + model, Linear Regression)
    pipeline_lr = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestRegressor())
        ]
    )

    # Entrenamiento y métricas
    ev = ModelEvaluation(X=X_train, y=y_train)
    ev.evaluate_model(model=pipeline_lr)

    return

if __name__ == "__main__":
    train()