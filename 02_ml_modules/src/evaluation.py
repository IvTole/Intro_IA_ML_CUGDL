import pandas as pd
import numpy as np

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error

# modules
from src.config import SEED

class ModelEvaluation:

    def __init__(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, shuffle: bool = True, seed:int = SEED):

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=shuffle,
            random_state=seed
        )

    def evaluate_model(self, model):

        model_type = type(model.named_steps["model"]).__name__
        print(f"Model Type: {model_type}")

        # Entrenamiento (con datos de entrenamiento, X_train, y_train)
        model.fit(self.X_train, self.y_train)

        # Predicciones (con datos de validacion, X_valid)
        y_pred = model.predict(self.X_valid)

        # Evaluacion (métricas)
        r2 = r2_score(y_pred=y_pred, y_true=self.y_valid)
        mse = mean_squared_error(y_pred=y_pred, y_true=self.y_valid)
        rmse = root_mean_squared_error(y_pred=y_pred, y_true=self.y_valid)
        mae = mean_absolute_error(y_pred=y_pred, y_true=self.y_valid)

        print(f"R2: {r2:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")

        