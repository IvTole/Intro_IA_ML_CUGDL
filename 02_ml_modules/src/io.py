
# Programacion orientada a objetos

import pandas as pd

from src.config import SEED, TRAIN_DATA_PATH, TARGET, FEATURES
from src.preprocessor import build_preprocessor

class Dataset:

    # Constructor
    # Atributos
    def __init__(self, num_samples: int = None, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed

    # Importacion de datos (método)
    def load_data(self):

        df = pd.read_csv(filepath_or_buffer=TRAIN_DATA_PATH)

        # Valores nulos
        df = df.dropna()

        return df
    
    def load_xy(self):

        df = self.load_data()

        X = df[FEATURES]
        y = df[TARGET]

        return X, y


