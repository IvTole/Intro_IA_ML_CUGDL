# Create train and test dataset

# Librerías
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split

from config import SEED

# main function
def train_test_dataset(path, filename, seed):
    """
    Arguments:
    Path: localizacion de los datos (path)

    Returns: two files
    train.csv
    test.csv
    """

    fullpath = os.path.join(path, filename)
    
    df = pd.read_csv(fullpath)

    # metodo train test split (scikit learn)
    df_train, df_test = train_test_split(df, test_size=0.20,
                                         shuffle=True,
                                         random_state=seed)
    
    # exportacion
    df_train.to_csv(os.path.join(path,"train.csv"), index = None)
    print("Train data created.")
    df_test.to_csv(os.path.join(path, "test.csv"), index = None)
    print("Test data created.")


    return 

if __name__ == "__main__":
    path = "../data/"
    filename = "housing.csv"
    train_test_dataset(path=path, filename=filename, seed=SEED)