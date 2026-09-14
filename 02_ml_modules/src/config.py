# Configuracion de mis variables utilizadas

# General
SEED = 42

# Path
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"

# FEATURES

GEO_FEATURES = [
    "longitude",
    "latitude"
]

NUM_FEATURES = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CAT_FEATURES = ["ocean_proximity"]

FEATURES =  GEO_FEATURES + NUM_FEATURES + CAT_FEATURES

# TARGET
TARGET = "median_house_value"

