from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import NUM_FEATURES, CAT_FEATURES, GEO_FEATURES

def build_preprocessor() -> ColumnTransformer:

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_FEATURES),
            ('cat', OneHotEncoder(), CAT_FEATURES)
        ],
        remainder='passthrough'
    )

    return preprocessor