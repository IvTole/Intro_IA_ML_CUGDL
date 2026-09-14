from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import NUM_FEATURES, CAT_FEATURES, GEO_FEATURES
from sklearn.preprocessing import MinMaxScaler

def build_preprocessor() -> ColumnTransformer:
    """
    Función que escala columnas numéricas y codifica columnas categóricas,

    Returns: ColumnTransformer. Objeto que condensa el flujo (preprocesamiento)
    de los datos de entrada.

    """

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), NUM_FEATURES),
            ('cat', OneHotEncoder(), CAT_FEATURES)
        ],
        remainder='passthrough'
    )

    return preprocessor