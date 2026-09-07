import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import pandas as pd # dataframes 
    import numpy as np # tensores, estadisticas, algebra lineal
    import matplotlib.pyplot as plt # graficos
    import seaborn as sns

    import geopandas as gpd
    import contextily as ctx

    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split

    # Modelos (regresion)
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    # Metricas (regresion)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error

    return (
        ColumnTransformer,
        OneHotEncoder,
        RandomForestRegressor,
        StandardScaler,
        ctx,
        gpd,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        mo,
        pd,
        plt,
        r2_score,
        root_mean_squared_error,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Importacion de datos
    """)
    return


@app.cell
def _(pd):
    url = "https://raw.githubusercontent.com/IvTole/Intro_IA_ML_CUGDL/refs/heads/main/data/housing/housing.csv"
    df_housing = pd.read_csv(url)

    # Promediar en terminos de población y numero de casas

    df_housing["total_bedrooms"] = df_housing["total_bedrooms"] / df_housing["households"]
    df_housing["total_rooms"] = df_housing["total_rooms"] / df_housing["households"]

    # df_housing = df_housing.drop(columns=["households"])

    # valores nulos
    df_housing = df_housing.dropna()


    df_housing.head()
    return (df_housing,)


@app.cell
def _(df_housing):
    df_housing.info()
    return


@app.cell
def _(df_housing):
    n_columns =  df_housing.shape[1]
    n_rows = df_housing.shape[0]
    return n_columns, n_rows


@app.cell
def _(mo, n_columns, n_rows):
    mo.hstack(
        [
    mo.stat(n_rows, label="Número de observaciones"),
    mo.stat(n_columns, label="Número de características")
        ],
        widths="equal"
    )
    return


@app.cell
def _(df_housing, gpd):
    ##
    gdf = gpd.GeoDataFrame(
        df_housing,
        geometry=gpd.points_from_xy(
            df_housing["longitude"], df_housing["latitude"]
        ),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    return (gdf,)


@app.cell
def _(ctx, df_housing, gdf, plt):
    ## plot (mapa)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    gdf.plot(
        ax = ax,
        column=df_housing["median_house_value"],
        cmap="RdPu",
        markersize=8,
        legend=True,
        alpha=0.2
    )
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Análisis Exploratorio de Datos (EDA)
    """)
    return


@app.cell
def _(df_housing):
    cols_num = df_housing.select_dtypes(include="number").columns.to_list()
    cols_num.remove("median_house_value")
    return (cols_num,)


@app.cell
def _(cols_num, mo):
    feature_selector = mo.ui.dropdown(
        options=cols_num,
        value="total_bedrooms",
        label="Variable"
    )
    feature_selector
    return (feature_selector,)


@app.cell
def _(df_housing, feature_selector, plt):
    ## variable elegida
    selected_feature = feature_selector.value

    fig_scatter = plt.figure()

    ax_scatter = fig_scatter.add_subplot(111)

    ax_scatter.set_ylabel("Median House Value")
    ax_scatter.set_xlabel(selected_feature)
    ax_scatter.set_title("Diagrama de correlacion con el target")

    ax_scatter.scatter(df_housing[selected_feature], df_housing["median_house_value"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocesamiento
    """)
    return


@app.cell
def _():
    cols_numeric = ['housing_median_age', 'total_rooms',
                'total_bedrooms', 'population', 'households',
                'median_income']
    cols_categoric = ['ocean_proximity']
    return cols_categoric, cols_numeric


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    cols_categoric,
    cols_numeric,
    df_housing,
):
    # Pipeline de preprocesamiento
    encoder = OneHotEncoder(drop=None, sparse_output=False) 
    scaler =  StandardScaler()

    # Objeto ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, cols_numeric),
            ('cat', encoder, cols_categoric)
        ],
        remainder="passthrough"
    )

    # salida a pandas
    preprocessor.set_output(transform="pandas")

    # Ajustarlo con los datos
    preprocessor_fitted = preprocessor.fit(df_housing)

    # Transformar el dataset
    df_housing_trans = preprocessor_fitted.transform(df_housing)

    df_housing_trans
    return (df_housing_trans,)


@app.cell
def _(df_housing_trans, plt, sns):
    # Matriz de correlacion
    plt.figure(figsize=(20,20))
    sns.heatmap(df_housing_trans.corr(method="spearman"),cmap="YlGnBu", annot = True)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento y evaluacion de un modelo de ML (scikit-learn)
    """)
    return


@app.cell
def _(
    RandomForestRegressor,
    df_housing_trans,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    train_test_split,
):
    # Paso 1 - Features, target
    X = df_housing_trans.drop(columns="remainder__median_house_value")
    y = df_housing_trans["remainder__median_house_value"]

    # Paso 2 - Division de entrenamiento / validacion
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.80)

    # Paso 3 - Instanciar modelo
    model = RandomForestRegressor()

    # Paso 4 - Entrenamiento (Training)
    model.fit(X=X_train, y=y_train)

    # Paso 5 - Predicciones
    y_pred = model.predict(X=X_valid)

    # Paso 6 -- Métricas de evaluacion
    mae = mean_absolute_error(y_pred=y_pred, y_true=y_valid)
    mse = mean_squared_error(y_pred=y_pred, y_true=y_valid)
    rmse = root_mean_squared_error(y_pred=y_pred, y_true=y_valid)
    r2 = r2_score(y_pred=y_pred, y_true=y_valid)
    mape = mean_absolute_percentage_error(y_pred=y_pred, y_true=y_valid)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
    print(f"MAPE: {mape}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
