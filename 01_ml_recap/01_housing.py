import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import pandas as pd # dataframes 
    import numpy as np # tensores, estadisticas, algebra lineal
    import matplotlib.pyplot as plt # graficos

    import geopandas as gpd
    import contextily as ctx

    return ctx, gpd, mo, pd, plt


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
