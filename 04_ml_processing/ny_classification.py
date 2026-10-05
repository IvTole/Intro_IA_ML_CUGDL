import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from datetime import datetime, timezone
    import requests
    import json
    import os

    # sklearn
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import KNNImputer, SimpleImputer

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    return (
        ColumnTransformer,
        HTTPAdapter,
        KNNImputer,
        OneHotEncoder,
        Path,
        Pipeline,
        Retry,
        SimpleImputer,
        StandardScaler,
        datetime,
        json,
        mo,
        np,
        os,
        pd,
        requests,
        timezone,
    )


@app.cell
def _(Path):
    Path(__file__).resolve().parent
    return


@app.cell
def _(HTTPAdapter, Path, Retry, datetime, json, os, requests, timezone):
    API = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
    COLUMNS = ["unique_key", "created_date", "closed_date", "complaint_type",
               "borough", "open_data_channel_type", "location_type"]
    QUERY = {
        "$select": ",".join(COLUMNS),
        "$where": "created_date >= '2025-01-01T00:00:00' AND "
                  "created_date < '2025-02-01T00:00:00' AND unique_key like '%0'",
        "$order": "created_date ASC, unique_key ASC",
    }
    CACHE = Path(__file__).resolve().parent / "data" / "nyc311_enero2025.json"

    def load_data():
        if CACHE.exists():
            payload = json.loads(CACHE.read_text())
            if payload["query"] != QUERY or payload["api"] != API:
                raise ValueError("La caché no corresponde a esta consulta. Renómbrala para descargar de nuevo.")
            return payload
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        token = os.environ.get("SOCRATA_APP_TOKEN")
        if token:
            session.headers["X-App-Token"] = token
        rows = []
        for offset in range(0, 100000, 5000):
            response = session.get(API, params={**QUERY, "$limit": 5000,
                                                "$offset": offset}, timeout=90)
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list):
                raise ValueError("Respuesta inesperada de Socrata.")
            rows.extend(page)
            if len(page) < 5000:
                break
        else:
            raise ValueError("Se alcanzó el límite de seguridad; no se guarda una muestra truncada.")
        if not rows:
            raise ValueError("La API no devolvió filas. Revisa la consulta.")
        payload = {"api": API, "query": QUERY,
                   "downloaded_at": datetime.now(timezone.utc).isoformat(), "rows": rows}
        CACHE.parent.mkdir(exist_ok=True)
        temporary = CACHE.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False))
        temporary.replace(CACHE)
        return payload

    return COLUMNS, load_data


@app.cell
def _(mo):
    download = mo.ui.run_button(label="1. Cargar datos (API o caché local)")
    download
    return (download,)


@app.cell
def _(COLUMNS, download, load_data, mo, pd):
    mo.stop(not download.value, mo.md("Pulsa **Cargar datos** para comenzar."))
    with mo.status.spinner(title="Leyendo solicitudes de NYC 311…"):
        payload = load_data()
    raw = pd.DataFrame(payload["rows"]).reindex(columns=COLUMNS)
    mo.vstack([mo.md(f"**{len(raw):,} filas** · Descarga: {payload['downloaded_at']}"),
               mo.ui.table(raw.head(10), selection=None)])
    return (raw,)


@app.cell
def _(raw):
    raw.head()
    return


@app.cell
def _(np, pd, raw):
    # Agragamos y corregimos variables

    # Pasar a tipo datetime
    raw["created_date"] = pd.to_datetime(raw["created_date"])
    raw["closed_date"] = pd.to_datetime(raw["closed_date"])

    # target
    raw["Diff"] = (raw["closed_date"] - raw["created_date"])

    # Filter
    raw_filtered = raw[raw["Diff"].dt.total_seconds() > 150]

    # Target
    raw_filtered["P_24h"] = np.where(raw_filtered["Diff"].dt.days>0, 0, 1)

    # Hora y dia de la semana
    raw_filtered["hour"] = raw_filtered["created_date"].dt.hour
    raw_filtered["weekday"] = raw_filtered["created_date"].dt.day_name()
    return (raw_filtered,)


@app.cell
def _(raw_filtered):
    raw_filtered.head(50)
    return


@app.cell
def _(raw_filtered):
    raw_filtered["P_24h"].value_counts()
    return


@app.cell
def _(raw):
    raw.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tipos de imputacion

    * media
    * moda
    * mediana
    * Imputacion por KNN
    * Interpolacion por métodos regresión
    """)
    return


@app.cell
def _():
    categorical = ["complaint_type", "borough", "open_data_channel_type", "location_type", "weekday"]
    numerical = ["hour"]
    target = ["P_24h"]
    return categorical, numerical, target


@app.cell
def _(
    ColumnTransformer,
    KNNImputer,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    categorical,
    numerical,
    raw_filtered,
    target,
):
    # Codificador
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Escalador
    scaler = StandardScaler()

    # Imputador
    imputer_num = KNNImputer(n_neighbors=5)
    imputer_cat = SimpleImputer(strategy="constant", fill_value="sin_dato")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", imputer_num),("scaler", scaler)]), numerical),
            ("cat", Pipeline([('imputer', imputer_cat),("encoder", encoder)]), categorical),
            ("target", "passthrough", target)
        ]
    )

    preprocessor.set_output(transform="pandas")

    df_clean = preprocessor.fit_transform(raw_filtered)

    df_clean.head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
