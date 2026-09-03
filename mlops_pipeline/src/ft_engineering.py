"""
Feature Engineering
--------------------
Transformaciones y creación de variables para el modelo de riesgo de mora
(Pago_atiempo) sobre la base de créditos.

Diseñado como una cadena de Transformers de scikit-learn (BaseEstimator +
TransformerMixin) para que todo el flujo de limpieza + features sea
reproducible, testeable por partes, y reutilizable dentro de un Pipeline.

Construido en 3 PR:
  PR1: limpieza base -> nulos, outliers, columnas irrelevantes
  PR2: variables derivadas y manejo de categorias
  PR3 (este bloque): ensamblaje del pipeline completo + separacion train/test
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# PR1: Limpieza base (nulos + outliers)
# ---------------------------------------------------------------------------

class ColumnasIrrelevantes(BaseEstimator, TransformerMixin):
    """
    Elimina columnas que no deben usarse como variables explicativas.

    - 'puntaje': tiene fuga de informacion (corr=0.92 con el target, ver
      Hallazgo 1 en comprension_eda.ipynb).
    - 'fecha_prestamo': fecha cruda en texto, no aporta como feature
      numerico sin transformar; se deja fuera del modelado por ahora.
    - 'tipo_credito': se reemplaza por 'tipo_credito_agrupado' (creada en
      NuevasVariables) para no duplicar la misma señal dos veces.
    """

    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop if cols_to_drop is not None else [
            "puntaje", "fecha_prestamo", "tipo_credito",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors="ignore")


class ColumnasNulos(BaseEstimator, TransformerMixin):
    """
    Corrige valores invalidos que representan datos faltantes, dejandolos
    como NaN explicito para que Imputacion los trate con criterio de
    negocio (Hallazgo 2 y 8).
    """

    CATEGORIAS_VALIDAS_TENDENCIA = ["Creciente", "Decreciente", "Estable"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "tendencia_ingresos" in X.columns:
            mask_invalida = (
                ~X["tendencia_ingresos"].isin(self.CATEGORIAS_VALIDAS_TENDENCIA)
                & X["tendencia_ingresos"].notna()
            )
            X.loc[mask_invalida, "tendencia_ingresos"] = np.nan

        if "salario_cliente" in X.columns and "cuota_pactada" in X.columns:
            mask_cero = X["salario_cliente"] == 0
            con_salario = X["salario_cliente"] > 0
            ratio = pd.Series(np.nan, index=X.index)
            ratio.loc[con_salario] = (
                X.loc[con_salario, "cuota_pactada"] / X.loc[con_salario, "salario_cliente"]
            )
            mask_ratio = ratio > 1
            X.loc[mask_cero | mask_ratio, "salario_cliente"] = np.nan

        return X


class Outliers(BaseEstimator, TransformerMixin):
    """
    Corrige el bloque de 150 filas con edad +100 anios (patron exacto,
    desviacion de 0.2 anios) y marca 'lote_datos_sospechoso' porque el
    salario y otros_prestamos de ese mismo bloque no tienen un factor de
    escala consistente entre si (Hallazgo 7).
    """

    EDAD_MAXIMA_PLAUSIBLE = 90

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "edad_cliente" not in X.columns:
            return X

        idx_bloque = X[X["edad_cliente"] > self.EDAD_MAXIMA_PLAUSIBLE].index
        X.loc[idx_bloque, "edad_cliente"] = X.loc[idx_bloque, "edad_cliente"] - 100

        X["lote_datos_sospechoso"] = 0
        X.loc[idx_bloque, "lote_datos_sospechoso"] = 1

        return X


class Imputacion(BaseEstimator, TransformerMixin):
    """
    Imputa nulos con logica de negocio diferenciada por variable
    (Hallazgo 3 y 5). Los nulos que no tienen una regla de negocio propia
    (salario_cliente, promedio_ingresos_datacredito) se dejan para el
    SimpleImputer generico del ColumnTransformer de modelado (mas abajo).
    """

    def fit(self, X, y=None):
        self.median_saldo_mora_ = X["saldo_mora"].median() if "saldo_mora" in X else np.nan
        self.median_saldo_total_ = X["saldo_total"].median() if "saldo_total" in X else np.nan
        self.median_puntaje_datacredito_ = (
            X["puntaje_datacredito"].median() if "puntaje_datacredito" in X else np.nan
        )
        return self

    def transform(self, X):
        X = X.copy()

        if "saldo_mora_codeudor" in X.columns:
            X["saldo_mora_codeudor"] = X["saldo_mora_codeudor"].fillna(0)

        if {"saldo_principal", "saldo_total", "saldo_mora"}.issubset(X.columns):
            mask_derivable = (
                X["saldo_principal"].isna()
                & X["saldo_total"].notna()
                & X["saldo_mora"].notna()
            )
            X.loc[mask_derivable, "saldo_principal"] = (
                X.loc[mask_derivable, "saldo_total"] - X.loc[mask_derivable, "saldo_mora"]
            )

        if "saldo_mora" in X.columns:
            X["saldo_mora"] = X["saldo_mora"].fillna(self.median_saldo_mora_)
        if "saldo_total" in X.columns:
            X["saldo_total"] = X["saldo_total"].fillna(self.median_saldo_total_)
        if "saldo_principal" in X.columns:
            X["saldo_principal"] = X["saldo_principal"].fillna(
                X["saldo_total"] - X["saldo_mora"]
            )
        if "puntaje_datacredito" in X.columns:
            X["puntaje_datacredito"] = X["puntaje_datacredito"].fillna(
                self.median_puntaje_datacredito_
            )

        return X


# ---------------------------------------------------------------------------
# PR2: Variables derivadas y manejo de categorias
# ---------------------------------------------------------------------------

class NuevasVariables(BaseEstimator, TransformerMixin):
    """
    Construye variables derivadas orientadas al negocio de riesgo de credito
    (ver Entregable 2, seccion "Variables derivadas").
    """

    MAPA_TIPO_CREDITO = {4: "4", 9: "9", 10: "10", 6: "6"}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "promedio_ingresos_datacredito" in X.columns:
            X["tiene_info_ingresos_buro"] = X["promedio_ingresos_datacredito"].notna().astype(int)

        if "tipo_credito" in X.columns:
            X["tipo_credito_agrupado"] = (
                X["tipo_credito"].map(self.MAPA_TIPO_CREDITO).fillna("Otros")
            )

        if {"cuota_pactada", "salario_cliente"}.issubset(X.columns):
            X["ratio_cuota_ingreso"] = X["cuota_pactada"] / X["salario_cliente"]

        if {"total_otros_prestamos", "saldo_total", "salario_cliente"}.issubset(X.columns):
            X["nivel_endeudamiento"] = (
                X["total_otros_prestamos"].fillna(0) + X["saldo_total"].fillna(0)
            ) / X["salario_cliente"]

        return X


class ToCategory(BaseEstimator, TransformerMixin):
    """
    Convierte columnas a tipo 'category' de pandas y rellena
    tendencia_ingresos faltante con "Sin_dato" (OneHotEncoder no maneja
    NaN directamente, y este nulo ya tiene interpretacion de negocio
    propia, ver Hallazgo 4).
    """

    def __init__(self, cols=None):
        self.cols = cols if cols is not None else [
            "tipo_laboral",
            "tipo_credito_agrupado",
            "tendencia_ingresos",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "tendencia_ingresos" in X.columns:
            X["tendencia_ingresos"] = X["tendencia_ingresos"].fillna("Sin_dato")
        for c in self.cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X


class EliminarCategorias(BaseEstimator, TransformerMixin):
    """
    Elimina filas de categorias estadisticamente inmanejables cuando se
    quiere que el modelo no las vea en absoluto. Por defecto no elimina
    nada (tipo_credito_agrupado ya resuelve esto fusionando en "Otros"
    sin perder filas); se deja disponible para casos puntuales futuros.
    """

    def __init__(self, target_col=None, cats_to_drop=None):
        self.target_col = target_col
        self.cats_to_drop = cats_to_drop if cats_to_drop is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.target_col or not self.cats_to_drop:
            return X
        return X[~X[self.target_col].isin(self.cats_to_drop)].copy()


# ---------------------------------------------------------------------------
# PR3: Ensamblaje final + separacion train/test
# ---------------------------------------------------------------------------

pipeline_basemodel = Pipeline(steps=[
    ("columnas_nulos", ColumnasNulos()),
    ("outliers", Outliers()),
    ("imputacion", Imputacion()),
    ("nuevas_variables", NuevasVariables()),
    ("to_category", ToCategory()),
    ("eliminar_categorias", EliminarCategorias()),
    ("columnas_irrelevantes", ColumnasIrrelevantes()),
])


# Variables finales para el modelo, tras limpieza + variables derivadas.
# Se excluyen: identificadores/fecha (fecha_prestamo), columnas con fuga
# de informacion (puntaje) y columnas redundantes (tipo_credito crudo,
# reemplazado por tipo_credito_agrupado). Todo esto ya lo maneja
# ColumnasIrrelevantes arriba.
NUMERIC_FEATURES = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito", "tiene_info_ingresos_buro",
    "lote_datos_sospechoso", "ratio_cuota_ingreso", "nivel_endeudamiento",
]

CATEGORICAL_FEATURES = ["tipo_laboral", "tipo_credito_agrupado", "tendencia_ingresos"]

TARGET_COL = "Pago_atiempo"


class ToDF(BaseEstimator, TransformerMixin):
    """
    Envuelve un ColumnTransformer (imputacion + escalado numerico,
    imputacion + one-hot categorico) y devuelve un DataFrame con nombres
    de columna legibles, en vez del array de numpy que entrega sklearn
    por defecto. Esto facilita interpretar coeficientes/importancias mas
    adelante en model_training_evaluation.py.
    """

    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.ct_ = None

    def fit(self, X, y=None):
        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self.ct_ = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.numeric_features),
                ("cat", categorical_pipe, self.categorical_features),
            ]
        )
        self.ct_.fit(X, y)
        return self

    def transform(self, X):
        Xt = self.ct_.transform(X)
        feat_names = self.ct_.get_feature_names_out()
        return pd.DataFrame(Xt, columns=feat_names, index=X.index)


def build_features(df, target_col=TARGET_COL, test_size=0.25, random_state=42):
    """
    Punto de entrada principal del modulo. Aplica la limpieza y las
    variables derivadas (pipeline_basemodel) sobre el dataframe crudo, y
    separa en conjuntos de entrenamiento/evaluacion estratificados por el
    target, listos para alimentar model_training_evaluation.py.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe crudo, tal como sale de Cargar_datos.ipynb.
    target_col : str
        Nombre de la columna objetivo (Pago_atiempo).
    test_size : float
        Proporcion del set de evaluacion.
    random_state : int
        Semilla para reproducibilidad del split.

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Variables explicativas ya limpias (sin escalar/codificar todavia;
        eso lo hace ToDF/preprocessor, tipicamente dentro del pipeline de
        modelado para evitar fuga de informacion entre train y test).
    y_train, y_test : pd.Series
        Variable objetivo correspondiente a cada conjunto.
    """
    df_limpio = pipeline_basemodel.fit_transform(df)

    y = df_limpio[target_col]
    X = df_limpio.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


# Pipeline de preprocesamiento ML, para usar dentro de model_training_evaluation.py
# junto con el estimador (ej. Pipeline([("preprocessor", preprocessor), ("model", modelo)])).
preprocessor = ToDF(numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES)


if __name__ == "__main__":
    print("Modulo de feature engineering - PR3 (pipeline completo)")
