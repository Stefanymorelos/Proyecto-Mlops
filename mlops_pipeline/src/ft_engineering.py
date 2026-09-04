"""
Feature Engineering
--------------------
Transformaciones y creación de variables para el modelo de riesgo de mora
(Pago_atiempo) sobre la base de créditos.

Diseñado como una cadena de Transformers de scikit-learn (BaseEstimator +
TransformerMixin) para que todo el flujo de limpieza + features sea
reproducible, testeable por partes, y reutilizable dentro de un Pipeline.

Este archivo se construye de forma incremental en 3 PR:
  PR1: limpieza base -> nulos, outliers, columnas irrelevantes
  PR2 (este bloque): variables derivadas y manejo de categorias
  PR3: ensamblaje del pipeline completo + separacion train/test
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# PR1: Limpieza base (nulos + outliers) — sin cambios respecto al PR anterior
# ---------------------------------------------------------------------------

class ColumnasIrrelevantes(BaseEstimator, TransformerMixin):
    """
    Elimina columnas que no deben usarse como variables explicativas.

    'puntaje' tiene fuga de informacion: correlacion de 0.92 con el target,
    frente a 0.09 con el score real (puntaje_datacredito). El valor de
    relleno (95,227787) nunca aparece en un credito en mora (Hallazgo 1).
    """

    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop if cols_to_drop is not None else ["puntaje"]

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
    (Hallazgo 3 y 5).
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
    (ver Entregable 2, seccion "Variables derivadas"):

    - tiene_info_ingresos_buro: si promedio_ingresos_datacredito no es nulo.
      Asociado a ser cliente Empleado (reporte automatico a la central de
      riesgo) vs Independiente (Hallazgo 4). En el EDA se encontro que los
      clientes SIN esta info tienen mas mora (5.73% vs 4.38%).
    - tipo_credito_agrupado: agrupa los codigos de tipo_credito con muestra
      estadisticamente inmanejable (7 y 68, n=2 y n=1) en "Otros", dejando
      visible el codigo 6 (42.9% de mora, Hallazgo 6) para que el modelo
      pueda aprender esa senal de riesgo.
    - ratio_cuota_ingreso: cuota_pactada / salario_cliente. Indicador
      clasico de capacidad de pago.
    - nivel_endeudamiento: (total_otros_prestamos + saldo_total) / salario_cliente.
      Compromiso total del ingreso con TODAS las deudas del cliente.
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
    Convierte columnas a tipo 'category' de pandas (mas eficiente en
    memoria y explicito para el ColumnTransformer de la etapa de modelado).
    Tambien rellena la categoria faltante de tendencia_ingresos con una
    etiqueta explicita "Sin_dato", ya que OneHotEncoder no maneja NaN
    directamente y este nulo si tiene una interpretacion de negocio propia
    (ver Hallazgo 4 / Hallazgo 2 en comprension_eda.ipynb).
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
    Elimina filas de categorias estadisticamente inmanejables cuando NO se
    quiere que el modelo las vea en absoluto (a diferencia de
    tipo_credito_agrupado, que las funde en "Otros" sin perder las filas).
    Por defecto no elimina nada; se deja disponible para casos puntuales
    (ej. codigos de tipo_credito con 1 sola observacion en todo el dataset).
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
# Pipeline PR2 (se extiende en PR3 con el ensamblaje final + train/test)
# ---------------------------------------------------------------------------

pipeline_basemodel_pr2 = Pipeline(steps=[
    ("columnas_irrelevantes", ColumnasIrrelevantes(cols_to_drop=["puntaje"])),
    ("columnas_nulos", ColumnasNulos()),
    ("outliers", Outliers()),
    ("imputacion", Imputacion()),
    ("nuevas_variables", NuevasVariables()),
    ("to_category", ToCategory()),
    ("eliminar_categorias", EliminarCategorias()),
])


if __name__ == "__main__":
    print("Modulo de feature engineering - PR2 (variables derivadas)")
