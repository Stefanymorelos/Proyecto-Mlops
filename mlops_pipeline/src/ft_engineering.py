"""
Feature Engineering
--------------------
Transformaciones y creación de variables para el modelo de riesgo de mora
(Pago_atiempo) sobre la base de créditos.

Diseñado como una cadena de Transformers de scikit-learn (BaseEstimator +
TransformerMixin) para que todo el flujo de limpieza + features sea
reproducible, testeable por partes, y reutilizable dentro de un Pipeline.

Este archivo se construye de forma incremental en 3 PR:
  PR1 (este bloque): limpieza base -> nulos, outliers, columnas irrelevantes
  PR2: variables derivadas y manejo de categorias
  PR3: ensamblaje del pipeline completo + separacion train/test
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# PR1: Limpieza base (nulos + outliers)
# ---------------------------------------------------------------------------

class ColumnasIrrelevantes(BaseEstimator, TransformerMixin):
    """
    Elimina columnas que no deben usarse como variables explicativas.

    En este dataset, 'puntaje' tiene fuga de informacion: su correlacion con
    el target (Pago_atiempo) es de 0.92, mientras que con el score real de
    la central de riesgo (puntaje_datacredito) es de apenas 0.09. El valor
    de relleno (95,227787) nunca aparece en un credito que cayo en mora, lo
    que confirma que la variable ya "conoce" el resultado -> no se debe usar
    como predictor (ver Entregable 2, Hallazgo 1).
    """

    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop if cols_to_drop is not None else ["puntaje"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors="ignore")


class ColumnasNulos(BaseEstimator, TransformerMixin):
    """
    Corrige valores invalidos/corruptos que en realidad representan datos
    faltantes, dejandolos como NaN explicito para que Imputacion los trate
    despues con criterio de negocio (nunca se rellenan a ciegas).

    Cubre:
    - tendencia_ingresos: 58 filas con numeros sueltos en vez de categoria
      (Creciente/Decreciente/Estable). Se valido que no es un corrimiento
      de columnas (todas las filas tienen 23 campos); son errores de
      captura aislados (Hallazgo 2).
    - salario_cliente: filas con salario = 0 (pero credito real aprobado,
      lo cual es incompatible con la verificacion de capacidad de pago) o
      con cuota_pactada > salario_cliente (35 filas en total, Hallazgo 8).
    """

    CATEGORIAS_VALIDAS_TENDENCIA = ["Creciente", "Decreciente", "Estable"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # tendencia_ingresos: valores no categoricos -> NaN
        if "tendencia_ingresos" in X.columns:
            mask_invalida = (
                ~X["tendencia_ingresos"].isin(self.CATEGORIAS_VALIDAS_TENDENCIA)
                & X["tendencia_ingresos"].notna()
            )
            X.loc[mask_invalida, "tendencia_ingresos"] = np.nan

        # salario_cliente: cero, o menor que la cuota pactada -> NaN
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
    Corrige (no elimina) el bloque de filas con edad distorsionada por un
    error sistematico de +100 anios (ej. 122 en vez de 22). El patron es
    matematicamente exacto: edad_registrada - 100 da una distribucion de
    edades normal (desviacion de solo 0.2 anios), por lo que se corrige
    con alta confianza en vez de descartar las filas (Hallazgo 7).

    Tambien marca esas filas con 'lote_datos_sospechoso', porque el mismo
    bloque tiene salario_cliente y total_otros_prestamos con factores de
    escala inconsistentes entre si (29.4x vs 16.1x) -> ese dato de ingreso
    no se corrige (no hay formula confiable), solo se deja senializado
    para que analisis posteriores lo puedan excluir si dependen del ingreso.
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
    Imputa nulos con logica de negocio diferenciada por variable, en vez de
    una unica estrategia generica (Hallazgo 3 y 5):

    - saldo_mora_codeudor: el 99.97% de los valores no nulos son 0 (casi
      nadie tiene codeudor en mora) -> se imputa con 0.
    - saldo_principal: cuando saldo_total y saldo_mora si existen, se
      deriva matematicamente con la identidad contable ya validada
      (saldo_total ~= saldo_principal + saldo_mora), en vez de imputar
      con la mediana a ciegas.
    - saldo_mora / saldo_total: si de verdad no hay ninguna pista (los 156
      casos donde los 4 campos de saldo faltan a la vez), se imputan con
      la mediana como ultimo recurso, ya que un modelo de sklearn no puede
      entrenarse con NaN.
    - puntaje_datacredito: nulo unicamente en clientes sin ningun rastro
      previo (cant_creditosvigentes=0 y huella_consulta=0, su primer
      credito) -> se imputa con la mediana general como piso conservador,
      documentando que representa "score minimo esperado sin historial".
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
            # ultimo recurso, ya con saldo_total/saldo_mora imputados arriba
            X["saldo_principal"] = X["saldo_principal"].fillna(
                X["saldo_total"] - X["saldo_mora"]
            )
        if "puntaje_datacredito" in X.columns:
            X["puntaje_datacredito"] = X["puntaje_datacredito"].fillna(
                self.median_puntaje_datacredito_
            )

        return X


# ---------------------------------------------------------------------------
# Pipeline PR1 (se ira extendiendo en PR2 y PR3)
# ---------------------------------------------------------------------------

pipeline_basemodel_pr1 = Pipeline(steps=[
    ("columnas_irrelevantes", ColumnasIrrelevantes(cols_to_drop=["puntaje"])),
    ("columnas_nulos", ColumnasNulos()),
    ("outliers", Outliers()),
    ("imputacion", Imputacion()),
])


if __name__ == "__main__":
    print("Modulo de feature engineering - PR1 (limpieza base)")
