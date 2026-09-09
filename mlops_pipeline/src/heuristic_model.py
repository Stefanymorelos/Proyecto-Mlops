"""
Modelo heuristico de riesgo de credito (baseline explicable).

Combina 4 senales de negocio que ya se validaron con evidencia en
comprension_eda.ipynb (relaciones monotonicas y limpias con la tasa de mora,
Hallazgos del EDA), SIN usar ningun modelo de aprendizaje automatico:

- puntaje_datacredito bajo   -> mas riesgo (relacion monotonica confirmada,
  66.7% de mora en "riesgo alto" vs 3.1% en "excelente")
- huella_consulta alta       -> mas riesgo (relacion monotonica creciente,
  3.6% -> 7.5% de mora segun numero de consultas recientes)
- edad_cliente baja          -> mas riesgo (18-25 anios: 8.5% de mora,
  decrece con la edad)
- tiene_info_ingresos_buro=0 -> mas riesgo (5.73% de mora vs 4.38% con info,
  asociado a informalidad laboral)

Deliberadamente NO se incluye ratio_cuota_ingreso como señal de riesgo: el
EDA encontro que esa variable no tiene relacion clara con la mora (Hallazgo
del EDA "lo que no confirmo la hipotesis"), asi que incluirla aqui seria
inventar una regla de negocio sin evidencia.

El calibrador de probabilidad y los umbrales se ajustan UNICAMENTE con datos
de entrenamiento (fit), nunca con datos de evaluacion -- misma disciplina
que ft_engineering.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


# (nombre, columna, direccion, es_binaria)
# direccion = +1 si "valor mas alto" = mas riesgo; -1 si "valor mas bajo" = mas riesgo
COMPONENTES: tuple[tuple[str, str, int, bool], ...] = (
    ("score_bajo", "puntaje_datacredito", -1, False),
    ("huella_alta", "huella_consulta", 1, False),
    ("edad_baja", "edad_cliente", -1, False),
    ("sin_info_ingresos", "tiene_info_ingresos_buro", -1, True),
)


class ModeloHeuristicoRiesgo(ClassifierMixin, BaseEstimator):
    """
    Clasificador heuristico compatible con sklearn (BaseEstimator + ClassifierMixin).

    No aprende patrones de los datos: combina 4 senales de negocio ya
    validadas en el EDA, convertidas a percentiles de riesgo (ajustados
    solo con el set de entrenamiento) y promediadas. Sirve como "piso
    minimo" que cualquier modelo de aprendizaje automatico debe superar.

    Parameters
    ----------
    review_fraction : float
        Fraccion de la poblacion de entrenamiento que se marca como
        "alto riesgo" en predict(). Ej. 0.20 = el 20% con mayor score
        de riesgo se predice como mora (clase 0).
    pesos_componentes : tuple de 4 floats
        Pesos de cada señal, en el orden de COMPONENTES. Por defecto,
        pesos iguales (1,1,1,1) -- el EDA no encontro evidencia de que
        una señal domine sobre las otras.

    Notes
    -----
    El target es Pago_atiempo: clase 0 = mora, clase 1 = pago a tiempo.
    predict_proba sigue el orden de sklearn: columna 0 = P(mora).
    """

    def __init__(
        self,
        review_fraction: float = 0.20,
        pesos_componentes: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        self.review_fraction = review_fraction
        self.pesos_componentes = pesos_componentes

    # ------------------------------------------------------------------
    # Validacion
    # ------------------------------------------------------------------
    def _validar_parametros(self) -> np.ndarray:
        if not 0 < self.review_fraction < 1:
            raise ValueError("review_fraction debe estar estrictamente entre 0 y 1")
        pesos = np.asarray(self.pesos_componentes, dtype=float)
        if pesos.shape != (len(COMPONENTES),):
            raise ValueError(f"pesos_componentes debe tener {len(COMPONENTES)} valores")
        if not np.isfinite(pesos).all() or (pesos < 0).any() or np.isclose(pesos.sum(), 0.0):
            raise ValueError("pesos_componentes debe ser finito, no negativo y no todo cero")
        return pesos

    def _seleccionar_columnas(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ModeloHeuristicoRiesgo espera un pandas DataFrame")
        requeridas = [col for _, col, _, _ in COMPONENTES]
        faltantes = [c for c in requeridas if c not in X.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas requeridas por el modelo heuristico: {faltantes}")
        return X.loc[:, requeridas].apply(pd.to_numeric, errors="coerce")

    # ------------------------------------------------------------------
    # Calculo de percentiles de riesgo
    # ------------------------------------------------------------------
    @staticmethod
    def _percentil(valores: np.ndarray, referencia: np.ndarray) -> np.ndarray:
        """Percentil de cada valor dentro de la distribucion de referencia (0 a 1)."""
        izq = np.searchsorted(referencia, valores, side="left")
        der = np.searchsorted(referencia, valores, side="right")
        return (izq + der + 1) / (2 * len(referencia))

    def _calcular_componentes(self, X: pd.DataFrame) -> pd.DataFrame:
        seleccion = self._seleccionar_columnas(X)
        salida = pd.DataFrame(index=X.index)
        for nombre, columna, direccion, es_binaria in COMPONENTES:
            valores = seleccion[columna].to_numpy(dtype=float)
            disponible = np.isfinite(valores)
            componente = np.full(len(X), np.nan, dtype=float)
            if es_binaria:
                # variable ya es 0/1: direccion=-1 significa "0 = mas riesgo"
                riesgo = valores if direccion == 1 else (1 - valores)
                componente[disponible] = riesgo[disponible]
            else:
                orientado = direccion * valores[disponible]
                referencia = self.referencias_.get(nombre)
                if referencia is not None and disponible.any():
                    componente[disponible] = self._percentil(orientado, referencia)
                # si no hay referencia de entrenamiento para esta señal, el
                # componente queda en NaN (se trata como "no disponible")
            salida[nombre] = componente
        return salida

    def _promediar_componentes(
        self, componentes: pd.DataFrame, *, rellenar_faltantes: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        valores = componentes.to_numpy(dtype=float)
        disponible = np.isfinite(valores)
        ponderado = np.where(disponible, valores * self.pesos_componentes_, 0.0)
        denominadores = np.where(disponible, self.pesos_componentes_, 0.0).sum(axis=1)
        scores = np.divide(
            ponderado.sum(axis=1), denominadores,
            out=np.full(len(componentes), np.nan), where=denominadores > 0,
        )
        n_disponibles = disponible.sum(axis=1)
        if rellenar_faltantes:
            scores = np.where(np.isfinite(scores), scores, self.score_neutral_)
        return scores, n_disponibles

    # ------------------------------------------------------------------
    # API de sklearn
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y) -> "ModeloHeuristicoRiesgo":
        self.pesos_componentes_ = self._validar_parametros()
        seleccion = self._seleccionar_columnas(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(seleccion):
            raise ValueError("y debe ser unidimensional y tener el mismo largo que X")
        if not np.isin(target, [0, 1]).all():
            raise ValueError("y debe contener unicamente las etiquetas 0 y 1 de Pago_atiempo")

        self.classes_ = np.asarray([0, 1], dtype=np.int8)
        self.feature_names_in_ = np.asarray([c for _, c, _, _ in COMPONENTES], dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)

        # referencias de percentil: solo para las señales no binarias
        self.referencias_: dict[str, np.ndarray] = {}
        for nombre, columna, direccion, es_binaria in COMPONENTES:
            if es_binaria:
                continue
            valores = seleccion[columna].to_numpy(dtype=float)
            disponible = np.isfinite(valores)
            if disponible.any():
                self.referencias_[nombre] = np.sort(direccion * valores[disponible])

        componentes_train = self._calcular_componentes(X)
        scores_train, n_disp = self._promediar_componentes(componentes_train, rellenar_faltantes=False)
        con_score = np.isfinite(scores_train)
        if not con_score.any():
            raise ValueError("Ninguna fila de entrenamiento tiene al menos una señal disponible")

        self.score_neutral_ = float(np.median(scores_train[con_score]))
        scores_train = np.where(np.isfinite(scores_train), scores_train, self.score_neutral_)
        self.umbral_revision_ = float(
            np.quantile(scores_train, 1 - self.review_fraction, method="higher")
        )
        self.scores_entrenamiento_ = scores_train

        # calibracion de probabilidad: regresion logistica 1D sobre el score de riesgo
        es_mora = (target == 0).astype(np.int8)
        self._calibrador_ = LogisticRegression()
        self._calibrador_.fit(scores_train.reshape(-1, 1), es_mora)

        return self

    def risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """Score de riesgo continuo (0-1, mas alto = mas riesgo), sin calibrar a probabilidad."""
        check_is_fitted(self, attributes=["referencias_", "umbral_revision_"])
        componentes = self._calcular_componentes(X)
        scores, _ = self._promediar_componentes(componentes, rellenar_faltantes=True)
        return scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice Pago_atiempo: 0 (mora) para el review_fraction de mayor riesgo, 1 (a tiempo) el resto."""
        scores = self.risk_score(X)
        return np.where(scores >= self.umbral_revision_, 0, 1).astype(np.int8)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad calibrada. Columna 0 = P(mora), columna 1 = P(a tiempo)."""
        scores = self.risk_score(X)
        proba_mora = self._calibrador_.predict_proba(scores.reshape(-1, 1))[:, 1]
        return np.column_stack([proba_mora, 1 - proba_mora])


if __name__ == "__main__":
    print("Modulo de modelo heuristico de riesgo de credito")
