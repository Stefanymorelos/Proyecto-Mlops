"""
Pruebas unitarias para mlops_pipeline/src/heuristic_model.py
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from heuristic_model import ModeloHeuristicoRiesgo


@pytest.fixture
def datos_entrenamiento():
    """20 filas sinteticas con senales claras de riesgo alto/bajo, y algunos NaN."""
    n = 20
    rng = np.random.RandomState(42)
    puntaje = np.concatenate([rng.uniform(200, 400, n // 2), rng.uniform(750, 950, n // 2)])
    huella = np.concatenate([rng.randint(5, 10, n // 2), rng.randint(0, 2, n // 2)])
    edad = np.concatenate([rng.randint(18, 25, n // 2), rng.randint(40, 60, n // 2)])
    tiene_info = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)  # primera mitad = mora

    X = pd.DataFrame({
        "puntaje_datacredito": puntaje,
        "huella_consulta": huella,
        "edad_cliente": edad,
        "tiene_info_ingresos_buro": tiene_info,
    })
    return X, pd.Series(y)


class TestValidacionParametros:
    def test_review_fraction_fuera_de_rango_lanza_error(self, datos_entrenamiento):
        X, y = datos_entrenamiento
        with pytest.raises(ValueError):
            ModeloHeuristicoRiesgo(review_fraction=1.5).fit(X, y)

    def test_pesos_negativos_lanzan_error(self, datos_entrenamiento):
        X, y = datos_entrenamiento
        with pytest.raises(ValueError):
            ModeloHeuristicoRiesgo(pesos_componentes=(-1, 1, 1, 1)).fit(X, y)

    def test_y_con_etiquetas_invalidas_lanza_error(self, datos_entrenamiento):
        X, _ = datos_entrenamiento
        y_invalido = pd.Series([2] * len(X))
        with pytest.raises(ValueError):
            ModeloHeuristicoRiesgo().fit(X, y_invalido)

    def test_columna_faltante_lanza_error(self, datos_entrenamiento):
        X, y = datos_entrenamiento
        with pytest.raises(ValueError):
            ModeloHeuristicoRiesgo().fit(X.drop(columns=["huella_consulta"]), y)


class TestAjusteYPrediccion:
    def test_fit_devuelve_self(self, datos_entrenamiento):
        X, y = datos_entrenamiento
        modelo = ModeloHeuristicoRiesgo()
        assert modelo.fit(X, y) is modelo

    def test_predict_devuelve_solo_0_o_1(self, datos_entrenamiento):
        X, y = datos_entrenamiento
        modelo = ModeloHeuristicoRiesgo(review_fraction=0.3).fit(X, y)
        pred = modelo.predict(X)
        assert set(np.unique(pred)) <= {0, 1}

    def test_predict_proba_suma_uno(self, datos_entrenamiento):
        X, y = datos_entrenamiento
        modelo = ModeloHeuristicoRiesgo().fit(X, y)
        proba = modelo.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_review_fraction_controla_proporcion_marcada(self, datos_entrenamiento):
        X, y = datos_entrenamiento
        modelo = ModeloHeuristicoRiesgo(review_fraction=0.2).fit(X, y)
        pred = modelo.predict(X)
        proporcion_marcada = (pred == 0).mean()
        # con method="higher" puede no ser exacto, pero debe estar cerca de 0.2
        assert 0.10 <= proporcion_marcada <= 0.30

    def test_clientes_de_alto_riesgo_reciben_mas_probabilidad_de_mora(self, datos_entrenamiento):
        X, y = datos_entrenamiento
        modelo = ModeloHeuristicoRiesgo().fit(X, y)
        proba = modelo.predict_proba(X)
        proba_mora = proba[:, 0]
        # la primera mitad de la muestra (mora=0, señales de alto riesgo)
        # debe tener, en promedio, mayor probabilidad de mora que la segunda mitad
        mitad = len(X) // 2
        assert proba_mora[:mitad].mean() > proba_mora[mitad:].mean()


class TestManejoDeFaltantes:
    def test_fila_con_una_senal_faltante_se_promedia_con_las_demas(self, datos_entrenamiento):
        X, y = datos_entrenamiento
        modelo = ModeloHeuristicoRiesgo().fit(X, y)
        X_con_nan = X.copy()
        X_con_nan.loc[0, "huella_consulta"] = np.nan
        # no debe lanzar error, y debe seguir devolviendo un score valido
        score = modelo.risk_score(X_con_nan)
        assert np.isfinite(score[0])

    def test_fila_sin_ninguna_senal_recibe_score_neutral(self, datos_entrenamiento):
        X, y = datos_entrenamiento
        modelo = ModeloHeuristicoRiesgo().fit(X, y)
        X_todo_nan = X.copy()
        X_todo_nan.iloc[0] = np.nan
        score = modelo.risk_score(X_todo_nan)
        assert score[0] == pytest.approx(modelo.score_neutral_)

    def test_ninguna_fila_de_entrenamiento_disponible_lanza_error(self):
        X = pd.DataFrame({
            "puntaje_datacredito": [np.nan, np.nan],
            "huella_consulta": [np.nan, np.nan],
            "edad_cliente": [np.nan, np.nan],
            "tiene_info_ingresos_buro": [np.nan, np.nan],
        })
        y = pd.Series([0, 1])
        with pytest.raises(ValueError):
            ModeloHeuristicoRiesgo().fit(X, y)


class TestCompatibilidadSklearn:
    def test_el_modelo_se_puede_clonar(self, datos_entrenamiento):
        X, y = datos_entrenamiento
        modelo = ModeloHeuristicoRiesgo(review_fraction=0.25)
        modelo_clonado = clone(modelo)
        assert modelo_clonado.review_fraction == 0.25
        # clone no debe copiar atributos aprendidos (los que terminan en "_")
        assert not hasattr(modelo_clonado, "referencias_")

    def test_expone_classes_y_feature_names_in_tras_fit(self, datos_entrenamiento):
        X, y = datos_entrenamiento
        modelo = ModeloHeuristicoRiesgo().fit(X, y)
        assert list(modelo.classes_) == [0, 1]
        assert len(modelo.feature_names_in_) == 4


class TestIntegracionConPipelineCompleto:
    def test_funciona_con_build_features_real(self):
        """Prueba de integracion: corre el pipeline completo de ft_engineering.py
        y entrena el modelo heuristico sobre su salida real.

        Se salta (no falla) si build_features todavia no existe en esta rama --
        eso pasa cuando se trabaja en paralelo y el PR3 (que agrega
        build_features a ft_engineering.py) todavia no se ha mergeado a
        developer. En cuanto se mergee, esta prueba empieza a correr sola.
        """
        from pathlib import Path

        try:
            from ft_engineering import build_features
        except ImportError:
            pytest.skip(
                "build_features no esta disponible todavia en ft_engineering.py "
                "(requiere que el PR3 este mergeado a developer)"
            )

        ruta_csv = Path(__file__).resolve().parent.parent / "Base_de_datos.csv"
        df = pd.read_csv(ruta_csv, sep=";", encoding="utf-8-sig")
        X_train, X_test, y_train, y_test = build_features(df)

        modelo = ModeloHeuristicoRiesgo(review_fraction=0.20).fit(X_train, y_train)
        pred = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test)

        assert len(pred) == len(X_test)
        assert np.allclose(proba.sum(axis=1), 1.0)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(1 - y_test, proba[:, 0])
        # el modelo heuristico debe ser mejor que adivinar al azar (AUC > 0.5)
        assert auc > 0.55
