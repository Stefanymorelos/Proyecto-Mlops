"""
Pruebas unitarias para mlops_pipeline/src/model_evaluation.py

Usa un modelo sintetico pequeno y rapido de sklearn (no el pipeline
completo de ft_engineering.py), para que las pruebas corran en segundos y
no dependan de que el PR3 este mergeado.
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from model_evaluation import (
    evaluar_modelo_desplegado,
    generar_reporte_html,
    resumir_log_predicciones,
)


@pytest.fixture
def modelo_entrenado(tmp_path):
    """Un LogisticRegression real, entrenado sobre datos sinteticos, guardado
    con joblib -- simula el .joblib que produce model_training_evaluation.py."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 100), "b": rng.uniform(0, 1, 100)})
    y = (X["a"] + X["b"] > 1).astype(int)

    modelo = LogisticRegression()
    modelo.fit(X, y)

    ruta = tmp_path / "modelo.joblib"
    joblib.dump(modelo, ruta)
    return {"ruta": ruta, "X": X, "y": y}


class TestEvaluarModeloDesplegado:
    def test_devuelve_metricas_esperadas(self, modelo_entrenado):
        resumen = evaluar_modelo_desplegado(
            modelo_entrenado["ruta"], modelo_entrenado["X"], modelo_entrenado["y"]
        )
        for clave in ["accuracy", "precision_mora", "recall_mora", "f1_mora", "roc_auc"]:
            assert clave in resumen
        assert 0 <= resumen["accuracy"] <= 1

    def test_usa_el_nombre_del_archivo_como_nombre_del_modelo(self, modelo_entrenado):
        resumen = evaluar_modelo_desplegado(
            modelo_entrenado["ruta"], modelo_entrenado["X"], modelo_entrenado["y"]
        )
        assert resumen["modelo"] == "modelo"


class TestResumirLogPredicciones:
    def test_devuelve_ceros_si_el_log_no_existe(self, tmp_path):
        resumen = resumir_log_predicciones(tmp_path / "no_existe.csv")
        assert resumen["total_predicciones"] == 0
        assert resumen["tasa_alto_riesgo"] is None

    def test_resume_un_log_real_correctamente(self, tmp_path):
        ruta_log = tmp_path / "log.csv"
        pd.DataFrame({
            "timestamp": ["2026-01-01T00:00:00+00:00"] * 4,
            "soportado": [True, True, True, False],
            "prediccion": ["mora", "a_tiempo", "mora", None],
            "probabilidad_mora": [0.7, 0.2, 0.6, None],
        }).to_csv(ruta_log, index=False)

        resumen = resumir_log_predicciones(ruta_log)
        assert resumen["total_predicciones"] == 4
        assert resumen["total_soportadas"] == 3
        assert resumen["tasa_alto_riesgo"] == pytest.approx(2 / 3)


class TestGenerarReporteHtml:
    def test_genera_archivo_con_metricas_y_log(self, tmp_path, modelo_entrenado):
        resumen = evaluar_modelo_desplegado(
            modelo_entrenado["ruta"], modelo_entrenado["X"], modelo_entrenado["y"]
        )
        resumen_log = {
            "total_predicciones": 10, "total_soportadas": 9, "tasa_alto_riesgo": 0.2,
            "probabilidad_mora_media": 0.3, "probabilidad_mora_std": 0.1,
            "primera_peticion": "2026-01-01", "ultima_peticion": "2026-01-02",
        }
        ruta = tmp_path / "reporte.html"
        generar_reporte_html(ruta, resumen_metricas=resumen, resumen_log=resumen_log)

        assert ruta.exists()
        contenido = ruta.read_text(encoding="utf-8")
        assert "Pestana de metricas" in contenido
        assert resumen["modelo"] in contenido

    def test_genera_archivo_aunque_no_haya_metricas_ni_log(self, tmp_path):
        ruta = tmp_path / "reporte_vacio.html"
        generar_reporte_html(ruta)
        assert ruta.exists()
