"""
Pruebas unitarias para mlops_pipeline/src/model_deploy.py

Usa un modelo entrenado real (rapido: solo regresion logistica, sin los
5 candidatos completos) contra una muestra del CSV real, para probar el
API de punta a punta con el TestClient de FastAPI (no hace falta un
servidor corriendo de verdad).
"""
import importlib
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def entorno_api(tmp_path_factory):
    """
    Entrena un modelo rapido real y lo deja listo en disco, junto con una
    copia del CSV historico, para que model_deploy.py los pueda cargar tal
    como lo haria en un despliegue real. Se salta si build_features todavia
    no esta disponible (requiere el PR3 de ft_engineering.py mergeado).
    """
    from pathlib import Path

    try:
        from ft_engineering import build_features  # noqa: F401
    except ImportError:
        pytest.skip("build_features no esta disponible todavia (requiere PR3 mergeado)")

    from sklearn.linear_model import LogisticRegression
    from model_training_evaluation import entrenar_y_comparar_modelos

    carpeta = tmp_path_factory.mktemp("despliegue")
    ruta_csv_original = Path(__file__).resolve().parent.parent / "Base_de_datos.csv"
    ruta_csv_copia = carpeta / "Base_de_datos.csv"
    ruta_csv_copia.write_bytes(ruta_csv_original.read_bytes())

    df = pd.read_csv(ruta_csv_copia, sep=";", encoding="utf-8-sig")
    ruta_modelo = carpeta / "modelo_prueba.joblib"
    entrenar_y_comparar_modelos(
        df, ruta_modelo_final=str(ruta_modelo),
        candidatos_ml={"regresion_logistica": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )},
        incluir_heuristico=False, cv=2,
    )

    return {"ruta_modelo": str(ruta_modelo), "ruta_csv": str(ruta_csv_copia)}


@pytest.fixture
def cliente(entorno_api, monkeypatch):
    """Cliente de pruebas con model_deploy.py apuntando al modelo/CSV de prueba."""
    monkeypatch.setenv("RUTA_MODELO", entorno_api["ruta_modelo"])
    monkeypatch.setenv("RUTA_DATOS_HISTORICOS", entorno_api["ruta_csv"])

    import model_deploy
    importlib.reload(model_deploy)  # para que relea las variables de entorno
    with TestClient(model_deploy.app) as client:
        yield client


@pytest.fixture
def creditos_de_muestra(entorno_api):
    df = pd.read_csv(entorno_api["ruta_csv"], sep=";", encoding="utf-8-sig")
    muestra = df.drop(columns=["Pago_atiempo"]).head(5).replace({np.nan: None})
    return muestra.to_dict(orient="records")


class TestSalud:
    def test_devuelve_ok_cuando_el_modelo_carga(self, cliente):
        r = cliente.get("/salud")
        assert r.status_code == 200
        assert r.json()["estado"] == "ok"


class TestPredecirLote:
    def test_predice_un_lote_valido(self, cliente, creditos_de_muestra):
        r = cliente.post("/predecir_lote", json={"creditos": creditos_de_muestra})
        assert r.status_code == 200
        cuerpo = r.json()
        assert cuerpo["total_recibidos"] == 5
        assert cuerpo["total_predichos"] == 5
        assert len(cuerpo["predicciones"]) == 5

    def test_probabilidades_estan_en_rango_valido(self, cliente, creditos_de_muestra):
        r = cliente.post("/predecir_lote", json={"creditos": creditos_de_muestra})
        for pred in r.json()["predicciones"]:
            assert 0.0 <= pred["probabilidad_mora"] <= 1.0

    def test_prediccion_coincide_con_riesgo_alto_segun_umbral(self, cliente, creditos_de_muestra):
        r = cliente.post("/predecir_lote", json={"creditos": creditos_de_muestra})
        cuerpo = r.json()
        umbral = cuerpo["umbral_decision"]
        for pred in cuerpo["predicciones"]:
            si_es_mora = pred["probabilidad_mora"] >= umbral
            assert pred["riesgo_alto"] == si_es_mora
            assert pred["prediccion"] == ("mora" if si_es_mora else "a_tiempo")

    def test_indices_se_preservan_en_el_mismo_orden_de_entrada(self, cliente, creditos_de_muestra):
        r = cliente.post("/predecir_lote", json={"creditos": creditos_de_muestra})
        indices = [p["indice"] for p in r.json()["predicciones"]]
        assert indices == list(range(len(creditos_de_muestra)))

    def test_lote_vacio_devuelve_error_400(self, cliente):
        r = cliente.post("/predecir_lote", json={"creditos": []})
        assert r.status_code == 400

    def test_tipo_credito_no_soportado_se_marca_explicito_sin_desaparecer(self, cliente, creditos_de_muestra):
        creditos = [dict(c) for c in creditos_de_muestra]
        creditos[1]["tipo_credito"] = 6  # codigo no soportado por el modelo

        r = cliente.post("/predecir_lote", json={"creditos": creditos})
        cuerpo = r.json()

        assert cuerpo["total_recibidos"] == 5
        assert cuerpo["total_predichos"] == 4
        assert len(cuerpo["predicciones"]) == 5  # nadie desaparece de la respuesta

        no_soportado = cuerpo["predicciones"][1]
        assert no_soportado["soportado"] is False
        assert no_soportado["prediccion"] is None
        assert "no soportado" in no_soportado["motivo_no_soportado"]

    def test_campo_requerido_faltante_devuelve_error_422(self, cliente, creditos_de_muestra):
        credito_incompleto = dict(creditos_de_muestra[0])
        del credito_incompleto["tipo_credito"]

        r = cliente.post("/predecir_lote", json={"creditos": [credito_incompleto]})
        assert r.status_code == 422  # error de validacion de Pydantic


class TestServicioPrediccion:
    def test_no_encontrar_el_modelo_lanza_error_claro(self, entorno_api):
        from model_deploy import ServicioPrediccion
        from pathlib import Path

        with pytest.raises(FileNotFoundError, match="No se encontro el modelo"):
            ServicioPrediccion(
                Path("no_existe.joblib"),
                Path(entorno_api["ruta_csv"]),
            )
