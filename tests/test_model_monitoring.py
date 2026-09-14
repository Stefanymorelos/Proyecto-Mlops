"""
Pruebas unitarias para mlops_pipeline/src/model_monitoring.py

Usa datos sinteticos con casos de borde claros (una version "sin cambios"
y otra con un corrimiento deliberado y grande), para verificar que la
deteccion de drift no da ni falsos positivos ni falsos negativos obvios.
"""
import numpy as np
import pandas as pd
import pytest

from model_monitoring import (
    calcular_drift,
    cargar_datos_para_drift,
    generar_reporte_drift,
)


@pytest.fixture
def datos_referencia():
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "plazo_meses": rng.randint(6, 36, 300),
        "edad_cliente": rng.randint(20, 65, 300),
        "salario_cliente": rng.uniform(1_000_000, 5_000_000, 300),
        "puntaje_datacredito": rng.uniform(300, 900, 300),
        "tipo_laboral": rng.choice(["Empleado", "Independiente"], 300, p=[0.7, 0.3]),
    })


class TestCalcularDrift:
    def test_datos_identicos_no_generan_drift(self, datos_referencia):
        resultado = calcular_drift(datos_referencia, datos_referencia.copy())
        assert resultado["columnas_con_drift"] == []

    def test_muestra_similar_de_la_misma_distribucion_casi_nunca_genera_drift(self, datos_referencia):
        """
        Con 5 columnas evaluadas a un umbral de significancia del 5% cada
        una, existe una probabilidad real (no nula) de que UNA columna
        dispare una falsa alarma solo por azar, aunque las distribuciones
        sean identicas (es la naturaleza de las pruebas de hipotesis con
        multiples variables). Por eso esta prueba tolera como maximo 1
        columna marcada, en vez de exigir cero absoluto -- exigir cero
        seria estadisticamente irreal, no una garantia de calidad real.
        """
        rng = np.random.RandomState(99)
        similar = pd.DataFrame({
            "plazo_meses": rng.randint(6, 36, 200),
            "edad_cliente": rng.randint(20, 65, 200),
            "salario_cliente": rng.uniform(1_000_000, 5_000_000, 200),
            "puntaje_datacredito": rng.uniform(300, 900, 200),
            "tipo_laboral": rng.choice(["Empleado", "Independiente"], 200, p=[0.7, 0.3]),
        })
        resultado = calcular_drift(datos_referencia, similar)
        assert len(resultado["columnas_con_drift"]) <= 1

    def test_corrimiento_numerico_grande_si_se_detecta(self, datos_referencia):
        con_drift = datos_referencia.copy()
        con_drift["edad_cliente"] = con_drift["edad_cliente"] + 40  # corrimiento fuerte

        resultado = calcular_drift(datos_referencia, con_drift)
        assert "edad_cliente" in resultado["columnas_con_drift"]
        assert resultado["resultados_numericos"]["edad_cliente"]["p_valor"] < 0.05

    def test_corrimiento_categorico_grande_si_se_detecta(self, datos_referencia):
        con_drift = datos_referencia.copy()
        con_drift["tipo_laboral"] = "Independiente"  # 100% independientes, vs ~30% en referencia

        resultado = calcular_drift(datos_referencia, con_drift)
        assert "tipo_laboral" in resultado["columnas_con_drift"]

    def test_columnas_faltantes_en_uno_de_los_dos_se_ignoran_sin_fallar(self, datos_referencia):
        nuevo_incompleto = datos_referencia.drop(columns=["puntaje_datacredito"])
        resultado = calcular_drift(datos_referencia, nuevo_incompleto)
        assert "puntaje_datacredito" not in resultado["resultados_numericos"]

    def test_registra_tamanos_de_muestra(self, datos_referencia):
        resultado = calcular_drift(datos_referencia, datos_referencia.head(50))
        assert resultado["n_referencia"] == 300
        assert resultado["n_nuevo"] == 50


class TestCargarDatosParaDrift:
    def test_devuelve_none_si_el_log_no_existe(self, tmp_path):
        resultado = cargar_datos_para_drift(tmp_path / "historico.csv", tmp_path / "no_existe.csv")
        assert resultado is None

    def test_devuelve_none_si_el_log_esta_vacio(self, tmp_path, datos_referencia):
        ruta_historico = tmp_path / "historico.csv"
        datos_referencia.to_csv(ruta_historico, sep=";", index=False)

        ruta_log = tmp_path / "log.csv"
        pd.DataFrame(columns=["a"]).to_csv(ruta_log, index=False)

        resultado = cargar_datos_para_drift(ruta_historico, ruta_log)
        assert resultado is None

    def test_carga_ambos_dataframes_correctamente(self, tmp_path, datos_referencia):
        ruta_historico = tmp_path / "historico.csv"
        datos_referencia.to_csv(ruta_historico, sep=";", index=False)

        ruta_log = tmp_path / "log.csv"
        datos_referencia.head(20).to_csv(ruta_log, index=False)

        resultado = cargar_datos_para_drift(ruta_historico, ruta_log)
        assert resultado is not None
        df_ref, df_nuevo = resultado
        assert len(df_ref) == 300
        assert len(df_nuevo) == 20


class TestGenerarReporteDrift:
    def test_genera_archivo_html(self, tmp_path, datos_referencia):
        con_drift = datos_referencia.copy()
        con_drift["edad_cliente"] = con_drift["edad_cliente"] + 40
        resultado = calcular_drift(datos_referencia, con_drift)

        ruta = tmp_path / "reporte_drift.html"
        generar_reporte_drift(ruta, resultado, datos_referencia, con_drift)

        assert ruta.exists()
        contenido = ruta.read_text(encoding="utf-8")
        assert "Data Drift" in contenido
        assert "edad_cliente" in contenido

    def test_reporte_sin_drift_no_tiene_seccion_de_graficas(self, tmp_path, datos_referencia):
        resultado = calcular_drift(datos_referencia, datos_referencia.copy())
        ruta = tmp_path / "reporte_sin_drift.html"
        generar_reporte_drift(ruta, resultado, datos_referencia, datos_referencia)

        contenido = ruta.read_text(encoding="utf-8")
        assert "No se detecto drift" in contenido
