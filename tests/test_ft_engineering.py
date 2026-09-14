"""
Pruebas unitarias para mlops_pipeline/src/ft_engineering.py

Cada clase de limpieza/transformacion se prueba de forma aislada, con
dataframes pequeños y controlados (no con la base completa), para que
las pruebas sean rapidas y deterministas. Los casos de prueba reflejan
los hallazgos documentados en comprension_eda.ipynb (Entregable 2).
"""
import numpy as np
import pandas as pd
import pytest

from ft_engineering import (
    ColumnasIrrelevantes,
    ColumnasNulos,
    Outliers,
    Imputacion,
    NuevasVariables,
    ToCategory,
    EliminarCategorias,
    ToDF,
    build_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)


# ---------------------------------------------------------------------------
# Fixture: dataframe minimo con las columnas que usa el pipeline
# ---------------------------------------------------------------------------

@pytest.fixture
def df_muestra():
    """Dataframe pequeño y controlado, con los casos de borde que ya
    documentamos en el Entregable 2 (edad inflada, salario invalido,
    tendencia_ingresos corrupta, tipos de credito minoritarios, etc.)."""
    return pd.DataFrame({
        "tipo_credito": [4, 9, 6, 4, 9, 10],
        "fecha_prestamo": ["1/01/2025"] * 6,
        "capital_prestado": [1000000, 2000000, 1500000, 1200000, 1800000, 900000],
        "plazo_meses": [12, 6, 24, 12, 6, 18],
        "edad_cliente": [30, 45, 122, 25, 38, 60],  # la fila 122 simula el bloque distorsionado
        "tipo_laboral": ["Empleado", "Independiente", "Empleado", "Empleado", "Independiente", "Empleado"],
        "salario_cliente": [2000000, 3000000, 0, 2500000, 2800000, 3200000],  # 0 = invalido
        "total_otros_prestamos": [500000, 1000000, 300000, 0, 700000, 400000],
        "cuota_pactada": [150000, 300000, 200000, 180000, 250000, 220000],
        "puntaje": ["95,22", "95,22", "10,50", "95,22", "95,22", "95,22"],
        "puntaje_datacredito": [780.0, 820.0, np.nan, 700.0, 750.0, 810.0],
        "cant_creditosvigentes": [2, 1, 0, 3, 1, 2],
        "huella_consulta": [1, 0, 5, 2, 1, 0],
        "saldo_mora": [0.0, 0.0, np.nan, 0.0, 0.0, 0.0],
        "saldo_total": [500000.0, 800000.0, np.nan, 600000.0, 700000.0, 400000.0],
        "saldo_principal": [500000.0, 800000.0, np.nan, 600000.0, 700000.0, 400000.0],
        "saldo_mora_codeudor": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "creditos_sectorFinanciero": [1, 0, 0, 2, 1, 0],
        "creditos_sectorCooperativo": [0, 1, 0, 0, 0, 1],
        "creditos_sectorReal": [0, 0, 1, 0, 1, 0],
        "promedio_ingresos_datacredito": [1900000, np.nan, 1400000, 2400000, np.nan, 3000000],
        "tendencia_ingresos": ["Creciente", "8315", "Estable", "Decreciente", "Creciente", "Estable"],
        "Pago_atiempo": [1, 1, 0, 1, 1, 1],
    })


# ---------------------------------------------------------------------------
# ColumnasIrrelevantes
# ---------------------------------------------------------------------------

class TestColumnasIrrelevantes:
    def test_elimina_columnas_por_defecto(self, df_muestra):
        resultado = ColumnasIrrelevantes().fit_transform(df_muestra)
        columnas_esperadas_fuera = ["puntaje", "fecha_prestamo", "tipo_credito",
                                     "capital_prestado", "saldo_mora_codeudor"]
        for col in columnas_esperadas_fuera:
            assert col not in resultado.columns, f"{col} deberia haberse eliminado"

    def test_no_afecta_otras_columnas(self, df_muestra):
        resultado = ColumnasIrrelevantes().fit_transform(df_muestra)
        assert "edad_cliente" in resultado.columns
        assert "Pago_atiempo" in resultado.columns

    def test_cols_to_drop_personalizado(self, df_muestra):
        resultado = ColumnasIrrelevantes(cols_to_drop=["edad_cliente"]).fit_transform(df_muestra)
        assert "edad_cliente" not in resultado.columns
        assert "puntaje" in resultado.columns  # no se toco, porque no estaba en la lista personalizada


# ---------------------------------------------------------------------------
# ColumnasNulos
# ---------------------------------------------------------------------------

class TestColumnasNulos:
    def test_tendencia_ingresos_invalida_se_vuelve_nan(self, df_muestra):
        resultado = ColumnasNulos().fit_transform(df_muestra)
        # la fila 1 tenia "8315" (invalido) -> debe quedar NaN
        assert pd.isna(resultado.loc[1, "tendencia_ingresos"])
        # las categorias validas no se tocan
        assert resultado.loc[0, "tendencia_ingresos"] == "Creciente"

    def test_salario_cero_se_vuelve_nan(self, df_muestra):
        resultado = ColumnasNulos().fit_transform(df_muestra)
        # la fila 2 tenia salario_cliente = 0
        assert pd.isna(resultado.loc[2, "salario_cliente"])

    def test_salario_valido_no_se_toca(self, df_muestra):
        resultado = ColumnasNulos().fit_transform(df_muestra)
        assert resultado.loc[0, "salario_cliente"] == 2000000


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

class TestOutliers:
    def test_corrige_edad_del_bloque_distorsionado(self, df_muestra):
        resultado = Outliers().fit_transform(df_muestra)
        # la fila 2 tenia edad_cliente = 122 -> deberia quedar en 22
        assert resultado.loc[2, "edad_cliente"] == 22

    def test_no_toca_edades_normales(self, df_muestra):
        resultado = Outliers().fit_transform(df_muestra)
        assert resultado.loc[0, "edad_cliente"] == 30

    def test_marca_lote_datos_sospechoso(self, df_muestra):
        resultado = Outliers().fit_transform(df_muestra)
        assert resultado.loc[2, "lote_datos_sospechoso"] == 1
        assert resultado.loc[0, "lote_datos_sospechoso"] == 0


# ---------------------------------------------------------------------------
# Imputacion
# ---------------------------------------------------------------------------

class TestImputacion:
    def test_imputa_puntaje_datacredito_con_mediana(self, df_muestra):
        imputador = Imputacion()
        resultado = imputador.fit_transform(df_muestra)
        assert not resultado["puntaje_datacredito"].isna().any()
        # el nulo original (fila 2) debe quedar con la mediana calculada en fit
        assert resultado.loc[2, "puntaje_datacredito"] == imputador.median_puntaje_datacredito_

    def test_imputa_saldo_total_y_mora_sin_nulos(self, df_muestra):
        resultado = Imputacion().fit_transform(df_muestra)
        assert not resultado["saldo_total"].isna().any()
        assert not resultado["saldo_mora"].isna().any()
        assert not resultado["saldo_principal"].isna().any()


# ---------------------------------------------------------------------------
# NuevasVariables
# ---------------------------------------------------------------------------

class TestNuevasVariables:
    def test_crea_tiene_info_ingresos_buro(self, df_muestra):
        resultado = NuevasVariables().fit_transform(df_muestra)
        assert "tiene_info_ingresos_buro" in resultado.columns
        assert resultado.loc[0, "tiene_info_ingresos_buro"] == 1  # tenia dato
        assert resultado.loc[1, "tiene_info_ingresos_buro"] == 0  # era NaN

    def test_crea_tipo_credito_agrupado(self, df_muestra):
        resultado = NuevasVariables().fit_transform(df_muestra)
        assert resultado.loc[0, "tipo_credito_agrupado"] == "4"
        assert resultado.loc[2, "tipo_credito_agrupado"] == "6"

    def test_crea_ratio_cuota_ingreso(self, df_muestra):
        resultado = NuevasVariables().fit_transform(df_muestra)
        esperado = 150000 / 2000000
        assert resultado.loc[0, "ratio_cuota_ingreso"] == pytest.approx(esperado)

    def test_crea_nivel_endeudamiento(self, df_muestra):
        resultado = NuevasVariables().fit_transform(df_muestra)
        assert "nivel_endeudamiento" in resultado.columns
        assert not resultado["nivel_endeudamiento"].isna().all()


# ---------------------------------------------------------------------------
# ToCategory
# ---------------------------------------------------------------------------

class TestToCategory:
    def test_rellena_tendencia_ingresos_faltante(self, df_muestra):
        limpio = ColumnasNulos().fit_transform(df_muestra)  # primero genera el NaN real
        resultado = ToCategory().fit_transform(limpio)
        assert not resultado["tendencia_ingresos"].isna().any()
        assert resultado.loc[1, "tendencia_ingresos"] == "Sin_dato"

    def test_convierte_a_category_dtype(self, df_muestra):
        resultado = ToCategory().fit_transform(df_muestra)
        assert str(resultado["tipo_laboral"].dtype) == "category"


# ---------------------------------------------------------------------------
# EliminarCategorias
# ---------------------------------------------------------------------------

class TestEliminarCategorias:
    def test_filtra_solo_tipo_credito_4_y_9(self, df_muestra):
        resultado = EliminarCategorias().fit_transform(df_muestra)
        assert set(resultado["tipo_credito"].unique()) == {4, 9}
        assert len(resultado) == 4  # de las 6 filas originales, 2 tenian 6 y 10

    def test_no_elimina_si_no_hay_configuracion(self, df_muestra):
        resultado = EliminarCategorias(target_col=None, cats_to_drop=None).fit_transform(df_muestra)
        assert len(resultado) == len(df_muestra)


# ---------------------------------------------------------------------------
# ToDF / preprocesador ML completo
# ---------------------------------------------------------------------------

class TestToDF:
    def test_devuelve_dataframe_sin_nulos(self, df_muestra):
        # Se corre el pipeline base completo primero, para que existan las
        # variables derivadas (tiene_info_ingresos_buro, ratio_cuota_ingreso, etc.)
        # antes de probar el preprocesador ML.
        limpio = ColumnasNulos().fit_transform(df_muestra)
        limpio = Outliers().fit_transform(limpio)
        limpio = Imputacion().fit_transform(limpio)
        limpio = NuevasVariables().fit_transform(limpio)
        limpio = ToCategory().fit_transform(limpio)

        X = limpio[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        preproc = ToDF(numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES)
        preproc.fit(X)
        resultado = preproc.transform(X)
        assert isinstance(resultado, pd.DataFrame)
        assert resultado.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# build_features (funcion principal, integracion de todo el pipeline)
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_retorna_train_test_con_proporcion_correcta(self, df_muestra):
        # se duplica la muestra para tener suficientes filas para el split estratificado
        df_grande = pd.concat([df_muestra] * 20, ignore_index=True)
        X_train, X_test, y_train, y_test = build_features(df_grande, test_size=0.25, random_state=42)

        total = len(X_train) + len(X_test)
        assert total <= len(df_grande)  # puede ser menor por el filtro de tipo_credito
        assert TARGET_COL not in X_train.columns
        assert TARGET_COL not in X_test.columns

    def test_no_quedan_columnas_eliminadas_en_X(self, df_muestra):
        df_grande = pd.concat([df_muestra] * 20, ignore_index=True)
        X_train, X_test, y_train, y_test = build_features(df_grande)
        for col in ["puntaje", "fecha_prestamo", "tipo_credito", "capital_prestado", "saldo_mora_codeudor"]:
            assert col not in X_train.columns
