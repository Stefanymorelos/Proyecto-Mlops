"""
Pruebas unitarias para mlops_pipeline/src/model_training_evaluation.py

Diseno de las pruebas: se prueban las piezas (summarize_classification,
build_model, buscar_umbral_optimo, curvas) por separado, con datos
sinteticos pequenos, para que la suite corra rapido. El flujo completo
(entrenar_y_comparar_modelos, que entrena 5 modelos incluyendo SVM con
validacion cruzada + curva de escalabilidad) tarda ~2 minutos con el
dataset real -- se prueba UNA vez, de forma liviana, usando solo el
modelo mas rapido (regresion logistica) en vez de correr los 5.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from model_training_evaluation import (
    build_model,
    build_model_heuristico,
    buscar_umbral_optimo,
    construir_preprocesador,
    curva_de_escalabilidad,
    entrenar_y_comparar_modelos,
    summarize_classification,
)


@pytest.fixture
def datos_sinteticos():
    """60 filas con las columnas que espera el preprocesador de este modulo."""
    n = 60
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "plazo_meses": rng.randint(2, 36, n),
        "edad_cliente": rng.randint(18, 70, n),
        "salario_cliente": rng.uniform(1_000_000, 6_000_000, n),
        "total_otros_prestamos": rng.uniform(0, 3_000_000, n),
        "cuota_pactada": rng.uniform(50_000, 800_000, n),
        "puntaje_datacredito": rng.uniform(150, 950, n),
        "cant_creditosvigentes": rng.randint(0, 6, n),
        "huella_consulta": rng.randint(0, 10, n),
        "saldo_mora": rng.uniform(0, 200_000, n),
        "saldo_total": rng.uniform(0, 3_000_000, n),
        "saldo_principal": rng.uniform(0, 3_000_000, n),
        "creditos_sectorFinanciero": rng.randint(0, 4, n),
        "creditos_sectorCooperativo": rng.randint(0, 3, n),
        "creditos_sectorReal": rng.randint(0, 3, n),
        "promedio_ingresos_datacredito": rng.uniform(1_000_000, 5_000_000, n),
        "tiene_info_ingresos_buro": rng.randint(0, 2, n),
        "lote_datos_sospechoso": np.zeros(n, dtype=int),
        "ratio_cuota_ingreso": rng.uniform(0.01, 0.4, n),
        "nivel_endeudamiento": rng.uniform(0, 2, n),
        "tipo_laboral": rng.choice(["Empleado", "Independiente"], n),
        "tipo_credito_agrupado": rng.choice(["4", "9"], n),
        "tendencia_ingresos": rng.choice(["Creciente", "Decreciente", "Estable"], n),
    })
    y = pd.Series(rng.choice([0, 1], n, p=[0.3, 0.7]))
    return df, y


class TestConstruirPreprocesador:
    def test_devuelve_column_transformer_ajustable(self, datos_sinteticos):
        X, _ = datos_sinteticos
        prep = construir_preprocesador()
        Xt = prep.fit_transform(X)
        assert Xt.shape[0] == len(X)
        assert not np.isnan(Xt.toarray() if hasattr(Xt, "toarray") else Xt).any()


class TestBuildModel:
    def test_devuelve_pipeline_entrenado(self, datos_sinteticos):
        X, y = datos_sinteticos
        resultado = build_model(
            "regresion_logistica",
            LogisticRegression(max_iter=500, class_weight="balanced"),
            X, y, cv=3,
        )
        assert resultado.nombre == "regresion_logistica"
        pred = resultado.pipeline.predict(X)
        assert len(pred) == len(X)

    def test_cv_scores_tiene_las_metricas_esperadas(self, datos_sinteticos):
        X, y = datos_sinteticos
        resultado = build_model(
            "regresion_logistica",
            LogisticRegression(max_iter=500, class_weight="balanced"),
            X, y, cv=3,
        )
        for metrica in ["test_roc_auc", "test_f1", "test_precision", "test_recall", "test_accuracy"]:
            assert metrica in resultado.cv_scores
            assert len(resultado.cv_scores[metrica]) == 3  # cv=3

    def test_tiempo_de_ajuste_se_registra(self, datos_sinteticos):
        X, y = datos_sinteticos
        resultado = build_model(
            "regresion_logistica",
            LogisticRegression(max_iter=500, class_weight="balanced"),
            X, y, cv=3,
        )
        assert resultado.tiempo_ajuste_total_s > 0


class TestBuildModelHeuristico:
    def test_funciona_con_columnas_crudas(self, datos_sinteticos):
        X, y = datos_sinteticos
        resultado = build_model_heuristico(X, y)
        assert resultado.nombre == "modelo_heuristico"
        pred = resultado.pipeline.predict(X)
        assert set(np.unique(pred)) <= {0, 1}


class TestSummarizeClassification:
    def test_devuelve_todas_las_metricas_esperadas(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 1])
        y_proba = np.array([0.2, 0.6, 0.9, 0.8, 0.3, 0.1, 0.7, 0.95])  # P(clase=1)

        resumen = summarize_classification(y_true, y_pred, y_proba, nombre="prueba")

        for clave in ["accuracy", "precision_mora", "recall_mora", "f1_mora", "roc_auc", "matriz_confusion"]:
            assert clave in resumen
        assert 0 <= resumen["accuracy"] <= 1
        assert 0 <= resumen["roc_auc"] <= 1
        assert resumen["matriz_confusion"].shape == (2, 2)

    def test_funciona_sin_probabilidades(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        resumen = summarize_classification(y_true, y_pred, y_proba=None, nombre="sin_proba")
        assert "roc_auc" not in resumen
        assert "accuracy" in resumen

    def test_prediccion_perfecta_da_metricas_maximas(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.05, 0.1, 0.95, 0.9])
        resumen = summarize_classification(y_true, y_pred, y_proba)
        assert resumen["accuracy"] == 1.0
        assert resumen["precision_mora"] == 1.0
        assert resumen["recall_mora"] == 1.0
        assert resumen["roc_auc"] == 1.0


class TestBuscarUmbralOptimo:
    def test_devuelve_umbral_dentro_del_rango_buscado(self):
        rng = np.random.RandomState(0)
        y_true = rng.choice([0, 1], 200, p=[0.2, 0.8])
        proba_mora = rng.uniform(0, 1, 200)
        umbral, costo = buscar_umbral_optimo(y_true, proba_mora)
        assert 0.05 <= umbral <= 0.95
        assert costo >= 0

    def test_penalizar_mas_los_falsos_negativos_baja_el_umbral(self):
        """Si un falso negativo (prestarle a quien no paga) es mucho mas caro,
        el umbral optimo deberia bajar (clasificar como mora mas facilmente,
        para no dejar pasar tantos falsos negativos)."""
        rng = np.random.RandomState(1)
        y_true = rng.choice([0, 1], 300, p=[0.3, 0.7])
        proba_mora = rng.uniform(0, 1, 300)

        umbral_penalizacion_alta, _ = buscar_umbral_optimo(
            y_true, proba_mora, costo_falso_negativo=20.0, costo_falso_positivo=1.0
        )
        umbral_penalizacion_baja, _ = buscar_umbral_optimo(
            y_true, proba_mora, costo_falso_negativo=1.0, costo_falso_positivo=20.0
        )
        assert umbral_penalizacion_alta <= umbral_penalizacion_baja


class TestCurvaDeEscalabilidad:
    def test_devuelve_un_tiempo_por_fraccion(self, datos_sinteticos):
        X, y = datos_sinteticos
        factory = lambda: LogisticRegression(max_iter=200, class_weight="balanced")
        # se envuelve en preprocesador+modelo via build_model normalmente;
        # aqui se prueba con el estimador crudo sobre columnas ya numericas
        # simplificadas, solo para validar la mecanica de la funcion.
        X_num = X[["edad_cliente", "salario_cliente", "puntaje_datacredito"]]
        resultado = curva_de_escalabilidad(
            "logistica_rapida", factory, X_num, y, fracciones=(0.5, 1.0)
        )
        assert len(resultado["tamanos_muestra"]) == 2
        assert len(resultado["tiempo_ajuste_s"]) == 2
        assert all(t >= 0 for t in resultado["tiempo_ajuste_s"])


class TestIntegracionConPipelineCompleto:
    def test_build_model_funciona_con_datos_reales(self):
        """Prueba de integracion liviana: usa build_features real + UN solo
        modelo rapido (regresion logistica), no los 5 candidatos completos
        (eso tarda ~2 minutos por el SVM, se corre manualmente, no en cada
        pytest). Se salta si build_features todavia no esta disponible
        (PR3 de ft_engineering.py sin mergear a developer)."""
        from pathlib import Path

        try:
            from ft_engineering import build_features
        except ImportError:
            pytest.skip("build_features no esta disponible todavia (requiere PR3 mergeado)")

        ruta_csv = Path(__file__).resolve().parent.parent / "Base_de_datos.csv"
        df = pd.read_csv(ruta_csv, sep=";", encoding="utf-8-sig")
        X_train, X_test, y_train, y_test = build_features(df)

        resultado = build_model(
            "regresion_logistica",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            X_train, y_train, cv=3,
        )
        pred = resultado.pipeline.predict(X_test)
        proba = resultado.pipeline.predict_proba(X_test)[:, 1]
        resumen = summarize_classification(y_test, pred, proba, nombre="regresion_logistica")

        assert resumen["roc_auc"] > 0.55  # mejor que adivinar al azar
        assert resultado.cv_scores["test_roc_auc"].mean() > 0.5

    def test_orquestador_completo_version_rapida(self, tmp_path):
        """Corre entrenar_y_comparar_modelos de punta a punta, pero en su
        version reducida (1 solo candidato de ML rapido, cv=2, pocas
        fracciones de escalabilidad) para poder ejercitar toda la logica
        del orquestador -- tabla comparativa, seleccion del mejor modelo,
        guardado con joblib -- en segundos, no en los ~2 minutos que
        tomaria con los 5 candidatos completos (incluyendo SVM)."""
        from pathlib import Path

        try:
            from ft_engineering import build_features  # noqa: F401
        except ImportError:
            pytest.skip("build_features no esta disponible todavia (requiere PR3 mergeado)")

        ruta_csv = Path(__file__).resolve().parent.parent / "Base_de_datos.csv"
        df = pd.read_csv(ruta_csv, sep=";", encoding="utf-8-sig")

        ruta_modelo = tmp_path / "modelo_prueba.joblib"
        resultado = entrenar_y_comparar_modelos(
            df,
            ruta_modelo_final=str(ruta_modelo),
            candidatos_ml={"regresion_logistica": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )},
            incluir_heuristico=True,
            cv=2,
            fracciones_escalabilidad=(0.3, 1.0),
        )

        assert "tabla_comparativa" in resultado
        assert len(resultado["tabla_comparativa"]) == 2  # heuristico + 1 candidato ML
        assert resultado["mejor_modelo"] in {"regresion_logistica", "modelo_heuristico"}
        assert ruta_modelo.exists()

        import joblib
        modelo_cargado = joblib.load(ruta_modelo)
        pred = modelo_cargado.predict(resultado["X_test"].iloc[:5])
        assert len(pred) == 5


class TestImportanciaVariables:
    def test_extrae_coeficientes_de_modelo_lineal(self, datos_sinteticos):
        from model_training_evaluation import extraer_importancia_variables
        X, y = datos_sinteticos
        resultado = build_model(
            "regresion_logistica",
            LogisticRegression(max_iter=500, class_weight="balanced"),
            X, y, cv=3,
        )
        tabla = extraer_importancia_variables(resultado.pipeline, "regresion_logistica", top_n=5)
        assert len(tabla) <= 5
        assert {"variable", "importancia"}.issubset(tabla.columns)
        # debe venir ordenada por magnitud (valor absoluto) descendente
        magnitudes = tabla["importancia"].abs().to_numpy()
        assert (magnitudes[:-1] >= magnitudes[1:]).all()

    def test_extrae_feature_importances_de_modelo_de_arbol(self, datos_sinteticos):
        from model_training_evaluation import extraer_importancia_variables
        from sklearn.ensemble import RandomForestClassifier
        X, y = datos_sinteticos
        resultado = build_model(
            "random_forest",
            RandomForestClassifier(n_estimators=20, class_weight="balanced", random_state=42),
            X, y, cv=3,
        )
        tabla = extraer_importancia_variables(resultado.pipeline, "random_forest")
        assert not tabla.empty
        assert (tabla["importancia"] >= 0).all()  # feature_importances_ siempre es no negativo

    def test_devuelve_vacio_si_el_modelo_no_expone_importancia(self, datos_sinteticos):
        from model_training_evaluation import extraer_importancia_variables
        from sklearn.svm import SVC
        X, y = datos_sinteticos
        resultado = build_model(
            "svm_rbf",
            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
            X, y, cv=3,
        )
        tabla = extraer_importancia_variables(resultado.pipeline, "svm_rbf")
        assert tabla.empty

    def test_graficar_importancia_variables_genera_archivo(self, tmp_path):
        from model_training_evaluation import graficar_importancia_variables
        tabla = pd.DataFrame({
            "variable": ["puntaje_datacredito", "huella_consulta", "edad_cliente"],
            "importancia": [0.8, -0.5, -0.3],
        })
        ruta = tmp_path / "importancia.png"
        graficar_importancia_variables(tabla, "modelo_prueba", ruta_salida=ruta)
        assert ruta.exists()

    def test_graficar_importancia_con_tabla_vacia_no_falla(self):
        from model_training_evaluation import graficar_importancia_variables
        resultado = graficar_importancia_variables(pd.DataFrame(), "modelo_sin_importancia")
        assert resultado is None


class TestGraficas:
    """Las funciones de graficas se prueban verificando que corren sin
    error y generan un archivo -- no se valida el contenido visual (eso ya
    se revisa manualmente), solo que la mecanica de graficado funciona."""

    @pytest.fixture
    def tabla_comparativa_sintetica(self):
        return pd.DataFrame({
            "modelo": ["regresion_logistica", "svm_rbf"],
            "roc_auc_holdout": [0.70, 0.68],
            "roc_auc_cv_media": [0.65, 0.64],
            "roc_auc_cv_std": [0.02, 0.03],
        })

    def test_graficar_comparacion_metricas_genera_archivo(self, tabla_comparativa_sintetica, tmp_path):
        from model_training_evaluation import graficar_comparacion_metricas
        ruta = tmp_path / "comparacion.png"
        graficar_comparacion_metricas(tabla_comparativa_sintetica, ruta_salida=ruta)
        assert ruta.exists()

    def test_graficar_curva_aprendizaje_genera_archivo(self, tmp_path):
        from model_training_evaluation import graficar_curva_aprendizaje
        consistencia = {
            "tamanos_muestra": np.array([100, 200, 300]),
            "train_media": np.array([0.75, 0.72, 0.70]),
            "train_std": np.array([0.02, 0.02, 0.01]),
            "val_media": np.array([0.60, 0.64, 0.66]),
            "val_std": np.array([0.04, 0.03, 0.02]),
        }
        ruta = tmp_path / "aprendizaje.png"
        graficar_curva_aprendizaje(consistencia, "modelo_prueba", ruta_salida=ruta)
        assert ruta.exists()

    def test_graficar_curva_escalabilidad_genera_archivo(self, tmp_path):
        from model_training_evaluation import graficar_curva_escalabilidad
        escalabilidad = {
            "regresion_logistica": {"tamanos_muestra": [100, 200], "tiempo_ajuste_s": [0.01, 0.02]},
            "svm_rbf": {"tamanos_muestra": [100, 200], "tiempo_ajuste_s": [0.5, 2.0]},
        }
        ruta = tmp_path / "escalabilidad.png"
        graficar_curva_escalabilidad(escalabilidad, ruta_salida=ruta)
        assert ruta.exists()

    def test_graficar_matriz_confusion_genera_archivo(self, tmp_path):
        from model_training_evaluation import graficar_matriz_confusion
        resumen = {"modelo": "prueba", "matriz_confusion": np.array([[10, 5], [8, 40]])}
        ruta = tmp_path / "matriz.png"
        graficar_matriz_confusion(resumen, ruta_salida=ruta)
        assert ruta.exists()

    def test_generar_todas_las_graficas_crea_los_4_archivos(self, tabla_comparativa_sintetica, tmp_path):
        from model_training_evaluation import generar_todas_las_graficas
        resultado_falso = {
            "tabla_comparativa": tabla_comparativa_sintetica,
            "mejor_modelo": "regresion_logistica",
            "consistencia": {
                "tamanos_muestra": np.array([100, 200]),
                "train_media": np.array([0.75, 0.70]),
                "train_std": np.array([0.02, 0.01]),
                "val_media": np.array([0.60, 0.65]),
                "val_std": np.array([0.04, 0.02]),
            },
            "escalabilidad": {
                "regresion_logistica": {"tamanos_muestra": [100, 200], "tiempo_ajuste_s": [0.01, 0.02]},
            },
            "resumenes_holdout": {
                "regresion_logistica": {"modelo": "regresion_logistica", "matriz_confusion": np.array([[10, 5], [8, 40]])},
            },
        }
        carpeta = tmp_path / "graficas"
        generar_todas_las_graficas(resultado_falso, carpeta)
        archivos = sorted(p.name for p in carpeta.iterdir())
        assert archivos == [
            "01_comparacion_metricas.png",
            "02_curva_aprendizaje.png",
            "03_curva_escalabilidad.png",
            "04_matriz_confusion_mejor_modelo.png",
        ]
