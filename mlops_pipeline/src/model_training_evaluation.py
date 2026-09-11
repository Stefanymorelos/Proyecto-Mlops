"""
Entrenamiento, comparacion y evaluacion de modelos de riesgo de credito.

Compara 5 candidatos sobre la salida de ft_engineering.build_features():
  - ModeloHeuristicoRiesgo   (piso sin ML, ver heuristic_model.py)
  - Regresion Logistica      (piso CON ml, interpretable)
  - Random Forest            (no linealidad, buena referencia tabular)
  - HistGradientBoosting     (mejor rendimiento tipico en datos tabulares)
  - SVM (kernel RBF)         (contraste de escalabilidad: notoriamente lento)

Sigue las funciones requeridas por el entregable: build_model y
summarize_classification. Evalua tres ejes: performance (metricas en
holdout), consistency (validacion cruzada / curva de aprendizaje) y
scalability (tiempo de ajuste vs. tamano de muestra).
"""

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from ft_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL, build_features
from heuristic_model import ModeloHeuristicoRiesgo

RANDOM_STATE = 42

# 'tiene_info_ingresos_buro' se excluye del set de modelado (aunque
# ft_engineering.py si la produce): coincide en el 99.4% de las filas con
# la categoria "Sin_dato" de tendencia_ingresos (ambas capturan la misma
# senal de informalidad laboral). Mantener las dos genera multicolinealidad
# que vuelve inestables los coeficientes individuales de la regresion
# logistica (se observo en la tabla de importancia de variables: signos
# casi contradictorios entre ambas). tendencia_ingresos_Sin_dato se
# conserva porque, ademas de esa senal, distingue tambien Creciente /
# Decreciente / Estable para quienes si tienen el dato.
NUMERIC_FEATURES_MODELO = [f for f in NUMERIC_FEATURES if f != "tiene_info_ingresos_buro"]


# ---------------------------------------------------------------------------
# Preprocesador ML compartido (mismo criterio que ft_engineering.ToDF, pero
# devolviendo un array de numpy dentro de un Pipeline con el estimador --
# asi cross_validate/learning_curve pueden clonar y reajustar todo junto,
# sin fuga de datos entre folds).
# ---------------------------------------------------------------------------

def construir_preprocesador() -> ColumnTransformer:
    numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numerico, NUMERIC_FEATURES_MODELO),
        ("cat", categorico, CATEGORICAL_FEATURES),
    ])


# Candidatos de ML "de verdad" (el heuristico se maneja aparte, no necesita
# preprocesador porque ya trabaja directo sobre las columnas crudas).
CANDIDATOS_ML: dict[str, "object"] = {
    "regresion_logistica": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "gradient_boosting": HistGradientBoostingClassifier(
        class_weight="balanced", random_state=RANDOM_STATE
    ),
    "svm_rbf": SVC(
        kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE
    ),
}


# ---------------------------------------------------------------------------
# build_model: entrena UN candidato (preprocesador + estimador en un solo
# Pipeline), con validacion cruzada para medir consistencia, y devuelve el
# pipeline ya reajustado con todos los datos de entrenamiento.
# ---------------------------------------------------------------------------

@dataclass
class ResultadoEntrenamiento:
    nombre: str
    pipeline: Pipeline
    cv_scores: dict[str, np.ndarray] = field(default_factory=dict)
    tiempo_ajuste_total_s: float = 0.0


def build_model(
    nombre: str,
    estimador,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> ResultadoEntrenamiento:
    """
    Construye y entrena un Pipeline (preprocesador + estimador) para UN
    candidato. Mide performance/consistencia con validacion cruzada
    estratificada (cv folds), y al final reajusta con TODO X_train/y_train
    para obtener el modelo final a evaluar en el holdout.

    El preprocesador va DENTRO del pipeline (no aparte) para que, en cada
    fold de la validacion cruzada, se ajuste solo con los datos de
    entrenamiento de ese fold -- evita fuga de datos entre folds.
    """
    pipeline = Pipeline(steps=[
        ("preprocesador", construir_preprocesador()),
        ("modelo", estimador),
    ])

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    cv_resultado = cross_validate(
        pipeline, X_train, y_train,
        cv=cv_splitter,
        scoring=["roc_auc", "f1", "precision", "recall", "accuracy"],
        return_train_score=True,
        n_jobs=1,  # SVM con RBF no siempre es thread-safe con n_jobs>1 en cross_validate
    )

    inicio = time.perf_counter()
    pipeline.fit(X_train, y_train)
    tiempo_total = time.perf_counter() - inicio

    return ResultadoEntrenamiento(
        nombre=nombre, pipeline=pipeline, cv_scores=cv_resultado,
        tiempo_ajuste_total_s=tiempo_total,
    )


def build_model_heuristico(X_train: pd.DataFrame, y_train: pd.Series) -> ResultadoEntrenamiento:
    """Version de build_model para ModeloHeuristicoRiesgo (no necesita
    preprocesador: trabaja directo sobre las columnas crudas de negocio)."""
    modelo = ModeloHeuristicoRiesgo(review_fraction=0.20)

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_resultado = cross_validate(
        modelo, X_train, y_train,
        cv=cv_splitter,
        scoring=["roc_auc", "f1", "precision", "recall", "accuracy"],
        return_train_score=True,
    )

    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    tiempo_total = time.perf_counter() - inicio

    return ResultadoEntrenamiento(
        nombre="modelo_heuristico", pipeline=modelo, cv_scores=cv_resultado,
        tiempo_ajuste_total_s=tiempo_total,
    )


# ---------------------------------------------------------------------------
# summarize_classification: metricas de PERFORMANCE sobre un conjunto dado
# (tipicamente el holdout de test, nunca visto durante el entrenamiento).
# ---------------------------------------------------------------------------

def summarize_classification(y_true, y_pred, y_proba=None, nombre: str = "") -> dict:
    """
    Resume el desempeno de un modelo ya entrenado sobre un conjunto de
    evaluacion. y_proba (si se da) debe ser la probabilidad de la clase
    positiva segun sklearn (columna 1 = Pago_atiempo=1); internamente se usa
    1-y_proba para medir la deteccion de mora (clase 0), que es la clase de
    interes de negocio.

    Returns
    -------
    dict con: accuracy, precision, recall, f1 (todas calculadas para la
    clase de mora=0, que es la minoritaria y la que le importa al negocio),
    roc_auc (si hay probabilidades), y la matriz de confusion.
    """
    resumen = {
        "modelo": nombre,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_mora": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_mora": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_mora": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }
    if y_proba is not None:
        proba_mora = 1 - np.asarray(y_proba)
        resumen["roc_auc"] = roc_auc_score(1 - np.asarray(y_true), proba_mora)
    return resumen


# ---------------------------------------------------------------------------
# Umbral de decision optimizado (no 0.5 por defecto), segun costo de negocio.
# ---------------------------------------------------------------------------

def buscar_umbral_optimo(
    y_true: np.ndarray,
    proba_mora: np.ndarray,
    costo_falso_negativo: float = 5.0,
    costo_falso_positivo: float = 1.0,
) -> tuple[float, float]:
    """
    Busca, entre 50 umbrales candidatos, el que minimiza el costo esperado
    de negocio. Por defecto, un falso negativo (predecir "paga a tiempo"
    cuando en realidad cae en mora -> se le presta a quien no debía) cuesta
    5 veces mas que un falso positivo (predecir "mora" cuando en realidad
    hubiera pagado -> se le niega un crédito bueno). Estos pesos son
    ajustables segun el apetito de riesgo real del negocio.

    Returns
    -------
    (umbral_optimo, costo_minimo)
    """
    y_true = np.asarray(y_true)
    es_mora = (y_true == 0).astype(int)
    mejores = (0.5, np.inf)

    for umbral in np.linspace(0.05, 0.95, 50):
        pred_mora = (proba_mora >= umbral).astype(int)
        falsos_negativos = ((pred_mora == 0) & (es_mora == 1)).sum()
        falsos_positivos = ((pred_mora == 1) & (es_mora == 0)).sum()
        costo = falsos_negativos * costo_falso_negativo + falsos_positivos * costo_falso_positivo
        if costo < mejores[1]:
            mejores = (float(umbral), float(costo))

    return mejores


# ---------------------------------------------------------------------------
# Consistency: curva de aprendizaje (que tan estable es el modelo segun el
# tamano de la muestra de entrenamiento -- varianza entre folds).
# ---------------------------------------------------------------------------

def curva_de_aprendizaje(pipeline, X_train, y_train, cv: int = 5) -> dict:
    tamanos, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        train_sizes=np.linspace(0.1, 1.0, 6),
        scoring="roc_auc",
        n_jobs=1,
    )
    return {
        "tamanos_muestra": tamanos,
        "train_media": train_scores.mean(axis=1),
        "train_std": train_scores.std(axis=1),
        "val_media": val_scores.mean(axis=1),
        "val_std": val_scores.std(axis=1),
    }


# ---------------------------------------------------------------------------
# Scalability: tiempo de ajuste vs. tamano de la muestra de entrenamiento.
# ---------------------------------------------------------------------------

def curva_de_escalabilidad(
    nombre: str, estimador_o_pipeline_factory, X_train, y_train,
    fracciones: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0),
) -> dict:
    """
    estimador_o_pipeline_factory: funcion sin argumentos que devuelve un
    Pipeline/estimador SIN entrenar (para poder instanciar uno nuevo en
    cada punto de la curva, sin reutilizar estado de un ajuste anterior).
    """
    n_total = len(X_train)
    tamanos, tiempos_ajuste, tiempos_prediccion = [], [], []

    rng = np.random.RandomState(RANDOM_STATE)
    indices_completos = np.arange(n_total)
    rng.shuffle(indices_completos)

    for frac in fracciones:
        n = max(int(n_total * frac), 20)
        idx = indices_completos[:n]
        X_sub = X_train.iloc[idx]
        y_sub = y_train.iloc[idx]

        modelo = estimador_o_pipeline_factory()
        inicio = time.perf_counter()
        modelo.fit(X_sub, y_sub)
        t_ajuste = time.perf_counter() - inicio

        inicio = time.perf_counter()
        modelo.predict(X_sub.iloc[: min(200, len(X_sub))])
        t_prediccion = time.perf_counter() - inicio

        tamanos.append(n)
        tiempos_ajuste.append(t_ajuste)
        tiempos_prediccion.append(t_prediccion)

    return {
        "modelo": nombre,
        "tamanos_muestra": tamanos,
        "tiempo_ajuste_s": tiempos_ajuste,
        "tiempo_prediccion_s": tiempos_prediccion,
    }

def _fabrica_pipeline(estimador):
    """Devuelve una funcion sin argumentos que construye un Pipeline nuevo
    (preprocesador + una COPIA sin entrenar del estimador dado). Se usa en
    curva_de_escalabilidad para poder instanciar un modelo fresco en cada
    punto de la curva, sin arrastrar estado de un ajuste anterior."""
    def _construir():
        return Pipeline(steps=[
            ("preprocesador", construir_preprocesador()),
            ("modelo", type(estimador)(**estimador.get_params())),
        ])
    return _construir

def extraer_importancia_variables(pipeline: Pipeline, nombre_modelo: str, top_n: int = 15) -> pd.DataFrame:
    """
    Extrae que tan importante fue cada variable para el modelo ya entrenado.
    Para modelos de arbol (Random Forest, Gradient Boosting) usa
    feature_importances_; para modelos lineales (Regresion Logistica, SVM
    lineal) usa los coeficientes (coef_). SVM con kernel RBF no expone
    ninguno de los dos (no es un modelo lineal en el espacio original), asi
    que devuelve un DataFrame vacio con una nota en vez de fallar.

    En credito esto no es un detalle opcional: un banco necesita poder
    explicar por que un modelo nego o aprobo un credito, no solo reportar
    metricas agregadas.

    Returns
    -------
    DataFrame con columnas [variable, importancia], ordenado de mayor a
    menor importancia absoluta, con las top_n variables mas relevantes.
    """
    preprocesador = pipeline.named_steps["preprocesador"]
    modelo = pipeline.named_steps["modelo"]
    nombres_variables = preprocesador.get_feature_names_out()

    if hasattr(modelo, "feature_importances_"):
        valores = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        valores = np.ravel(modelo.coef_)
    else:
        # SVM con kernel RBF (u otro modelo no lineal) no expone importancia
        # de variables de forma directa -- se devuelve vacio a proposito,
        # en vez de fallar o inventar un numero.
        return pd.DataFrame(columns=["variable", "importancia"])

    tabla = pd.DataFrame({"variable": nombres_variables, "importancia": valores})
    tabla["importancia_abs"] = tabla["importancia"].abs()
    tabla = tabla.sort_values("importancia_abs", ascending=False).drop(columns="importancia_abs")
    return tabla.head(top_n).reset_index(drop=True)


def graficar_importancia_variables(tabla_importancia: pd.DataFrame, nombre_modelo: str, ruta_salida=None):
    import matplotlib.pyplot as plt

    if tabla_importancia.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    datos = tabla_importancia.sort_values("importancia")
    colores = [COLOR_MORA if v < 0 else COLOR_OK for v in datos["importancia"]]
    ax.barh(datos["variable"], datos["importancia"], color=colores)
    ax.set_xlabel("Importancia (coeficiente o feature_importances_)")
    ax.set_title(f"Variables mas relevantes — {nombre_modelo}", fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    if ruta_salida:
        plt.savefig(ruta_salida, dpi=130)
        plt.close()
    return fig


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def entrenar_y_comparar_modelos(
    df: pd.DataFrame,
    ruta_modelo_final: str | Path | None = None,
    candidatos_ml: dict | None = None,
    incluir_heuristico: bool = True,
    cv: int = 5,
    fracciones_escalabilidad: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0),
) -> dict:
    """
    Ejecuta el flujo completo: separa train/test (via ft_engineering),
    entrena el modelo heuristico + los candidatos de ML, evalua a todos
    en el holdout con summarize_classification, calcula curva de
    aprendizaje del mejor modelo (consistency) y curva de escalabilidad de
    todos (scalability, con foco en contrastar SVM vs el resto), elige el
    mejor segun ROC-AUC en el holdout, y lo guarda con joblib.

    candidatos_ml, incluir_heuristico, cv y fracciones_escalabilidad son
    configurables para poder correr una version reducida y rapida en
    pruebas automatizadas (ej. un solo modelo, cv=2) sin esperar los ~2
    minutos que toma la comparacion completa con los 5 candidatos.

    Returns
    -------
    dict con: tabla_comparativa (DataFrame), resultados_individuales,
    consistencia (curva de aprendizaje del ganador), escalabilidad (curva
    de todos los modelos), mejor_modelo (nombre), pipeline_final.
    """
    if candidatos_ml is None:
        candidatos_ml = CANDIDATOS_ML

    X_train, X_test, y_train, y_test = build_features(df)

    resultados: dict[str, ResultadoEntrenamiento] = {}
    if incluir_heuristico:
        resultados["modelo_heuristico"] = build_model_heuristico(X_train, y_train)
    for nombre, estimador in candidatos_ml.items():
        resultados[nombre] = build_model(nombre, estimador, X_train, y_train, cv=cv)

    filas_comparativas = []
    resumenes_holdout = {}
    for nombre, resultado in resultados.items():
        modelo = resultado.pipeline
        pred = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test)[:, 1]  # P(Pago_atiempo=1)
        resumen = summarize_classification(y_test, pred, proba, nombre=nombre)
        resumenes_holdout[nombre] = resumen

        umbral_opt, costo_opt = buscar_umbral_optimo(y_test, 1 - proba)

        filas_comparativas.append({
            "modelo": nombre,
            "roc_auc_holdout": resumen.get("roc_auc"),
            "roc_auc_cv_media": resultado.cv_scores.get("test_roc_auc", np.array([np.nan])).mean(),
            "roc_auc_cv_std": resultado.cv_scores.get("test_roc_auc", np.array([np.nan])).std(),
            "precision_mora_holdout": resumen["precision_mora"],
            "recall_mora_holdout": resumen["recall_mora"],
            "f1_mora_holdout": resumen["f1_mora"],
            "tiempo_ajuste_s": resultado.tiempo_ajuste_total_s,
            "umbral_optimo_negocio": umbral_opt,
        })

    tabla_comparativa = pd.DataFrame(filas_comparativas).sort_values(
        "roc_auc_holdout", ascending=False
    ).reset_index(drop=True)

    mejor_modelo_nombre = tabla_comparativa.iloc[0]["modelo"]
    mejor_resultado = resultados[mejor_modelo_nombre]

    consistencia = None
    if mejor_modelo_nombre != "modelo_heuristico":
        consistencia = curva_de_aprendizaje(mejor_resultado.pipeline, X_train, y_train, cv=cv)

    importancia_variables = pd.DataFrame()
    if mejor_modelo_nombre != "modelo_heuristico":
        importancia_variables = extraer_importancia_variables(mejor_resultado.pipeline, mejor_modelo_nombre)

    escalabilidad = {}
    for nombre, estimador in candidatos_ml.items():
        escalabilidad[nombre] = curva_de_escalabilidad(
            nombre, _fabrica_pipeline(estimador), X_train, y_train, fracciones=fracciones_escalabilidad
        )

    if ruta_modelo_final is not None:
        joblib.dump(mejor_resultado.pipeline, ruta_modelo_final)

    return {
        "tabla_comparativa": tabla_comparativa,
        "resultados_individuales": resultados,
        "resumenes_holdout": resumenes_holdout,
        "consistencia": consistencia,
        "escalabilidad": escalabilidad,
        "importancia_variables": importancia_variables,
        "mejor_modelo": mejor_modelo_nombre,
        "pipeline_final": mejor_resultado.pipeline,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    print("Modulo de entrenamiento y evaluacion de modelos")


# ---------------------------------------------------------------------------
# Graficas comparativas (requeridas por el entregable)
# ---------------------------------------------------------------------------

COLOR_MORA = "#E63946"
COLOR_OK = "#2E86AB"
COLOR_NAVY = "#1E2761"


def graficar_comparacion_metricas(tabla_comparativa: pd.DataFrame, ruta_salida=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tabla_comparativa))
    ancho = 0.35
    ax.bar(x - ancho / 2, tabla_comparativa["roc_auc_holdout"], ancho, label="ROC-AUC (holdout)", color=COLOR_NAVY)
    ax.bar(x + ancho / 2, tabla_comparativa["roc_auc_cv_media"], ancho, label="ROC-AUC (media CV)", color=COLOR_OK)
    ax.errorbar(
        x + ancho / 2, tabla_comparativa["roc_auc_cv_media"],
        yerr=tabla_comparativa["roc_auc_cv_std"], fmt="none", ecolor="black", capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tabla_comparativa["modelo"], rotation=20, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Comparacion de performance: holdout vs. validacion cruzada", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if ruta_salida:
        plt.savefig(ruta_salida, dpi=130)
        plt.close()
    return fig


def graficar_curva_aprendizaje(consistencia: dict, nombre_modelo: str, ruta_salida=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    tam = consistencia["tamanos_muestra"]
    ax.plot(tam, consistencia["train_media"], "o-", color=COLOR_OK, label="Entrenamiento")
    ax.fill_between(
        tam, consistencia["train_media"] - consistencia["train_std"],
        consistencia["train_media"] + consistencia["train_std"], alpha=0.15, color=COLOR_OK,
    )
    ax.plot(tam, consistencia["val_media"], "o-", color=COLOR_MORA, label="Validacion")
    ax.fill_between(
        tam, consistencia["val_media"] - consistencia["val_std"],
        consistencia["val_media"] + consistencia["val_std"], alpha=0.15, color=COLOR_MORA,
    )
    ax.set_xlabel("Tamano de la muestra de entrenamiento")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Curva de aprendizaje (consistencia) — {nombre_modelo}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if ruta_salida:
        plt.savefig(ruta_salida, dpi=130)
        plt.close()
    return fig


def graficar_curva_escalabilidad(escalabilidad: dict, ruta_salida=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colores = {"regresion_logistica": COLOR_OK, "random_forest": "#6A994E",
               "gradient_boosting": "#BC6C25", "svm_rbf": COLOR_MORA}
    for nombre, esc in escalabilidad.items():
        ax.plot(
            esc["tamanos_muestra"], esc["tiempo_ajuste_s"], "o-",
            label=nombre, color=colores.get(nombre, "gray"),
        )
    ax.set_xlabel("Tamano de la muestra de entrenamiento")
    ax.set_ylabel("Tiempo de ajuste (segundos)")
    ax.set_title("Escalabilidad: tiempo de ajuste vs. tamano de muestra", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if ruta_salida:
        plt.savefig(ruta_salida, dpi=130)
        plt.close()
    return fig


def graficar_matriz_confusion(resumen: dict, ruta_salida=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=resumen["matriz_confusion"], display_labels=["Mora (0)", "A tiempo (1)"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusion — {resumen['modelo']}", fontweight="bold")
    plt.tight_layout()
    if ruta_salida:
        plt.savefig(ruta_salida, dpi=130)
        plt.close()
    return fig


def generar_todas_las_graficas(resultado_comparacion: dict, carpeta_salida: str | Path):
    """Genera y guarda las 4 graficas comparativas en la carpeta indicada."""
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    graficar_comparacion_metricas(
        resultado_comparacion["tabla_comparativa"],
        ruta_salida=carpeta_salida / "01_comparacion_metricas.png",
    )
    if resultado_comparacion["consistencia"] is not None:
        graficar_curva_aprendizaje(
            resultado_comparacion["consistencia"], resultado_comparacion["mejor_modelo"],
            ruta_salida=carpeta_salida / "02_curva_aprendizaje.png",
        )
    graficar_curva_escalabilidad(
        resultado_comparacion["escalabilidad"],
        ruta_salida=carpeta_salida / "03_curva_escalabilidad.png",
    )
    graficar_matriz_confusion(
        resultado_comparacion["resumenes_holdout"][resultado_comparacion["mejor_modelo"]],
        ruta_salida=carpeta_salida / "04_matriz_confusion_mejor_modelo.png",
    )
    tabla_importancia = resultado_comparacion.get("importancia_variables")
    if tabla_importancia is not None and not tabla_importancia.empty:
        graficar_importancia_variables(
            tabla_importancia, resultado_comparacion["mejor_modelo"],
            ruta_salida=carpeta_salida / "05_importancia_variables.png",
        )
