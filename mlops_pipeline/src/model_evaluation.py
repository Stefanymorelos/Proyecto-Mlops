"""
Evaluacion del modelo YA DESPLEGADO (no del entrenamiento -- eso ya lo
hace model_training_evaluation.py). Responde a la pregunta: "el modelo que
esta sirviendo model_deploy.py ahora mismo, que tan bien esta funcionando?"

Genera una "pestana de metricas" -- un reporte HTML de un solo archivo
(sin dependencias externas para verlo, se abre directo en el navegador)
con:
  1. Metricas de desempeno del modelo desplegado contra un conjunto de
     evaluacion con verdad conocida (holdout de entrenamiento, ya que
     todavia no hay trafico real de produccion con desenlaces conocidos).
  2. Resumen de la actividad real del API: cuantas predicciones se han
     hecho, que fraccion se marco como alto riesgo, distribucion de
     probabilidades -- a partir del log que escribe model_deploy.py en
     cada peticion (RUTA_LOG_PREDICCIONES).
"""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from model_training_evaluation import summarize_classification


def evaluar_modelo_desplegado(ruta_modelo: str | Path, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict:
    """
    Carga el modelo tal como esta guardado para produccion (el mismo
    .joblib que carga model_deploy.py) y calcula sus metricas de
    desempeno contra un conjunto de evaluacion con verdad conocida.

    A proposito NO reentrena nada -- evalua el objeto exacto que esta
    sirviendo el API ahora mismo, para detectar si por ejemplo alguien
    reemplazo el archivo por otro modelo sin avisar.
    """
    pipeline = joblib.load(ruta_modelo)
    pred = pipeline.predict(X_eval)
    proba = pipeline.predict_proba(X_eval)[:, 1]
    return summarize_classification(y_eval, pred, proba, nombre=Path(ruta_modelo).stem)


def resumir_log_predicciones(ruta_log: str | Path) -> dict:
    """
    Resume la actividad real del API a partir del log que escribe
    model_deploy.py (RUTA_LOG_PREDICCIONES) en cada peticion. No necesita
    verdad conocida -- son solo estadisticas descriptivas de lo que el
    modelo ha estado prediciendo.

    Returns
    -------
    dict con total de predicciones, cuantas fueron soportadas, tasa de
    alto riesgo, y estadisticos de la probabilidad de mora. Si el log
    todavia no existe (API sin trafico aun), devuelve un resumen vacio
    en vez de fallar.
    """
    ruta_log = Path(ruta_log)
    if not ruta_log.exists():
        return {
            "total_predicciones": 0,
            "total_soportadas": 0,
            "tasa_alto_riesgo": None,
            "probabilidad_mora_media": None,
            "probabilidad_mora_std": None,
            "primera_peticion": None,
            "ultima_peticion": None,
        }

    log = pd.read_csv(ruta_log)
    soportadas = log[log["soportado"] == True] if "soportado" in log.columns else log  # noqa: E712

    return {
        "total_predicciones": len(log),
        "total_soportadas": len(soportadas),
        "tasa_alto_riesgo": (
            float((soportadas["prediccion"] == "mora").mean()) if len(soportadas) else None
        ),
        "probabilidad_mora_media": (
            float(soportadas["probabilidad_mora"].mean()) if len(soportadas) else None
        ),
        "probabilidad_mora_std": (
            float(soportadas["probabilidad_mora"].std()) if len(soportadas) else None
        ),
        "primera_peticion": log["timestamp"].min() if "timestamp" in log.columns and len(log) else None,
        "ultima_peticion": log["timestamp"].max() if "timestamp" in log.columns and len(log) else None,
    }


def _figura_a_base64(fig) -> str:
    """Convierte una figura de matplotlib a una cadena base64 embebible
    directo en HTML (sin tener que guardar/servir archivos .png aparte)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _grafica_matriz_confusion(resumen: dict):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(4, 3.6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=resumen["matriz_confusion"], display_labels=["Mora (0)", "A tiempo (1)"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusion")
    plt.tight_layout()
    return fig


def generar_reporte_html(
    ruta_salida: str | Path,
    resumen_metricas: dict | None = None,
    resumen_log: dict | None = None,
) -> Path:
    """
    Genera la "pestana de metricas" del modelo desplegado: un archivo HTML
    autocontenido (sin CSS/JS externos, sin necesidad de servidor) que se
    abre directo en cualquier navegador.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    bloque_metricas = "<p><em>No hay conjunto de evaluacion con verdad conocida disponible.</em></p>"
    if resumen_metricas is not None:
        img_b64 = _figura_a_base64(_grafica_matriz_confusion(resumen_metricas))
        bloque_metricas = f"""
        <table>
          <tr><th>Modelo</th><td>{resumen_metricas['modelo']}</td></tr>
          <tr><th>Accuracy</th><td>{resumen_metricas['accuracy']:.4f}</td></tr>
          <tr><th>Precision (mora)</th><td>{resumen_metricas['precision_mora']:.4f}</td></tr>
          <tr><th>Recall (mora)</th><td>{resumen_metricas['recall_mora']:.4f}</td></tr>
          <tr><th>F1 (mora)</th><td>{resumen_metricas['f1_mora']:.4f}</td></tr>
          <tr><th>ROC-AUC</th><td>{resumen_metricas.get('roc_auc', float('nan')):.4f}</td></tr>
        </table>
        <img src="data:image/png;base64,{img_b64}" alt="Matriz de confusion">
        """

    bloque_actividad = "<p><em>Sin datos de actividad todavia.</em></p>"
    if resumen_log is not None and resumen_log["total_predicciones"] > 0:
        bloque_actividad = f"""
        <table>
          <tr><th>Total de predicciones</th><td>{resumen_log['total_predicciones']}</td></tr>
          <tr><th>Total soportadas</th><td>{resumen_log['total_soportadas']}</td></tr>
          <tr><th>Tasa marcada como alto riesgo</th>
              <td>{resumen_log['tasa_alto_riesgo']:.2%}</td></tr>
          <tr><th>Probabilidad de mora (media ± std)</th>
              <td>{resumen_log['probabilidad_mora_media']:.4f} ± {resumen_log['probabilidad_mora_std']:.4f}</td></tr>
          <tr><th>Primera / ultima peticion</th>
              <td>{resumen_log['primera_peticion']} — {resumen_log['ultima_peticion']}</td></tr>
        </table>
        """

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Pestana de metricas — Modelo de riesgo de credito</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; max-width: 800px; margin: 2rem auto; color: #1a1a2e; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; border-bottom: 2px solid #1E2761; padding-bottom: 0.3rem; margin-top: 2rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ text-align: left; padding: 0.4rem 1rem; border-bottom: 1px solid #ddd; }}
  th {{ color: #555; font-weight: 600; }}
  .generado {{ color: #888; font-size: 0.85rem; }}
</style>
</head>
<body>
  <h1>Pestana de metricas — Modelo de riesgo de credito</h1>
  <p class="generado">Generado: {datetime.now(timezone.utc).isoformat()}</p>

  <h2>Desempeno contra conjunto de evaluacion</h2>
  {bloque_metricas}

  <h2>Actividad real del API (log de predicciones)</h2>
  {bloque_actividad}
</body>
</html>"""

    ruta_salida.write_text(html, encoding="utf-8")
    return ruta_salida


if __name__ == "__main__":
    print("Modulo de evaluacion del modelo desplegado")
