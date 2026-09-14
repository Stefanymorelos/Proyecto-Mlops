"""
Monitoreo de *data drift*: compara la distribucion de los datos que estan
llegando DE VERDAD al API (el log que escribe model_deploy.py en cada
peticion) contra la distribucion de los datos con los que se entreno el
modelo (Base_de_datos.csv). Si empiezan a diferir mucho, es una senal de
alerta -- el mundo real cambio, y el modelo entrenado con datos viejos
puede estar perdiendo precision sin que nadie se de cuenta todavia
(no hace falta esperar a tener las verdaderas etiquetas de pago para
sospechar esto).

Sobre la "periodicidad" que pide el enunciado: este modulo expone la
funcion de computo (calcular_drift / generar_reporte_drift) que se puede
invocar cuantas veces se quiera -- la periodicidad real (ej. correrlo
cada noche) la define quien lo agende (cron, un scheduler de Airflow,
una tarea de GitHub Actions programada, etc.), no este archivo. Aqui se
construye el bloque de computo reutilizable, no el reloj.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Columnas numericas y categoricas crudas (las que de verdad llegan en cada
# peticion al API, antes de cualquier limpieza) sobre las que tiene sentido
# medir drift. No se incluyen columnas de identificacion ni el target.
COLUMNAS_NUMERICAS_DRIFT = [
    "plazo_meses", "edad_cliente", "salario_cliente", "total_otros_prestamos",
    "cuota_pactada", "puntaje_datacredito", "cant_creditosvigentes",
    "huella_consulta", "saldo_mora", "saldo_total", "saldo_principal",
    "creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]
COLUMNAS_CATEGORICAS_DRIFT = ["tipo_laboral", "tendencia_ingresos"]

UMBRAL_P_VALOR = 0.05  # por debajo de esto, se considera que si hay drift
UMBRAL_DIFERENCIA_PROPORCION = 0.10  # 10 puntos porcentuales, para categoricas


def _drift_numerico(referencia: pd.Series, nuevo: pd.Series) -> dict:
    """
    Prueba de Kolmogorov-Smirnov de 2 muestras: compara si dos conjuntos
    de valores numericos vienen de la misma distribucion. No asume
    normalidad (a diferencia de una prueba t), por eso es la eleccion
    estandar para deteccion de drift en variables continuas.
    """
    referencia = referencia.dropna()
    nuevo = nuevo.dropna()
    if len(referencia) < 2 or len(nuevo) < 2:
        return {"estadistico": None, "p_valor": None, "hay_drift": False, "motivo": "datos insuficientes"}

    estadistico, p_valor = ks_2samp(referencia, nuevo)
    return {
        "estadistico": float(estadistico),
        "p_valor": float(p_valor),
        "hay_drift": bool(p_valor < UMBRAL_P_VALOR),
        "media_referencia": float(referencia.mean()),
        "media_nueva": float(nuevo.mean()),
    }


def _drift_categorico(referencia: pd.Series, nuevo: pd.Series) -> dict:
    """
    Compara la proporcion de cada categoria entre referencia y datos
    nuevos. Se marca drift si alguna categoria cambio su proporcion en
    mas de UMBRAL_DIFERENCIA_PROPORCION (10 puntos porcentuales por
    defecto) -- mas facil de explicar a alguien de negocio que un
    chi-cuadrado, y suficiente para una alerta temprana.
    """
    referencia = referencia.dropna()
    nuevo = nuevo.dropna()
    if len(referencia) < 2 or len(nuevo) < 2:
        return {"hay_drift": False, "motivo": "datos insuficientes", "diferencias": {}}

    prop_referencia = referencia.value_counts(normalize=True)
    prop_nuevo = nuevo.value_counts(normalize=True)
    categorias = set(prop_referencia.index) | set(prop_nuevo.index)

    diferencias = {
        cat: float(prop_nuevo.get(cat, 0.0) - prop_referencia.get(cat, 0.0))
        for cat in categorias
    }
    max_diferencia = max(abs(d) for d in diferencias.values()) if diferencias else 0.0

    return {
        "hay_drift": bool(max_diferencia > UMBRAL_DIFERENCIA_PROPORCION),
        "diferencia_maxima": max_diferencia,
        "diferencias": diferencias,
    }


def calcular_drift(df_referencia: pd.DataFrame, df_nuevo: pd.DataFrame) -> dict:
    """
    Calcula drift columna por columna, comparando df_referencia (tipicamente
    Base_de_datos.csv, los datos de entrenamiento) contra df_nuevo
    (tipicamente el log de predicciones reales del API).

    Returns
    -------
    dict con:
      - resultados_numericos: {columna: resultado de _drift_numerico}
      - resultados_categoricos: {columna: resultado de _drift_categorico}
      - columnas_con_drift: lista de nombres de columnas que dispararon alerta
      - n_referencia / n_nuevo: tamanos de cada conjunto comparado
    """
    resultados_numericos = {}
    for col in COLUMNAS_NUMERICAS_DRIFT:
        if col in df_referencia.columns and col in df_nuevo.columns:
            resultados_numericos[col] = _drift_numerico(df_referencia[col], df_nuevo[col])

    resultados_categoricos = {}
    for col in COLUMNAS_CATEGORICAS_DRIFT:
        if col in df_referencia.columns and col in df_nuevo.columns:
            resultados_categoricos[col] = _drift_categorico(df_referencia[col], df_nuevo[col])

    columnas_con_drift = [
        col for col, r in {**resultados_numericos, **resultados_categoricos}.items()
        if r.get("hay_drift")
    ]

    return {
        "resultados_numericos": resultados_numericos,
        "resultados_categoricos": resultados_categoricos,
        "columnas_con_drift": columnas_con_drift,
        "n_referencia": len(df_referencia),
        "n_nuevo": len(df_nuevo),
        "calculado_en": datetime.now(timezone.utc).isoformat(),
    }


def cargar_datos_para_drift(
    ruta_datos_historicos: str | Path, ruta_log_predicciones: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Carga los 2 conjuntos a comparar. Devuelve None (en vez de fallar) si
    el log de predicciones todavia no existe o esta vacio -- es una
    situacion normal (API recien desplegado, sin trafico aun), no un
    error.
    """
    ruta_log_predicciones = Path(ruta_log_predicciones)
    if not ruta_log_predicciones.exists():
        return None

    df_referencia = pd.read_csv(ruta_datos_historicos, sep=";", encoding="utf-8-sig")
    df_nuevo = pd.read_csv(ruta_log_predicciones)
    if df_nuevo.empty:
        return None

    return df_referencia, df_nuevo


def _grafica_comparacion_numerica(referencia: pd.Series, nuevo: pd.Series, columna: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(referencia.dropna(), bins=20, alpha=0.5, density=True, label="Entrenamiento", color="#2E86AB")
    ax.hist(nuevo.dropna(), bins=20, alpha=0.5, density=True, label="Datos nuevos (API)", color="#E63946")
    ax.set_title(f"Distribucion: {columna}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def generar_reporte_drift(
    ruta_salida: str | Path, resultado_drift: dict, df_referencia: pd.DataFrame | None = None,
    df_nuevo: pd.DataFrame | None = None,
) -> Path:
    """Genera un reporte HTML autocontenido con el resultado del drift,
    con graficas de comparacion para las columnas que si dispararon alerta."""
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    filas_numericas = ""
    for col, r in resultado_drift["resultados_numericos"].items():
        alerta = "⚠️ DRIFT" if r.get("hay_drift") else "OK"
        p_valor = r.get("p_valor")
        filas_numericas += (
            f"<tr><td>{col}</td><td>{alerta}</td>"
            f"<td>{p_valor:.4f}</td></tr>\n" if p_valor is not None
            else f"<tr><td>{col}</td><td>{alerta}</td><td>-</td></tr>\n"
        )

    filas_categoricas = ""
    for col, r in resultado_drift["resultados_categoricos"].items():
        alerta = "⚠️ DRIFT" if r.get("hay_drift") else "OK"
        diff = r.get("diferencia_maxima")
        filas_categoricas += (
            f"<tr><td>{col}</td><td>{alerta}</td>"
            f"<td>{diff:.2%}</td></tr>\n" if diff is not None
            else f"<tr><td>{col}</td><td>{alerta}</td><td>-</td></tr>\n"
        )

    graficas_html = ""
    if df_referencia is not None and df_nuevo is not None:
        import matplotlib
        matplotlib.use("Agg")
        for col in resultado_drift["columnas_con_drift"]:
            if col in COLUMNAS_NUMERICAS_DRIFT and col in df_referencia.columns and col in df_nuevo.columns:
                fig = _grafica_comparacion_numerica(df_referencia[col], df_nuevo[col], col)
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
                buffer.seek(0)
                img_b64 = base64.b64encode(buffer.read()).decode("ascii")
                graficas_html += f'<img src="data:image/png;base64,{img_b64}" alt="{col}">\n'
                import matplotlib.pyplot as plt
                plt.close(fig)

    hay_alertas = len(resultado_drift["columnas_con_drift"]) > 0
    resumen_alerta = (
        f"⚠️ Se detecto drift en {len(resultado_drift['columnas_con_drift'])} columna(s): "
        f"{', '.join(resultado_drift['columnas_con_drift'])}"
        if hay_alertas else "✅ No se detecto drift significativo en ninguna columna."
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte de Data Drift — Modelo de riesgo de credito</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; max-width: 800px; margin: 2rem auto; color: #1a1a2e; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; border-bottom: 2px solid #1E2761; padding-bottom: 0.3rem; margin-top: 2rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
  th, td {{ text-align: left; padding: 0.4rem 1rem; border-bottom: 1px solid #ddd; }}
  th {{ color: #555; font-weight: 600; }}
  .alerta {{ padding: 0.8rem; border-radius: 6px; background: {"#fdecea" if hay_alertas else "#eafaf1"}; }}
  .generado {{ color: #888; font-size: 0.85rem; }}
  img {{ max-width: 100%; margin: 0.5rem 0; }}
</style>
</head>
<body>
  <h1>Reporte de Data Drift — Modelo de riesgo de credito</h1>
  <p class="generado">Calculado: {resultado_drift['calculado_en']}
     · Referencia: {resultado_drift['n_referencia']} filas
     · Datos nuevos: {resultado_drift['n_nuevo']} filas</p>

  <p class="alerta">{resumen_alerta}</p>

  <h2>Variables numericas (prueba Kolmogorov-Smirnov)</h2>
  <table>
    <tr><th>Variable</th><th>Estado</th><th>p-valor</th></tr>
    {filas_numericas}
  </table>

  <h2>Variables categoricas (diferencia maxima de proporcion)</h2>
  <table>
    <tr><th>Variable</th><th>Estado</th><th>Diferencia maxima</th></tr>
    {filas_categoricas}
  </table>

  {"<h2>Distribuciones con drift detectado</h2>" + graficas_html if graficas_html else ""}
</body>
</html>"""

    ruta_salida.write_text(html, encoding="utf-8")
    return ruta_salida


if __name__ == "__main__":
    print("Modulo de monitoreo de data drift")
