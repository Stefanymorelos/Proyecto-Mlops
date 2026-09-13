"""
Despliegue del modelo de riesgo de credito como un servicio (API), no como
una aplicacion visual. Expone un endpoint HTTP que recibe un LOTE de
creditos nuevos (datos crudos, mismo esquema que Base_de_datos.csv sin el
target) y devuelve la prediccion de cada uno.

"App" aqui es el termino tecnico para el programa que sirve el modelo
(literalmente `app = FastAPI()`), no una interfaz visual como Streamlit --
esto es infraestructura para que OTROS sistemas consuman el modelo, no un
producto para que un humano llene un formulario.

Se puede correr localmente para pruebas:
    uvicorn model_deploy:app --reload
o empaquetarse en una imagen de Docker (ver Dockerfile) para desplegarlo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.base import clone

try:
    from ft_engineering import pipeline_basemodel
except ImportError:
    # pipeline_basemodel todavia no existe en esta rama de ft_engineering.py
    # (requiere el PR3 mergeado a developer -- ahi se ensambla con ese
    # nombre exacto). El modulo SI se puede importar igual (para que las
    # rutas del API queden registradas y las pruebas que no dependen de
    # esto puedan correr); ServicioPrediccion falla con un mensaje claro
    # si de verdad se intenta usar sin el PR3 disponible.
    pipeline_basemodel = None

RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", "mejor_modelo_final.joblib"))
RUTA_DATOS_HISTORICOS = Path(os.environ.get("RUTA_DATOS_HISTORICOS", "Base_de_datos.csv"))
UMBRAL_DECISION = float(os.environ.get("UMBRAL_DECISION", "0.5"))
TIPOS_CREDITO_SOPORTADOS = {4, 9}  # los unicos que el modelo vio en entrenamiento (ver ft_engineering.EliminarCategorias)


# ---------------------------------------------------------------------------
# Esquemas de entrada/salida (contrato del API)
# ---------------------------------------------------------------------------

class CreditoNuevo(BaseModel):
    """
    Datos crudos de un credito nuevo, en el MISMO esquema que
    Base_de_datos.csv (sin la columna objetivo, que todavia no existe para
    un cliente nuevo). No se piden 'puntaje' (fuga de informacion, ver
    Hallazgo 1 del EDA), 'fecha_prestamo', 'capital_prestado' ni
    'saldo_mora_codeudor' -- ft_engineering.py las descarta de todas formas,
    asi que no tiene sentido exigirselas a quien consuma el API.
    """

    tipo_credito: int = Field(..., description="Codigo del tipo de credito (el modelo solo soporta 4 y 9)")
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: float | None = None
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float | None = None
    saldo_total: float | None = None
    saldo_principal: float | None = None
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float | None = None
    tendencia_ingresos: str | None = None


class LotePrediccion(BaseModel):
    creditos: list[CreditoNuevo]


class PrediccionCredito(BaseModel):
    indice: int
    soportado: bool
    motivo_no_soportado: str | None = None
    probabilidad_mora: float | None = None
    prediccion: Literal["mora", "a_tiempo"] | None = None
    riesgo_alto: bool | None = None


class RespuestaLote(BaseModel):
    modelo: str
    umbral_decision: float
    total_recibidos: int
    total_predichos: int
    predicciones: list[PrediccionCredito]


# ---------------------------------------------------------------------------
# Servicio de prediccion: encapsula el estado "pesado" (modelo + limpiador
# ya ajustado) para no tener que recargarlos en cada peticion.
# ---------------------------------------------------------------------------

class ServicioPrediccion:
    def __init__(self, ruta_modelo: Path, ruta_datos_historicos: Path):
        if pipeline_basemodel is None:
            raise ImportError(
                "ft_engineering.pipeline_basemodel no esta disponible en esta rama "
                "(requiere el PR3 mergeado a developer)."
            )
        if not ruta_modelo.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo en {ruta_modelo}. "
                "Corre model_training_evaluation.py primero para generarlo."
            )
        self.nombre_modelo = ruta_modelo.stem
        self.pipeline_ml = joblib.load(ruta_modelo)

        df_historico = pd.read_csv(ruta_datos_historicos, sep=";", encoding="utf-8-sig")
        objetivo = "Pago_atiempo"
        if objetivo in df_historico.columns:
            df_historico = df_historico.drop(columns=[objetivo])

        # El limpiador se ajusta UNA sola vez, con los datos historicos --
        # igual que en entrenamiento -- y de ahi en adelante solo se usa
        # transform() sobre cada peticion nueva. Nunca se reajusta con datos
        # de un cliente nuevo: eso filtraria informacion de una peticion
        # hacia el preprocesamiento de otra.
        self.limpiador = clone(pipeline_basemodel)
        self.limpiador.fit(df_historico)

    def _transformar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica cada paso del limpiador ya ajustado, en secuencia manual.

        No se usa self.limpiador.transform(df) directo porque el ultimo paso
        (ColumnasIrrelevantes) no guarda ningun atributo propio al ajustarse
        -- es video, solo elimina columnas -- y sklearn interpreta eso como
        "el pipeline nunca se ajusto", aunque si se llamo fit(). Esto solo
        aparece en el patron fit() + transform() por separado (como aqui, en
        despliegue); nunca se noto en entrenamiento porque ahi siempre se usa
        fit_transform() de una sola vez, que no hace esa validacion.
        """
        resultado = df
        for _, paso in self.limpiador.steps:
            resultado = paso.transform(resultado)
        return resultado

    def predecir_lote(self, registros: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la limpieza + el modelo sobre un lote de creditos nuevos.

        Nota importante: el limpiador (via EliminarCategorias) descarta
        filas con tipo_credito fuera de {4, 9} -- el modelo nunca aprendio
        a predecir esos casos. En vez de dejar que esas filas desaparezcan
        en silencio (lo cual desalinearia los indices de la respuesta), se
        detectan ANTES de transformar y se marcan explicitamente como "no
        soportado" en la respuesta final.
        """
        soportado = registros["tipo_credito"].isin(TIPOS_CREDITO_SOPORTADOS)

        resultado = pd.DataFrame(index=registros.index)
        resultado["soportado"] = soportado
        resultado["motivo_no_soportado"] = None
        resultado.loc[~soportado, "motivo_no_soportado"] = (
            "tipo_credito no soportado por el modelo (solo se entreno con 4 y 9)"
        )
        resultado["probabilidad_mora"] = None
        resultado["prediccion"] = None
        resultado["riesgo_alto"] = None

        if soportado.any():
            limpio = self._transformar(registros.loc[soportado])
            proba_a_tiempo = self.pipeline_ml.predict_proba(limpio)[:, 1]
            proba_mora = 1 - proba_a_tiempo

            resultado.loc[soportado, "probabilidad_mora"] = proba_mora
            resultado.loc[soportado, "prediccion"] = [
                "mora" if p >= UMBRAL_DECISION else "a_tiempo" for p in proba_mora
            ]
            resultado.loc[soportado, "riesgo_alto"] = proba_mora >= UMBRAL_DECISION

        return resultado


# ---------------------------------------------------------------------------
# Aplicacion FastAPI
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    obtener_servicio()  # cargar el modelo al iniciar
    yield


app = FastAPI(
    title="API de Riesgo de Credito",
    description="Sirve el mejor modelo entrenado en model_training_evaluation.py para predicciones por lote.",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)

_servicio: ServicioPrediccion | None = None


def obtener_servicio() -> ServicioPrediccion:
    global _servicio
    if _servicio is None:
        _servicio = ServicioPrediccion(RUTA_MODELO, RUTA_DATOS_HISTORICOS)
    return _servicio


@app.get("/salud")
def salud():
    """Chequeo simple de que el servicio esta arriba y el modelo cargado."""
    try:
        servicio = obtener_servicio()
        return {"estado": "ok", "modelo": servicio.nombre_modelo}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predecir_lote", response_model=RespuestaLote)
def predecir_lote(lote: LotePrediccion) -> RespuestaLote:
    if not lote.creditos:
        raise HTTPException(status_code=400, detail="El lote no puede venir vacio")

    servicio = obtener_servicio()
    registros = pd.DataFrame([c.model_dump() for c in lote.creditos])
    resultado = servicio.predecir_lote(registros)

    predicciones = [
        PrediccionCredito(
            indice=i,
            soportado=bool(fila["soportado"]),
            motivo_no_soportado=fila["motivo_no_soportado"],
            probabilidad_mora=fila["probabilidad_mora"],
            prediccion=fila["prediccion"],
            riesgo_alto=fila["riesgo_alto"],
        )
        for i, fila in resultado.reset_index(drop=True).iterrows()
    ]

    return RespuestaLote(
        modelo=servicio.nombre_modelo,
        umbral_decision=UMBRAL_DECISION,
        total_recibidos=len(lote.creditos),
        total_predichos=int(resultado["soportado"].sum()),
        predicciones=predicciones,
    )


if __name__ == "__main__":
    import uvicorn
    # Solo para pruebas rapidas locales (python model_deploy.py). Docker
    # NUNCA ejecuta este bloque -- el contenedor arranca directo con el
    # CMD del Dockerfile (uvicorn por linea de comandos, con --host
    # 0.0.0.0, que si es necesario ahi para el mapeo de puertos). Aqui,
    # para uso local, no hay razon para exponerse a toda la red: basta con
    # localhost.
    uvicorn.run(app, host="127.0.0.1", port=8000)
