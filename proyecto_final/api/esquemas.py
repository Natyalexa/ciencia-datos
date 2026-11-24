"""Esquemas de entrada y salida del servicio.

La entrada reproduce la firma registrada en MLflow para el champion:
    comentario_limpio: string | banco: string | log_n_palabras: double | n_neg_lexico: long
    es_solo_simbolos: long | version_app_desconocida: long
con sus rangos y categorías permitidas. log_n_palabras y n_neg_lexico son funciones del texto:
si el cliente no las envía, el servicio las calcula con la misma función de src/pipeline.py.
review_id y lote_id no entran al modelo: sirven para rastrear la predicción.
"""
import math
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.config import BANCOS

Banco = Literal[tuple(BANCOS)]  # type: ignore[valid-type]

MAX_PALABRAS = 500  # la reseña más larga de la OBT tiene menos de 130 palabras; 500 deja margen


class EntradaPrediccion(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "review_id": "gp-0001", "lote_id": 3,
        "comentario_limpio": "no me deja ingresar desde la actualizacion, pesimo servicio",
        "banco": "Banco Pichincha", "es_solo_simbolos": 0, "version_app_desconocida": 0}]})

    review_id: Optional[str] = Field(None, max_length=128, description="Id de la reseña (trazabilidad)")
    lote_id: Optional[int] = Field(None, ge=0, description="Lote de la simulación (trazabilidad)")
    comentario_limpio: str = Field(..., max_length=5000, description="Texto limpio (Silver) de la reseña")
    banco: Banco = Field(..., description="Una de las 8 entidades vistas en entrenamiento")
    log_n_palabras: Optional[float] = Field(None, ge=0.0, le=math.log1p(MAX_PALABRAS))
    n_neg_lexico: Optional[int] = Field(None, ge=0, le=100)
    es_solo_simbolos: int = Field(..., ge=0, le=1)
    version_app_desconocida: int = Field(..., ge=0, le=1)


class SalidaPrediccion(BaseModel):
    id_prediccion: str
    review_id: Optional[str] = None
    probabilidad: Optional[float] = Field(None, description="P(reseña negativa). Nula si decidió la regla de respaldo")
    decision: Literal["priorizar", "cola_normal"]
    accion: str
    umbral: Optional[float] = None
    modelo: str
    version_modelo: str
    alias: Optional[str] = None
    run_id: Optional[str] = None
    respaldo: bool = Field(..., description="True si decidió la regla de negocio y no el modelo")
    motivo_respaldo: Optional[str] = None
    latencia_ms: float


class EstadoSalud(BaseModel):
    estado: Literal["ok", "degradado"]
    modelo_cargado: bool
    modo_respaldo: bool
    motivo_respaldo: Optional[str] = None
    version_modelo: Optional[str] = None
    registro_destino: str


class InfoModelo(BaseModel):
    nombre: str
    version: Optional[str]
    alias: Optional[str]
    run_id: Optional[str]
    uri: str
    umbral: Optional[float]
    cargado_en: Optional[str]
    modo_respaldo: bool


class CambioRespaldo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    activo: bool
    motivo: str = Field("activado manualmente", max_length=300)
