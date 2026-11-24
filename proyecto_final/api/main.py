"""Servicio de predicción: priorización de reseñas negativas de apps bancarias.

Endpoints mínimos: GET /salud, GET /modelo, POST /predecir.
Administración (requieren el encabezado X-Token-Admin = ADMIN_TOKEN):
    POST /admin/recargar          vuelve a resolver el alias y carga la versión a la que apunta
    POST /admin/respaldo          interruptor: {"activo": true} hace decidir a la regla de negocio
    POST /admin/vaciar-registro   escribe ya en Snowflake las filas pendientes
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.esquemas import CambioRespaldo, EntradaPrediccion, EstadoSalud, InfoModelo, SalidaPrediccion
from api.registro import Registro
from api.servicio import ServicioModelo
from src import config

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

servicio = ServicioModelo()
registro = Registro()


@asynccontextmanager
async def ciclo_vida(app: FastAPI):
    if registro.destino == "snowflake":
        try:
            from src.tablas import asegurar_tablas
            asegurar_tablas()
        except Exception:  # noqa: BLE001
            log.exception("No se pudieron verificar las tablas de PRODUCCION")
    servicio.cargar(forzar=True)
    registro.iniciar()
    yield
    registro.detener()


app = FastAPI(title="Priorización de reseñas negativas · apps bancarias de Ecuador",
              version="1.0.0", lifespan=ciclo_vida)


def _utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------- registro de cada solicitud
@app.middleware("http")
async def medir(request: Request, call_next):
    t0 = time.perf_counter()
    estado, error = 500, None
    try:
        respuesta = await call_next(request)
        estado = respuesta.status_code
        return respuesta
    except Exception as e:  # noqa: BLE001
        error = type(e).__name__
        raise
    finally:
        if request.url.path not in ("/salud", "/docs", "/openapi.json", "/favicon.ico"):
            lote = request.headers.get("x-lote-id")
            registro.agregar("solicitudes", {
                "id_solicitud": str(uuid.uuid4()), "ts": _utc_naive(), "metodo": request.method,
                "ruta": request.url.path, "estado_http": estado,
                "latencia_ms": round((time.perf_counter() - t0) * 1000, 3),
                "lote_id": int(lote) if lote and lote.isdigit() else None, "error": error})


@app.exception_handler(RequestValidationError)
async def entrada_invalida(request: Request, exc: RequestValidationError):
    """422 con el campo exacto que falló, en lenguaje claro."""
    detalle = [{"campo": ".".join(str(p) for p in e["loc"] if p != "body"),
                "error": e["msg"], "tipo": e["type"],
                **({"recibido": e["input"]} if not isinstance(e.get("input"), dict) else {})}
               for e in exc.errors()]
    return JSONResponse(status_code=422, content={"mensaje": "Entrada inválida", "detalle": detalle})


def token_admin(x_token_admin: str | None = Header(default=None)):
    esperado = os.environ.get("ADMIN_TOKEN")
    if not esperado or x_token_admin != esperado:
        raise HTTPException(status_code=401, detail="Token de administración inválido")


# ---------------------------------------------------------------- endpoints
@app.get("/salud", response_model=EstadoSalud)
def salud():
    servicio.revisar_alias()
    return EstadoSalud(estado="degradado" if servicio.en_respaldo else "ok",
                       modelo_cargado=servicio.modelo is not None, modo_respaldo=servicio.en_respaldo,
                       motivo_respaldo=servicio.motivo_respaldo, version_modelo=servicio.version,
                       registro_destino=registro.destino)


@app.get("/modelo", response_model=InfoModelo)
def modelo():
    servicio.revisar_alias()
    return InfoModelo(nombre=config.NOMBRE_MODELO, version=servicio.version, alias=servicio.alias,
                      run_id=servicio.run_id, uri=servicio.uri, umbral=servicio.umbral,
                      cargado_en=servicio.cargado_en, modo_respaldo=servicio.en_respaldo)


@app.post("/predecir", response_model=SalidaPrediccion)
def predecir(entrada: EntradaPrediccion):
    t0 = time.perf_counter()
    servicio.revisar_alias()
    r = servicio.predecir(entrada.model_dump())
    latencia = round((time.perf_counter() - t0) * 1000, 3)
    salida = SalidaPrediccion(
        id_prediccion=str(uuid.uuid4()), review_id=entrada.review_id, probabilidad=r["probabilidad"],
        decision=r["decision"], accion=config.ACCIONES[r["decision"]], umbral=r["umbral"],
        modelo=config.NOMBRE_MODELO, version_modelo=r["version_modelo"], alias=r["alias"],
        run_id=r["run_id"], respaldo=r["respaldo"], motivo_respaldo=r["motivo_respaldo"], latencia_ms=latencia)
    registro.agregar("predicciones", {
        "id_prediccion": salida.id_prediccion, "ts_prediccion": _utc_naive(), "review_id": entrada.review_id,
        "lote_id": entrada.lote_id, "entradas": json.dumps(r["entradas"], ensure_ascii=False, default=float),
        "probabilidad": r["probabilidad"], "decision": r["decision"], "umbral": r["umbral"],
        "modelo": config.NOMBRE_MODELO, "version_modelo": r["version_modelo"], "alias": r["alias"],
        "run_id": r["run_id"], "respaldo": r["respaldo"], "motivo_respaldo": r["motivo_respaldo"],
        "latencia_ms": latencia})
    return salida


@app.post("/admin/recargar", response_model=InfoModelo, dependencies=[Depends(token_admin)])
def recargar():
    servicio.cargar(forzar=True)
    return modelo()


@app.post("/admin/respaldo", response_model=EstadoSalud, dependencies=[Depends(token_admin)])
def cambiar_respaldo(cambio: CambioRespaldo):
    servicio.activar_respaldo(cambio.activo, cambio.motivo)
    log.warning("Interruptor de respaldo %s: %s", "ACTIVADO" if cambio.activo else "desactivado", cambio.motivo)
    return salud()


@app.post("/admin/vaciar-registro", dependencies=[Depends(token_admin)])
def vaciar_registro():
    return {"escritas": registro.vaciar(), "pendientes": registro.pendientes(), "errores": registro.errores}
