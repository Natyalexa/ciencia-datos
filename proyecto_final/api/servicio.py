"""Carga del modelo por alias, recarga en caliente, respaldo e interruptor.

- MODELO_URI = models:/<nombre>@champion (por defecto) o una ruta local (pruebas/CI).
- Cada RECARGA_SEG segundos se consulta a qué versión apunta el alias; si cambió, se carga la
  nueva versión sin reiniciar ni reconstruir la imagen (promoción y rollback).
- Si el modelo no carga, el servicio no se cae: responde con la regla de negocio (línea base
  del PSet #3) y lo indica en cada respuesta (respaldo=True).
- El interruptor de respaldo (variable MODO_RESPALDO o POST /admin/respaldo) fuerza la regla.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config
from src.pipeline import FEATURES, conteo_lexico

log = logging.getLogger("api.servicio")


def _verdadero(v: str | None) -> bool:
    return str(v).strip().lower() in {"1", "true", "si", "sí", "yes", "on"}


class ServicioModelo:
    def __init__(self, uri: str | None = None):
        self.uri = uri or os.environ.get("MODELO_URI", config.MODELO_URI)
        self.recarga_seg = float(os.environ.get("RECARGA_SEG", "30"))
        self.modelo = None
        self.version: str | None = None
        self.alias: str | None = None
        self.run_id: str | None = None
        self.umbral: float | None = None
        self.cargado_en: str | None = None
        self.error_carga: str | None = None
        self.respaldo_manual = _verdadero(os.environ.get("MODO_RESPALDO"))
        self.motivo_manual = "variable de entorno MODO_RESPALDO" if self.respaldo_manual else None
        self._lock = threading.Lock()
        self._ultima_revision = 0.0

    # ------------------------------------------------------------ carga
    @property
    def es_registro(self) -> bool:
        return self.uri.startswith("models:/")

    def _resolver_registro(self):
        """Devuelve (version, run_id, umbral, alias) del alias en el Model Registry."""
        import mlflow
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        cli = mlflow.MlflowClient()
        nombre, alias = self.uri[len("models:/"):].split("@")
        mv = cli.get_model_version_by_alias(nombre, alias)
        met = cli.get_run(mv.run_id).data.metrics
        umbral = met.get("umbral_aplicado", met.get("umbral_costo", config.UMBRAL_POR_DEFECTO))
        return str(mv.version), mv.run_id, float(umbral), alias

    def _resolver_local(self):
        meta_path = os.path.join(self.uri, "metadata.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        return (str(meta.get("version", "local")), meta.get("run_id"),
                float(meta.get("umbral", config.UMBRAL_POR_DEFECTO)), meta.get("alias", "local"))

    def cargar(self, forzar: bool = False) -> bool:
        """Carga (o recarga) el modelo. Nunca lanza excepción: si falla, queda en respaldo."""
        with self._lock:
            try:
                import mlflow.sklearn
                version, run_id, umbral, alias = (self._resolver_registro() if self.es_registro
                                                  else self._resolver_local())
                if not forzar and self.modelo is not None and version == self.version:
                    return True
                uri = f"models:/{config.NOMBRE_MODELO}/{version}" if self.es_registro else self.uri
                modelo = mlflow.sklearn.load_model(uri)
                self.modelo, self.version, self.run_id, self.umbral, self.alias = modelo, version, run_id, umbral, alias
                self.cargado_en = datetime.now(timezone.utc).isoformat()
                self.error_carga = None
                log.info("Modelo cargado: %s versión %s (run %s, umbral %.2f)", config.NOMBRE_MODELO, version, run_id, umbral)
                return True
            except Exception as e:  # noqa: BLE001 - cualquier fallo lleva al respaldo
                self.error_carga = f"{type(e).__name__}: {e}"[:300]
                log.error("No se pudo cargar el modelo (%s). Se usará la regla de respaldo.", self.error_carga)
                if self.modelo is None:
                    self.version = self.run_id = None
                return False

    def revisar_alias(self) -> None:
        """Recarga perezosa: como mucho una consulta al registro cada RECARGA_SEG segundos."""
        if not self.es_registro or time.time() - self._ultima_revision < self.recarga_seg:
            return
        self._ultima_revision = time.time()
        self.cargar()

    # ------------------------------------------------------------ respaldo
    @property
    def en_respaldo(self) -> bool:
        return self.respaldo_manual or self.modelo is None

    @property
    def motivo_respaldo(self) -> str | None:
        if self.respaldo_manual:
            return f"interruptor activado: {self.motivo_manual}"
        if self.modelo is None:
            return f"el modelo no cargó ({self.error_carga})"
        return None

    def activar_respaldo(self, activo: bool, motivo: str) -> None:
        self.respaldo_manual = activo
        self.motivo_manual = motivo if activo else None

    # ------------------------------------------------------------ predicción
    @staticmethod
    def preparar(entrada: dict) -> pd.DataFrame:
        texto = entrada["comentario_limpio"] or ""
        n_pal = len(texto.split())
        fila = {
            "comentario_limpio": texto,
            "banco": entrada["banco"],
            "log_n_palabras": float(entrada["log_n_palabras"]) if entrada.get("log_n_palabras") is not None
            else float(np.log1p(n_pal)),
            "n_neg_lexico": int(entrada["n_neg_lexico"]) if entrada.get("n_neg_lexico") is not None
            else int(conteo_lexico(texto)),
            "es_solo_simbolos": int(entrada["es_solo_simbolos"]),
            "version_app_desconocida": int(entrada["version_app_desconocida"]),
        }
        return pd.DataFrame([fila])[FEATURES]

    def predecir(self, entrada: dict) -> dict:
        """Devuelve probabilidad (o None), decisión y metadatos de la versión que decidió."""
        X = self.preparar(entrada)
        if not self.en_respaldo:
            try:
                p = float(self.modelo.predict_proba(X)[:, 1][0])
                dec = config.DECISION_POSITIVA if p >= self.umbral else config.DECISION_NEGATIVA
                return {"probabilidad": round(p, 6), "decision": dec, "umbral": self.umbral,
                        "version_modelo": self.version, "alias": self.alias, "run_id": self.run_id,
                        "respaldo": False, "motivo_respaldo": None, "entradas": X.iloc[0].to_dict()}
            except Exception as e:  # noqa: BLE001
                log.exception("Falló la predicción del modelo; se responde con la regla de respaldo")
                motivo = f"error al predecir: {type(e).__name__}"
        else:
            motivo = self.motivo_respaldo
        # Línea base de negocio del PSet #3: priorizar si el texto contiene alguna palabra de queja.
        dec = config.DECISION_POSITIVA if int(X["n_neg_lexico"].iloc[0]) >= 1 else config.DECISION_NEGATIVA
        return {"probabilidad": None, "decision": dec, "umbral": None, "version_modelo": "regla-negocio",
                "alias": None, "run_id": None, "respaldo": True, "motivo_respaldo": motivo,
                "entradas": X.iloc[0].to_dict()}
