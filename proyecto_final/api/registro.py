"""Registro de predicciones (tabla PREDICCIONES) y de solicitudes (tabla SOLICITUDES).

Escribir fila por fila en Snowflake costaría ~300 ms por predicción, así que el servicio
acumula las filas en memoria y las vacía en bloque (write_pandas) cada VACIAR_SEG segundos,
cuando hay VACIAR_FILAS filas o cuando se llama a POST /admin/vaciar-registro.
Sin Snowflake (pruebas, CI) las filas se agregan a archivos CSV en REGISTRO_DIR.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import pandas as pd

from src import config, datos

log = logging.getLogger("api.registro")


class Registro:
    def __init__(self):
        self.destino = os.environ.get("REGISTRO_DESTINO", "auto").lower()
        if self.destino == "auto":
            self.destino = "snowflake" if datos.snowflake_configurado() else "csv"
        self.dir = os.environ.get("REGISTRO_DIR", "/tmp/registro")
        self.vaciar_seg = float(os.environ.get("VACIAR_SEG", "10"))
        self.vaciar_filas = int(os.environ.get("VACIAR_FILAS", "500"))
        self._buf = {"predicciones": [], "solicitudes": []}
        self._lock = threading.Lock()
        self._parar = threading.Event()
        self._hilo = None
        self.errores = 0

    # ------------------------------------------------------------ ciclo de vida
    def iniciar(self):
        if self._hilo is None:
            self._hilo = threading.Thread(target=self._bucle, daemon=True)
            self._hilo.start()

    def detener(self):
        self._parar.set()
        self.vaciar()

    def _bucle(self):
        while not self._parar.wait(self.vaciar_seg):
            self.vaciar()

    # ------------------------------------------------------------ escritura
    def agregar(self, tipo: str, fila: dict):
        with self._lock:
            self._buf[tipo].append(fila)
            lleno = len(self._buf[tipo]) >= self.vaciar_filas
        if lleno:
            threading.Thread(target=self.vaciar, daemon=True).start()

    def pendientes(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._buf.values())

    def vaciar(self) -> dict:
        with self._lock:
            lotes = {k: v for k, v in self._buf.items() if v}
            self._buf = {"predicciones": [], "solicitudes": []}
        escritas = {}
        for tipo, filas in lotes.items():
            df = pd.DataFrame(filas)
            try:
                escritas[tipo] = self._escribir(tipo, df)
            except Exception:  # noqa: BLE001 - no perder filas si Snowflake falla
                self.errores += 1
                log.exception("No se pudo escribir %s; se reintenta en el siguiente vaciado", tipo)
                with self._lock:
                    self._buf[tipo] = filas + self._buf[tipo]
                escritas[tipo] = 0
        return escritas

    def _escribir(self, tipo: str, df: pd.DataFrame) -> int:
        if self.destino == "snowflake":
            tabla = config.T_PREDICCIONES if tipo == "predicciones" else config.T_SOLICITUDES
            return datos.escribir_df(df, tabla)
        os.makedirs(self.dir, exist_ok=True)
        ruta = os.path.join(self.dir, f"{tipo}.csv")
        df.to_csv(ruta, mode="a", index=False, header=not os.path.exists(ruta))
        return len(df)


def ahora_ms() -> float:
    return time.perf_counter() * 1000
