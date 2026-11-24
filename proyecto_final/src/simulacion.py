"""Simulación de producción (sección 5).

    python -m src.simulacion preparar [--reiniciar]   # arma los lotes, la deriva y la referencia
    python -m src.simulacion puntuar --lote 3         # envía el lote 3 a la API (como producción)

Construcción de los lotes
- Fecha de corte = la reseña más reciente del entrenamiento del champion (split temporal 80/20 del
  snapshot de la OBT usado en el PSet #3). Todo lo posterior a esa fecha nunca lo vio el modelo.
- Un lote = una semana (lunes a domingo), el ritmo de la ingesta y de la reunión de atención al
  cliente. Las semanas con menos de MIN_FILAS_LOTE reseñas se unen a la siguiente. Si quedan
  menos de 8 semanas, se divide el período en MAX_LOTES tramos consecutivos de igual tamaño.
- El target de un lote está disponible HORIZONTE_LOTES lote(s) después de la predicción.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config, datos
from src.pipeline import FEATURES, TARGET, agregar_features, conteo_lexico, split_temporal

COLS_LOTE = ["review_id", "lote_id", "periodo_inicio", "periodo_fin", "fecha_target_disponible",
             "fecha_resena", "banco", "comentario_limpio", "n_palabras", "log_n_palabras", "n_neg_lexico",
             "es_solo_simbolos", "version_app_desconocida", "es_negativa", "con_deriva",
             "alterado_texto", "alterado_banco", "fecha_corte_entrenamiento", "creado_en"]


def kestra_outputs(**vals):
    """Kestra lee las salidas de un script desde una línea ::{"outputs": {...}}::"""
    print("::" + json.dumps({"outputs": vals}, default=str) + "::")


# ---------------------------------------------------------------- lotes
def asignar_lotes(post: pd.DataFrame) -> pd.DataFrame:
    d = post.sort_values(["fecha_resena", "review_id"], kind="mergesort").reset_index(drop=True)
    semana = d["fecha_resena"].dt.to_period("W-SUN").dt.start_time
    tam = semana.value_counts().sort_index()
    # unir semanas chicas con la siguiente (la última, con la anterior)
    grupos, actual, acumulado = {}, 0, 0
    for s, n in tam.items():
        grupos[s] = actual
        acumulado += n
        if acumulado >= config.MIN_FILAS_LOTE:
            actual, acumulado = actual + 1, 0
    if acumulado and actual > 0:  # cola chica al final
        for s in [k for k, v in grupos.items() if v == actual]:
            grupos[s] = actual - 1
    d["lote_id"] = semana.map(grupos).astype(int) + 1
    d["criterio"] = "semana"
    if d["lote_id"].nunique() < config.MIN_LOTES:
        d["lote_id"] = (np.arange(len(d)) * config.MAX_LOTES // len(d)) + 1
        d["criterio"] = f"{config.MAX_LOTES} tramos consecutivos de igual tamaño"
    lim = d.groupby("lote_id")["fecha_resena"].agg(["min", "max"])
    d["periodo_inicio"] = d["lote_id"].map(lim["min"]).dt.normalize()
    d["periodo_fin"] = d["lote_id"].map(lim["max"]).dt.normalize()
    return d


def inducir_deriva(d: pd.DataFrame) -> pd.DataFrame:
    """Deriva inducida, documentada en docs/deriva_inducida.md.

    En los últimos DRIFT_LOTES_FINALES lotes:
    1. Recorte de texto: DRIFT_FRAC_TRUNCADAS de las reseñas con más de DRIFT_PALABRAS palabras se
       recortan a sus primeras DRIFT_PALABRAS palabras (simula que la fuente empieza a entregar
       solo el inicio del comentario). Se recalculan n_palabras, log_n_palabras y n_neg_lexico.
       El target (las estrellas) no cambia: cambia la relación entre el texto visible y el target.
    2. Banco nuevo: DRIFT_FRAC_BANCO_NUEVO de las filas pasan a un banco que el modelo no conoce.
    """
    rng = np.random.default_rng(config.SEMILLA)
    n_lotes = d["lote_id"].max()
    desde = n_lotes - config.DRIFT_LOTES_FINALES + 1
    d["con_deriva"] = d["lote_id"] >= desde
    d["alterado_texto"] = False
    d["alterado_banco"] = False
    idx = d.index[d["con_deriva"] & (d["n_palabras"] > config.DRIFT_PALABRAS)]
    trunc = rng.choice(idx, size=int(len(idx) * config.DRIFT_FRAC_TRUNCADAS), replace=False)
    d.loc[trunc, "comentario_limpio"] = d.loc[trunc, "comentario_limpio"].map(
        lambda t: " ".join(str(t).split()[:config.DRIFT_PALABRAS]))
    d.loc[trunc, "alterado_texto"] = True
    idx_b = d.index[d["con_deriva"]]
    nuevo = rng.choice(idx_b, size=int(len(idx_b) * config.DRIFT_FRAC_BANCO_NUEVO), replace=False)
    d.loc[nuevo, "banco"] = config.DRIFT_BANCO_NUEVO
    d.loc[nuevo, "alterado_banco"] = True
    # recalcular las features que dependen del texto
    d["n_palabras"] = d["comentario_limpio"].fillna("").map(lambda t: len(str(t).split()))
    d["log_n_palabras"] = np.log1p(d["n_palabras"].astype(float))
    d["n_neg_lexico"] = d["comentario_limpio"].fillna("").map(conteo_lexico)
    return d


def cargar_snapshot() -> tuple[pd.DataFrame, str]:
    tabla = config.TABLA_OBT_SNAPSHOT
    if datos.existe_tabla(tabla):
        return datos.cargar_obt(tabla), tabla
    print(f"AVISO: no existe {tabla}; se usa la OBT actual para reconstruir el corte del entrenamiento.")
    return datos.cargar_obt(config.TABLA_OBT), config.TABLA_OBT


def preparar(reiniciar: bool = False) -> dict:
    from src.tablas import asegurar_tablas
    asegurar_tablas(reiniciar=reiniciar)
    snap, origen = cargar_snapshot()
    snap = agregar_features(snap.assign(fecha_resena=pd.to_datetime(snap["fecha_resena"])))
    train, _ = split_temporal(snap, 0.2)
    corte = train["fecha_resena"].max()

    actual = datos.cargar_obt(config.TABLA_OBT)
    actual = agregar_features(actual.assign(fecha_resena=pd.to_datetime(actual["fecha_resena"])))
    post = actual[actual["fecha_resena"] > corte].copy()
    if len(post) < config.MIN_LOTES * 50:
        sys.exit(f"Solo hay {len(post)} reseñas posteriores al corte {corte}: no alcanza para 8 lotes.")

    d = inducir_deriva(asignar_lotes(post))
    d["fecha_target_disponible"] = d.groupby("lote_id")["periodo_fin"].transform("max") + pd.Timedelta(
        days=7 * config.HORIZONTE_LOTES)
    d["fecha_corte_entrenamiento"] = corte
    d["creado_en"] = datetime.now(timezone.utc).replace(tzinfo=None)
    for c in ("periodo_inicio", "periodo_fin", "fecha_target_disponible"):
        d[c] = pd.to_datetime(d[c]).dt.date
    datos.escribir_df(d[COLS_LOTE], config.T_LOTES, sobrescribir=True)

    resumen = (d.groupby("lote_id").agg(inicio=("periodo_inicio", "min"), fin=("periodo_fin", "max"),
                                        n=("review_id", "size"), pct_negativa=(TARGET, "mean"),
                                        con_deriva=("con_deriva", "max"), recortadas=("alterado_texto", "sum"),
                                        banco_nuevo=("alterado_banco", "sum")))
    print(f"Origen del corte: {origen} | fecha de corte del entrenamiento: {corte}")
    print(f"Criterio de lotes: {d['criterio'].iloc[0]} | lotes: {d['lote_id'].nunique()} | filas: {len(d)}")
    print(resumen.round(3).to_string())

    from monitoreo.referencia import construir_referencia
    construir_referencia(train)
    lotes = sorted(int(x) for x in d["lote_id"].unique())
    kestra_outputs(lotes=lotes, n_lotes=len(lotes), corte=str(corte))
    return {"lotes": lotes, "corte": str(corte)}


# ---------------------------------------------------------------- puntuación
def puntuar(lote_id: int, api_url: str | None = None) -> dict:
    """Envía cada reseña del lote a POST /predecir, como lo haría la integración en producción."""
    import requests
    api_url = api_url or os.environ.get("API_URL", "http://api:8000")
    df = datos.leer_sql(f"SELECT * FROM {config.T_LOTES} WHERE lote_id = %s ORDER BY fecha_resena", (lote_id,))
    if df.empty:
        sys.exit(f"El lote {lote_id} no existe en {config.T_LOTES}")
    s = requests.Session()
    s.headers.update({"X-Lote-Id": str(lote_id)})
    estados, t0 = {}, time.time()
    for fila in df.to_dict("records"):
        cuerpo = {"review_id": fila["review_id"], "lote_id": int(lote_id),
                  "comentario_limpio": fila["comentario_limpio"] or "", "banco": fila["banco"],
                  "log_n_palabras": float(fila["log_n_palabras"]), "n_neg_lexico": int(fila["n_neg_lexico"]),
                  "es_solo_simbolos": int(fila["es_solo_simbolos"]),
                  "version_app_desconocida": int(fila["version_app_desconocida"])}
        for intento in range(3):
            try:
                r = s.post(f"{api_url}/predecir", json=cuerpo, timeout=30)
                break
            except requests.RequestException:
                if intento == 2:
                    raise
                time.sleep(2 ** intento)
        estados[r.status_code] = estados.get(r.status_code, 0) + 1
    token = os.environ.get("ADMIN_TOKEN", "")
    vaciado = s.post(f"{api_url}/admin/vaciar-registro", headers={"X-Token-Admin": token}, timeout=120)
    info = s.get(f"{api_url}/modelo", timeout=30).json()
    print(f"Lote {lote_id}: {len(df)} reseñas en {time.time() - t0:.1f} s | estados HTTP {estados} | "
          f"modelo versión {info.get('version')} | registro {vaciado.json() if vaciado.ok else vaciado.status_code}")
    kestra_outputs(lote_id=lote_id, n=len(df), estados={str(k): v for k, v in estados.items()},
                   version_modelo=info.get("version"))
    return estados


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("preparar")
    p1.add_argument("--reiniciar", action="store_true", help="vacía PREDICCIONES, MONITOREO, etc.")
    p2 = sub.add_parser("puntuar")
    p2.add_argument("--lote", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "preparar":
        preparar(a.reiniciar)
    else:
        puntuar(a.lote)
