"""Monitoreo después de cada lote (lo ejecuta kestra/monitoreo.yml).

    python -m monitoreo.ejecutar --lote 5

En el lote t se calcula:
- Sistema y predicciones del lote t (latencia, errores, volumen, distribución y tasa de decisión).
- Datos del lote t (faltantes, fuera de rango, categorías nuevas, PSI/KS/chi²).
- Modelo y negocio del lote t − HORIZONTE: su target recién llegó.
- Diagnóstico (tabla 6.2), alertas y la regla de disparo del reentrenamiento.
Escribe en MONITOREO y ALERTAS y deja las salidas para Kestra.
"""
from __future__ import annotations

import argparse
import json
import math
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from api.esquemas import MAX_PALABRAS
from monitoreo import deriva
from monitoreo.alertas import CATALOGO, regla_disparo
from monitoreo.referencia import CATEGORICAS, CONTINUAS, leer_referencia, preparar_columnas
from src import config as C
from src import datos
from src.pipeline import costo_negocio
from src.simulacion import kestra_outputs


class Acumulador:
    def __init__(self, lote_calculo: int):
        self.lote_calculo = lote_calculo
        self.ts = datetime.now(timezone.utc).replace(tzinfo=None)
        self.metricas, self.alertas = [], []

    def m(self, lote, tipo, metrica, valor, umbral=None, estado="ok", feature=None, version=None, detalle=None):
        self.metricas.append({"lote_id": lote, "lote_calculo": self.lote_calculo, "ts": self.ts, "tipo": tipo,
                              "metrica": metrica, "feature": feature,
                              "valor": None if valor is None or (isinstance(valor, float) and math.isnan(valor))
                              else float(valor),
                              "umbral": None if umbral is None else float(umbral), "estado": estado,
                              "version_modelo": version, "detalle": detalle})

    def a(self, clave, lote, valor, umbral=None, feature=None):
        d = CATALOGO[clave]
        self.alertas.append({"id_alerta": str(uuid.uuid4()), "ts": self.ts, "lote_id": lote,
                             "lote_calculo": self.lote_calculo, "nombre": d.nombre, "metrica": d.metrica,
                             "feature": feature, "valor": None if valor is None else float(valor),
                             "umbral": None if umbral is None else float(umbral), "ventana": d.ventana,
                             "severidad": d.severidad, "receptor": d.receptor, "accion": d.accion})


def metricas_version(version: str, ref: dict) -> dict:
    """Media y desviación de la CV registradas para la versión que hizo las predicciones."""
    import mlflow
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    cli = mlflow.MlflowClient()
    try:
        met = cli.get_run(cli.get_model_version(C.NOMBRE_MODELO, str(version)).run_id).data.metrics
    except Exception:  # noqa: BLE001
        met = {}
    cv = ref.get("cv_champion", {})
    return {k: met.get(k, cv.get(k)) for k in ("cv_roc_auc_mean", "cv_roc_auc_std", "cv_pr_auc_mean",
                                               "cv_pr_auc_std", "cv_costo_mean", "cv_costo_std")}


def version_champion_actual() -> str:
    import mlflow
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    return str(mlflow.MlflowClient().get_model_version_by_alias(C.NOMBRE_MODELO, C.ALIAS_PRODUCCION).version)


# ---------------------------------------------------------------- bloques de métricas
def sistema(acc: Acumulador, lote: int):
    s = datos.leer_sql(f"SELECT estado_http, latencia_ms FROM {C.T_SOLICITUDES} "
                       f"WHERE lote_id = %s AND ruta = '/predecir'", (lote,))
    if s.empty:
        return
    n = len(s)
    p50, p95 = np.percentile(s["latencia_ms"], [50, 95])
    t4 = float(((s["estado_http"] >= 400) & (s["estado_http"] < 500)).mean())
    t5 = float((s["estado_http"] >= 500).mean())
    acc.m(lote, "sistema", "volumen_solicitudes", n)
    acc.m(lote, "sistema", "latencia_p50_ms", p50)
    acc.m(lote, "sistema", "latencia_p95_ms", p95, C.LATENCIA_P95_MS, "alerta" if p95 > C.LATENCIA_P95_MS else "ok")
    acc.m(lote, "sistema", "tasa_4xx", t4, C.TASA_4XX_MAX, "alerta" if t4 > C.TASA_4XX_MAX else "ok")
    acc.m(lote, "sistema", "tasa_5xx", t5, C.TASA_5XX_MAX, "alerta" if t5 > C.TASA_5XX_MAX else "ok")
    if p95 > C.LATENCIA_P95_MS:
        acc.a("latencia", lote, p95, C.LATENCIA_P95_MS)
    if t4 > C.TASA_4XX_MAX:
        acc.a("errores_4xx", lote, t4, C.TASA_4XX_MAX)
    if t5 > C.TASA_5XX_MAX:
        acc.a("errores_5xx", lote, t5, C.TASA_5XX_MAX)


def predicciones(acc: Acumulador, lote: int, ref: dict) -> str | None:
    p = datos.leer_sql(f"SELECT probabilidad, decision, version_modelo, respaldo FROM {C.T_PREDICCIONES} "
                       f"WHERE lote_id = %s", (lote,))
    if p.empty:
        return None
    version = p["version_modelo"].mode().iloc[0]
    meta = ref["meta"]
    tasa = float((p["decision"] == C.DECISION_POSITIVA).mean())
    resp = float(p["respaldo"].astype(bool).mean())
    acc.m(lote, "modelo", "n_predicciones", len(p), version=version)
    acc.m(lote, "modelo", "tasa_priorizar", tasa, meta["tasa_priorizar_ref"], version=version)
    acc.m(lote, "modelo", "tasa_cola_normal", 1 - tasa, version=version)
    acc.m(lote, "modelo", "pct_respaldo", resp, 0, "alerta" if resp > 0 else "ok", version=version)
    probs = p["probabilidad"].dropna().astype(float)
    if len(probs):
        for q in (10, 25, 50, 75, 90):
            acc.m(lote, "modelo", f"probabilidad_p{q}", np.percentile(probs, q), version=version)
        acc.m(lote, "modelo", "probabilidad_media", probs.mean(), version=version)
        r = deriva.comparar_continua(ref["perfil"]["probabilidad"], probs)
        est = "alerta" if r["psi"] > C.PSI_ALERTA else ("aviso" if r["psi"] > C.PSI_AVISO else "estable")
        acc.m(lote, "deriva", "psi", r["psi"], C.PSI_ALERTA, est, feature="probabilidad", version=version)
        if est == "alerta":
            acc.a("psi_prediccion", lote, r["psi"], C.PSI_ALERTA, feature="probabilidad")
    cambio = abs(tasa - meta["tasa_priorizar_ref"])
    if cambio > C.CAMBIO_TASA_DECISION_MAX:
        acc.a("tasa_decision", lote, tasa, meta["tasa_priorizar_ref"])
    if resp > 0:
        acc.a("respaldo_activo", lote, resp, 0)
    return version


def datos_lote(acc: Acumulador, lote: int, ref: dict) -> bool:
    """Calidad y deriva de datos del lote. Devuelve True si las entradas cambiaron (alerta)."""
    d = datos.leer_sql(f"SELECT * FROM {C.T_LOTES} WHERE lote_id = %s", (lote,))
    if d.empty:
        return False
    cols = ["comentario_limpio", "banco", "log_n_palabras", "n_neg_lexico", "es_solo_simbolos",
            "version_app_desconocida"]
    for c in cols:
        acc.m(lote, "datos", "pct_faltantes", d[c].isna().mean(), feature=c)
    vacio = float((d["comentario_limpio"].fillna("").str.strip() == "").mean())
    lim_vacio = ref["meta"]["pct_texto_vacio_ref"] + C.FALTANTES_MAX
    acc.m(lote, "datos", "pct_texto_vacio", vacio, lim_vacio, "alerta" if vacio > lim_vacio else "ok",
          feature="comentario_limpio")
    if vacio > lim_vacio:
        acc.a("faltantes", lote, vacio, lim_vacio, feature="comentario_limpio")
    fuera = float(((d["log_n_palabras"] < 0) | (d["log_n_palabras"] > math.log1p(MAX_PALABRAS))
                   | (d["n_neg_lexico"] < 0) | ~d["es_solo_simbolos"].isin([0, 1])
                   | ~d["version_app_desconocida"].isin([0, 1])).mean())
    acc.m(lote, "datos", "pct_fuera_de_rango", fuera, 0, "alerta" if fuera > 0 else "ok")
    if fuera > 0:
        acc.a("fuera_rango", lote, fuera, 0)
    nuevas = float((~d["banco"].isin(C.BANCOS)).mean())
    acc.m(lote, "datos", "pct_categorias_no_vistas", nuevas, C.CATEGORIAS_NUEVAS_MAX,
          "alerta" if nuevas > C.CATEGORIAS_NUEVAS_MAX else "ok", feature="banco")
    if nuevas > C.CATEGORIAS_NUEVAS_MAX:
        acc.a("categorias_nuevas", lote, nuevas, C.CATEGORIAS_NUEVAS_MAX, feature="banco")

    perfil = {k: v for k, v in ref["perfil"].items() if k in CONTINUAS + CATEGORICAS}
    res = deriva.deriva_lote(perfil, preparar_columnas(d))
    completo = deriva.deriva_lote(ref["perfil_entrenamiento_completo"], preparar_columnas(d))
    cambio = False
    for r in res.to_dict("records"):
        est = deriva.nivel_psi(r["psi"], r["p_valor"], C.PSI_AVISO, C.PSI_ALERTA, C.P_VALOR)
        acc.m(lote, "deriva", "psi", r["psi"], C.PSI_ALERTA, est, feature=r["feature"])
        acc.m(lote, "deriva", f"p_valor_{r['prueba']}", r["p_valor"], C.P_VALOR, feature=r["feature"])
        if est == "alerta":
            cambio = True
            acc.a("psi_feature", lote, r["psi"], C.PSI_ALERTA, feature=r["feature"])
    for r in completo.to_dict("records"):
        acc.m(lote, "deriva", "psi_vs_entrenamiento_completo", r["psi"], feature=r["feature"], estado="info")
    return cambio


def modelo_negocio(acc: Acumulador, lote: int, ref: dict) -> dict | None:
    """Desempeño y negocio de un lote cuyo target ya llegó."""
    df = datos.leer_sql(
        f"SELECT p.probabilidad, p.decision, p.version_modelo, l.es_negativa, l.n_neg_lexico, l.banco, "
        f"l.alterado_texto FROM {C.T_PREDICCIONES} p JOIN {C.T_LOTES} l "
        f"ON p.review_id = l.review_id AND p.lote_id = l.lote_id WHERE p.lote_id = %s", (lote,))
    if df.empty:
        return None
    version = df["version_modelo"].mode().iloc[0]
    y = df["es_negativa"].astype(int).values
    pred = (df["decision"] == C.DECISION_POSITIVA).astype(int).values
    regla = (df["n_neg_lexico"] >= 1).astype(int).values
    costos = {"costo_modelo": costo_negocio(y, pred), "costo_regla_negocio": costo_negocio(y, regla),
              "costo_no_hacer_nada": costo_negocio(y, np.zeros_like(y)),
              "costo_revisar_todas": costo_negocio(y, np.ones_like(y))}
    for k, v in costos.items():
        acc.m(lote, "negocio", k, v, version=version)
    tp = int(((pred == 1) & (y == 1)).sum())
    acc.m(lote, "negocio", "quejas_reales", int(y.sum()), version=version)
    acc.m(lote, "negocio", "quejas_priorizadas_modelo", tp, version=version)
    acc.m(lote, "negocio", "quejas_priorizadas_regla", int(((regla == 1) & (y == 1)).sum()), version=version)
    acc.m(lote, "negocio", "revisiones_innecesarias_modelo", int(((pred == 1) & (y == 0)).sum()), version=version)
    acc.m(lote, "modelo", "recall", tp / max(y.sum(), 1), version=version)
    acc.m(lote, "modelo", "precision", tp / max(pred.sum(), 1), version=version)
    acc.m(lote, "modelo", "prevalencia", y.mean(), version=version)

    cv = metricas_version(version, ref)
    ok = df["probabilidad"].notna().values
    resultado = {"lote": lote, "version": version, "alerta": False, **costos}
    if ok.sum() > 30 and len(np.unique(y[ok])) == 2:
        p = df.loc[ok, "probabilidad"].astype(float).values
        roc, pr = roc_auc_score(y[ok], p), average_precision_score(y[ok], p)
        sd_roc = deriva.bootstrap_metrica(y[ok], p, "roc_auc", C.N_BOOTSTRAP)
        sd_pr = deriva.bootstrap_metrica(y[ok], p, "pr_auc", C.N_BOOTSTRAP)
        lim_roc = deriva.umbral_desempeno(cv["cv_roc_auc_mean"], cv["cv_roc_auc_std"], sd_roc, C.K_SIGMA)
        lim_pr = deriva.umbral_desempeno(cv["cv_pr_auc_mean"], cv["cv_pr_auc_std"], sd_pr, C.K_SIGMA)
        alerta = roc < lim_roc
        acc.m(lote, "modelo", "roc_auc", roc, lim_roc, "alerta" if alerta else "ok", version=version,
              detalle=json.dumps({"media_cv": cv["cv_roc_auc_mean"], "sd_cv": cv["cv_roc_auc_std"], "sd_lote": sd_roc}))
        acc.m(lote, "modelo", "pr_auc", pr, lim_pr, "aviso" if pr < lim_pr else "ok", version=version,
              detalle=json.dumps({"media_cv": cv["cv_pr_auc_mean"], "sd_cv": cv["cv_pr_auc_std"], "sd_lote": sd_pr,
                                  "nota": "el PR-AUC depende de la prevalencia; la alerta usa ROC-AUC"}))
        if alerta:
            acc.a("desempeno", lote, roc, lim_roc)
        resultado.update(roc_auc=roc, pr_auc=pr, alerta=alerta)
    if costos["costo_modelo"] >= costos["costo_regla_negocio"]:
        acc.a("costo_vs_regla", lote, costos["costo_modelo"], costos["costo_regla_negocio"])
    # desglose por subgrupo (sección 8): banco
    for b, g in df.groupby("banco"):
        yb, pb = g["es_negativa"].astype(int).values, (g["decision"] == C.DECISION_POSITIVA).astype(int).values
        acc.m(lote, "subgrupo", "recall", ((pb == 1) & (yb == 1)).sum() / max(yb.sum(), 1), feature=b, version=version)
        acc.m(lote, "subgrupo", "tasa_priorizar", pb.mean(), feature=b, version=version)
        acc.m(lote, "subgrupo", "costo", costo_negocio(yb, pb), feature=b, version=version)
    return resultado


def historial_desempeno(excluir_lote: int) -> list:
    h = datos.leer_sql(f"SELECT lote_id, version_modelo, estado FROM {C.T_MONITOREO} "
                       f"WHERE tipo = 'modelo' AND metrica = 'roc_auc' AND lote_id <> %s "
                       f"QUALIFY ROW_NUMBER() OVER (PARTITION BY lote_id ORDER BY ts DESC) = 1 "
                       f"ORDER BY lote_id", (excluir_lote,))
    return [(int(r["lote_id"]), str(r["version_modelo"]), r["estado"] == "alerta") for r in h.to_dict("records")]


def lotes_desde_ultimo_reentreno(lote: int) -> int:
    r = datos.leer_sql(f"SELECT MAX(lote_disparo) AS u FROM {C.T_REENTRENAMIENTOS} WHERE decision = 'aprobado'")
    ultimo = r["u"].iloc[0] if not r.empty and pd.notna(r["u"].iloc[0]) else 0
    return lote - int(ultimo)


def lote_con_deriva_registrada(lote: int) -> bool:
    r = datos.leer_sql(f"SELECT COUNT(*) AS n FROM {C.T_ALERTAS} WHERE lote_id = %s "
                       f"AND nombre IN ('Deriva de una feature')", (lote,))
    return int(r["n"].iloc[0]) > 0


# ---------------------------------------------------------------- principal
def ejecutar(lote: int) -> dict:
    ref = leer_referencia()
    acc = Acumulador(lote)
    existe = not datos.leer_sql(f"SELECT 1 FROM {C.T_LOTES} WHERE lote_id = %s LIMIT 1", (lote,)).empty
    if existe:
        sistema(acc, lote)
        predicciones(acc, lote, ref)
        datos_lote(acc, lote, ref)

    k = lote - C.HORIZONTE_LOTES        # lote cuyo target acaba de llegar
    eval_k = modelo_negocio(acc, k, ref) if k >= 1 else None
    disparar, motivo = False, ""
    if eval_k:
        cambio_k = lote_con_deriva_registrada(k)
        diag, accion = deriva.diagnostico(cambio_k, bool(eval_k["alerta"]))
        acc.m(k, "diagnostico", "diagnostico", None, estado=diag, version=eval_k["version"], detalle=accion)
        hist = historial_desempeno(k) + [(k, str(eval_k["version"]), bool(eval_k["alerta"]))]
        disparar, motivo = regla_disparo(hist, version_champion_actual(), lotes_desde_ultimo_reentreno(lote))

    datos.escribir_df(pd.DataFrame(acc.metricas), C.T_MONITOREO)
    if acc.alertas:
        datos.escribir_df(pd.DataFrame(acc.alertas), C.T_ALERTAS)

    sev_orden = {"Crítica": 3, "Alta": 2, "Media": 1}
    resumen = [f"[{a['severidad']}] lote {a['lote_id']} · {a['nombre']}"
               + (f" ({a['feature']})" if a["feature"] else "")
               + f": {a['valor']:.4f} vs {a['umbral']:.4f} → {a['receptor']}: {a['accion']}"
               for a in sorted(acc.alertas, key=lambda a: -sev_orden.get(a["severidad"], 0))]
    print(f"Monitoreo del lote {lote}: {len(acc.metricas)} métricas, {len(acc.alertas)} alertas")
    for r in resumen:
        print("ALERTA", r)
    if eval_k:
        print(f"Lote {k} (target disponible): costo modelo {eval_k['costo_modelo']:.3f} | regla "
              f"{eval_k['costo_regla_negocio']:.3f} | ROC-AUC {eval_k.get('roc_auc', float('nan')):.3f} | "
              f"alerta de desempeño: {eval_k['alerta']}")
    print("Reentrenamiento:", f"SÍ ({motivo})" if disparar else "no")
    max_sev = max([sev_orden.get(a["severidad"], 0) for a in acc.alertas], default=0)
    kestra_outputs(n_alertas=len(acc.alertas), alertas="\n".join(resumen) or "sin alertas",
                   severidad_maxima={3: "Crítica", 2: "Alta", 1: "Media", 0: "ninguna"}[max_sev],
                   disparar_reentrenamiento=disparar, motivo=motivo, lote_evaluacion=k if eval_k else None)
    return {"alertas": acc.alertas, "disparar": disparar, "motivo": motivo}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lote", type=int, required=True)
    ejecutar(ap.parse_args().lote)
