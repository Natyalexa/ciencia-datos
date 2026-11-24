"""Distribución de referencia para el monitoreo (se calcula una vez, al preparar la simulación).

Referencia de deriva = el REF_FRACCION más reciente del conjunto de entrenamiento del champion.
Por qué no todo el entrenamiento: la ingesta toma como máximo 3 000 reseñas por banco, así que
los bancos con mucho volumen (Pichincha) solo aparecen en los últimos meses y los de poco volumen
(JEP, Austro) cubren años. La mezcla de bancos del entrenamiento completo no describe ningún
momento real; la de sus últimos meses sí describe el presente en que se desplegó el modelo.
El PSI contra el entrenamiento completo también se guarda, como información.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from monitoreo import deriva
from src import config, datos
from src.pipeline import FEATURES, TARGET, costo_negocio

CONTINUAS = ["log_n_palabras"]
CATEGORICAS = ["banco", "n_neg_lexico_cat", "es_solo_simbolos", "version_app_desconocida"]


def preparar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["n_neg_lexico_cat"] = out["n_neg_lexico"].clip(upper=3).astype(int).astype(str).replace({"3": "3+"})
    for c in ("es_solo_simbolos", "version_app_desconocida"):
        out[c] = out[c].astype(int).astype(str)
    return out


def _cv_champion(train: pd.DataFrame, modelo, umbral: float) -> dict:
    """CV temporal del champion con su misma configuración: media y desviación por pliegue de
    PR-AUC, ROC-AUC y costo (el PSet #3 solo registró la desviación del PR-AUC)."""
    from sklearn.base import clone
    from src.experimento import validar_cv
    cv = validar_cv(clone(modelo), train[FEATURES], train[TARGET])
    pliegues_y = np.array_split(cv["oof_y"], len(cv["va_pr"]))
    pliegues_p = np.array_split(cv["oof_p"], len(cv["va_pr"]))
    costos = [costo_negocio(y, (p >= umbral).astype(int)) for y, p in zip(pliegues_y, pliegues_p)]
    return {"cv_pr_auc_mean": float(np.mean(cv["va_pr"])), "cv_pr_auc_std": float(np.std(cv["va_pr"])),
            "cv_roc_auc_mean": float(np.mean(cv["va_roc"])), "cv_roc_auc_std": float(np.std(cv["va_roc"])),
            "cv_costo_mean": float(np.mean(costos)), "cv_costo_std": float(np.std(costos)),
            "pliegues_pr_auc": [float(x) for x in cv["va_pr"]],
            "pliegues_roc_auc": [float(x) for x in cv["va_roc"]], "pliegues_costo": costos}


def construir_referencia(train: pd.DataFrame, calcular_cv: bool = True) -> dict:
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    cli = mlflow.MlflowClient()
    mv = cli.get_model_version_by_alias(config.NOMBRE_MODELO, config.ALIAS_PRODUCCION)
    run = cli.get_run(mv.run_id)
    umbral = float(run.data.metrics.get("umbral_aplicado", config.UMBRAL_POR_DEFECTO))
    modelo = mlflow.sklearn.load_model(f"models:/{config.NOMBRE_MODELO}/{mv.version}")

    ref = train.tail(int(len(train) * config.REF_FRACCION)).copy()
    ref["probabilidad"] = modelo.predict_proba(ref[FEATURES])[:, 1]
    refc = preparar_columnas(ref)
    perfil = deriva.perfil(refc, CONTINUAS + ["probabilidad"], CATEGORICAS)
    perfil_completo = deriva.perfil(preparar_columnas(train), CONTINUAS, CATEGORICAS)

    # PSI "natural": cada quinto del entrenamiento contra la referencia (justifica los umbrales)
    previo = preparar_columnas(train.iloc[: len(train) - len(ref)])
    cortes = np.linspace(0, len(previo), 5).astype(int)
    bloques = [previo.iloc[a:b] for a, b in zip(cortes[:-1], cortes[1:])]
    psi_nat = pd.concat([deriva.deriva_lote({k: v for k, v in perfil.items() if k != "probabilidad"}, b)
                         .assign(bloque=i + 1) for i, b in enumerate(bloques)])

    meta = {"version_champion": mv.version, "run_id": mv.run_id, "umbral": umbral, "n_ref": len(ref),
            "ref_desde": str(ref["fecha_resena"].min()), "ref_hasta": str(ref["fecha_resena"].max()),
            "tasa_priorizar_ref": float((ref["probabilidad"] >= umbral).mean()),
            "pct_negativa_ref": float(ref[TARGET].mean()),
            "pct_texto_vacio_ref": float((ref["comentario_limpio"].fillna("").str.strip() == "").mean()),
            "corte": str(train["fecha_resena"].max())}
    filas = [("perfil", perfil), ("perfil_entrenamiento_completo", perfil_completo),
             ("psi_natural", psi_nat.to_dict("records")), ("meta", meta)]

    if calcular_cv:
        cv = _cv_champion(train, modelo, umbral)
        for k in ("cv_roc_auc_mean", "cv_roc_auc_std", "cv_costo_mean", "cv_costo_std"):
            cli.log_metric(mv.run_id, k, cv[k])   # completa el run del champion (trazable en MLflow)
        filas.append(("cv_champion", cv))

    ahora = datetime.now(timezone.utc).replace(tzinfo=None)
    df = pd.DataFrame([{"clave": k, "valor": json.dumps(v, default=str), "creado_en": ahora} for k, v in filas])
    datos.escribir_df(df, config.T_REFERENCIA, sobrescribir=True)
    print("Referencia guardada:", json.dumps(meta, default=str))
    print("PSI natural máximo por feature (bloques del entrenamiento vs referencia):")
    print(psi_nat.groupby("feature")["psi"].max().round(3).to_string())
    return {"meta": meta}


def leer_referencia() -> dict:
    df = datos.leer_sql(f"SELECT clave, valor FROM {config.T_REFERENCIA}")
    return {r["clave"]: json.loads(r["valor"]) for r in df.to_dict("records")}
