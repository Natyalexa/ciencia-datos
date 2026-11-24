"""Reentrenamiento del modelo (sección 7.2). Lo ejecuta kestra/reentrenamiento.yml.

    python -m src.reentrenar entrenar --lote-disparo 8 --lote-eval 7 --motivo "desempeño ..."
    python -m src.reentrenar decidir --id <id> --aprobar true --motivo "..." [--usuario nataly]

entrenar
  1. Ventana: todo el entrenamiento original (peso 1) + los lotes de producción cuyo target ya
     llegó y que son anteriores al lote de evaluación (peso PESO_PRODUCCION). Se conserva la
     historia (cubre los 8 bancos y años de reseñas) pero se da más peso al régimen reciente.
  2. Misma preparación (src/pipeline.py) e hiperparámetros del champion: el pipeline se clona del
     champion; no se repite el tuning.
  3. CV temporal con pesos → umbral de costo y desviación del costo entre pliegues.
  4. Registro en MLflow (tag fase=reentrenamiento y motivo) y en el Model Registry con alias challenger.
  5. Champion y challenger se evalúan en el mismo lote etiquetado más reciente, que ninguno vio.
  6. Regla de promoción (sección 4): el challenger se promueve si su costo en ese lote es menor que el
     del champion por más de una desviación estándar del costo en la CV, y su ROC-AUC no es peor
     que la del champion por más de una desviación de la CV.
decidir
  Aplica la decisión humana: mueve el alias champion (aprobado) o deja todo igual (rechazado).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src import config as C
from src import datos
from src.pipeline import FEATURES, TARGET, agregar_features, costo_negocio, mejor_umbral, split_temporal
from src.simulacion import cargar_snapshot, kestra_outputs


def _mlflow():
    import mlflow
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    mlflow.set_experiment(C.NOMBRE_EXPERIMENTO)
    return mlflow, mlflow.MlflowClient()


def armar_ventana(lote_eval: int):
    snap, origen = cargar_snapshot()
    snap = agregar_features(snap.assign(fecha_resena=pd.to_datetime(snap["fecha_resena"])))
    hist, _ = split_temporal(snap, 0.2)
    hist = hist.assign(peso=C.PESO_HISTORICO, origen="historico", lote_id=0)
    lotes = datos.leer_sql(f"SELECT * FROM {C.T_LOTES} WHERE lote_id < %s", (lote_eval,))
    lotes["fecha_resena"] = pd.to_datetime(lotes["fecha_resena"])
    lotes = lotes[lotes["banco"].isin(C.BANCOS)]   # el esquema de la API no admite bancos nuevos
    prod = lotes.assign(peso=C.PESO_PRODUCCION, origen="produccion")
    cols = FEATURES + [TARGET, "fecha_resena", "review_id", "peso", "origen", "lote_id"]
    ventana = pd.concat([hist[cols], prod[cols]], ignore_index=True)
    ventana = ventana.sort_values(["fecha_resena", "review_id"], kind="mergesort").reset_index(drop=True)
    for c in ("es_solo_simbolos", "version_app_desconocida", "n_neg_lexico"):
        ventana[c] = ventana[c].astype(int)
    ventana["log_n_palabras"] = ventana["log_n_palabras"].astype(float)
    ventana[TARGET] = ventana[TARGET].astype(int)
    return ventana, origen, sorted(int(x) for x in prod["lote_id"].unique())


def cv_ponderada(pipe, X, y, w) -> dict:
    pr, roc, costos, oof_y, oof_p, tiempos = [], [], [], [], [], []
    plieg = list(TimeSeriesSplit(C.CV_SPLITS_REENTRENO).split(X))
    for tr, va in plieg:
        m = clone(pipe)
        t0 = time.time()
        m.fit(X.iloc[tr], y.iloc[tr], modelo__sample_weight=w.iloc[tr].values)
        tiempos.append(time.time() - t0)
        p = m.predict_proba(X.iloc[va])[:, 1]
        pr.append(average_precision_score(y.iloc[va], p))
        roc.append(roc_auc_score(y.iloc[va], p))
        oof_y.append(y.iloc[va].values)
        oof_p.append(p)
    umbral, costo = mejor_umbral(np.concatenate(oof_y), np.concatenate(oof_p))
    costos = [costo_negocio(a, (b >= umbral).astype(int)) for a, b in zip(oof_y, oof_p)]
    return {"cv_pr_auc_mean": float(np.mean(pr)), "cv_pr_auc_std": float(np.std(pr)),
            "cv_roc_auc_mean": float(np.mean(roc)), "cv_roc_auc_std": float(np.std(roc)),
            "cv_costo_mean": float(np.mean(costos)), "cv_costo_std": float(np.std(costos)),
            "umbral_costo": umbral, "costo_cv_umbral": costo,
            "segundos_cv_por_pliegue": float(np.mean(tiempos))}


def evaluar(modelo, umbral, X, y) -> dict:
    p = modelo.predict_proba(X)[:, 1]
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "costo": costo_negocio(y, (p >= umbral).astype(int)),
            "recall": float(((p >= umbral) & (y == 1)).sum() / max((y == 1).sum(), 1))}


def regla_promocion(champ: dict, chall: dict, cv: dict) -> tuple[bool, str]:
    mejora = champ["costo"] - chall["costo"]
    sd = cv["cv_costo_std"]
    no_peor = chall["roc_auc"] >= champ["roc_auc"] - cv["cv_roc_auc_std"]
    if mejora > sd and no_peor:
        return True, (f"promover: el costo baja {mejora:.4f} por reseña (> 1 σ de la CV = {sd:.4f}) "
                      f"y la ROC-AUC no empeora ({chall['roc_auc']:.4f} vs {champ['roc_auc']:.4f})")
    razones = []
    if mejora <= sd:
        razones.append(f"la mejora de costo ({mejora:.4f}) no supera 1 σ de la CV ({sd:.4f})")
    if not no_peor:
        razones.append(f"la ROC-AUC empeora más de 1 σ ({chall['roc_auc']:.4f} vs {champ['roc_auc']:.4f})")
    return False, "no promover: " + "; ".join(razones)


def entrenar(lote_disparo: int, lote_eval: int, motivo: str) -> dict:
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from src.experimento import RUTA_SRC, commit_git
    mlflow, cli = _mlflow()

    champ_mv = cli.get_model_version_by_alias(C.NOMBRE_MODELO, C.ALIAS_PRODUCCION)
    champ_run = cli.get_run(champ_mv.run_id)
    champion = mlflow.sklearn.load_model(f"models:/{C.NOMBRE_MODELO}/{champ_mv.version}")
    umbral_champ = float(champ_run.data.metrics.get("umbral_aplicado", C.UMBRAL_POR_DEFECTO))

    ventana, origen, lotes_usados = armar_ventana(lote_eval)
    ev = datos.leer_sql(f"SELECT * FROM {C.T_LOTES} WHERE lote_id = %s", (lote_eval,))
    ev = ev[ev["banco"].isin(C.BANCOS)]
    if ev.empty or ev["es_negativa"].nunique() < 2:
        sys.exit(f"El lote de evaluación {lote_eval} no tiene target utilizable")
    for c in ("es_solo_simbolos", "version_app_desconocida", "n_neg_lexico"):
        ev[c] = ev[c].astype(int)
    Xev, yev = ev[FEATURES], ev["es_negativa"].astype(int).values

    # instantánea inmutable de los datos de producción usados (trazabilidad)
    snap_lotes = f"{C.ESQUEMA_PROD}.LOTES_SNAP_L{lote_disparo}_{datetime.now():%Y%m%d%H%M%S}"
    datos.ejecutar_sql(f"CREATE TABLE {snap_lotes} CLONE {C.T_LOTES}")

    X, y, w = ventana[FEATURES], ventana[TARGET], ventana["peso"]
    pipe = clone(champion)   # misma preparación y mismos hiperparámetros, sin ajustar
    print(f"Ventana: {len(ventana)} filas ({(ventana.origen == 'historico').sum()} históricas con peso "
          f"{C.PESO_HISTORICO:g}, {(ventana.origen == 'produccion').sum()} de los lotes {lotes_usados} con peso "
          f"{C.PESO_PRODUCCION:g}). Evaluación: lote {lote_eval} ({len(ev)} reseñas).")
    cv = cv_ponderada(pipe, X, y, w)
    final = clone(pipe)
    t0 = time.time()
    final.fit(X, y, modelo__sample_weight=w.values)
    seg = time.time() - t0

    m_champ = evaluar(champion, umbral_champ, Xev, yev)
    m_chall = evaluar(final, cv["umbral_costo"], Xev, yev)
    promueve, explicacion = regla_promocion(m_champ, m_chall, cv)

    with mlflow.start_run(run_name=f"reentrenamiento-lote{lote_disparo}") as run:
        mlflow.set_tags({"fase": "reentrenamiento", "motivo_disparo": motivo, "familia": "lineal",
                         "modelo": "Embeddings+LogReg", "lote_disparo": lote_disparo, "lote_evaluacion": lote_eval,
                         "commit_git": commit_git(), "champion_comparado": champ_mv.version,
                         "regla_promocion": explicacion})
        mlflow.log_params({"ventana": "historico+lotes_etiquetados_ponderados", "peso_historico": C.PESO_HISTORICO,
                           "peso_produccion": C.PESO_PRODUCCION, "lotes_produccion": json.dumps(lotes_usados),
                           "n_train": len(ventana), "obt_snapshot": origen, "lotes_snapshot": snap_lotes,
                           "tabla_origen": C.TABLA_OBT, "hash_dataset": datos.hash_dataframe(ventana[FEATURES + [TARGET, "peso"]]),
                           "fecha_corte_snapshot": str(ventana["fecha_resena"].max()),
                           "hiperparametros": "los del champion v" + str(champ_mv.version), "semilla": C.SEMILLA,
                           "cv": f"TimeSeriesSplit({C.CV_SPLITS_REENTRENO}) ponderada"})
        mlflow.log_metrics({**{k: v for k, v in cv.items()}, "umbral_aplicado": cv["umbral_costo"],
                            "segundos_entrenamiento": seg,
                            **{f"eval_champion_{k}": v for k, v in m_champ.items()},
                            **{f"eval_challenger_{k}": v for k, v in m_chall.items()},
                            "regla_promueve": float(promueve)})
        muestra = X.head(100)
        info = mlflow.sklearn.log_model(final, name="modelo", input_example=X.head(5),
                                        signature=infer_signature(muestra, final.predict_proba(muestra)),
                                        code_paths=[RUTA_SRC], serialization_format="cloudpickle",
                                        registered_model_name=C.NOMBRE_MODELO)
        version = str(info.registered_model_version)
        cli.set_registered_model_alias(C.NOMBRE_MODELO, C.ALIAS_CANDIDATO, version)
        cli.set_model_version_tag(C.NOMBRE_MODELO, version, "fase", "reentrenamiento")
        cli.set_model_version_tag(C.NOMBRE_MODELO, version, "decision", "pendiente")
        run_id = run.info.run_id

    id_re = str(uuid.uuid4())
    fila = {"id": id_re, "ts": datetime.now(timezone.utc).replace(tzinfo=None), "lote_disparo": lote_disparo,
            "motivo": motivo, "lotes_entrenamiento": json.dumps(lotes_usados), "lote_evaluacion": lote_eval,
            "run_id": run_id, "version_challenger": version, "version_champion": str(champ_mv.version),
            "costo_champion": m_champ["costo"], "costo_challenger": m_chall["costo"],
            "roc_auc_champion": m_champ["roc_auc"], "roc_auc_challenger": m_chall["roc_auc"],
            "pr_auc_champion": m_champ["pr_auc"], "pr_auc_challenger": m_chall["pr_auc"],
            "sd_costo_cv": cv["cv_costo_std"], "regla_promueve": promueve, "decision": "pendiente",
            "motivo_decision": None, "ts_decision": None}
    datos.escribir_df(pd.DataFrame([fila]), C.T_REENTRENAMIENTOS)

    tabla = (f"| | champion v{champ_mv.version} | challenger v{version} |\n|---|---|---|\n"
             f"| Costo por reseña (lote {lote_eval}) | {m_champ['costo']:.4f} | {m_chall['costo']:.4f} |\n"
             f"| ROC-AUC | {m_champ['roc_auc']:.4f} | {m_chall['roc_auc']:.4f} |\n"
             f"| PR-AUC | {m_champ['pr_auc']:.4f} | {m_chall['pr_auc']:.4f} |\n"
             f"| Recall | {m_champ['recall']:.4f} | {m_chall['recall']:.4f} |\n"
             f"| Umbral | {umbral_champ:.2f} | {cv['umbral_costo']:.2f} |")
    print(tabla)
    print("Regla de promoción →", explicacion)
    kestra_outputs(id_reentrenamiento=id_re, run_id=run_id, version_challenger=version,
                   version_champion=str(champ_mv.version), regla_promueve=promueve, explicacion=explicacion,
                   comparacion=tabla)
    return fila


def decidir(id_re: str, aprobar: bool, motivo: str, usuario: str = "revisor") -> str:
    _, cli = _mlflow()
    r = datos.leer_sql(f"SELECT * FROM {C.T_REENTRENAMIENTOS} WHERE id = %s", (id_re,))
    if r.empty:
        sys.exit(f"No existe el reentrenamiento {id_re}")
    r = r.iloc[0]
    nueva, anterior = str(r["version_challenger"]), str(r["version_champion"])
    if aprobar:
        cli.set_registered_model_alias(C.NOMBRE_MODELO, "champion_anterior", anterior)
        cli.set_registered_model_alias(C.NOMBRE_MODELO, C.ALIAS_PRODUCCION, nueva)
        decision = "aprobado"
        print(f"Alias champion: v{anterior} → v{nueva} (champion_anterior = v{anterior}, para rollback)")
    else:
        decision = "rechazado"
        print(f"Champion intacto (v{anterior}). Motivo del rechazo: {motivo}")
    cli.set_model_version_tag(C.NOMBRE_MODELO, nueva, "decision", decision)
    cli.set_model_version_tag(C.NOMBRE_MODELO, nueva, "motivo_decision", motivo)
    cli.set_model_version_tag(C.NOMBRE_MODELO, nueva, "decidido_por", usuario)
    cli.set_tag(r["run_id"], "decision", decision)
    cli.set_tag(r["run_id"], "motivo_decision", motivo)
    motivo_sql = motivo.replace("'", "''")
    datos.ejecutar_sql(f"UPDATE {C.T_REENTRENAMIENTOS} SET decision = '{decision}', "
                       f"motivo_decision = '{motivo_sql} (por {usuario})', ts_decision = CURRENT_TIMESTAMP()::TIMESTAMP_NTZ "
                       f"WHERE id = '{id_re}'")
    kestra_outputs(decision=decision, version_champion=nueva if aprobar else anterior)
    return decision


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("entrenar")
    e.add_argument("--lote-disparo", type=int, required=True)
    e.add_argument("--lote-eval", type=int, required=True)
    e.add_argument("--motivo", default="manual")
    d = sub.add_parser("decidir")
    d.add_argument("--id", required=True)
    d.add_argument("--aprobar", required=True, choices=["true", "false"])
    d.add_argument("--motivo", default="sin comentario")
    d.add_argument("--usuario", default="revisor")
    a = ap.parse_args()
    if a.cmd == "entrenar":
        entrenar(a.lote_disparo, a.lote_eval, a.motivo)
    else:
        decidir(a.id, a.aprobar == "true", a.motivo, a.usuario)
