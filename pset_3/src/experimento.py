"""Validación y registro de corridas en MLflow (compartido por los notebooks 03-05)."""
import os
import subprocess
import time

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.datos import hash_dataframe
from src.pipeline import SEED, costo_negocio, mejor_umbral

NOMBRE_EXPERIMENTO = "resenas-bancos-ec"
NOMBRE_MODELO = "resenas-bancos-ec-modelo"
TABLA_ORIGEN = "PSET2_DB.OBT.OBT_RESENAS"
N_SPLITS = 5
RUTA_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def commit_git() -> str:
    """Commit de Git: variable GIT_COMMIT si existe; si no, intenta git; si no, 'no-disponible'."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        out = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=RUTA_SRC, timeout=5)
        return out.stdout.strip() or "no-disponible"
    except Exception:
        return "no-disponible"


def configurar():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(NOMBRE_EXPERIMENTO)


def limpiar_fase(fase: str, familias=None):
    """Borra corridas previas de una fase (y de ciertas familias) para que 'Run All' no duplique evidencia."""
    exp = mlflow.get_experiment_by_name(NOMBRE_EXPERIMENTO)
    if exp is None:
        return
    runs = mlflow.search_runs([exp.experiment_id], filter_string=f"tags.fase = '{fase}'")
    if familias and len(runs):
        runs = runs[runs["tags.familia"].isin(list(familias))]
    for rid in runs["run_id"]:
        mlflow.delete_run(rid)


def validar_cv(pipe, X, y, n_splits: int = N_SPLITS) -> dict:
    """CV temporal. Devuelve métricas por pliegue y predicciones fuera de muestra."""
    tr_pr, va_pr, va_roc, va_f1, tiempos, oof_y, oof_p = [], [], [], [], [], [], []
    for tr, va in TimeSeriesSplit(n_splits).split(X):
        m = clone(pipe)
        t0 = time.time()
        m.fit(X.iloc[tr], y.iloc[tr])
        tiempos.append(time.time() - t0)
        p_tr = m.predict_proba(X.iloc[tr])[:, 1]
        p_va = m.predict_proba(X.iloc[va])[:, 1]
        tr_pr.append(average_precision_score(y.iloc[tr], p_tr))
        va_pr.append(average_precision_score(y.iloc[va], p_va))
        va_roc.append(roc_auc_score(y.iloc[va], p_va))
        va_f1.append(f1_score(y.iloc[va], (p_va >= 0.5).astype(int)))
        oof_y.append(y.iloc[va].values); oof_p.append(p_va)
    return dict(tr_pr=tr_pr, va_pr=va_pr, va_roc=va_roc, va_f1=va_f1, tiempos=tiempos,
                oof_y=np.concatenate(oof_y), oof_p=np.concatenate(oof_p))


def correr(nombre, pipe, fase, familia, X, y, params=None, fecha_corte="", nested=False,
           log_modelo=True, tags=None, prueba=None, registrar=None, alias=None):
    """Ejecuta CV + ajuste final y registra todo en una corrida de MLflow.

    prueba=(X_test, y_test): SOLO en la fase final; evalúa una vez en el hold-out con el umbral
    elegido en CV. registrar=<nombre>: registra el modelo en el Model Registry; alias=<alias>.
    """
    with mlflow.start_run(run_name=nombre, nested=nested) as run:
        mlflow.log_params({**(params or {}), "semilla": SEED, "cv": f"TimeSeriesSplit({N_SPLITS})"})
        mlflow.log_params({"tabla_origen": TABLA_ORIGEN, "fecha_corte_snapshot": fecha_corte,
                           "hash_dataset": hash_dataframe(X.assign(_y=y.values)), "n_train": len(X)})
        mlflow.set_tags({"fase": fase, "familia": familia, "commit_git": commit_git(), **(tags or {})})

        cv = validar_cv(pipe, X, y)
        umbral, costo = mejor_umbral(cv["oof_y"], cv["oof_p"])

        final = clone(pipe)
        t0 = time.time()
        final.fit(X, y)
        seg = time.time() - t0
        p_train = final.predict_proba(X)[:, 1]
        train_pr = average_precision_score(y, p_train)
        cv_pr = float(np.mean(cv["va_pr"]))

        m = dict(
            train_pr_auc=train_pr, cv_pr_auc_mean=cv_pr, cv_pr_auc_std=float(np.std(cv["va_pr"])),
            brecha_pr_auc=train_pr - cv_pr, cv_roc_auc_mean=float(np.mean(cv["va_roc"])),
            cv_f1_mean=float(np.mean(cv["va_f1"])), cv_f1_std=float(np.std(cv["va_f1"])),
            umbral_costo=umbral, costo_cv_umbral=costo,
            costo_cv_050=costo_negocio(cv["oof_y"], (cv["oof_p"] >= 0.5).astype(int)),
            segundos_entrenamiento=seg, segundos_cv_por_pliegue=float(np.mean(cv["tiempos"])),
        )
        if prueba is not None:
            Xte, yte = prueba
            t0 = time.time()
            p_te = final.predict_proba(Xte)[:, 1]
            ms = (time.time() - t0) / len(Xte) * 1000
            pred = (p_te >= umbral).astype(int)
            m.update(test_pr_auc=average_precision_score(yte, p_te), test_roc_auc=roc_auc_score(yte, p_te),
                     test_f1=f1_score(yte, pred), test_precision=precision_score(yte, pred),
                     test_recall=recall_score(yte, pred), test_costo=costo_negocio(yte, pred),
                     test_costo_050=costo_negocio(yte, (p_te >= 0.5).astype(int)),
                     umbral_aplicado=umbral, ms_inferencia_por_resena=ms)
        mlflow.log_metrics(m)
        version = None
        if log_modelo:
            muestra = X.head(100)
            info = mlflow.sklearn.log_model(final, name="modelo", input_example=X.head(5),
                                            signature=infer_signature(muestra, final.predict_proba(muestra)),
                                            code_paths=[RUTA_SRC], serialization_format="cloudpickle",
                                            registered_model_name=registrar)
            version = getattr(info, "registered_model_version", None)
            if registrar and alias and version:
                mlflow.MlflowClient().set_registered_model_alias(registrar, alias, str(version))
        out = {"run_id": run.info.run_id, "nombre": nombre, "model_version": version, **m}
        if prueba is not None:
            out["_modelo"] = final
        return out
