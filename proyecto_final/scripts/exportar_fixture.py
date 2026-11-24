"""Exporta el champion y una muestra fija a tests/fixtures/ para las pruebas de CI.

    docker compose exec jupyter python scripts/exportar_fixture.py

En CI no hay servidor de MLflow: las pruebas cargan esta copia (mlflow.sklearn.save_model).
Además registra en el run del champion la métrica sobre la muestra fija (muestra_ci_pr_auc,
muestra_ci_costo), que es la que test_modelo.py debe reproducir.
"""
import json
import os
import shutil
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import sklearn
from sklearn.metrics import average_precision_score

from src import config as C
from src.datos import cargar_obt
from src.pipeline import FEATURES, TARGET, agregar_features, costo_negocio, split_temporal
from src.simulacion import cargar_snapshot

DESTINO = os.path.join("tests", "fixtures")
N_MUESTRA = 400

mlflow.set_tracking_uri(C.MLFLOW_URI)
cli = mlflow.MlflowClient()
mv = cli.get_model_version_by_alias(C.NOMBRE_MODELO, C.ALIAS_PRODUCCION)
run = cli.get_run(mv.run_id)
umbral = float(run.data.metrics.get("umbral_aplicado", C.UMBRAL_POR_DEFECTO))
modelo = mlflow.sklearn.load_model(f"models:/{C.NOMBRE_MODELO}/{mv.version}")

# muestra fija: reseñas del hold-out temporal del PSet #3 (nunca vistas en entrenamiento)
snap, origen = cargar_snapshot()
_, test = split_temporal(agregar_features(snap), 0.2)
import pandas as pd
muestra = (pd.concat([test[test[TARGET] == k].sample(N_MUESTRA // 2, random_state=C.SEMILLA) for k in (0, 1)])
           .sort_values("review_id"))[["review_id"] + FEATURES + [TARGET]]
p = modelo.predict_proba(muestra[FEATURES])[:, 1]
pr_auc = float(average_precision_score(muestra[TARGET], p))
costo = float(costo_negocio(muestra[TARGET], (p >= umbral).astype(int)))
cli.log_metric(mv.run_id, "muestra_ci_pr_auc", pr_auc)
cli.log_metric(mv.run_id, "muestra_ci_costo", costo)

ruta_modelo = os.path.join(DESTINO, "modelo_champion")
shutil.rmtree(ruta_modelo, ignore_errors=True)
mlflow.sklearn.save_model(modelo, ruta_modelo, serialization_format="cloudpickle")
os.makedirs(DESTINO, exist_ok=True)
muestra.to_csv(os.path.join(DESTINO, "muestra.csv"), index=False)
meta = {"nombre": C.NOMBRE_MODELO, "version": str(mv.version), "alias": C.ALIAS_PRODUCCION, "run_id": mv.run_id,
        "umbral": umbral, "muestra_ci_pr_auc": pr_auc, "muestra_ci_costo": costo, "n_muestra": len(muestra),
        "test_pr_auc_registrado": run.data.metrics.get("test_pr_auc"), "obt_origen": origen,
        "scikit_learn": sklearn.__version__, "exportado_en": datetime.now(timezone.utc).isoformat()}
with open(os.path.join(ruta_modelo, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(json.dumps(meta, indent=2, ensure_ascii=False))
req = mlflow.pyfunc.get_model_dependencies(f"models:/{C.NOMBRE_MODELO}/{mv.version}")
print("\nVersiones con las que se entrenó (fíjalas en requirements.txt):")
print("".join(l for l in open(req) if l.split("==")[0].strip() in {"scikit-learn", "numpy", "pandas", "sentence-transformers", "torch", "mlflow", "cloudpickle"}))
