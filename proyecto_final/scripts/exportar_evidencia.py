"""Exporta la evidencia a docs/ en CSV (sección «Repositorio» del enunciado).

    docker compose exec jupyter python scripts/exportar_evidencia.py

docs/predicciones.csv, monitoreo.csv, alertas.csv           ← tablas de Snowflake
docs/corridas.csv                                            ← todas las corridas (mlflow.search_runs)
docs/reentrenamientos.csv                                    ← corridas con fase = reentrenamiento
docs/reentrenamientos_decisiones.csv                         ← tabla REENTRENAMIENTOS (decisión humana)
docs/solicitudes_resumen.csv                                 ← métricas de sistema por lote
"""
import os

import mlflow

from src import config as C
from src import datos

os.makedirs("docs", exist_ok=True)
for tabla, archivo in [(C.T_PREDICCIONES, "predicciones.csv"), (C.T_MONITOREO, "monitoreo.csv"),
                       (C.T_ALERTAS, "alertas.csv"), (C.T_REENTRENAMIENTOS, "reentrenamientos_decisiones.csv")]:
    df = datos.leer_sql(f"SELECT * FROM {tabla}")
    df.to_csv(f"docs/{archivo}", index=False)
    print(f"{archivo}: {len(df)} filas")

s = datos.leer_sql(f"""SELECT lote_id, COUNT(*) AS solicitudes,
    APPROX_PERCENTILE(latencia_ms, 0.5) AS p50_ms, APPROX_PERCENTILE(latencia_ms, 0.95) AS p95_ms,
    AVG(IFF(estado_http BETWEEN 400 AND 499, 1, 0)) AS tasa_4xx, AVG(IFF(estado_http >= 500, 1, 0)) AS tasa_5xx
    FROM {C.T_SOLICITUDES} WHERE ruta = '/predecir' GROUP BY lote_id ORDER BY lote_id""")
s.to_csv("docs/solicitudes_resumen.csv", index=False)

mlflow.set_tracking_uri(C.MLFLOW_URI)
exp = mlflow.get_experiment_by_name(C.NOMBRE_EXPERIMENTO)
runs = mlflow.search_runs([exp.experiment_id])
runs.to_csv("docs/corridas.csv", index=False)
re = runs[runs.get("tags.fase") == "reentrenamiento"] if "tags.fase" in runs else runs.iloc[0:0]
re.to_csv("docs/reentrenamientos.csv", index=False)
print(f"corridas.csv: {len(runs)} | reentrenamientos.csv: {len(re)}")
print(runs.groupby("tags.fase").size().to_string())
