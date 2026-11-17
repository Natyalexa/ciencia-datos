"""Prueba de humo: lee la OBT de Snowflake y registra una corrida de prueba en MLflow.

Uso (dentro del contenedor):  python -m src.prueba_conexion
"""
import mlflow

from src.datos import cargar_obt, hash_dataframe

df = cargar_obt()
print("Filas y columnas:", df.shape)
print(df.dtypes.to_string())
print("\nPrimeras filas:")
print(df.head(3).T.to_string())

mlflow.set_experiment("prueba-conexion")
with mlflow.start_run(run_name="smoke-test"):
    mlflow.log_param("tabla_origen", "PSET2_DB.OBT.OBT_RESENAS")
    mlflow.log_param("hash_dataset", hash_dataframe(df))
    mlflow.log_metric("filas", len(df))
    mlflow.log_metric("pct_negativo_estrellas", float((df["sentimiento_estrellas"] == "negativo").mean()))
print("\nOK: Snowflake y MLflow responden.")
