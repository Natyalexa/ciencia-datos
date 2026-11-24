"""
Construye la One Big Table (OBT) de reseñas a partir de las tablas Gold.

Grano de la OBT: UNA FILA POR RESEÑA (review_id), el mismo grano que fct_resena.
Todas las dimensiones se unen con LEFT JOIN desde la tabla de hechos, de modo que
ninguna reseña pueda perderse, y se valida que los joins no hayan duplicado ni
descartado filas antes de escribir el resultado en Snowflake (esquema OBT).
Si alguna validación falla, el proceso termina con error (Kestra lo marca en rojo).
"""
import os
import re
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ACCOUNT = os.environ["SNOWFLAKE_ACCOUNT"]
KEY_PATH = os.environ.get("SNOWFLAKE_KEY_PATH", "/secrets/rsa_key.p8")


def leer_llave_privada(ruta):
    """Devuelve la llave PKCS#8 sin encabezado/pie ni saltos de línea (formato del conector)."""
    with open(ruta, "r") as f:
        texto = f.read()
    return re.sub(r"-----(BEGIN|END) [A-Z ]*-----|\s", "", texto)


SF = {
    "sfURL": f"{ACCOUNT}.snowflakecomputing.com",
    "sfUser": "PSET2_SVC",
    "pem_private_key": leer_llave_privada(KEY_PATH),
    "sfDatabase": "PSET2_DB",
    "sfSchema": "GOLD",
    "sfWarehouse": "PSET2_WH",
    "sfRole": "PSET2_ROLE",
}

spark = SparkSession.builder.appName("pset2_obt_resenas").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


def leer(tabla, esquema="GOLD"):
    return (
        spark.read.format("snowflake")
        .options(**{**SF, "sfSchema": esquema})
        .option("dbtable", tabla)
        .load()
    )


# ---------------------------------------------------------------- lectura de Gold
fct = leer("FCT_RESENA")
dim_banco = leer("DIM_BANCO").select("BANCO_KEY", "BANCO", "APP_ID")
dim_fecha = leer("DIM_FECHA").select(
    "FECHA_KEY", "FECHA", "ANIO", "TRIMESTRE", "MES", "ANIO_MES",
    "SEMANA_INICIO", "DIA_SEMANA_ISO",
)
dim_version = leer("DIM_VERSION_APP").select(
    "VERSION_APP_KEY", "VERSION_APP",
    F.col("ES_DESCONOCIDA").alias("VERSION_APP_DESCONOCIDA"),
    F.col("PRIMERA_RESENA").alias("VERSION_PRIMERA_RESENA"),
)

# ---------------------------------------------------------------- construcción
obt = (
    fct
    .join(dim_banco, "BANCO_KEY", "left")
    .join(dim_fecha, "FECHA_KEY", "left")
    .join(dim_version, "VERSION_APP_KEY", "left")
    .drop("BANCO_KEY", "FECHA_KEY", "VERSION_APP_KEY")
)

# ---------------------------------------------------------------- validación de joins
n_fct = fct.count()
n_obt = obt.count()
n_ids = obt.select("REVIEW_ID").distinct().count()
sin_dimension = obt.filter(
    F.col("BANCO").isNull() | F.col("FECHA").isNull() | F.col("VERSION_APP").isNull()
).count()

por_banco_fct = (
    fct.join(dim_banco, "BANCO_KEY").groupBy("BANCO").count().withColumnRenamed("count", "N_FCT")
)
por_banco_obt = obt.groupBy("BANCO").count().withColumnRenamed("count", "N_OBT")
difieren = (
    por_banco_fct.join(por_banco_obt, "BANCO", "full_outer")
    .filter(F.col("N_FCT").isNull() | F.col("N_OBT").isNull() | (F.col("N_FCT") != F.col("N_OBT")))
    .count()
)

print("=" * 60)
print("VALIDACIÓN DE LA OBT")
print(f"Filas en FCT_RESENA .................. {n_fct}")
print(f"Filas en la OBT ...................... {n_obt}")
print(f"REVIEW_ID distintos en la OBT ........ {n_ids}")
print(f"Filas sin banco/fecha/versión ........ {sin_dimension}")
print(f"Bancos con conteo distinto (fct/obt) . {difieren}")
print("=" * 60)

errores = []
if n_obt != n_fct:
    errores.append(f"La OBT tiene {n_obt} filas y la tabla de hechos {n_fct}.")
if n_ids != n_obt:
    errores.append(f"Hay {n_obt - n_ids} review_id duplicados en la OBT.")
if sin_dimension != 0:
    errores.append(f"{sin_dimension} filas no encontraron su dimensión.")
if difieren != 0:
    errores.append(f"{difieren} bancos tienen conteos distintos entre FCT y OBT.")

if errores:
    for e in errores:
        print("ERROR:", e)
    spark.stop()
    sys.exit(1)

# ---------------------------------------------------------------- escritura en Snowflake
(
    obt.write.format("snowflake")
    .options(**{**SF, "sfSchema": "OBT"})
    .option("dbtable", "OBT_RESENAS")
    .mode("overwrite")
    .save()
)

n_escrito = leer("OBT_RESENAS", esquema="OBT").count()
print(f"OBT escrita en PSET2_DB.OBT.OBT_RESENAS: {n_escrito} filas")
if n_escrito != n_obt:
    print(f"ERROR: se escribieron {n_escrito} filas y se esperaban {n_obt}.")
    spark.stop()
    sys.exit(1)

spark.stop()
