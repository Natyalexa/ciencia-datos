"""Acceso a Snowflake: lectura de la OBT, escritura de tablas de producción y trazabilidad.

La llave privada se lee de SNOWFLAKE_PRIVATE_KEY_PEM (Kestra la inyecta desde un secreto)
o, si no existe, del archivo SNOWFLAKE_KEY_PATH (montado en /secrets en Docker Compose).
"""
import hashlib
import os

import pandas as pd

from src import config

LLAVE_PRIVADA = os.environ.get("SNOWFLAKE_KEY_PATH", "/secrets/rsa_key.p8")
TABLA_OBT = config.TABLA_OBT


def snowflake_configurado() -> bool:
    return bool(os.environ.get("SNOWFLAKE_ACCOUNT")) and (
        bool(os.environ.get("SNOWFLAKE_PRIVATE_KEY_PEM")) or os.path.exists(LLAVE_PRIVADA))


def _llave_der() -> bytes:
    """Llave privada PKCS#8 (PEM) convertida al formato DER que pide el conector."""
    from cryptography.hazmat.primitives import serialization
    pem = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PEM")
    if pem:
        datos = pem.encode()
    else:
        with open(LLAVE_PRIVADA, "rb") as f:
            datos = f.read()
    llave = serialization.load_pem_private_key(datos, password=None)
    return llave.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def conectar():
    """Conexión con el usuario de servicio del PSet #2 (autenticación por llave RSA)."""
    import snowflake.connector
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ.get("SNOWFLAKE_USER", "PSET2_SVC"),
        private_key=_llave_der(),
        role=os.environ.get("SNOWFLAKE_ROLE", "PSET2_ROLE"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "PSET2_WH"),
        database=config.BASE,
        session_parameters={"QUERY_TAG": "proyecto_final"},
    )


def leer_sql(sql: str, params=None) -> pd.DataFrame:
    con = conectar()
    try:
        cur = con.cursor()
        cur.execute(sql, params)
        df = cur.fetch_pandas_all()
    finally:
        con.close()
    df.columns = [c.lower() for c in df.columns]
    return df


def ejecutar_sql(*sentencias: str) -> None:
    con = conectar()
    try:
        cur = con.cursor()
        for s in sentencias:
            cur.execute(s)
    finally:
        con.close()


def cargar_obt(tabla: str = TABLA_OBT) -> pd.DataFrame:
    """Descarga la OBT completa (una fila por reseña) con columnas en minúsculas."""
    return leer_sql(f"SELECT * FROM {tabla}")


def existe_tabla(tabla: str) -> bool:
    base, esquema, nombre = tabla.split(".")
    df = leer_sql(
        f"SELECT COUNT(*) AS n FROM {base}.INFORMATION_SCHEMA.TABLES "
        f"WHERE TABLE_SCHEMA = '{esquema}' AND TABLE_NAME = '{nombre}'")
    return int(df["n"].iloc[0]) > 0


def escribir_df(df: pd.DataFrame, tabla: str, sobrescribir: bool = False) -> int:
    """Escribe un DataFrame en Snowflake (crea la tabla si no existe). Columnas en MAYÚSCULAS."""
    from snowflake.connector.pandas_tools import write_pandas
    if df.empty:
        return 0
    base, esquema, nombre = tabla.split(".")
    out = df.dropna(axis=1, how="all").copy()   # columnas vacias quedan NULL en Snowflake
    out.columns = [c.upper() for c in out.columns]
    con = conectar()
    try:
        cur = con.cursor()
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {base}.{esquema}")
        if sobrescribir:  # se vacía y se conserva la tabla con sus tipos (ver src/tablas.py)
            cur.execute(f"TRUNCATE TABLE IF EXISTS {tabla}")
        ok, _, n, _ = write_pandas(con, out, nombre, database=base, schema=esquema,
                                   auto_create_table=True, overwrite=False,
                                   use_logical_type=True, quote_identifiers=False)
    finally:
        con.close()
    return int(n) if ok else 0


def hash_dataframe(df: pd.DataFrame) -> str:
    """Huella (SHA-256) del contenido del dataset, para registrarla en MLflow."""
    h = pd.util.hash_pandas_object(df.sort_index(axis=1), index=True).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def cargar_obt_entrenamiento() -> pd.DataFrame:
    """OBT con la que se entrenó el champion: el snapshot inmutable si existe (ver
    docs/setup_produccion.sql); si no, la OBT actual. Los notebooks 01-05 la usan para que
    'Restart & Run All' reproduzca el PSet #3 aunque la ingesta haya agregado reseñas nuevas."""
    if existe_tabla(config.TABLA_OBT_SNAPSHOT):
        return cargar_obt(config.TABLA_OBT_SNAPSHOT)
    print(f"AVISO: no existe {config.TABLA_OBT_SNAPSHOT}; se usa {TABLA_OBT}")
    return cargar_obt(TABLA_OBT)
