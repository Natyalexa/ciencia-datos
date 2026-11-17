"""Acceso a los datos: lectura de la OBT desde Snowflake y utilidades de trazabilidad."""
import hashlib
import os

import pandas as pd
import snowflake.connector
from cryptography.hazmat.primitives import serialization

LLAVE_PRIVADA = os.environ.get("SNOWFLAKE_KEY_PATH", "/secrets/rsa_key.p8")
TABLA_OBT = "PSET2_DB.OBT.OBT_RESENAS"


def _llave_der() -> bytes:
    """Lee la llave privada PKCS#8 (PEM) y la entrega en el formato DER que pide el conector."""
    with open(LLAVE_PRIVADA, "rb") as f:
        llave = serialization.load_pem_private_key(f.read(), password=None)
    return llave.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def conectar():
    """Conexión a Snowflake con el usuario de servicio del PSet #2 (autenticación por llave RSA)."""
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ.get("SNOWFLAKE_USER", "PSET2_SVC"),
        private_key=_llave_der(),
        role=os.environ.get("SNOWFLAKE_ROLE", "PSET2_ROLE"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "PSET2_WH"),
    )


def cargar_obt() -> pd.DataFrame:
    """Descarga la OBT completa (una fila por reseña) con columnas en minúsculas."""
    con = conectar()
    try:
        cur = con.cursor()
        cur.execute(f"SELECT * FROM {TABLA_OBT}")
        df = cur.fetch_pandas_all()
    finally:
        con.close()
    df.columns = [c.lower() for c in df.columns]
    return df


def hash_dataframe(df: pd.DataFrame) -> str:
    """Huella (SHA-256) del contenido del dataset, para registrarla en MLflow."""
    h = pd.util.hash_pandas_object(df.sort_index(axis=1), index=True).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]
