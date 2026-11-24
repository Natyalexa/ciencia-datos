"""Configuración común de las pruebas.

En CI no hay MLflow ni Snowflake: la API carga la copia del champion exportada en
tests/fixtures/modelo_champion (scripts/exportar_fixture.py) y registra en CSV temporales.
"""
import json
import os
import sys
import tempfile

import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
FIXTURES = os.path.join(RAIZ, "tests", "fixtures")
RUTA_MODELO = os.path.join(FIXTURES, "modelo_champion")

# Variables ANTES de importar la API (el servicio se crea al importar api.main)
os.environ["MODELO_URI"] = RUTA_MODELO
os.environ["REGISTRO_DESTINO"] = "csv"
os.environ["REGISTRO_DIR"] = tempfile.mkdtemp(prefix="registro_")
os.environ["ADMIN_TOKEN"] = "token-de-prueba"
os.environ["MODO_RESPALDO"] = "false"
os.environ.setdefault("EMB_CACHE", "")
for var in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_PRIVATE_KEY_PEM"):
    os.environ.pop(var, None)

MENSAJE = ("Falta tests/fixtures/modelo_champion. Genéralo con "
           "`docker compose exec jupyter python scripts/exportar_fixture.py` y súbelo al repositorio.")


@pytest.fixture(scope="session")
def metadata():
    ruta = os.path.join(RUTA_MODELO, "metadata.json")
    if not os.path.exists(ruta):
        pytest.fail(MENSAJE)
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def modelo(metadata):
    import mlflow.sklearn
    return mlflow.sklearn.load_model(RUTA_MODELO)


@pytest.fixture(scope="session")
def muestra():
    return pd.read_csv(os.path.join(FIXTURES, "muestra.csv"), keep_default_na=False)


@pytest.fixture(scope="session")
def cliente(metadata):
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def entrada_valida():
    return {"review_id": "prueba-1", "lote_id": 1,
            "comentario_limpio": "la app no me deja ingresar desde la actualizacion, pesimo servicio",
            "banco": "Banco Pichincha", "es_solo_simbolos": 0, "version_app_desconocida": 0}
