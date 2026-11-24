"""1. Contrato: entradas válidas → 200 con el esquema esperado; entradas inválidas → 422 con el campo."""
import pytest

from api.esquemas import SalidaPrediccion

HDR = {"X-Token-Admin": "token-de-prueba"}


def test_salud_y_modelo(cliente, metadata):
    s = cliente.get("/salud").json()
    assert s["estado"] == "ok" and s["modelo_cargado"] and not s["modo_respaldo"]
    m = cliente.get("/modelo").json()
    assert m["version"] == metadata["version"]
    assert m["run_id"] == metadata["run_id"]
    assert m["alias"] == metadata["alias"]
    assert m["nombre"] == "resenas-bancos-ec-modelo"


def test_entrada_valida_devuelve_200_con_esquema(cliente, entrada_valida, metadata):
    r = cliente.post("/predecir", json=entrada_valida)
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == set(SalidaPrediccion.model_fields)
    assert 0.0 <= cuerpo["probabilidad"] <= 1.0
    assert cuerpo["decision"] in {"priorizar", "cola_normal"}
    assert cuerpo["accion"]
    assert cuerpo["version_modelo"] == metadata["version"]
    assert cuerpo["respaldo"] is False
    assert cuerpo["umbral"] == pytest.approx(metadata["umbral"])
    esperado = "priorizar" if cuerpo["probabilidad"] >= cuerpo["umbral"] else "cola_normal"
    assert cuerpo["decision"] == esperado


def test_features_derivadas_opcionales(cliente, entrada_valida):
    """Enviar log_n_palabras y n_neg_lexico calculados igual que el servicio da la misma probabilidad."""
    import math
    from src.pipeline import conteo_lexico
    a = cliente.post("/predecir", json=entrada_valida).json()
    texto = entrada_valida["comentario_limpio"]
    b = cliente.post("/predecir", json={**entrada_valida, "log_n_palabras": math.log1p(len(texto.split())),
                                        "n_neg_lexico": conteo_lexico(texto)}).json()
    assert a["probabilidad"] == pytest.approx(b["probabilidad"], abs=1e-9)


def _campos(r):
    return {d["campo"] for d in r.json()["detalle"]}


def test_campo_faltante_422(cliente, entrada_valida):
    entrada_valida.pop("comentario_limpio")
    r = cliente.post("/predecir", json=entrada_valida)
    assert r.status_code == 422 and "comentario_limpio" in _campos(r)


def test_tipo_incorrecto_422(cliente, entrada_valida):
    r = cliente.post("/predecir", json={**entrada_valida, "es_solo_simbolos": "talvez"})
    assert r.status_code == 422 and "es_solo_simbolos" in _campos(r)


def test_categoria_desconocida_422(cliente, entrada_valida):
    r = cliente.post("/predecir", json={**entrada_valida, "banco": "Banco Internacional"})
    assert r.status_code == 422 and "banco" in _campos(r)


@pytest.mark.parametrize("campo,valor", [("log_n_palabras", 50.0), ("version_app_desconocida", 2),
                                         ("n_neg_lexico", -1)])
def test_fuera_de_rango_422(cliente, entrada_valida, campo, valor):
    r = cliente.post("/predecir", json={**entrada_valida, campo: valor})
    assert r.status_code == 422 and campo in _campos(r)


def test_campo_extra_422(cliente, entrada_valida):
    r = cliente.post("/predecir", json={**entrada_valida, "calificacion": 1})   # fuga: no existe al predecir
    assert r.status_code == 422 and "calificacion" in _campos(r)


def test_prediccion_queda_registrada(cliente, entrada_valida, metadata):
    import os
    import pandas as pd
    r = cliente.post("/predecir", json=entrada_valida).json()
    cliente.post("/admin/vaciar-registro", headers=HDR)
    reg = pd.read_csv(os.path.join(os.environ["REGISTRO_DIR"], "predicciones.csv"))
    fila = reg[reg["id_prediccion"] == r["id_prediccion"]]
    assert len(fila) == 1
    for col in ("ts_prediccion", "entradas", "probabilidad", "decision", "version_modelo", "latencia_ms"):
        assert pd.notna(fila[col].iloc[0]), col
    assert str(fila["version_modelo"].iloc[0]) == metadata["version"]
    sol = pd.read_csv(os.path.join(os.environ["REGISTRO_DIR"], "solicitudes.csv"))
    assert {200, 422} <= set(sol["estado_http"])   # también se registran los rechazos (métricas de sistema)


def test_admin_requiere_token(cliente):
    assert cliente.post("/admin/respaldo", json={"activo": True}).status_code == 401
