"""4. Respaldo: si el modelo no carga, el servicio responde con la regla de negocio y lo indica.
También prueba el interruptor manual de respaldo (sección 8.4)."""
import pytest

import api.main as main
from api.servicio import ServicioModelo

HDR = {"X-Token-Admin": "token-de-prueba"}


@pytest.fixture
def sin_modelo(cliente, monkeypatch):
    roto = ServicioModelo(uri="/ruta/que/no/existe")
    roto.cargar(forzar=True)
    monkeypatch.setattr(main, "servicio", roto)
    return cliente


def test_salud_degradada_si_no_carga(sin_modelo):
    s = sin_modelo.get("/salud").json()
    assert s["modelo_cargado"] is False and s["modo_respaldo"] is True and s["estado"] == "degradado"


@pytest.mark.parametrize("texto,esperado", [("no funciona la app", "priorizar"),
                                            ("excelente aplicacion", "cola_normal")])
def test_responde_con_regla_de_negocio(sin_modelo, entrada_valida, texto, esperado):
    r = sin_modelo.post("/predecir", json={**entrada_valida, "comentario_limpio": texto})
    assert r.status_code == 200
    c = r.json()
    assert c["respaldo"] is True and c["probabilidad"] is None
    assert c["version_modelo"] == "regla-negocio" and "no cargó" in c["motivo_respaldo"]
    assert c["decision"] == esperado


def test_interruptor_de_respaldo(cliente, entrada_valida):
    r = cliente.post("/admin/respaldo", json={"activo": True, "motivo": "prueba"}, headers=HDR)
    assert r.status_code == 200 and r.json()["modo_respaldo"] is True
    c = cliente.post("/predecir", json=entrada_valida).json()
    assert c["respaldo"] is True and c["version_modelo"] == "regla-negocio"
    assert "interruptor" in c["motivo_respaldo"]
    cliente.post("/admin/respaldo", json={"activo": False}, headers=HDR)
    c = cliente.post("/predecir", json=entrada_valida).json()
    assert c["respaldo"] is False and c["probabilidad"] is not None
