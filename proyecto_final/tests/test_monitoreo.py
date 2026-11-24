"""Pruebas unitarias del monitoreo y de las reglas de decisión (no necesitan el modelo)."""
import numpy as np
import pandas as pd
import pytest

from monitoreo import deriva
from monitoreo.alertas import regla_disparo
from src.reentrenar import regla_promocion


def test_psi_cero_si_misma_distribucion():
    rng = np.random.default_rng(0)
    ref = deriva.perfil_continuo(pd.Series(rng.normal(size=5000)))
    assert deriva.comparar_continua(ref, pd.Series(rng.normal(size=2000)))["psi"] < 0.02


def test_psi_alto_si_se_desplaza():
    rng = np.random.default_rng(0)
    ref = deriva.perfil_continuo(pd.Series(rng.normal(size=5000)))
    r = deriva.comparar_continua(ref, pd.Series(rng.normal(1.0, size=2000)))
    assert r["psi"] > 0.25 and r["p_valor"] < 0.01


def test_categoria_nueva_cuenta_como_no_vista():
    ref = deriva.perfil_categorico(pd.Series(["a"] * 50 + ["b"] * 50))
    r = deriva.comparar_categorica(ref, pd.Series(["a"] * 45 + ["b"] * 45 + ["z"] * 10))
    assert r["pct_no_vistas"] == pytest.approx(0.10) and r["psi"] > 0.25


def test_nivel_psi_usa_prueba_como_respaldo():
    assert deriva.nivel_psi(0.30, 0.001, 0.1, 0.25, 0.01) == "alerta"
    assert deriva.nivel_psi(0.30, 0.20, 0.1, 0.25, 0.01) == "aviso"     # PSI alto pero puede ser ruido
    assert deriva.nivel_psi(0.05, 0.001, 0.1, 0.25, 0.01) == "estable"


def test_umbral_desempeno_suma_ruido_del_lote():
    assert deriva.umbral_desempeno(0.95, 0.01, 0.0, 2) == pytest.approx(0.93)
    assert deriva.umbral_desempeno(0.95, 0.01, 0.01, 2) < 0.93


def test_regla_disparo_requiere_lotes_seguidos_de_la_version_en_uso():
    assert regla_disparo([(6, "1", True), (7, "1", True)], "1", 7)[0]
    assert not regla_disparo([(6, "1", True), (7, "1", False)], "1", 7)[0]
    assert not regla_disparo([(7, "1", True), (8, "1", True)], "3", 1)[0]   # ya se reemplazó la versión


def test_deriva_sin_caida_no_dispara():
    assert not regla_disparo([(5, "1", False), (6, "1", False)], "1", 6)[0]


def test_regla_promocion():
    cv = {"cv_costo_std": 0.02, "cv_roc_auc_std": 0.01}
    assert regla_promocion({"costo": 0.40, "roc_auc": 0.85}, {"costo": 0.30, "roc_auc": 0.88}, cv)[0]
    assert not regla_promocion({"costo": 0.40, "roc_auc": 0.85}, {"costo": 0.39, "roc_auc": 0.88}, cv)[0]
    assert not regla_promocion({"costo": 0.40, "roc_auc": 0.90}, {"costo": 0.30, "roc_auc": 0.85}, cv)[0]


def test_diagnostico_tabla_6_2():
    assert deriva.diagnostico(True, False)[0] == "Deriva de datos sin impacto"
    assert deriva.diagnostico(True, True)[0] == "Deriva de datos con impacto"
    assert deriva.diagnostico(False, True)[0] == "Deriva conceptual"
