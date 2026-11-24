"""2. Modelo: la copia exportada reproduce la métrica registrada en MLflow sobre la muestra fija.

Tolerancias:
- PR-AUC: 0,002. El modelo y los datos son los mismos; solo cambian la CPU y la compilación de
  torch, que mueven los embeddings en el orden de 1e-6. Una diferencia mayor que 0,002 (diez
  veces menos que la desviación entre pliegues de la CV, 0,0145) indica otro modelo u otra
  preparación de los datos.
- Costo: una decisión de diferencia (5 / 400 = 0,0125). Un cambio de 1e-6 en la probabilidad
  solo puede voltear una reseña que esté justo en el umbral.
"""
import pytest
from sklearn.metrics import average_precision_score

from src.pipeline import FEATURES, TARGET, costo_negocio

TOL_PR_AUC = 0.002
TOL_COSTO = 5 / 400 + 1e-9


def test_reproduce_metrica_registrada(modelo, muestra, metadata):
    p = modelo.predict_proba(muestra[FEATURES])[:, 1]
    pr = average_precision_score(muestra[TARGET], p)
    costo = costo_negocio(muestra[TARGET], (p >= metadata["umbral"]).astype(int))
    assert pr == pytest.approx(metadata["muestra_ci_pr_auc"], abs=TOL_PR_AUC)
    assert costo == pytest.approx(metadata["muestra_ci_costo"], abs=TOL_COSTO)


def test_supera_linea_base_en_la_muestra(modelo, muestra, metadata):
    """El champion debe costar menos que la regla de negocio (léxico) en la misma muestra."""
    p = modelo.predict_proba(muestra[FEATURES])[:, 1]
    costo_modelo = costo_negocio(muestra[TARGET], (p >= metadata["umbral"]).astype(int))
    costo_regla = costo_negocio(muestra[TARGET], (muestra["n_neg_lexico"] >= 1).astype(int))
    assert costo_modelo < costo_regla
