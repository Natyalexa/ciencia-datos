"""Deriva de datos (PSI + KS / chi-cuadrado) y deriva conceptual (métrica del lote vs CV).

Convenciones:
- PSI = Σ (p_lote − p_ref) · ln(p_lote / p_ref), con bins fijados en la referencia.
- Variables continuas: bins por deciles de la referencia (bordes abiertos en los extremos).
- Variables discretas o categóricas: un bin por valor; los valores no vistos van a "__nuevo__".
- ε = 1e-4 evita log(0) cuando un bin queda vacío.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-4
NUEVO = "__nuevo__"


# ---------------------------------------------------------------- referencia
def perfil_continuo(x: pd.Series, n_bins: int = 10) -> dict:
    """Bordes por cuantiles de la referencia y proporción por bin."""
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    bordes = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    bordes[0], bordes[-1] = -np.inf, np.inf
    if len(bordes) < 3:  # variable casi constante
        bordes = np.array([-np.inf, float(x.median()), np.inf])
    cuenta = np.histogram(x, bins=bordes)[0]
    return {"tipo": "continua", "bordes": [float(b) for b in bordes],
            "prop": (cuenta / cuenta.sum()).tolist(), "muestra": x.sample(min(len(x), 5000), random_state=42).tolist()}


def perfil_categorico(x: pd.Series) -> dict:
    p = x.astype(str).value_counts(normalize=True)
    return {"tipo": "categorica", "categorias": p.index.tolist(), "prop": p.values.tolist(),
            "n": int(len(x))}


def perfil(df: pd.DataFrame, continuas, categoricas) -> dict:
    out = {c: perfil_continuo(df[c]) for c in continuas}
    out.update({c: perfil_categorico(df[c]) for c in categoricas})
    return out


# ---------------------------------------------------------------- PSI y pruebas
def _psi(p_ref: np.ndarray, p_act: np.ndarray) -> float:
    p_ref = np.clip(np.asarray(p_ref, float), EPS, None)
    p_act = np.clip(np.asarray(p_act, float), EPS, None)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def comparar_continua(ref: dict, x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    cuenta = np.histogram(x, bins=np.asarray(ref["bordes"]))[0]
    p_act = cuenta / max(cuenta.sum(), 1)
    ks = stats.ks_2samp(x, np.asarray(ref["muestra"])) if len(x) else None
    return {"psi": _psi(ref["prop"], p_act), "prueba": "KS",
            "estadistico": float(ks.statistic) if ks else np.nan,
            "p_valor": float(ks.pvalue) if ks else np.nan}


def comparar_categorica(ref: dict, x: pd.Series) -> dict:
    x = x.astype(str)
    cats = list(ref["categorias"])
    act = x.where(x.isin(cats), NUEVO).value_counts()
    obs = np.array([act.get(c, 0) for c in cats] + [act.get(NUEVO, 0)], float)
    p_ref = np.array(list(ref["prop"]) + [0.0])
    psi = _psi(p_ref, obs / max(obs.sum(), 1))
    # chi-cuadrado de bondad de ajuste contra las proporciones de referencia
    esperado = np.clip(p_ref, EPS, None)
    esperado = esperado / esperado.sum() * obs.sum()
    chi = stats.chisquare(obs, esperado) if obs.sum() else None
    return {"psi": psi, "prueba": "chi2",
            "estadistico": float(chi.statistic) if chi else np.nan,
            "p_valor": float(chi.pvalue) if chi else np.nan,
            "pct_no_vistas": float(obs[-1] / max(obs.sum(), 1))}


def deriva_lote(ref_perfil: dict, lote: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for col, ref in ref_perfil.items():
        if col not in lote:
            continue
        r = comparar_continua(ref, lote[col]) if ref["tipo"] == "continua" else comparar_categorica(ref, lote[col])
        filas.append({"feature": col, **r})
    return pd.DataFrame(filas)


def nivel_psi(psi: float, p_valor: float, aviso: float, alerta: float, alfa: float) -> str:
    """El PSI decide; la prueba estadística respalda (descarta diferencias que pueden ser ruido)."""
    if psi > alerta and (np.isnan(p_valor) or p_valor < alfa):
        return "alerta"
    if psi > aviso:
        return "aviso"
    return "estable"


# ---------------------------------------------------------------- deriva conceptual
def bootstrap_metrica(y, p, metrica: str = "roc_auc", n: int = 300, semilla: int = 42) -> float:
    """Desviación estándar de la métrica del lote por bootstrap (ruido por tamaño de muestra)."""
    y, p = np.asarray(y), np.asarray(p)
    f = roc_auc_score if metrica == "roc_auc" else average_precision_score
    rng = np.random.default_rng(semilla)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(f(y[i], p[i]))
    return float(np.std(vals)) if vals else np.nan


def umbral_desempeno(media_cv: float, sd_cv: float, sd_lote: float, k: float) -> float:
    """Límite inferior: media de la CV − k·σ_total, con σ_total² = σ_CV² + σ_lote².

    σ_CV mide cuánto varía el modelo entre períodos de entrenamiento; σ_lote, el ruido por
    evaluar en un lote de pocos cientos de reseñas. Sin σ_lote, cualquier lote pequeño
    dispararía falsas alarmas.
    """
    return float(media_cv - k * np.sqrt(sd_cv ** 2 + (0 if np.isnan(sd_lote) else sd_lote) ** 2))


def diagnostico(entradas_cambian: bool, desempeno_cae: bool) -> tuple[str, str]:
    """Tabla 6.2 del enunciado aplicada al proyecto."""
    if entradas_cambian and not desempeno_cae:
        return ("Deriva de datos sin impacto",
                "Registrar y vigilar; no reentrenar. Revisar con el equipo de datos si el cambio es de la fuente.")
    if entradas_cambian and desempeno_cae:
        return ("Deriva de datos con impacto",
                "Revisar la fuente (¿error de ingesta?) y, si el cambio es real, reentrenar con lotes etiquetados recientes.")
    if not entradas_cambian and desempeno_cae:
        return ("Deriva conceptual",
                "Reentrenar con mayor peso en lo reciente y revisar con atención al cliente si cambió qué es una queja.")
    return ("Sin deriva", "Ninguna.")
