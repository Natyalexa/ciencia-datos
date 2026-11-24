"""Preparación, features y utilidades de evaluación (PSet 3).

Todo lo que se ajusta con datos (TF-IDF, one-hot, escalado) vive dentro de un
Pipeline de scikit-learn, así solo se ajusta con el conjunto de entrenamiento.
Las features derivadas son funciones fila a fila (no aprenden de otras filas).
"""
import os
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

SEED = 42
TARGET = "es_negativa"
TEXTO = "comentario_limpio"
FECHA = "fecha_resena"

# Variables que NO se usan como features:
#  - calificacion / sentimiento_estrellas: definen el target.
#  - me_gusta: se acumula DESPUÉS de publicar la reseña (no existe al predecir).
#  - tiene_respuesta_banco / respuesta_inconsistente / horas_hasta_respuesta:
#    ocurren DESPUÉS de la reseña y el banco responde más a las negativas.
EXCLUIDAS = ["calificacion", "sentimiento_estrellas", "me_gusta",
             "tiene_respuesta_banco", "respuesta_inconsistente", "horas_hasta_respuesta"]

CAT = ["banco"]
NUM = ["log_n_palabras", "n_neg_lexico"]
BIN = ["es_solo_simbolos", "version_app_desconocida"]

# Léxico simple de quejas (también es la línea base de negocio).
LEXICO_NEG = [
    "pesim", "horrible", "malo", "mala", "peor", "basura", "estafa", "ladron",
    "no funciona", "no sirve", "no me deja", "no puedo", "no abre", "no carga",
    "se cae", "se cuelga", "error", "falla", "fallas", "problema", "problemas",
    "lento", "demora", "bloque", "pesimo", "inservible", "deficiente", "cobran",
    "reclamo", "no entra", "no permite", "mejoren", "arreglen", "solucionen",
]
_RE_NEG = re.compile("|".join(re.escape(p) for p in LEXICO_NEG))


def _sin_acentos(s: str) -> str:
    tabla = str.maketrans("áéíóúüñ", "aeiouun")
    return s.lower().translate(tabla)


def conteo_lexico(texto: str) -> int:
    return len(_RE_NEG.findall(_sin_acentos(texto or "")))


def agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derivadas fila a fila (sin aprender de otras filas => sin fuga)."""
    out = df.copy()
    out[TARGET] = (out["calificacion"] <= 2).astype(int) if "calificacion" in out else out.get(TARGET)
    out["log_n_palabras"] = np.log1p(out["n_palabras"].astype(float))
    out["n_neg_lexico"] = out[TEXTO].fillna("").map(conteo_lexico)
    for c in BIN:
        out[c] = out[c].astype(int)
    out[TEXTO] = out[TEXTO].fillna("")
    return out


def split_temporal(df: pd.DataFrame, frac_test: float = 0.2):
    """Hold-out temporal: el 20% más reciente por fecha. Devuelve (train, test) ordenados por fecha."""
    d = df.sort_values([FECHA, "review_id"], kind="mergesort").reset_index(drop=True)
    corte = int(len(d) * (1 - frac_test))
    return d.iloc[:corte].copy(), d.iloc[corte:].copy()


def _a_float32_denso(X):
    return X.toarray().astype(np.float32) if hasattr(X, "toarray") else np.asarray(X, dtype=np.float32)


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Embeddings de un modelo preentrenado (sentence-transformers). No aprende de los datos,
    así que no hay fuga. Cachea por texto para no recalcular en cada pliegue."""

    _modelos = {}   # a nivel de clase: no se serializa con el pipeline
    _cache = {}
    _disco_cargado = False
    RUTA_CACHE = os.environ.get("EMB_CACHE", "/work/.cache/embeddings.pkl")

    @classmethod
    def _leer_disco(cls):
        if cls._disco_cargado:
            return
        cls._disco_cargado = True
        try:
            with open(cls.RUTA_CACHE, "rb") as f:
                cls._cache.update(pickle.load(f))
        except Exception:
            pass

    @classmethod
    def _guardar_disco(cls):
        try:
            os.makedirs(os.path.dirname(cls.RUTA_CACHE), exist_ok=True)
            with open(cls.RUTA_CACHE, "wb") as f:
                pickle.dump(cls._cache, f, protocol=4)
        except Exception:
            pass

    def __init__(self, modelo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", batch_size=128):
        self.modelo = modelo
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def _cargar(self):
        if self.modelo not in EmbeddingTransformer._modelos:
            from sentence_transformers import SentenceTransformer
            EmbeddingTransformer._modelos[self.modelo] = SentenceTransformer(self.modelo, device="cpu")
        return EmbeddingTransformer._modelos[self.modelo]

    def transform(self, X):
        EmbeddingTransformer._leer_disco()
        textos = [str(t) for t in np.asarray(X).ravel()]
        nuevos = [t for t in dict.fromkeys(textos) if (self.modelo, t) not in EmbeddingTransformer._cache]
        if nuevos:
            vec = self._cargar().encode(nuevos, batch_size=self.batch_size, show_progress_bar=False,
                                        convert_to_numpy=True, normalize_embeddings=True)
            for t, v in zip(nuevos, vec):
                EmbeddingTransformer._cache[(self.modelo, t)] = v.astype(np.float32)
            EmbeddingTransformer._guardar_disco()
        return np.vstack([EmbeddingTransformer._cache[(self.modelo, t)] for t in textos])


def construir_preprocesador(max_features: int = 3000, texto: str = "tfidf") -> ColumnTransformer:
    vec = (EmbeddingTransformer() if texto == "embeddings" else
           TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_features,
                           sublinear_tf=True, strip_accents="unicode"))
    return ColumnTransformer([
        ("texto", vec, TEXTO),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM),
        ("bin", "passthrough", BIN),
    ])


def construir_pipeline(modelo, max_features: int = 3000, denso: bool = False, texto: str = "tfidf") -> Pipeline:
    pasos = [("prep", construir_preprocesador(max_features, texto))]
    if denso:  # p.ej. HistGradientBoosting / MLP lo requieren
        pasos.append(("denso", FunctionTransformer(_a_float32_denso, accept_sparse=True)))
    pasos.append(("modelo", modelo))
    return Pipeline(pasos)


FEATURES = [TEXTO] + CAT + NUM + BIN


# ---------------------------------------------------------------- negocio
COSTO_FN = 5.0  # reseña negativa que nadie atiende
COSTO_FP = 1.0  # reseña revisada que no era negativa


def costo_negocio(y_true, y_pred, c_fn: float = COSTO_FN, c_fp: float = COSTO_FP) -> float:
    """Costo promedio por reseña."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return float((c_fn * fn + c_fp * fp) / len(y_true))


def mejor_umbral(y_true, proba, c_fn: float = COSTO_FN, c_fp: float = COSTO_FP):
    """Umbral que minimiza el costo. Se debe elegir con datos de entrenamiento/CV, no con la prueba."""
    umbrales = np.linspace(0.05, 0.95, 91)
    costos = [costo_negocio(y_true, (proba >= u).astype(int), c_fn, c_fp) for u in umbrales]
    i = int(np.argmin(costos))
    return float(umbrales[i]), float(costos[i])


class ReglaNegocio(BaseEstimator, ClassifierMixin):
    """Línea base de negocio: marcar negativa si el texto contiene alguna palabra de queja."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (X[TEXTO].fillna("").map(conteo_lexico) >= 1).astype(float).values
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def regla_negocio(df: pd.DataFrame) -> np.ndarray:
    return ReglaNegocio().fit(df).predict(df)
