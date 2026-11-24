"""Catálogo de alertas (sección 6.3) y regla de disparo del reentrenamiento (sección 7.1).

Cada alerta: métrica, umbral, ventana, severidad, quién la recibe y acción. El catálogo se
exporta tal cual a la tabla del documento técnico (python -m monitoreo.alertas).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from src import config as C


@dataclass(frozen=True)
class DefAlerta:
    nombre: str
    metrica: str
    umbral: str
    ventana: str
    severidad: str
    receptor: str
    accion: str


CATALOGO = {
    "psi_feature": DefAlerta(
        "Deriva de una feature", "PSI de cada feature vs referencia (KS / chi² como respaldo)",
        f"PSI > {C.PSI_ALERTA} y p < {C.P_VALOR}", "1 lote", "Alta", "Equipo de datos",
        "Revisar la fuente y la ingesta; si el cambio es real, evaluar reentrenamiento"),
    "psi_prediccion": DefAlerta(
        "Deriva de las predicciones", "PSI de la probabilidad predicha vs referencia",
        f"PSI > {C.PSI_ALERTA}", "1 lote", "Media", "Equipo de ciencia de datos",
        "Revisar qué features explican el cambio y la carga del equipo de atención"),
    "categorias_nuevas": DefAlerta(
        "Categorías no vistas", "% de filas con banco no visto en entrenamiento",
        f"> {C.CATEGORIAS_NUEVAS_MAX:.0%}", "1 lote", "Media", "Equipo de datos",
        "Incorporar la entidad al esquema y al entrenamiento; mientras tanto la API la rechaza (422)"),
    "faltantes": DefAlerta(
        "Texto faltante", "% de reseñas sin texto o con features nulas",
        f"> {C.FALTANTES_MAX:.0%} sobre la referencia", "1 lote", "Media", "Equipo de datos",
        "Revisar el extractor de Google Play y la capa Silver"),
    "fuera_rango": DefAlerta(
        "Valores fuera de rango", "% de filas fuera del rango de la firma", "> 0 %", "1 lote", "Media",
        "Equipo de datos", "Corregir la transformación que produjo el valor"),
    "desempeno": DefAlerta(
        "Caída de desempeño (deriva conceptual)", "ROC-AUC del lote con target vs CV del modelo en uso",
        f"< media_CV − {C.K_SIGMA:g}·σ_total", "1 lote", "Alta", "Equipo de ciencia de datos",
        "Diagnosticar (tabla 6.2). Con 2 lotes seguidos se dispara el reentrenamiento"),
    "costo_vs_regla": DefAlerta(
        "Modelo peor que la regla actual", "Costo por reseña del modelo vs regla de negocio (léxico)",
        "costo modelo ≥ costo regla", "1 lote", "Crítica", "Jefatura de atención digital + ciencia de datos",
        "Activar el interruptor de respaldo (la regla decide) hasta reentrenar o retirar"),
    "tasa_decision": DefAlerta(
        "Cambio en la carga de trabajo", "Tasa de 'priorizar' vs referencia",
        f"± {C.CAMBIO_TASA_DECISION_MAX:.0%} (puntos porcentuales)", "1 lote", "Media",
        "Jefatura de atención digital", "Ajustar la capacidad del equipo o revisar el umbral"),
    "latencia": DefAlerta(
        "Latencia alta", "Latencia p95 de POST /predecir", f"> {C.LATENCIA_P95_MS:.0f} ms", "1 lote", "Media",
        "Equipo de plataforma", "Revisar recursos del contenedor y el tamaño del modelo"),
    "errores_5xx": DefAlerta(
        "Errores del servicio", "Tasa de respuestas 5xx", f"> {C.TASA_5XX_MAX:.0%}", "1 lote", "Alta",
        "Equipo de plataforma", "Revisar logs; si persiste, activar el respaldo"),
    "errores_4xx": DefAlerta(
        "Entradas rechazadas", "Tasa de respuestas 4xx (422)", f"> {C.TASA_4XX_MAX:.0%}", "1 lote", "Media",
        "Equipo de datos / integración", "Revisar qué campo falla (detalle del 422) y corregir el origen"),
    "respaldo_activo": DefAlerta(
        "Servicio en modo respaldo", "% de predicciones hechas por la regla de respaldo", "> 0 %", "1 lote",
        "Alta", "Equipo de plataforma + ciencia de datos", "Verificar MLflow y el alias champion"),
}


def regla_disparo(historial_desempeno, version_actual: str, lotes_desde_ultimo: int | None) -> tuple[bool, str]:
    """¿Se dispara el reentrenamiento?

    historial_desempeno: lista de (lote_id, version_modelo, alerta_bool), ordenada por lote.
    - Por desempeño: las últimas LOTES_CONSECUTIVOS_DISPARO evaluaciones de la versión EN USO
      tienen alerta. Solo cuentan lotes que predijo la versión actual: si un lote malo lo predijo
      un champion ya reemplazado, no vuelve a disparar.
    - Por calendario: pasaron REENTRENO_CALENDARIO_LOTES lotes desde el último reentrenamiento.
    La deriva de datos sola NO dispara (no entra en esta regla).
    """
    propios = [h for h in historial_desempeno if str(h[1]) == str(version_actual)]
    n = C.LOTES_CONSECUTIVOS_DISPARO
    if len(propios) >= n and all(h[2] for h in propios[-n:]):
        lotes = ", ".join(str(h[0]) for h in propios[-n:])
        return True, f"desempeño: ROC-AUC bajo el umbral en {n} lotes seguidos ({lotes}) con la versión {version_actual}"
    if lotes_desde_ultimo is not None and lotes_desde_ultimo >= C.REENTRENO_CALENDARIO_LOTES:
        return True, f"calendario: {lotes_desde_ultimo} lotes desde el último reentrenamiento"
    return False, ""


def tabla_markdown() -> str:
    filas = ["| Alerta | Métrica | Umbral | Ventana | Severidad | Quién la recibe | Acción |",
             "|---|---|---|---|---|---|---|"]
    for a in CATALOGO.values():
        d = asdict(a)
        filas.append("| " + " | ".join(d[k] for k in ("nombre", "metrica", "umbral", "ventana", "severidad",
                                                        "receptor", "accion")) + " |")
    return "\n".join(filas)


if __name__ == "__main__":
    print(tabla_markdown())
