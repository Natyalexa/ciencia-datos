"""Configuración única del proyecto (PSet #4).

Todas las decisiones numéricas de operación viven aquí para que el código, las
pruebas, los flujos de Kestra y el documento técnico usen los mismos valores.
"""
import os

# ---------------------------------------------------------------- identidad
PROYECTO = "resenas-bancos-ec"
NOMBRE_EXPERIMENTO = PROYECTO
NOMBRE_MODELO = f"{PROYECTO}-modelo"
ALIAS_PRODUCCION = "champion"
ALIAS_CANDIDATO = "challenger"
MODELO_URI = os.environ.get("MODELO_URI", f"models:/{NOMBRE_MODELO}@{ALIAS_PRODUCCION}")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# ---------------------------------------------------------------- Snowflake
BASE = "PSET2_DB"
TABLA_OBT = f"{BASE}.OBT.OBT_RESENAS"
# Copia inmutable (zero-copy CLONE) de la OBT con la que se entrenó el champion del PSet #3.
TABLA_OBT_SNAPSHOT = os.environ.get("OBT_SNAPSHOT", f"{BASE}.OBT.OBT_RESENAS_SNAP_PSET3")
ESQUEMA_PROD = f"{BASE}.PRODUCCION"
T_LOTES = f"{ESQUEMA_PROD}.LOTES"
T_PREDICCIONES = f"{ESQUEMA_PROD}.PREDICCIONES"
T_SOLICITUDES = f"{ESQUEMA_PROD}.SOLICITUDES"
T_MONITOREO = f"{ESQUEMA_PROD}.MONITOREO"
T_ALERTAS = f"{ESQUEMA_PROD}.ALERTAS"
T_REENTRENAMIENTOS = f"{ESQUEMA_PROD}.REENTRENAMIENTOS"
T_REFERENCIA = f"{ESQUEMA_PROD}.REFERENCIA"

# ---------------------------------------------------------------- formulación (PSet #1 / #3)
BANCOS = [
    "Banco Pichincha", "Banco Guayaquil", "Produbanco", "Diners Club (blu)",
    "Banco del Pacífico", "Banco Bolivariano", "Banco del Austro", "Cooperativa JEP",
]
COSTO_FN = 5.0          # queja que nadie atiende
COSTO_FP = 1.0          # reseña revisada que no era queja
UMBRAL_POR_DEFECTO = 0.22   # umbral del champion (elegido en CV, PSet #3); el real se lee del run
DECISION_POSITIVA = "priorizar"
DECISION_NEGATIVA = "cola_normal"
ACCIONES = {
    DECISION_POSITIVA: "Enviar a la cola prioritaria del equipo de atención digital: revisar y responder en menos de 24 h.",
    DECISION_NEGATIVA: "Dejar en la cola normal: revisión semanal por muestreo.",
}

# ---------------------------------------------------------------- simulación (sección 5)
HORIZONTE_LOTES = 1         # el target (estrellas confirmadas por atención al cliente) llega 1 lote (1 semana) después
MIN_LOTES = 8
MAX_LOTES = 10
MIN_FILAS_LOTE = 150        # semanas con menos reseñas se unen a la siguiente
SEMILLA = 42
# Deriva inducida (documentada en docs/deriva_inducida.md). Se aplica desde el lote DRIFT_DESDE
# hasta el último: simula un cambio persistente en la fuente (la tienda empieza a mostrar
# reseñas recortadas) y la llegada de un banco nuevo que el modelo no conoce.
DRIFT_LOTES_FINALES = 5         # los últimos 5 lotes tienen deriva (con 10 lotes: 6..10)
DRIFT_FRAC_TRUNCADAS = 0.60     # fracción de reseñas con más de 3 palabras que se recortan
DRIFT_PALABRAS = 3              # se conservan solo las 3 primeras palabras
DRIFT_FRAC_BANCO_NUEVO = 0.03   # fracción de filas reasignadas a un banco no visto
DRIFT_BANCO_NUEVO = "Banco Internacional"

# ---------------------------------------------------------------- monitoreo (sección 6)
REF_FRACCION = 0.25        # referencia de deriva = el 25 % más reciente del entrenamiento (ver memo §7)
PSI_AVISO = 0.10
PSI_ALERTA = 0.25
P_VALOR = 0.01             # respaldo estadístico (KS / chi-cuadrado)
K_SIGMA = 2.0              # desempeño: alerta si métrica < media_CV - K·σ_total
N_BOOTSTRAP = 300
LATENCIA_P95_MS = 300.0
TASA_5XX_MAX = 0.01
TASA_4XX_MAX = 0.02
CAMBIO_TASA_DECISION_MAX = 0.15   # 15 puntos porcentuales frente a la referencia
FALTANTES_MAX = 0.05
CATEGORIAS_NUEVAS_MAX = 0.01

# ---------------------------------------------------------------- reentrenamiento (sección 7)
LOTES_CONSECUTIVOS_DISPARO = 2   # 2 lotes seguidos con alerta de desempeño
REENTRENO_CALENDARIO_LOTES = 13  # además, uno por trimestre (13 semanas) aunque no haya alertas
PESO_HISTORICO = 1.0
PESO_PRODUCCION = 5.0            # los lotes recientes pesan 5 veces más (ver memo §6)
CV_SPLITS_REENTRENO = 5
