# Registro de cambios (PSets #1 a #3 → proyecto final)

## A. Observaciones recibidas del profesor

| PSet | Observación | Qué cambié | Archivo(s) |
|---|---|---|---|
| #1 | Sin observaciones registradas | Las cifras del pitch se reemplazan por las de la simulación (sección 9 del memo) | `docs/PSet4_presentacion_Cuichan_Ortiz.pdf` |
| #2 · Kestra (8,5) | Integrar dbt y Spark en el flujo para que el DAG continúe automáticamente después de la ingesta | El flujo `ingesta` ahora ejecuta `dbt_build` (Bronze → Silver → Gold + 43 pruebas) y `spark_obt` (OBT validada) después de cargar RAW. Ejecución verificada: 8 tareas en verde en 2 min 16 s | `kestra/ingesta.yml`, `dbt/Dockerfile`, `spark/Dockerfile`, `docs/evidencias/07_kestra_ingesta_dbt_spark_ok.png` |
| #2 · Infraestructura (9,5) | Automatizar la importación del flujo y la ejecución de las transformaciones; requerían pasos manuales | Kestra carga los flujos de `kestra/` al arrancar (`--flow-path`) y los vuelve a cargar cada vez que se guarda un archivo (`micronaut.io.watch`): ya no se copian en la interfaz. Las transformaciones las ejecuta el propio flujo (fila anterior) y las imágenes de dbt y Spark llevan el código dentro | `docker-compose.yml` (servicio `kestra`), `README.md` |
| #3 · EDA y features (8,5) | Guardar las salidas del notebook 02 para verificar las pruebas de las features y las exclusiones por fuga | El notebook 02 se ejecuta con Restart & Run All y se guarda con sus salidas | `notebooks/02_features.ipynb` |
| #3 · Líneas base y candidatos (8,5) | Falta ejecutar un LLM por prompting; la falta de presupuesto para la API no descarta modelos locales | Nuevo notebook con un LLM local (Qwen2.5-1.5B-Instruct en Ollama, en Docker, sin costo) en modo zero-shot, evaluado en la misma muestra fija del hold-out que el champion y la regla, con umbral elegido en el entrenamiento; registrado en MLflow (`familia = llm_local`) | `notebooks/03c_llm_local.ipynb`, `docker-compose.yml` (servicio `ollama`, perfil `llm`) |
| #3 · Reproducibilidad (8,5) | Guardar las salidas de los notebooks 02 y 03b y eliminar los archivos duplicados del repositorio | Notebooks 02 y 03b ejecutados y guardados con salidas. El proyecto final no copia los duplicados de la raíz de `pset_3/` (`Dockerfile (1)`, `requirements (2).txt`, notebooks y `.py` repetidos); notebooks solo en `notebooks/` y código solo en `src/` | `notebooks/`, `src/` |

## B. Decisiones actualizadas

| # | Decisión anterior | Decisión nueva | Por qué | Archivo(s) |
|---|---|---|---|---|
| 1 | Deriva conceptual y alertas de desempeño con PR-AUC (métrica técnica del PSet #3) | Se mantiene PR-AUC como métrica técnica, pero **la alerta de desempeño usa ROC-AUC**. El PR-AUC se reporta como aviso | El PR-AUC depende de la prevalencia: la proporción semanal de negativas varía entre 19 % y 50 % (EDA, H4), así que un lote con menos quejas "cae" en PR-AUC sin que el modelo empeore. La ROC-AUC no depende de la prevalencia | `monitoreo/ejecutar.py`, `src/config.py` |
| 2 | Umbral de desempeño: media de la CV − k·σ_CV | media_CV − 2·√(σ_CV² + σ_lote²), con σ_lote por bootstrap | Un lote semanal tiene unos cientos de reseñas: sin el ruido de muestreo, lotes normales darían falsas alarmas | `monitoreo/deriva.py` |
| 3 | La OBT se leía completa (`OBT_RESENAS`) en cada notebook | Los notebooks 01-05 leen la instantánea `OBT_RESENAS_SNAP_PSET3` (zero-copy CLONE) | La ingesta reescribe la OBT con reseñas nuevas: sin la instantánea, "Restart & Run All" cambiaría el split y las métricas del PSet #3 | `src/datos.py`, `notebooks/01-05`, `docs/setup_produccion.sql` |
| 4 | El notebook 05 borraba y recreaba el modelo registrado | Reutiliza las corridas finales si existen | Volver a ejecutarlo borraba las versiones de reentrenamiento y el historial de alias | `notebooks/05_seleccion.ipynb` |
| 5 | `06_verificacion_champion.ipynb` | Renombrado a `05b_verificacion_champion.ipynb`; verifica la versión del champion del PSet #3, no el alias actual | El enunciado reserva 06 y 07 para simulación y monitoreo; el alias champion ahora puede apuntar a una versión reentrenada | `notebooks/05b_verificacion_champion.ipynb` |
| 6 | Kestra solo hacía la ingesta; dbt y Spark se corrían a mano (limitación del PSet #2) | `kestra/ingesta.yml` encadena ingesta → `dbt build` → Spark | Una sola ejecución lleva la fuente hasta la OBT | `kestra/ingesta.yml`, `dbt/Dockerfile`, `spark/Dockerfile` |
| 7 | La cuenta de Snowflake se editaba dentro del flujo de Kestra | Se lee de `.env` (`ENV_SNOWFLAKE_ACCOUNT`) | Evita editar el flujo y subir datos de la cuenta | `kestra/*.yml`, `.env.example` |
| 8 | Llave privada solo como archivo `/secrets/rsa_key.p8` | También por variable (`SNOWFLAKE_PRIVATE_KEY_PEM`) desde el secreto de Kestra | Los contenedores que lanza Kestra no montan carpetas | `src/datos.py`, `dbt/profiles.yml`, `spark/build_obt.py` |
| 9 | El run del champion registraba solo la desviación del PR-AUC | Se agregan `cv_roc_auc_std`, `cv_costo_mean` y `cv_costo_std` al mismo run (misma CV temporal) | Los necesitan la alerta de desempeño y la regla de promoción | `monitoreo/referencia.py` |
| 10 | Copias sueltas en la raíz de `pset_3/` (`Dockerfile (1)`, `requirements (2).txt`, notebooks y `.py` duplicados) | No se copian al proyecto final | Una sola fuente de verdad: `notebooks/` y `src/` | — |
| 11 | Requisitos distintos en Jupyter y en la API | Un solo `requirements.txt` para Jupyter/trabajos, API y CI | Un pickle de scikit-learn solo es confiable con la misma versión | `requirements.txt`, `api/requirements.txt`, `docker/ds/requirements.txt` |
| 12 | Cifras del pitch del PSet #1 | Se reemplazan por las de la simulación (sección 9 del memo) | Coherencia pedida en la sección 10.7 | `docs/PSet4_presentacion_*.pdf` |

## C. Pendientes de verificar

- Resuelto: el champion v1 registrado es `972c63a668334b73a195ce78b0bb44ea` (el del memo del PSet #3); la salida antigua del notebook de verificación mostraba otro `run_id` de una re-ejecución anterior. El 05b vuelve a ejecutarse.
- Confirmar con el memo del PSet #1 la acción exacta asociada a cada decisión (texto en `src/config.py::ACCIONES`).
