# PSet #3 · Experimentación y selección del modelo

Pregunta de negocio: **priorizar las reseñas negativas** (calificación ≤ 2) de Google Play de 8 bancos del Ecuador para que reciban atención primero.
Fuente: `PSET2_DB.OBT.OBT_RESENAS` (Snowflake, PSet #2). Etiqueta derivada de las estrellas (no hay etiquetado manual).

## 1. Requisitos
- Docker Desktop.
- La llave privada del usuario de servicio del PSet #2 en `secrets/rsa_key.p8` (no se versiona).

## 2. Configuración
1. Copia `.env.example` como `.env` y completa: `SNOWFLAKE_ACCOUNT`, `MLFLOW_DB_PASSWORD`, `JUPYTER_TOKEN` y `GIT_COMMIT` (SHA del commit usado).
2. Coloca la llave en `secrets/rsa_key.p8`.

> **Nunca subas** `.env`, `secrets/`, `.cache/` ni `mlruns/` al repositorio (ya están en `.gitignore`).

## 3. Levantar MLflow y Jupyter
```
docker compose up -d --build
```
- MLflow (tracking + registro de modelos): http://localhost:5000
- JupyterLab: http://localhost:8888 (token de `.env`)

## 4. Probar conexiones
```
docker compose exec jupyter python -m src.prueba_conexion
```

## 5. Orden de ejecución de los notebooks
`01_eda` → `02_features` → `03_baseline_candidatos` → `03b_embeddings` → `04_tuning` → `05_seleccion` → `06_verificacion_champion`

Ejecución limpia de todos (Restart + Run All, con salidas guardadas):
```
for %n in (01_eda 02_features 03_baseline_candidatos 03b_embeddings 04_tuning 05_seleccion 06_verificacion_champion) do docker compose exec jupyter jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/%n.ipynb
```
(en cmd; en un `.bat` usa `%%n`).

## 6. Decisiones de diseño
- Target `es_negativa = calificacion <= 2`. Excluidas por fuga/no disponibles en inferencia: calificacion, sentimiento_estrellas, me_gusta, tiene_respuesta_banco, respuesta_inconsistente, horas_hasta_respuesta.
- Validación: hold-out temporal (20 % más reciente) + `TimeSeriesSplit(5)` en entrenamiento.
- Métricas: PR-AUC (técnica) y costo por reseña con FN=5, FP=1 (sensibilidad 3:1, 5:1, 10:1). Umbral elegido sobre predicciones out-of-fold.
- Experimento: 2 baselines, 5 familias (LogReg, RF, LightGBM, MLP, Embeddings+LogReg), tuning de 3 modelos (30 configuraciones c/u, corridas anidadas) y corrida final.
- Regla de selección (pre-registrada): elegibles = dentro de 1 desv. estándar del mejor PR-AUC en CV; champion = menor costo en CV; challenger = siguiente.
- No se probó un LLM por prompting por falta de presupuesto de API (documentado en el memo).

## 7. Registro de modelos
Modelo `resenas-bancos-ec-modelo` con alias `champion` y `challenger`. `06_verificacion_champion` lo carga desde el registro y comprueba que reproduce las métricas.

## 8. Estructura
`src/` (datos, pipeline, experimento) · `notebooks/` · `docs/corridas.csv` · `docs/figs/` · `docs/PSet3_memo_Cuichan_Ortiz.pdf`
