# Deriva inducida en la simulación

Código: `src/simulacion.py::inducir_deriva` · parámetros en `src/config.py` · semilla 42 (reproducible).

| | Qué se alteró | Lotes | Por qué |
|---|---|---|---|
| 1. Recorte de texto | El 60 % de las reseñas con más de 3 palabras se recortan a sus **3 primeras palabras**. Se recalculan `n_palabras`, `log_n_palabras` y `n_neg_lexico`. **El target (estrellas) no cambia.** | Los últimos 5 lotes | Simula que la fuente (Google Play o el extractor) empieza a entregar solo el inicio del comentario. Cambia la distribución de `log_n_palabras` (deriva de datos) y la relación entre el texto visible y el target: una queja larga pasa a verse como un texto corto, y en el entrenamiento los textos cortos eran mayoritariamente positivos (11 % de negativas con ≤ 2 palabras). |
| 2. Banco nuevo | El 3 % de las filas pasan a `banco = "Banco Internacional"`. | Los últimos 5 lotes | Simula la incorporación de una entidad que el modelo no conoce. El esquema de la API la rechaza (422) y el monitoreo la cuenta como categoría no vista. |

Qué debe pasar (y se verifica en `notebooks/07_analisis_monitoreo.ipynb`):

1. En el primer lote con deriva se disparan las alertas de PSI de `log_n_palabras` (y de `n_neg_lexico` y de la probabilidad), la de categorías no vistas y la de tasa de 4xx. En los lotes sin alterar no deberían dispararse.
2. Cuando llega el target de los lotes con deriva, la ROC-AUC cae bajo `media_CV − 2·σ_total` → diagnóstico **deriva de datos con impacto**.
3. Con 2 lotes seguidos con caída (predichos por la misma versión) se dispara `reentrenamiento.yml`: el challenger se entrena con el histórico y los lotes etiquetados (peso 5), se compara con el champion en el lote etiquetado más reciente y espera la aprobación.
4. Después de aprobar, los lotes siguientes los predice la versión nueva: la deriva de datos sigue (el texto sigue recortado), pero el desempeño debería recuperarse → **deriva de datos sin impacto**, que no vuelve a disparar el reentrenamiento.

Si el challenger no supera al champion, la regla lo rechaza y se documenta la comparación (tabla `PRODUCCION.REENTRENAMIENTOS`).
