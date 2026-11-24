-- PSet #4 · objetos de operación en Snowflake (ejecutar una vez, con PSET2_ROLE).
USE ROLE PSET2_ROLE;
USE WAREHOUSE PSET2_WH;
USE DATABASE PSET2_DB;

-- 1) Instantánea inmutable de la OBT con la que se entrenó el champion del PSet #3.
--    CLONE no copia datos (zero-copy) y no cambia aunque la ingesta reescriba la OBT.
--    Ejecútalo ANTES de volver a correr la ingesta. Verifica que tenga 21 698 filas.
CREATE TABLE IF NOT EXISTS OBT.OBT_RESENAS_SNAP_PSET3 CLONE OBT.OBT_RESENAS;
SELECT COUNT(*) AS filas, MIN(fecha_resena), MAX(fecha_resena) FROM OBT.OBT_RESENAS_SNAP_PSET3;

-- 2) Esquema de operación. Las tablas (LOTES, PREDICCIONES, SOLICITUDES, MONITOREO, ALERTAS,
--    REENTRENAMIENTOS, REFERENCIA) las crea el código con tipos explícitos (src/tablas.py).
CREATE SCHEMA IF NOT EXISTS PRODUCCION;

-- 3) Trazabilidad de una predicción hasta los datos (sección 10.6)
-- SELECT p.id_prediccion, p.version_modelo, p.run_id, p.ts_prediccion, l.review_id, l.lote_id
-- FROM PRODUCCION.PREDICCIONES p JOIN PRODUCCION.LOTES l USING (review_id, lote_id)
-- WHERE p.id_prediccion = '<id>';
-- run_id → MLflow: parámetros obt_snapshot / lotes_snapshot / hash_dataset / tabla_origen
-- OBT_RESENAS_SNAP_PSET3 ← GOLD.FCT_RESENA + DIM_BANCO + DIM_FECHA + DIM_VERSION_APP (spark/build_obt.py)
