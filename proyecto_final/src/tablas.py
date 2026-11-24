"""DDL de las tablas de operación (esquema PSET2_DB.PRODUCCION).

Se crean con tipos explícitos para que write_pandas no infiera tipos distintos entre
cargas (por ejemplo, una columna toda nula en un lote de respaldo).
"""
from src import config

DDL = {
    config.T_LOTES: """
        review_id STRING, lote_id NUMBER, periodo_inicio DATE, periodo_fin DATE,
        fecha_target_disponible DATE, fecha_resena TIMESTAMP_NTZ, banco STRING,
        comentario_limpio STRING, n_palabras NUMBER, log_n_palabras FLOAT, n_neg_lexico NUMBER,
        es_solo_simbolos NUMBER, version_app_desconocida NUMBER, es_negativa NUMBER,
        con_deriva BOOLEAN, alterado_texto BOOLEAN, alterado_banco BOOLEAN,
        fecha_corte_entrenamiento TIMESTAMP_NTZ, creado_en TIMESTAMP_NTZ""",
    config.T_PREDICCIONES: """
        id_prediccion STRING, ts_prediccion TIMESTAMP_NTZ, review_id STRING, lote_id NUMBER,
        entradas STRING, probabilidad FLOAT, decision STRING, umbral FLOAT, modelo STRING,
        version_modelo STRING, alias STRING, run_id STRING, respaldo BOOLEAN,
        motivo_respaldo STRING, latencia_ms FLOAT""",
    config.T_SOLICITUDES: """
        id_solicitud STRING, ts TIMESTAMP_NTZ, metodo STRING, ruta STRING, estado_http NUMBER,
        latencia_ms FLOAT, lote_id NUMBER, error STRING""",
    config.T_MONITOREO: """
        lote_id NUMBER, lote_calculo NUMBER, ts TIMESTAMP_NTZ, tipo STRING, metrica STRING,
        feature STRING, valor FLOAT, umbral FLOAT, estado STRING, version_modelo STRING, detalle STRING""",
    config.T_ALERTAS: """
        id_alerta STRING, ts TIMESTAMP_NTZ, lote_id NUMBER, lote_calculo NUMBER, nombre STRING,
        metrica STRING, feature STRING, valor FLOAT, umbral FLOAT, ventana STRING, severidad STRING,
        receptor STRING, accion STRING""",
    config.T_REENTRENAMIENTOS: """
        id STRING, ts TIMESTAMP_NTZ, lote_disparo NUMBER, motivo STRING, lotes_entrenamiento STRING,
        lote_evaluacion NUMBER, run_id STRING, version_challenger STRING, version_champion STRING,
        costo_champion FLOAT, costo_challenger FLOAT, roc_auc_champion FLOAT, roc_auc_challenger FLOAT,
        pr_auc_champion FLOAT, pr_auc_challenger FLOAT, sd_costo_cv FLOAT, regla_promueve BOOLEAN,
        decision STRING, motivo_decision STRING, ts_decision TIMESTAMP_NTZ""",
    config.T_REFERENCIA: """
        clave STRING, valor STRING, creado_en TIMESTAMP_NTZ""",
}


def sentencias(reiniciar: bool = False):
    base, esquema = config.ESQUEMA_PROD.split(".")
    out = [f"CREATE SCHEMA IF NOT EXISTS {base}.{esquema}"]
    for tabla, cols in DDL.items():
        verbo = "CREATE OR REPLACE TABLE" if reiniciar else "CREATE TABLE IF NOT EXISTS"
        out.append(f"{verbo} {tabla} ({' '.join(cols.split())})")
    return out


def asegurar_tablas(reiniciar: bool = False):
    from src.datos import ejecutar_sql
    ejecutar_sql(*sentencias(reiniciar))


if __name__ == "__main__":
    import sys
    asegurar_tablas(reiniciar="--reiniciar" in sys.argv)
    print("Tablas de PRODUCCION listas" + (" (reiniciadas)" if "--reiniciar" in sys.argv else ""))
