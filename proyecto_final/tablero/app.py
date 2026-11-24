"""Tablero de monitoreo lote por lote (Streamlit).

Lee PRODUCCION.MONITOREO, ALERTAS y REENTRENAMIENTOS de Snowflake. Sin conexión, usa los CSV
exportados en docs/ (monitoreo.csv, alertas.csv, reentrenamientos_decisiones.csv).
"""
import os

import altair as alt
import pandas as pd
import streamlit as st

from src import config as C

st.set_page_config(page_title="Monitoreo · reseñas bancarias", layout="wide")


@st.cache_data(ttl=60)
def cargar():
    from src import datos
    if datos.snowflake_configurado():
        try:
            m = datos.leer_sql(f"SELECT * FROM {C.T_MONITOREO}")
            a = datos.leer_sql(f"SELECT * FROM {C.T_ALERTAS}")
            r = datos.leer_sql(f"SELECT * FROM {C.T_REENTRENAMIENTOS}")
            return m, a, r, "Snowflake"
        except Exception as e:  # noqa: BLE001
            st.warning(f"No se pudo leer Snowflake ({type(e).__name__}); se usan los CSV de docs/")
    leer = lambda n: pd.read_csv(f"docs/{n}") if os.path.exists(f"docs/{n}") else pd.DataFrame()
    m, a, r = leer("monitoreo.csv"), leer("alertas.csv"), leer("reentrenamientos_decisiones.csv")
    for df in (m, a, r):
        df.columns = [c.lower().replace("tags.", "").replace("metrics.", "") for c in df.columns]
    return m, a, r, "CSV (docs/)"


m, a, r, fuente = cargar()
st.title("Monitoreo del modelo de priorización de reseñas")
st.caption(f"Fuente: {fuente} · un lote = una semana · el target llega {C.HORIZONTE_LOTES} lote después de la predicción")
if m.empty:
    st.info("Aún no hay métricas. Ejecuta el flujo produccion_lotes en Kestra.")
    st.stop()

# última versión de cada métrica (si un lote se recalculó)
m = (m.sort_values("ts").drop_duplicates(["lote_id", "tipo", "metrica", "feature"], keep="last")
     if "ts" in m else m)
reent = r[r.get("decision", pd.Series(dtype=str)).astype(str).isin(["aprobado", "rechazado", "pendiente"])] if not r.empty else r


def marcas():
    """Reglas verticales: reentrenamientos (verde = aprobado, gris = rechazado/pendiente)."""
    if reent.empty:
        return None
    d = reent.assign(x=reent["lote_disparo"].astype(float) + 0.5,
                     etiqueta=lambda x: "reentreno " + x["decision"].astype(str) + " → v" + x["version_challenger"].astype(str))
    return alt.Chart(d).mark_rule(strokeDash=[6, 3], strokeWidth=2).encode(
        x="x:Q", color=alt.Color("decision:N", scale=alt.Scale(domain=["aprobado", "rechazado", "pendiente"],
                                                                range=["#2e7d32", "#9e9e9e", "#f9a825"]),
                                 legend=alt.Legend(title="Reentrenamiento")),
        tooltip=["etiqueta:N", "motivo:N"])


def grafico(df, titulo, y="valor", color="metrica", umbral=None, formato=".3f"):
    if df.empty:
        st.caption(f"{titulo}: sin datos")
        return
    base = alt.Chart(df).encode(x=alt.X("lote_id:Q", title="Lote", axis=alt.Axis(tickMinStep=1)))
    capas = [base.mark_line(point=True).encode(y=alt.Y(f"{y}:Q", title=None, axis=alt.Axis(format=formato)),
                                               color=alt.Color(f"{color}:N", title=None),
                                               tooltip=["lote_id", color, alt.Tooltip(y, format=formato), "estado"])]
    if "estado" in df:
        al = df[df["estado"] == "alerta"]
        if not al.empty:
            capas.append(alt.Chart(al).mark_point(shape="triangle-up", size=160, color="#c62828", filled=True)
                         .encode(x="lote_id:Q", y=f"{y}:Q", tooltip=["lote_id", color, "estado"]))
    if umbral is not None:
        capas.append(alt.Chart(pd.DataFrame({"u": [umbral]})).mark_rule(color="#c62828", strokeDash=[2, 2]).encode(y="u:Q"))
    mk = marcas()
    if mk is not None:
        capas.append(mk)
    st.markdown(f"**{titulo}**")
    st.altair_chart(alt.layer(*capas).properties(height=240), use_container_width=True)


def sel(tipo, metricas=None, feature=None):
    d = m[m["tipo"] == tipo]
    if metricas:
        d = d[d["metrica"].isin(metricas)]
    if feature is not None:
        d = d[d["feature"] == feature]
    return d.assign(lote_id=d["lote_id"].astype(int), valor=d["valor"].astype(float))


lotes = sorted(m["lote_id"].dropna().astype(int).unique())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lotes monitoreados", len(lotes))
c2.metric("Alertas", 0 if a.empty else len(a))
c3.metric("Reentrenamientos", 0 if reent.empty else len(reent))
neg = sel("negocio", ["costo_modelo", "costo_regla_negocio"])
if not neg.empty:
    piv = neg.pivot_table(index="lote_id", columns="metrica", values="valor")
    c4.metric("Costo por reseña: modelo vs regla", f"{piv['costo_modelo'].mean():.3f}",
              f"{piv['costo_modelo'].mean() - piv['costo_regla_negocio'].mean():+.3f} vs regla", delta_color="inverse")

t1, t2, t3, t4, t5 = st.tabs(["Negocio y modelo", "Deriva de datos", "Sistema", "Subgrupos", "Alertas y reentrenamientos"])
with t1:
    grafico(sel("negocio", ["costo_modelo", "costo_regla_negocio", "costo_no_hacer_nada", "costo_revisar_todas"]),
            "Costo por reseña (FN = 5, FP = 1): modelo vs alternativas")
    col = st.columns(2)
    with col[0]:
        grafico(sel("modelo", ["roc_auc"]), "ROC-AUC del lote (cuando llega el target) y su umbral de alerta")
    with col[1]:
        grafico(sel("modelo", ["pr_auc", "recall", "precision"]), "PR-AUC, recall y precisión")
    col = st.columns(2)
    with col[0]:
        grafico(sel("modelo", ["tasa_priorizar"]), "Tasa de decisión 'priorizar'", formato=".0%")
    with col[1]:
        grafico(sel("modelo", ["probabilidad_p25", "probabilidad_p50", "probabilidad_p75"]), "Distribución de la probabilidad")
with t2:
    psi = sel("deriva", ["psi"])
    grafico(psi, "PSI por feature (línea roja = 0,25)", color="feature", umbral=C.PSI_ALERTA)
    col = st.columns(2)
    with col[0]:
        grafico(sel("datos", ["pct_categorias_no_vistas", "pct_texto_vacio", "pct_fuera_de_rango"]),
                "Calidad de datos", formato=".1%")
    with col[1]:
        grafico(sel("deriva", ["psi_vs_entrenamiento_completo"]), "PSI contra el entrenamiento completo (informativo)",
                color="feature")
    diag = m[m["tipo"] == "diagnostico"][["lote_id", "estado", "detalle", "version_modelo"]]
    st.markdown("**Diagnóstico por lote (tabla 6.2)**")
    st.dataframe(diag.rename(columns={"estado": "diagnóstico", "detalle": "acción"}).sort_values("lote_id"),
                 hide_index=True, use_container_width=True)
with t3:
    col = st.columns(3)
    with col[0]:
        grafico(sel("sistema", ["latencia_p50_ms", "latencia_p95_ms"]), "Latencia (ms)", formato=".0f")
    with col[1]:
        grafico(sel("sistema", ["tasa_4xx", "tasa_5xx"]), "Tasa de errores", formato=".1%")
    with col[2]:
        grafico(sel("sistema", ["volumen_solicitudes"]), "Volumen de solicitudes", formato=".0f")
with t4:
    sub = m[m["tipo"] == "subgrupo"]
    if not sub.empty:
        met = st.selectbox("Métrica", ["recall", "tasa_priorizar", "costo"])
        d = sub[sub["metrica"] == met].assign(lote_id=lambda x: x["lote_id"].astype(int), valor=lambda x: x["valor"].astype(float))
        grafico(d, f"{met} por banco", color="feature")
with t5:
    st.markdown("**Alertas**")
    st.dataframe(a.sort_values(["lote_id", "severidad"]) if not a.empty else a, hide_index=True, use_container_width=True)
    st.markdown("**Reentrenamientos**")
    st.dataframe(reent, hide_index=True, use_container_width=True)
