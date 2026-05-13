import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

st.set_page_config(
    page_title="🎮 Video Games EDA",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Cargar datos ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("video_games_clean.csv")
    df["year"] = df["year"].astype("Int64")
    df_scored = df[df["critic_score"].notna()].copy()
    return df, df_scored

df, df_scored = load_data()
# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎮 Video Games Sales")
st.sidebar.markdown("**Dataset: 1980 – 2024**")

year_min = int(df["year"].min())
year_max = int(df["year"].max())

col_s1, col_s2 = st.sidebar.columns(2)
year_start = col_s1.number_input("Año inicial", min_value=year_min, max_value=year_max, value=year_min, step=1)
year_end   = col_s2.number_input("Año final",   min_value=year_min, max_value=year_max, value=year_max, step=1)


if year_start > year_end:
    st.sidebar.error("⚠️ El año inicial no puede ser mayor al final.")
    year_range = (year_min, year_max)
else:
    year_range = (year_start, year_end)

all_genres = sorted(df["genre"].dropna().unique().tolist())
selected_genres = st.sidebar.multiselect(
    "Géneros",
    all_genres,
    default=all_genres,
    key="genres_select"
)

all_consoles = sorted(df["console"].dropna().unique().tolist())
selected_consoles = st.sidebar.multiselect(
    "Consolas",
    all_consoles,
    default=all_consoles,
    key="consoles_select"
)

if st.sidebar.button("↺ Restablecer Filtros", use_container_width=True):
    st.session_state["genres_select"]   = sorted(df["genre"].dropna().unique().tolist())
    st.session_state["consoles_select"] = sorted(df["console"].dropna().unique().tolist())

st.sidebar.markdown(
    """
    <style>
    section[data-testid="stSidebar"] > div { overflow-y: hidden !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ── Filtrar ───────────────────────────────────────────────────────────────────
mask = (
    df["year"].between(*year_range) &
    df["genre"].isin(selected_genres) &
    df["console"].isin(selected_consoles)
)
dff = df[mask].copy()
dff_scored = df_scored[
    df_scored["year"].between(*year_range) &
    df_scored["genre"].isin(selected_genres) &
    df_scored["console"].isin(selected_consoles)
].copy()

# ── Métricas ──────────────────────────────────────────────────────────────────
st.title("🎮 Video Games Sales — Análisis Exploratorio")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("🕹️ Juegos", f"{len(dff):,}")
col2.metric("📦 Consolas", f"{dff['console'].nunique()}")
col3.metric("🏢 Publishers", f"{dff['publisher'].nunique()}")
col4.metric("💰 Ventas totales", f"{dff['total_sales'].sum():,.0f} M")

st.markdown("---")

# ── Sección 1: Ventas por año ─────────────────────────────────────────────────
st.subheader("📈 Ventas globales totales por año")
ventas_anio = dff.groupby("year")["total_sales"].sum().reset_index()
fig1 = px.line(
    ventas_anio, x="year", y="total_sales",
    labels={"year": "Año", "total_sales": "Ventas (M unidades)"},
    color_discrete_sequence=["#6c63ff"],
    markers=True
)
fig1.update_layout(hovermode="x unified", height=380)
st.plotly_chart(fig1, use_container_width=True)

# ── Sección 2: Géneros y Consolas ─────────────────────────────────────────────
st.subheader("🎯 Ventas por Género y Consola")
col_a, col_b = st.columns(2)

with col_a:
    generos = dff.groupby("genre")["total_sales"].sum().sort_values(ascending=True).reset_index()
    fig2 = px.bar(
        generos, x="total_sales", y="genre", orientation="h",
        labels={"total_sales": "Ventas (M)", "genre": "Género"},
        color="total_sales", color_continuous_scale="Viridis"
    )
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    consolas = dff.groupby("console")["total_sales"].sum().sort_values(ascending=False).head(20).reset_index()
    fig3 = px.bar(
        consolas, x="console", y="total_sales",
        labels={"console": "Consola", "total_sales": "Ventas (M)"},
        color="total_sales", color_continuous_scale="Plasma"
    )
    fig3.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

# ── Sección 3: Publishers y Regiones ─────────────────────────────────────────
st.subheader("🏢 Publishers y Distribución Regional")
col_c, col_d = st.columns(2)

with col_c:
    publishers = (
        dff[dff["publisher"] != "Unknown"]
        .groupby("publisher")["total_sales"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .reset_index()
    )
    fig4 = px.pie(
        publishers, names="publisher", values="total_sales",
        title="Top 12 Publishers", hole=0.4
    )
    fig4.update_layout(height=420)
    st.plotly_chart(fig4, use_container_width=True)

with col_d:
    regiones = {
        "Norteamérica": dff["na_sales"].sum(),
        "Europa/PAL": dff["pal_sales"].sum(),
        "Japón": dff["jp_sales"].sum(),
        "Otras regiones": dff["other_sales"].sum(),
    }
    fig5 = px.pie(
        names=list(regiones.keys()),
        values=list(regiones.values()),
        title="Distribución regional", hole=0.35,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig5.update_layout(height=420)
    st.plotly_chart(fig5, use_container_width=True)

# ── Sección 4: Evolución de géneros ──────────────────────────────────────────
st.subheader("📊 Evolución de ventas por género (Top 6)")
top6_genres = dff.groupby("genre")["total_sales"].sum().nlargest(6).index.tolist()
evolucion = (
    dff[dff["genre"].isin(top6_genres)]
    .groupby(["year", "genre"])["total_sales"]
    .sum()
    .reset_index()
)
fig6 = px.line(
    evolucion, x="year", y="total_sales", color="genre",
    labels={"year": "Año", "total_sales": "Ventas (M)", "genre": "Género"}
)
fig6.update_layout(hovermode="x unified", height=400)
st.plotly_chart(fig6, use_container_width=True)

# ── Sección 5: Crítica vs Ventas ──────────────────────────────────────────────
st.subheader("⭐ Puntuación de crítica vs Ventas totales")
if len(dff_scored) > 0:
    fig7 = px.scatter(
        dff_scored, x="critic_score", y="total_sales",
        color="genre", hover_data=["title", "console", "publisher"],
        labels={"critic_score": "Puntuación crítica", "total_sales": "Ventas (M)", "genre": "Género"},
        opacity=0.6, trendline="ols"
    )
    fig7.update_layout(height=450)
    st.plotly_chart(fig7, use_container_width=True)
    corr = dff_scored[["critic_score", "total_sales"]].corr().iloc[0, 1]
    st.info(f"📐 Correlación de Pearson (crítica vs ventas): **{corr:.3f}**")
else:
    st.warning("No hay datos con critic_score para el filtro seleccionado.")

# ── Sección 6: Top 20 juegos ──────────────────────────────────────────────────
st.subheader("🏆 Top 20 juegos más vendidos")
top20 = dff.nlargest(20, "total_sales")[
    ["title", "console", "genre", "publisher", "year", "total_sales"]
].sort_values("total_sales", ascending=False)

fig8 = px.bar(
    top20, x="total_sales", y="title", orientation="h",
    color="genre", hover_data=["console", "publisher", "year"],
    labels={"total_sales": "Ventas (M)", "title": ""}
)
fig8.update_layout(height=600, showlegend=True)
fig8.update_yaxes(categoryorder="total ascending")
st.plotly_chart(fig8, use_container_width=True)

# ── Sección 7: Heatmap ────────────────────────────────────────────────────────
st.subheader("🔥 Heatmap: Géneros por Consola")

top_consolas_idx = dff.groupby("console")["total_sales"].sum().nlargest(12).index

heatmap_data = (
    dff[dff["console"].isin(top_consolas_idx)]
    .groupby(["console", "genre"])["total_sales"]
    .sum()
    .unstack(fill_value=0)
)

fig9 = px.imshow(
    heatmap_data,
    color_continuous_scale="YlOrRd",
    labels={"x": "Género", "y": "Consola", "color": "Ventas (M)"},
    aspect="auto"
)
fig9.update_layout(height=450)
st.plotly_chart(fig9, use_container_width=True)

# ── Resumen final ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Resumen del dataset (filtrado)")
summary_cols = st.columns(3)
with summary_cols[0]:
    best_game = dff.loc[dff["total_sales"].idxmax(), "title"] if len(dff) else "—"
    best_sales = dff["total_sales"].max() if len(dff) else 0
    st.metric("🏆 Juego más vendido", best_game, f"{best_sales:.1f} M")
with summary_cols[1]:
    top_genre = dff.groupby("genre")["total_sales"].sum().idxmax() if len(dff) else "—"
    st.metric("🎯 Género líder", top_genre)
with summary_cols[2]:
    top_console = dff.groupby("console")["total_sales"].sum().idxmax() if len(dff) else "—"
    st.metric("🕹️ Consola líder", top_console)

st.caption("Dataset: Video Games Sales 1980–2024 | EDA implementado con Streamlit + Plotly")
