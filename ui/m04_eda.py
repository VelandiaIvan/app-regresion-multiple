"""Módulo 4 · Análisis exploratorio de datos (AED)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import eda, profiling
from ui import charts
from ui import components as C
from ui import state


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 4", "Análisis exploratorio de datos",
           "Entienda la forma de cada variable y cómo se relacionan antes de modelar. Aquí se usan todos los datos de forma DESCRIPTIVA; "
           "la selección formal de variables explicativas (módulo 6) usará solo el conjunto de entrenamiento.")
    C.need(state.has_data(), "Primero cargue los datos.", "m02_datos")
    prev = state.preview()
    df = prev.df
    cls = profiling.classify_columns(df)
    num_cols = cls[(cls["grupo"].isin(["numérica"])) | (cls["detalle"].str.startswith("binaria"))]["variable"].tolist()
    num_cols = [c for c in num_cols if cls.set_index("variable").loc[c, "grupo"] != "identificador"]
    cat_cols = cls[(cls["grupo"] == "categórica") & (cls["n_únicos"] >= 2)]["variable"].tolist()
    if len(num_cols) < 2:
        st.warning("Se necesitan al menos dos variables numéricas para el análisis de relaciones.")
    target_default = num_cols.index(ss.y) if ss.y in num_cols else 0
    target = st.selectbox("Variable de referencia (la que se quiere estimar)", num_cols, index=target_default, key="m4_target",
                          help="Solo para esta exploración. La variable Y definitiva se confirma en el módulo 5.")

    tabs = st.tabs(["Descriptivas", "Distribuciones", "Categóricas", "Dispersión", "Correlaciones", "Tendencias y patrones", "Lectura de negocio"])

    with tabs[0]:
        d = profiling.descriptives(df)
        C.table(d, 4)
        C.how_to_read("**CV (coeficiente de variación)** = desviación / media: compara la dispersión entre variables de distinta escala. "
                      "**Asimetría** > 0 indica cola larga a la derecha (pocos valores muy altos); |asimetría| > 1 es fuerte. "
                      "**Curtosis** alta indica colas pesadas (más atípicos).")

    with tabs[1]:
        v = st.selectbox("Variable numérica", num_cols, key="m4_dist_var")
        c1, c2 = st.columns(2)
        with c1:
            C.chart(charts.histogram(df[v], v), "m4_hist")
        with c2:
            C.chart(charts.boxplot(df[v], v), "m4_box")
        dt = eda.distribution_table(df[num_cols])
        if not dt.empty:
            C.table(dt, 3)
        C.how_to_read("Un histograma asimétrico sugiere probar una transformación (ln, √). Si media y mediana difieren mucho, hay colas o atípicos. "
                      "La regresión **no exige** que X o Y sean normales: exige (aprox.) que los **residuos** lo sean.")

    with tabs[2]:
        if not cat_cols:
            st.info("No hay variables categóricas con al menos dos categorías.")
        else:
            cv = st.selectbox("Variable categórica", cat_cols, key="m4_cat_var")
            fr = eda.categorical_frequencies(df, cv)
            c1, c2 = st.columns([1, 1])
            with c1:
                C.table(fr, 3)
            with c2:
                C.chart(charts.bars(fr, "categoría", "frecuencia", f"Frecuencias de {cv}", fmt=",d"), "m4_freq")
            if target != cv and pd.api.types.is_numeric_dtype(df[target]):
                ce = eda.category_effect(df, cv, target)
                st.markdown(f"**¿Cambia «{target}» según «{cv}»?**")
                C.table(ce["table"], 4)
                if not pd.isna(ce["anova_p"]):
                    st.caption(f"ANOVA p = {ce['anova_p']:.3g} · Kruskal-Wallis p = {ce['kruskal_p']:.3g} · η² = {ce['eta2']:.2f} "
                               "(proporción de la variación de Y asociada a la categoría).")
                    if ce["eta2"] > 0.1 and ce["anova_p"] < 0.05:
                        C.callout(f"«{cv}» se asocia con diferencias relevantes en «{target}». Una variable indicadora derivada de ella podría ser una X candidata "
                                  "(módulo 3, codificación), o un motivo para segmentar el modelo.", "info")

    with tabs[3]:
        c1, c2, c3 = st.columns(3)
        xs = [c for c in num_cols if c != target]
        if not xs:
            st.info("No hay otra variable numérica.")
        else:
            xv = c1.selectbox("Variable X", xs, key="m4_sc_x")
            loess = c2.toggle("Mostrar tendencia suavizada (LOWESS)", True)
            color = c3.selectbox("Colorear por (≤ 3 categorías)", ["(ninguna)"] + [c for c in cat_cols if df[c].nunique() <= 3])
            C.chart(charts.scatter_eda(df, xv, target, loess, None if color == "(ninguna)" else color), "m4_scatter")
            ps = eda.pair_stats(df[xv], df[target])
            lin = eda.linearity_screen(df[xv], df[target])
            C.kpis([("Pearson r", f"{ps['pearson_r']:+.3f}", f"p = {ps['pearson_p']:.3g}"), ("Spearman ρ", f"{ps['spearman_rho']:+.3f}", f"p = {ps['spearman_p']:.3g}"),
                    ("n pares", f"{ps['n']:,}"), ("Mejor forma", lin["mejor"], f"+{lin['mejora'] * 100:.1f} pts de R² ajust. vs recta")])
            C.callout(lin["mensaje"], "warn" if lin["no_lineal"] else "ok", "Evaluación preliminar de linealidad")
            if not lin["formas"].empty:
                with st.expander("Comparación de formas funcionales (mismo Y)"):
                    C.table(lin["formas"], 4)
            C.how_to_read("La **recta (negra)** es la relación de esa variable sola con Y (sin considerar las demás). La **curva naranja (LOWESS)** sigue los datos sin imponer forma: "
                          "si se separa claramente de la recta, la relación es curva. Si los puntos forman un abanico (más dispersión a un lado), habrá heterocedasticidad.")

    with tabs[4]:
        c1, c2 = st.columns([1, 3])
        meth = c1.radio("Método", ["pearson", "spearman"], format_func=lambda m: {"pearson": "Pearson (lineal)", "spearman": "Spearman (monótona)"}[m])
        pick = c2.multiselect("Variables en la matriz", num_cols, default=num_cols[:12])
        if len(pick) >= 2:
            C.chart(charts.corr_heatmap(eda.correlation_matrix(df, meth, pick), f"Matriz de correlación · {meth}"), "m4_corr")
        ct = eda.correlation_table(df, target, [c for c in num_cols if c != target])
        st.markdown(f"**Correlación de cada variable con «{target}»**")
        if ct.empty:
            st.info("No hay suficientes pares completos.")
        else:
            C.table(ct, 3)
            C.how_to_read("**Pearson** mide relación lineal; **Spearman** mide si al subir X sube (o baja) Y de forma consistente, sin exigir línea recta. "
                          "Si |ρ| supera claramente a |r|, la relación es monótona pero curva. **Una correlación casi perfecta (|r| > 0,98) es una señal de alarma**: "
                          "suele ser una variable derivada de Y (fuga de información). Correlación alta ≠ variable útil para decidir ≠ causalidad.")

    with tabs[5]:
        if not ss.date_col or ss.date_col not in df.columns:
            st.info("No hay columna de fecha seleccionada (módulo 2). Sin ella no se analizan tendencias ni estacionalidad.")
        else:
            tp = eda.time_patterns(df, ss.date_col, target)
            if not tp["ok"]:
                st.info("Muy pocas observaciones con fecha para analizar tendencias.")
            else:
                C.chart(charts.time_series(tp["serie"], ss.date_col, target), "m4_ts")
                C.kpis([("Tendencia por año", f"{tp['pendiente_por_año']:+,.4g}", f"p = {tp['p_tendencia']:.3g}"),
                        ("R² de la tendencia", f"{tp['r2_tendencia']:.2f}"), ("Autocorrelación (rezago 1)", f"{tp['autocorr_lag1']:.2f}")])
                if tp["p_tendencia"] < 0.05:
                    C.callout(f"«{target}» muestra una <b>tendencia {'creciente' if tp['pendiente_por_año'] > 0 else 'decreciente'}</b> significativa. "
                              "Si X también tiene tendencia, la correlación puede ser espuria (ambas crecen con el tiempo sin que una explique a la otra).", "warn")
                if "estacionalidad_mensual" in tp:
                    st.caption(f"Fuerza de la estacionalidad mensual: {tp['fuerza_estacional']:.0%} de la variación.")
                if tp["autocorr_lag1"] > 0.5:
                    st.warning("Alta autocorrelación: los errores del modelo probablemente no serán independientes (se verificará en el módulo 8).")

    with tabs[6]:
        ct = eda.correlation_table(df, target, [c for c in num_cols if c != target])
        st.markdown(f"##### Lectura para «{target}»")
        if ct.empty:
            st.info("Sin relaciones calculables.")
        else:
            for _, r in ct.head(4).iterrows():
                st.markdown("- " + eda.describe_relationship(r, r["variable"], target))
        ctx = ss.ctx
        if ctx.restricciones.strip():
            C.callout(f"Restricciones declaradas: <i>{ctx.restricciones}</i>. Al interpretar relaciones recuerde que ningún análisis estadístico las impone por sí solo.", "info")
        C.callout("Una correlación alta no prueba causalidad ni garantiza que la variable sea útil para decidir: debe ser <b>conocida con antelación</b>, "
                  "<b>controlable</b> (si se quiere actuar sobre ella) y <b>coherente</b> con la lógica del negocio.", "edu")
    C.next_step("Continuar: definir la variable a estimar (Y) ▶", "m05_y")
