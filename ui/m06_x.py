"""Módulo 6 · Identificación y selección de variables independientes (X)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import context, eda, profiling
from ui import charts
from ui import components as C
from ui import state

SIGN_LABEL = {"sin expectativa": None, "positivo (+)": 1, "negativo (−)": -1}
SIGN_BACK = {None: "sin expectativa", 1: "positivo (+)", -1: "negativo (−)"}


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 6", "Variables independientes (X)",
           "Elija las variables candidatas: la aplicación probará combinaciones de DOS O MÁS a la vez y se quedará con la mejor. La preselección usa SOLO el "
           "conjunto de entrenamiento (así la elección no «espía» los datos de prueba).")
    C.need(state.has_data(), "Primero cargue los datos.", "m02_datos")
    C.need(bool(ss.y), "Defina primero la variable Y.", "m05_y")

    st.markdown("#### Partición entrenamiento / prueba")
    c1, c2, c3, c4 = st.columns([1.3, 1.2, 1, 1.5])
    method = c1.radio("Tipo de partición", ["aleatoria", "temporal"], index=["aleatoria", "temporal"].index(ss.split["method"]), horizontal=True,
                      help="Temporal: las últimas observaciones (según la fecha o el orden de las filas) quedan como prueba. Es lo adecuado si el modelo se usará para predecir el futuro.")
    test_size = c2.slider("% para prueba", 10, 40, int(ss.split["test_size"] * 100), 5) / 100
    seed = int(c3.number_input("Semilla", 0, 9999, ss.split["seed"]))
    ss.split = {"method": method, "test_size": test_size, "seed": seed}
    c4.caption(f"Columna de fecha: **{ss.date_col or 'ninguna'}**. "
               + ("Con partición temporal y sin fecha se usa el orden de las filas." if method == "temporal" and not ss.date_col else ""))
    md, err = state.modeling()
    if md is None:
        st.error(f"No se pudo preparar los datos: {err}")
        st.stop()
    C.kpis([("Filas con Y válida", f"{md.n_with_y:,}", f"de {md.n_raw:,} tras limpieza estructural"), ("Entrenamiento", f"{len(md.train_idx):,}", "ajusta los modelos"),
            ("Prueba", f"{len(md.test_idx):,}", "solo evalúa" if md.test_idx else "no se reserva (muestra pequeña)")])
    for w in md.warnings:
        st.warning(w)

    train = md.df.loc[md.train_idx]
    cls = profiling.classify_columns(md.df)
    drop_cols = {md.y_source, md.y_col} | {c for c, t in md.transforms.items() if t["source"] == md.y_source}
    cands = [c for c in cls[cls["usable_como_X"]]["variable"] if c not in drop_cols and pd.api.types.is_numeric_dtype(md.df[c])
             and not str(c).endswith("__atípico")]
    if not cands:
        st.error("No hay variables numéricas utilizables como X. Revise el módulo 3 (por ejemplo, codifique categóricas).")
        st.stop()
    if len(cands) < 2:
        st.error(f"Solo hay una variable numérica utilizable como X («{cands[0]}»). La regresión MÚLTIPLE necesita al menos 2: agregue columnas "
                 "al archivo o convierta variables categóricas en indicadoras 0/1 en el módulo 3.")
        st.stop()
    corr = eda.correlation_table(train, md.y_col, cands)
    expected = {k: v["sign"] for k, v in ss.x_meta.items() if v.get("sign")}
    rank = context.rank_predictor_candidates(md.df.loc[md.train_idx], md.y_source, cls, ss.ctx, corr, expected)
    if rank.empty:
        st.error("Ninguna candidata tiene suficientes pares completos con Y (mínimo 4).")
        st.stop()

    st.markdown("#### Candidatas (calculado solo con entrenamiento)")
    prev_sel = set(ss.x_cands)
    default_sel = prev_sel or set(rank[rank["riesgo_fuga"] != "alta"].head(8)["variable"])   # el modelo decide cuáles aportan
    if len(default_sel) < 2:
        default_sel = set(rank.head(3)["variable"])
    ed = rank[["variable", "puntaje_preliminar", "pearson_r", "spearman_rho", "p_valor", "riesgo_fuga", "a_favor", "alertas"]].copy()
    ed.insert(0, "usar", ed["variable"].isin(default_sel))
    ed["signo esperado"] = [SIGN_BACK.get(ss.x_meta.get(v, {}).get("sign")) for v in ed["variable"]]
    ed["se conoce antes de decidir"] = [ss.x_meta.get(v, {}).get("known", True) for v in ed["variable"]]
    edited = st.data_editor(
        ed, hide_index=True, key=f"m6_editor_{abs(hash(str(ss._md_sig))) % 10**8}",
        disabled=["variable", "puntaje_preliminar", "pearson_r", "spearman_rho", "p_valor", "riesgo_fuga", "a_favor", "alertas"],
        column_config={
            "usar": st.column_config.CheckboxColumn("Usar", help="Candidata a entrar en el modelo múltiple"),
            "puntaje_preliminar": st.column_config.ProgressColumn("Puntaje prelim.", min_value=0, max_value=100, format="%.0f"),
            "pearson_r": st.column_config.NumberColumn("r", format="%.3f"), "spearman_rho": st.column_config.NumberColumn("ρ", format="%.3f"),
            "p_valor": st.column_config.NumberColumn("p", format="%.3g"),
            "signo esperado": st.column_config.SelectboxColumn("Signo esperado", options=list(SIGN_LABEL), help="¿Cómo debería afectar X a Y según su conocimiento del negocio?"),
            "se conoce antes de decidir": st.column_config.CheckboxColumn("Se conoce antes", help="Desmarque si X solo se conoce después de necesitar estimar Y (fuga temporal)."),
        })
    ss.x_cands = edited.loc[edited["usar"], "variable"].tolist()
    ss.x_meta = {r["variable"]: {"sign": SIGN_LABEL[r["signo esperado"]], "known": bool(r["se conoce antes de decidir"])} for _, r in edited.iterrows()}
    flagged = rank[(rank["riesgo_fuga"] != "ninguna") & rank["variable"].isin(ss.x_cands)]
    for _, r in flagged.iterrows():
        sev = "bad" if r["riesgo_fuga"] == "alta" else "warn"
        C.callout(f"<b>{r['variable']}</b>: riesgo de fuga <b>{r['riesgo_fuga']}</b>. {r['alertas']}", sev)
        ok = st.checkbox(f"Confirmo que «{r['variable']}» se conoce ANTES de estimar Y y no se calcula a partir de Y", value=r["variable"] in ss.leak_cleared,
                         key=f"m6_leak_{r['variable']}")
        (ss.leak_cleared.add if ok else ss.leak_cleared.discard)(r["variable"])
    C.how_to_read("El **puntaje preliminar** pondera: fuerza de la relación (35), significancia (15), calidad y cantidad de datos (15), sentido de negocio (15: la mencionó "
                  "como factor o es una palanca típica), coherencia con el signo esperado (10) y forma lineal (10). Resta si hay riesgo de fuga. **No es solo la correlación más alta**: "
                  "una variable con correlación 0,99 casi siempre es una consecuencia de Y, no una explicación. Marque todas las que tengan sentido: el siguiente paso descarta las que no aportan al combinarlas.")
    top3 = [c for c in ss.x_cands][:3]
    if top3:
        st.markdown("#### Vista rápida de las primeras seleccionadas (entrenamiento)")
        cols = st.columns(len(top3))
        for c, v in zip(cols, top3):
            with c:
                C.chart(charts.scatter_eda(train, v, md.y_col, loess=True), f"m6_sc_{v}")
    if len(ss.x_cands) < 2:
        st.warning("Seleccione al menos DOS variables: un modelo de regresión múltiple combina 2 o más X.")
    else:
        n_comb = 2 ** len(ss.x_cands) - len(ss.x_cands) - 1
        st.caption(f"Con {len(ss.x_cands)} candidatas hay {n_comb:,} combinaciones posibles de 2 o más variables; la aplicación las evalúa y compara las mejores.")
        C.next_step(f"Construir modelos múltiples con {len(ss.x_cands)} candidatas ▶", "m07_modelos")
