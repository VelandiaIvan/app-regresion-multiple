"""Módulo 5 · Revisión de la variable dependiente (Y) después de preparar los datos."""
from __future__ import annotations

import streamlit as st
from scipy import stats

from core import context, profiling
from ui import charts
from ui import components as C
from ui import state


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 5", "Variable a estimar (Y)",
           "Confirme la variable que el modelo va a estimar y revise su comportamiento. La aplicación ya propuso una en el módulo 2; "
           "aquí puede cambiarla.")
    C.need(state.has_data(), "Primero cargue los datos.", "m02_datos")
    df = state.preview().df
    cls = profiling.classify_columns(df)
    rank = context.rank_target_candidates(df, cls, ss.ctx)
    if rank.empty:
        st.error("No hay variables numéricas continuas utilizables como Y.")
        st.stop()

    top = rank.iloc[0]
    opts = rank["variable"].tolist()
    current = ss.y if ss.y in opts else top["variable"]
    C.callout(f"Sugerida: <b>{top['variable']}</b> (puntaje {top['puntaje']:.0f}/100). Motivos: {top['razones']}."
              + (f" <br><b>Ojo:</b> {top['advertencias']}." if top["advertencias"] else ""), "ok", "Sugerencia de la aplicación")
    y = st.selectbox("Variable dependiente (Y) confirmada", opts, index=opts.index(current),
                     format_func=lambda v: f"{v}{'  ★ sugerida' if v == top['variable'] else ''}")
    if y != ss.y:
        ss.y = y
        ss.cmp = None
        ss.chosen = None
        ss.x_cands = [c for c in ss.x_cands if c != y]
    ss.y_transform = None                      # Y se modela en su propia escala: los resultados se leen directamente
    with st.expander("Ver todas las candidatas y su puntaje"):
        C.table(rank, 3)

    s = df[y]
    st.markdown(f"#### Revisión de «{y}»")
    c1, c2 = st.columns([2, 1])
    with c1:
        C.chart(charts.histogram(s, y), "m5_hist")
    with c2:
        miss = int(s.isna().sum())
        sk = float(stats.skew(s.dropna())) if s.notna().sum() > 2 else 0.0
        C.kpis([("Valores válidos", f"{s.notna().sum():,}", f"{miss} faltantes → se descartan")])
        C.kpis([("Media / mediana", f"{s.mean():,.4g} / {s.median():,.4g}")])
        C.kpis([("Mínimo – máximo", f"{s.min():,.4g} – {s.max():,.4g}", f"asimetría {sk:+.2f}")])
    if miss:
        C.callout(f"Hay {miss} filas sin valor de Y. <b>No se imputan</b>: rellenar la variable que se quiere estimar sería inventar la respuesta; "
                  "se descartan al modelar.", "warn")
    n_x = int(cls["usable_como_X"].sum()) - (1 if y in set(cls[cls["usable_como_X"]]["variable"]) else 0)
    if n_x < 2:
        C.callout("Además de Y hay menos de 2 variables utilizables como X. La regresión múltiple necesita al menos 2: "
                  "en el módulo 3 puede convertir variables categóricas en indicadoras 0/1.", "warn")
    C.next_step("Continuar: elegir variables explicativas (X) ▶", "m06_x")
