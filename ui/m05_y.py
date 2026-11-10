"""Módulo 5 · Identificación de la variable dependiente (Y)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from core import context, profiling
from ui import charts
from ui import components as C
from ui import state


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 5", "Variable dependiente (Y)",
           "La variable a estimar la define el objetivo del negocio, no los datos. La aplicación sugiere candidatas según su descripción y la calidad de los datos; "
           "usted confirma o cambia.")
    C.need(state.has_data(), "Primero cargue los datos.", "m02_datos")
    df = state.preview().df
    cls = profiling.classify_columns(df)
    rank = context.rank_target_candidates(df, cls, ss.ctx)
    if rank.empty:
        st.error("No hay variables numéricas continuas utilizables como Y.")
        st.stop()

    st.markdown("#### Candidatas recomendadas")
    top = rank.iloc[0]
    C.callout(f"Se recomienda <b>{top['variable']}</b> (puntaje {top['puntaje']:.0f}/100). Motivos: {top['razones']}."
              + (f" <br><b>Ojo:</b> {top['advertencias']}." if top["advertencias"] else ""), "ok", "Sugerencia")
    C.table(rank, 3)
    C.how_to_read("El puntaje combina: coincidencia con el resultado que usted describió (45%), si el nombre corresponde a un concepto típico de resultado "
                  "(ventas, costos, tiempos, consumo…), calidad de los datos (completitud y variabilidad) y qué tan conectada está con las demás variables. "
                  "Es una **guía**, no una regla: si su objetivo es otro, elija otra variable.", "¿Cómo se calcula el puntaje?")

    opts = rank["variable"].tolist()
    current = ss.y if ss.y in opts else top["variable"]
    y = st.selectbox("Variable dependiente (Y) confirmada", opts, index=opts.index(current))
    if y != ss.y:
        ss.y = y
        ss.cmp = None
        ss.chosen = None
        ss.x_cands = [c for c in ss.x_cands if c != y]
    s = df[y]
    st.markdown(f"#### Revisión de «{y}»")
    c1, c2 = st.columns([2, 1])
    with c1:
        C.chart(charts.histogram(s, y), "m5_hist")
    with c2:
        miss = int(s.isna().sum())
        C.kpis([("Valores válidos", f"{s.notna().sum():,}", f"{miss} faltantes → se descartan"),
                ("Media / mediana", f"{s.mean():,.4g} / {s.median():,.4g}"),
                ("Asimetría", f"{stats.skew(s.dropna()):+.2f}")])
    if miss:
        C.callout(f"Hay {miss} filas sin valor de Y. <b>No se imputan</b>: rellenar la variable que se quiere estimar sería inventar la respuesta; se descartan al modelar.", "warn")
    can_ln = bool((s.dropna() > 0).all())
    sk = stats.skew(s.dropna())
    sugg = "ln" if (sk > 1 and can_ln) else None
    opt_lbl = {"ninguna": None, "ln": "ln", "sqrt": "sqrt"}
    labels = ["ninguna", "ln"] if can_ln else ["ninguna"]
    if (s.dropna() >= 0).all():
        labels.append("sqrt")
    cur = {None: "ninguna", "ln": "ln", "sqrt": "sqrt"}.get(ss.y_transform, "ninguna")
    t = st.radio("Transformación de Y para el modelo", labels, index=labels.index(cur) if cur in labels else 0, horizontal=True,
                 format_func={"ninguna": "Ninguna (Y en su escala)", "ln": "Logaritmo natural", "sqrt": "Raíz cuadrada"}.get,
                 help="Los resultados (estimaciones, errores, simulación) siempre se expresan en la escala original de Y.")
    ss.y_transform = opt_lbl[t]
    if sugg and ss.y_transform is None:
        C.callout(f"Y tiene asimetría fuerte ({sk:.2f}) y es positiva: probar <b>ln(Y)</b> suele estabilizar la varianza y mejorar el ajuste (relaciones porcentuales). "
                  "Puede comparar ambas versiones en los módulos 7 y 8.", "info")
    if ss.y_transform:
        st.caption("Con ln(Y) los coeficientes se leen como cambios porcentuales; los intervalos se transforman a la escala original de Y.")
    C.next_step("Continuar: elegir variables explicativas (X) ▶", "m06_x")
