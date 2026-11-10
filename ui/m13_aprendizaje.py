"""Módulo 13 · Centro de aprendizaje: regresión lineal (de la simple a la múltiple)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.education import CONCEPTS
from ui import charts
from ui import components as C


def _demo() -> None:
    st.markdown("#### 🎛 Laboratorio: ajuste usted la recta")
    st.caption("Los datos (simulados) siguen una tendencia. Mueva el intercepto y la pendiente: la suma de errores al cuadrado (SSE) baja cuando la recta se acerca "
               "a la de mínimos cuadrados, que es la que minimiza el SSE.")
    rng = np.random.default_rng(3)
    n = 40
    x = rng.uniform(0, 10, n)
    y = 4 + 2.2 * x + rng.normal(0, 3, n)
    sl = np.cov(x, y, bias=True)[0, 1] / x.var()
    ic = y.mean() - sl * x.mean()
    c1, c2, c3 = st.columns([1, 1, 1])
    b0 = c1.slider("Intercepto β0", -10.0, 20.0, 0.0, 0.1)
    b1 = c2.slider("Pendiente β1", -2.0, 6.0, 1.0, 0.05)
    show_res = c3.toggle("Mostrar residuos", True)
    yh = b0 + b1 * x
    sse, sse_min = float(((y - yh) ** 2).sum()), float(((y - (ic + sl * x)) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    fig = go.Figure()
    if show_res:
        xs, ys = [], []
        for xi, a, b in zip(x, y, yh):
            xs += [xi, xi, None]
            ys += [a, b, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="residuos (errores)", line=dict(color=charts.ORANGE, width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="datos", marker=dict(size=8, color=charts.BLUE, line=dict(color=charts.SURFACE, width=1))))
    gx = np.array([0, 10])
    fig.add_trace(go.Scatter(x=gx, y=b0 + b1 * gx, mode="lines", name="su recta", line=dict(color=charts.INK, width=2.5)))
    fig.add_trace(go.Scatter(x=gx, y=ic + sl * gx, mode="lines", name="recta óptima (MCO)", line=dict(color=charts.AQUA, width=2.5)))
    C.chart(charts._base(fig, "Su recta frente a la recta de mínimos cuadrados", 400, "X", "Y"), "m13_demo")
    C.kpis([("SSE de su recta", f"{sse:,.1f}"), ("SSE mínimo posible", f"{sse_min:,.1f}"), ("R² de su recta", f"{1 - sse / sst:.2f}", "puede ser negativo si es peor que el promedio"),
            ("Recta óptima", f"{ic:.2f} + {sl:.2f}·X")])
    if sse <= sse_min * 1.05:
        st.success("¡Muy cerca del mínimo! Esa es la idea de los mínimos cuadrados.")


def render() -> None:
    C.hero("Módulo 13", "Centro de aprendizaje · regresión lineal",
           "25 conceptos con definición sencilla, explicación técnica, ejemplo empresarial, interpretación y utilidad práctica. Más un laboratorio interactivo.")
    tab_c, tab_l = st.tabs(["Conceptos", "Laboratorio interactivo"])
    with tab_c:
        q = st.text_input("Buscar un concepto", placeholder="p. ej. residuos, p-valor, extrapolación…")
        items = [c for c in CONCEPTS if not q or q.lower() in (c["titulo"] + c["simple"] + c["tecnica"]).lower()]
        if not items:
            st.info("Sin resultados.")
            return
        titles = [f"{c['n']}. {c['titulo']}" for c in items]
        pick = st.radio("Concepto", titles, key="m13_pick", label_visibility="collapsed", horizontal=False) if len(items) <= 6 else st.selectbox("Concepto", titles, key="m13_pick")
        c = items[titles.index(pick)]
        st.markdown(f"### {c['n']}. {c['titulo']}")
        if c.get("formula"):
            st.latex(c["formula"])
        C.callout(c["simple"], "ok", "Definición sencilla")
        a, b = st.columns(2)
        with a:
            C.callout(c["tecnica"], "info", "Explicación técnica")
            C.callout(c["interpretacion"], "edu", "Cómo se interpreta")
        with b:
            C.callout(c["ejemplo"], "warn", "Ejemplo empresarial")
            C.callout(c["utilidad"], "ok", "Utilidad práctica")
        nav = st.columns(2)
        i = c["n"] - 1
        if i > 0:
            nav[0].caption(f"← Anterior: {CONCEPTS[i - 1]['titulo']}")
        if i < len(CONCEPTS) - 1:
            nav[1].caption(f"Siguiente: {CONCEPTS[i + 1]['titulo']} →")
        with st.expander("Ver el índice completo"):
            for k in CONCEPTS:
                st.markdown(f"{k['n']}. {k['titulo']}")
    with tab_l:
        _demo()
