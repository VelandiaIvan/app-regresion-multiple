"""Asistente de modelamiento y estimación · Regresión lineal MÚLTIPLE (modelos con 2 o más variables explicativas).

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui import components as C  # noqa: E402
from ui import state  # noqa: E402

st.set_page_config(page_title="Asistente de Regresión Lineal Múltiple", page_icon="🧮", layout="wide",
                   initial_sidebar_state="expanded")
C.inject_css()
state.init()

# El flujo empieza por los DATOS: la aplicación los lee, explica de qué tratan y propone la variable a estimar.
MODULES = [
    ("m02_datos", "1 · Carga de datos"),
    ("m01_contexto", "2 · Qué dicen sus datos"),
    ("m03_preparacion", "3 · Preparación de datos"),
    ("m04_eda", "4 · Análisis exploratorio"),
    ("m05_y", "5 · Variable a estimar (Y)"),
    ("m06_x", "6 · Variables explicativas (X)"),
    ("m07_modelos", "7 · Modelos de regresión"),
    ("m08_evaluacion", "8 · Evaluación y selección"),
    ("m09_interpretacion", "9 · Interpretación"),
    ("m10_simulacion", "10 · Simulación"),
    ("m11_sensibilidad", "11 · Sensibilidad"),
    ("m12_informe", "12 · Informe ejecutivo"),
    ("m13_aprendizaje", "13 · Centro de aprendizaje"),
]
LABELS = dict(MODULES)


def done(key: str) -> bool:
    ss = st.session_state
    return {
        "m02_datos": state.has_data(),
        "m01_contexto": state.has_data() and "m01_contexto" in ss.visited,
        "m03_preparacion": len(ss.plan) > 0 or "m03_preparacion" in ss.visited and state.has_data(),
        "m04_eda": "m04_eda" in ss.visited and state.has_data(),
        "m05_y": bool(ss.y) and "m05_y" in ss.visited,
        "m06_x": len(ss.x_cands) >= 2,
        "m07_modelos": ss.cmp is not None,
        "m08_evaluacion": bool(ss.chosen),
        "m09_interpretacion": "m09_interpretacion" in ss.visited and bool(ss.chosen),
        "m10_simulacion": "m10_simulacion" in ss.visited and bool(ss.chosen),
        "m11_sensibilidad": "m11_sensibilidad" in ss.visited and bool(ss.chosen),
        "m12_informe": "m12_informe" in ss.visited and bool(ss.chosen),
        "m13_aprendizaje": "m13_aprendizaje" in ss.visited,
    }[key]


ss = st.session_state
if "nav" not in ss:
    ss["nav"] = MODULES[0][0]

with st.sidebar:
    st.markdown("### 🧮 Regresión lineal múltiple")
    st.caption("Estime un resultado con 2 o más variables a la vez · modelar, estimar y decidir")
    st.radio("Módulos", [k for k, _ in MODULES], key="nav",
             format_func=lambda k: f"{'✅' if done(k) else '○'}  {LABELS[k]}", label_visibility="collapsed")
    st.divider()
    st.selectbox("Nivel de confianza de los intervalos", [90, 95, 99], key="conf", format_func=lambda v: f"{v}%")
    with st.expander("Unidad de Y (opcional)"):
        ss.units["y"] = st.text_input("Unidad de Y", ss.units.get("y", ""), placeholder="p. ej. millones de pesos")
    if state.has_data():
        st.caption(f"Datos: {ss.source_name}  \n{ss.loaded.df.shape[0]:,} filas × {ss.loaded.df.shape[1]} columnas")
    if st.button("↺ Reiniciar análisis", help="Borra el plan de limpieza, la selección de variables y los modelos (conserva los datos)."):
        state.reset_analysis()
        st.rerun()
    st.divider()
    st.caption("Proyecto académico. Las recomendaciones automáticas son heurísticas explicables, no un modelo de lenguaje: la decisión final es suya.")

key = ss["nav"]
ss.visited.add(key)
module = importlib.import_module(f"ui.{key}")
module.render()
