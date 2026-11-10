"""Módulo 1 · Identificación del problema de negocio."""
from __future__ import annotations

import streamlit as st

from core.context import interpret_context
from sample_data import generate as G
from ui import components as C
from ui import state

FIELDS = [
    ("sector", "Sector económico", "p. ej. Comercio minorista, Logística, Salud, Agroindustria, Educación, Finanzas públicas…", "input"),
    ("actividad", "Actividad principal de la empresa", "¿A qué se dedica y en qué escala opera?", "input"),
    ("core", "Core del negocio", "¿Qué hace valioso al negocio? ¿Qué quiere optimizar?", "area"),
    ("problema", "Problema que desea resolver", "Describa la situación que motiva el análisis.", "area"),
    ("objetivo_y", "Variable o resultado que quiere estimar", "En sus palabras: ventas, costo, tiempo de entrega, consumo, rendimiento…", "input"),
    ("factores", "Posibles factores explicativos", "¿Qué cree que influye en ese resultado? (separe con comas)", "area"),
    ("restricciones", "Restricciones y reglas de negocio", "Topes, rangos válidos, políticas, condiciones que el modelo debe respetar.", "area"),
    ("pregunta", "Pregunta de negocio", "La pregunta concreta que el informe debe responder.", "area"),
]


def _load_example() -> None:
    state.load_sample(st.session_state["sample_sel"])


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 1", "Identificación del problema de negocio",
           "Antes de analizar números, defina el contexto. Todo el proceso —qué variable estimar, qué factores considerar y cómo "
           "interpretar los resultados— se orienta con lo que escriba aquí.")

    with st.expander(f"🧪 ¿Quiere probar con un ejemplo? (datos sintéticos de {len(G.DATASETS)} sectores)", expanded=not state.has_data()):
        key = st.selectbox("Caso de ejemplo", list(G.DATASETS), key="sample_sel")
        st.caption(G.DATASETS[key][2])
        st.button("Cargar ejemplo y contexto de negocio", on_click=_load_example, type="primary")
        if state.has_data() and "sintéticos" in ss.source_name:
            st.success(f"Cargado: {ss.source_name}")

    st.markdown("#### Cuéntenos su negocio")
    for fld, _, _, _ in FIELDS:
        ss.setdefault(f"ctx_{fld}", getattr(ss.ctx, fld))
    left, right = st.columns(2)
    for i, (fld, label, hint, kind) in enumerate(FIELDS):
        with (left if i < 4 else right):
            if kind == "input":
                st.text_input(label, key=f"ctx_{fld}", placeholder=hint)
            else:
                st.text_area(label, key=f"ctx_{fld}", placeholder=hint, height=92)
    for fld, *_ in FIELDS:
        setattr(ss.ctx, fld, ss[f"ctx_{fld}"])
    ss.ctx.y_higher_is_better = st.radio(
        "Para el negocio, un valor MÁS ALTO del resultado a estimar es…", ["mejor (p. ej. ventas, rendimiento)", "peor (p. ej. costos, tiempos, defectos)"],
        index=0 if ss.ctx.y_higher_is_better else 1, horizontal=True,
        help="Se usa para nombrar los escenarios pesimista y optimista de forma coherente.").startswith("mejor")

    st.markdown("#### Lo que la aplicación entendió")
    cols = list(ss.loaded.df.columns) if state.has_data() else []
    info = interpret_context(ss.ctx, cols)
    if ss.ctx.filled_fields() == 0:
        C.callout("Complete al menos el sector, el resultado que quiere estimar y la pregunta de negocio para obtener recomendaciones mejor orientadas.", "info")
    else:
        C.callout(info["resumen"], "info", "Resumen")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Conceptos clave detectados:** " + (", ".join(f"`{t}`" for t in info["conceptos_clave"]) or "—"))
            if info["restricciones"]:
                st.markdown("**Restricciones registradas:**")
                for r in info["restricciones"]:
                    st.markdown(f"- {r}")
        with c2:
            if cols:
                st.markdown("**Columnas que parecen corresponder al resultado a estimar:** " + (", ".join(f"`{c}`" for c in info["columnas_para_Y"]) or "ninguna coincide aún"))
                st.markdown("**Columnas que parecen corresponder a los factores:** " + (", ".join(f"`{c}`" for c in info["columnas_para_X"]) or "ninguna coincide aún"))
            else:
                st.caption("Cargue datos en el módulo 2 para cruzar su descripción con las columnas.")
        if info["campos_sin_completar"]:
            st.caption("Sin completar: " + ", ".join(info["campos_sin_completar"]))
    C.how_to_read("La aplicación **no usa un modelo de lenguaje**: compara su texto con los nombres de las columnas mediante normalización, raíces de palabras "
                  "y sinónimos multisectoriales. Es una guía explicable; usted confirma o modifica las variables en los módulos 5 y 6. "
                  "Las restricciones que escriba se citan en el informe y se tienen en cuenta al interpretar escenarios, pero no se imponen automáticamente al modelo.",
                  "¿Cómo se usa lo que escribo?")
    C.next_step("Continuar: cargar los datos ▶", "m02_datos")
