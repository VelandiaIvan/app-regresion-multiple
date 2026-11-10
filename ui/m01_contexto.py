"""Módulo 2 · Qué dicen sus datos: la aplicación explica el archivo y propone la variable a estimar (Y).

Antes este módulo pedía describir el negocio en ocho campos. Ahora la aplicación lee los datos y propone; el usuario
solo confirma. El contexto escrito es OPCIONAL (dos preguntas) y sirve para afinar la sugerencia.
"""
from __future__ import annotations

import streamlit as st

from core import context, profiling, understanding
from ui import components as C
from ui import state


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 2", "Qué dicen sus datos",
           "La aplicación leyó su archivo: aquí le explica de qué trata, qué parece medir cada columna y cuál es, probablemente, "
           "el resultado que conviene estimar. Usted solo confirma o corrige.")
    C.need(state.has_data(), "Primero cargue un archivo de datos o un ejemplo.", "m02_datos")

    df = state.preview().df
    cls = profiling.classify_columns(df)
    info = understanding.describe_dataset(df, cls, ss.date_col, ss.source_name)

    st.markdown("#### De qué tratan los datos")
    C.callout(f"{info['tema']}<br>{info['unidad']}<br>{info['tamaño']}", "info", "Lectura automática")
    if info["evidencia"]:
        st.caption("Pistas encontradas en los nombres de las columnas → " + info["evidencia"])
    for note in info["calidad"]:
        st.markdown(f"- {note}")
    with st.expander("Qué parece medir cada columna", expanded=True):
        C.table(info["columnas"], 3)

    st.markdown("#### Variable que conviene estimar (Y)")
    rank = context.rank_target_candidates(df, cls, ss.ctx)
    if rank.empty:
        st.error("No hay columnas numéricas continuas que puedan ser la variable a estimar. Revise el archivo o la preparación de datos.")
        st.stop()
    expl = context.explainability(df, rank["variable"].tolist())
    exp = understanding.explain_target(rank, expl)
    C.callout(exp["texto"] + (f"<br><b>Ojo:</b> {exp['advertencias']}." if exp.get("advertencias") else ""), "ok", f"Sugerencia: {exp['y']}")
    opts = rank["variable"].tolist()
    current = ss.y if ss.y in opts else exp["y"]
    y = st.selectbox("Confirme la variable a estimar (Y)", opts, index=opts.index(current),
                     format_func=lambda v: f"{v}{'  ★ sugerida' if v == exp['y'] else ''}")
    if y != ss.y:
        ss.y = y
        ss.cmp = None
        ss.chosen = None
        ss.x_cands = [c for c in ss.x_cands if c != y]
    xs = expl.get(y, (0, []))[1]
    if xs:
        st.markdown("**Variables que podrían explicarla (X):** " + ", ".join(f"`{c}`" for c in xs[:8])
                    + ". Se confirmarán en el módulo 6; el modelo final usará 2 o más.")
    with st.expander("Ver el puntaje de todas las candidatas a Y"):
        C.table(rank, 3)
        st.caption("Sin descripción escrita, el puntaje combina: qué tan bien se explica la columna con las demás (45), si su nombre corresponde a un "
                   "resultado típico de negocio (20), la calidad de sus datos (20) y su relación con otras variables (15).")

    with st.expander("✍️ Contexto opcional: dos preguntas para afinar la sugerencia"):
        st.caption("No es obligatorio. Si ya sabe qué quiere estimar, escríbalo y la sugerencia se ajusta.")
        ss.setdefault("ctx_objetivo_y", ss.ctx.objetivo_y)
        ss.setdefault("ctx_pregunta", ss.ctx.pregunta)
        st.text_input("¿Qué resultado quiere estimar?", key="ctx_objetivo_y", placeholder="En sus palabras: ventas, costo, tiempo de entrega, consumo…")
        st.text_input("¿Qué pregunta quiere responder? (aparece en el informe)", key="ctx_pregunta",
                      placeholder="p. ej. ¿Cuánto cambian las ventas si subo el precio?")
        ss.ctx.objetivo_y, ss.ctx.pregunta = ss["ctx_objetivo_y"], ss["ctx_pregunta"]
        ss.ctx.y_higher_is_better = st.radio(
            "Para usted, un valor MÁS ALTO de Y es…", ["mejor (p. ej. ventas, rendimiento)", "peor (p. ej. costos, tiempos, defectos)"],
            index=0 if ss.ctx.y_higher_is_better else 1, horizontal=True,
            help="Solo se usa para nombrar los escenarios pesimista y optimista.").startswith("mejor")

    C.how_to_read("La aplicación **no usa un modelo de lenguaje**. Reconoce conceptos en los nombres de las columnas (en español e inglés), mira los tipos de "
                  "dato y calcula qué columna se explica mejor con las demás. Es una guía: si la sugerencia no coincide con su objetivo, cámbiela.",
                  "¿Cómo llegó la aplicación a estas conclusiones?")
    C.next_step("Continuar: preparar los datos ▶", "m03_preparacion")
