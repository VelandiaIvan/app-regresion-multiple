"""Módulo 11 · Análisis de sensibilidad: qué variable mueve más el resultado."""
from __future__ import annotations

import streamlit as st

from core import simulation_multi as sim_
from ui import charts
from ui import components as C
from ui.m09_interpretacion import chosen_model


def _parse(txt: str) -> list[float]:
    out = []
    for t in txt.replace(";", ",").split(","):
        t = t.strip().replace("%", "")
        if t:
            try:
                out.append(float(t))
            except ValueError:
                st.warning(f"Valor ignorado: «{t}»")
    return out


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 11", "Análisis de sensibilidad",
           "¿Qué variable mueve más a Y? Se cambia UNA variable a la vez en un porcentaje, dejando las demás quietas, y se compara el efecto.")
    cm, md = chosen_model()
    res = cm.res
    yl = res.y_name
    if ss.cmp.recommendation.status == "no confiable":
        C.callout("Este modelo fue calificado como <b>no confiable</b>: la sensibilidad es solo ilustrativa.", "bad")

    scen = sim_.scenario_values(res, ss.ctx.y_higher_is_better)
    c1, c2 = st.columns([1, 2])
    start = c1.selectbox("Punto de partida", ["base", "pesimista", "optimista"], format_func=lambda k: f"Escenario {k}")
    base = scen[start]
    with c2:
        std = [5, 10, -5, -10]
        picks = st.multiselect("Variaciones (%)", std, default=std, format_func=lambda v: f"{v:+d}%")
        custom = _parse(st.text_input("Variaciones personalizadas (%), separadas por coma", "", placeholder="Ejemplo: 15, -20"))
    changes = list(picks) + custom
    y0 = float(res.predict(base)["y_est"].iloc[0])
    C.kpis([(f"{yl} estimado en el punto de partida", f"{y0:,.4g}", ss.units.get("y", "")),
            ("Variables del modelo", f"{res.k}", ", ".join(res.x_names)[:60])])

    st.markdown("#### ¿Qué variable pesa más? (±10 % cada una)")
    tor = sim_.tornado_table(res, base, 10.0)
    C.chart(charts.tornado_chart(tor, yl, 10.0), "m11_tornado")
    if len(tor):
        top = tor.iloc[0]
        C.callout(f"Ante un mismo cambio de ±10 %, la variable que más mueve «{yl}» es <b>{top['variable']}</b> "
                  f"({top['+10%']:+,.4g} si sube; {top['-10%']:+,.4g} si baja).", "ok", "Palanca principal")

    if not changes:
        st.info("Seleccione al menos una variación para ver el detalle.")
        st.stop()
    st.markdown("#### Detalle por variable")
    vars_sel = st.multiselect("Variables a incluir", res.x_names, default=res.x_names)
    tab = sim_.sensitivity_table(res, base, changes, vars_sel or res.x_names)
    ss["last_sens"], ss["last_sens_key"] = tab, cm.key
    st.dataframe(tab[["variable", "escenario", "x_inicial", "x_nuevo", "y_inicial", "y_nuevo", "dif_absoluta", "var_%", "ip_inf", "ip_sup", "extrapolación"]],
                 hide_index=True, column_config={"var_%": st.column_config.NumberColumn("Variación %", format="%+.2f"),
                                                 "dif_absoluta": st.column_config.NumberColumn("Diferencia", format="%+.4g")})
    C.chart(charts.sensitivity_multi_chart(tab, yl), "m11_chart")
    st.markdown("#### Interpretación para el negocio")
    for _, r in tab.iterrows():
        st.markdown(f"- {r['interpretación']}")
    if (tab["extrapolación"] != "dentro").any():
        st.warning("Algunas variaciones llevan una variable fuera de su rango observado: esos resultados son extrapolaciones.")
    C.callout("La sensibilidad refleja la <b>asociación estimada</b> y supone que las demás variables no cambian. Antes de mover una palanca real, "
              "considere su costo, sus límites operativos y valide con una prueba piloto.", "edu")
    C.next_step("Continuar: informe ejecutivo ▶", "m12_informe")
