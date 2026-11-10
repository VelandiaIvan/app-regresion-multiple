"""Módulo 11 · Análisis de sensibilidad."""
from __future__ import annotations

import numpy as np
import streamlit as st

from core import simulation as sim_
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
           "¿Cuánto se mueve Y cuando X sube o baja un porcentaje? Compare variaciones del 5 % y 10 % (o las suyas) alrededor de un punto de partida.")
    cm, md = chosen_model()
    res = cm.res
    xl, yl = res.x_source, res.y_source
    if ss.cmp.recommendation.status == "no confiable":
        C.callout("Este modelo fue calificado como <b>no confiable</b>: la sensibilidad es solo ilustrativa.", "bad")

    c1, c2 = st.columns([1, 2])
    scen = sim_.scenario_values(res, ss.ctx.y_higher_is_better)
    start = c1.selectbox("Punto de partida (X inicial)", ["Escenario base (mediana)", "Promedio observado", "Valor exacto"])
    if start.startswith("Escenario"):
        x0 = scen["base"]
    elif start.startswith("Promedio"):
        x0 = float(np.mean(res.x_obs))
    else:
        x0 = c1.number_input("X inicial", value=float(np.median(res.x_obs)), format="%.6g")
    with c2:
        std = [5, 10, -5, -10]
        picks = st.multiselect("Variaciones estándar de X (%)", std, default=std, format_func=lambda v: f"{v:+d}%")
        custom = _parse(st.text_input("Variaciones personalizadas (%), separadas por coma", "15, -20", help="Ejemplo: 15, -20, 30"))
    tab = sim_.sensitivity_table(res, x0, picks, custom, x_label=xl, y_label=yl)
    ss["last_sens"], ss["last_sens_key"] = tab, cm.key
    p0 = res.predict([x0]).iloc[0]
    up = res.predict([x0 * 1.01]).iloc[0]["y_est"] if abs(x0) > 0 else np.nan
    elast = ((up - p0["y_est"]) / abs(p0["y_est"]) * 100) if (abs(p0["y_est"]) > 1e-12 and not np.isnan(up)) else np.nan
    C.kpis([("X inicial", f"{x0:,.4g}", ss.units.get("x", "")), (f"{yl} estimado inicial", f"{p0['y_est']:,.4g}", ss.units.get("y", "")),
            ("Elasticidad en el punto", f"{elast:+.2f}" if not np.isnan(elast) else "n/a", "% de cambio de Y por cada +1% de X")])
    if tab.empty:
        st.info("Seleccione al menos una variación.")
        st.stop()
    st.dataframe(tab[["escenario", "x_inicial", "x_nuevo", "y_inicial", "y_nuevo", "dif_absoluta", "var_%", "ip_inf", "ip_sup", "extrapolación"]], hide_index=True,
                 column_config={"var_%": st.column_config.NumberColumn("Variación %", format="%+.2f"), "dif_absoluta": st.column_config.NumberColumn("Diferencia", format="%+.4g")})
    C.chart(charts.sensitivity_chart(tab, yl), "m11_chart")
    st.markdown("#### Interpretación para el negocio")
    for _, r in tab.iterrows():
        st.markdown(f"- {r['interpretación']}")
    if (tab["extrapolación"] != "dentro").any():
        st.warning("Algunas variaciones llevan X fuera del rango observado: esos resultados son extrapolaciones y no están respaldados por los datos.")
    if res.x_transform or res.y_transform:
        st.caption("Con transformaciones, el efecto no es simétrico: +10% y −10% no producen cambios iguales en sentido contrario.")
    elif elast == elast and abs(elast) < 0.5 and res.p_b1 < res.alpha:
        st.caption(f"La elasticidad es baja ({elast:.2f}): Y reacciona proporcionalmente poco a X en este punto; el intercepto (nivel base) pesa más que el factor.")
    C.callout("La sensibilidad refleja la <b>asociación estimada</b>. Antes de mover una palanca real, considere su costo, sus límites operativos y valide con una prueba piloto.", "edu")
    C.next_step("Continuar: informe ejecutivo ▶", "m12_informe")
