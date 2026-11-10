"""Módulo 10 · Simulación de escenarios y estimaciones."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import simulation as sim_
from ui import charts
from ui import components as C
from ui.m09_interpretacion import chosen_model


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 10", "Simulación de escenarios y estimaciones",
           "Mueva X y vea cómo cambia el valor estimado de Y, con su incertidumbre y alertas cuando se sale del rango que los datos respaldan.")
    cm, md = chosen_model()
    res = cm.res
    xu, yu = ss.units.get("x", ""), ss.units.get("y", "")
    xl, yl = res.x_source, res.y_source
    if ss.cmp.recommendation.status == "no confiable":
        C.callout("Este modelo fue calificado como <b>no confiable</b>: las simulaciones son solo ilustrativas.", "bad")
    lvl = int(round((1 - res.alpha) * 100))

    q = np.quantile(res.x_obs, [0.10, 0.25, 0.5, 0.75, 0.90])
    C.kpis([("Mínimo observado de X", f"{res.x_min:,.4g}"), ("P10", f"{q[0]:,.4g}"), ("Mediana", f"{q[2]:,.4g}"), ("P90", f"{q[4]:,.4g}"),
            ("Máximo observado de X", f"{res.x_max:,.4g}", xu)])
    if ss.ctx.restricciones.strip():
        st.caption(f"Reglas de negocio declaradas: {ss.ctx.restricciones.strip()}  — el modelo no las aplica automáticamente; verifíquelas al elegir valores.")

    st.markdown("#### Escenarios pesimista · base · optimista")
    mode = st.radio("Cómo se definen", ["Rango observado (P10 / mediana / P90)", "Valores definidos por usted"], horizontal=True)
    hib = ss.ctx.y_higher_is_better
    if mode.startswith("Rango"):
        scen = sim_.scenario_values(res, hib)
    else:
        base_def = sim_.scenario_values(res, hib)
        c = st.columns(3)
        scen = {}
        for col, k in zip(c, ["pesimista", "base", "optimista"]):
            scen[k] = col.number_input(f"X {k}", value=float(base_def[k]), key=f"sc_{k}_{cm.key}", format="%.6g")
    rows = []
    for k, xv in scen.items():
        p = res.predict([xv]).iloc[0]
        ex = sim_.extrapolation_status(res, xv)
        rows.append({"escenario": k, "X": xv, "Y estimado": p["y_est"], f"IC {lvl}% inf": p["ic_inf"], f"IC {lvl}% sup": p["ic_sup"],
                     f"IP {lvl}% inf": p["ip_inf"], f"IP {lvl}% sup": p["ip_sup"], "rango": {"dentro": "✔ dentro", "leve": "⚠ extrap. leve", "severa": "🛑 extrap. severa"}[ex.status]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    ss["last_scen"], ss["last_scen_key"] = scen, cm.key
    st.caption(f"Pesimista = el X que produce el peor resultado (aquí un {yl} {'menor' if hib else 'mayor'} es peor). IC = intervalo del promedio; IP = intervalo de una observación individual.")

    st.markdown("#### Simulación interactiva")
    c1, c2 = st.columns([1, 2])
    base_name = c1.selectbox("Escenario de partida", list(scen), index=list(scen).index("base"))
    x_base = float(scen[base_name])
    lo_g, hi_g = res.x_min - 0.4 * (res.x_max - res.x_min), res.x_max + 0.4 * (res.x_max - res.x_min)
    if res.x_transform in {"ln", "sqrt", "ln1p", "inv"}:
        lo_g = max(lo_g, res.x_min * 0.3 if res.x_min > 0 else 1e-6)
    with c2:
        ctrl = st.radio("Control de X simulado", ["Deslizador", "Valor exacto"], horizontal=True, key="m10_ctrl")
        key_x = f"m10_x_{cm.key}"
        if ctrl == "Deslizador":
            x_new = st.slider(f"{xl} simulado ({xu})" if xu else f"{xl} simulado", float(lo_g), float(hi_g), float(np.clip(x_base * 1.1, lo_g, hi_g)),
                              step=float((hi_g - lo_g) / 400), key=key_x + "_s", format="%.6g")
        else:
            x_new = st.number_input(f"{xl} simulado", value=float(x_base * 1.1), key=key_x + "_n", format="%.6g")
        st.caption(f"Rango observado: [{res.x_min:,.4g}; {res.x_max:,.4g}]. La zona naranja del gráfico es extrapolación.")
    sim = sim_.simulate(res, x_base, x_new)
    ss["last_sim"], ss["last_sim_key"] = sim, cm.key
    for ex, lab in ((sim["extrap_new"], "simulado"),):
        if ex.status == "leve":
            st.warning(ex.message)
        elif ex.status == "severa":
            st.error(ex.message)
        else:
            st.success(ex.message)
    C.kpis([(f"{yl} estimado · base", f"{sim['y_base']:,.4g}", f"X = {sim['x_base']:,.4g}"),
            (f"{yl} estimado · simulado", f"{sim['y_new']:,.4g}", f"X = {sim['x_new']:,.4g}"),
            ("Variación absoluta", f"{sim['dy']:+,.4g}", yu),
            ("Variación porcentual", f"{sim['dy_pct']:+.1f}%" if not np.isnan(sim["dy_pct"]) else "n/a", f"X cambia {sim['dx_pct']:+.1f}%" if not np.isnan(sim["dx_pct"]) else "")])
    st.dataframe(pd.DataFrame([
        {"": "Base", "X": sim["x_base"], "Ŷ": sim["y_base"], f"IC {lvl}%": f"[{sim['base_ci'][0]:,.4g}; {sim['base_ci'][1]:,.4g}]",
         f"IP {lvl}%": f"[{sim['base_pi'][0]:,.4g}; {sim['base_pi'][1]:,.4g}]"},
        {"": "Simulado", "X": sim["x_new"], "Ŷ": sim["y_new"], f"IC {lvl}%": f"[{sim['new_ci'][0]:,.4g}; {sim['new_ci'][1]:,.4g}]",
         f"IP {lvl}%": f"[{sim['new_pi'][0]:,.4g}; {sim['new_pi'][1]:,.4g}]"}]), hide_index=True)
    C.callout(sim_.interpret_simulation(res, sim, xl, yl), "info", "Interpretación")
    C.chart(charts.scenario_chart(res, scen, sim), "m10_chart")
    C.how_to_read("**Estimación puntual:** el mejor valor único de Y. **Intervalo de confianza:** dónde estaría el *promedio* de muchos casos con ese X. "
                  "**Intervalo de predicción:** dónde caería *un caso individual*. Para presupuestar usted querría el primero; para prometer un resultado a un cliente, el segundo.")

    st.markdown("#### ¿Qué X necesito para alcanzar una meta de Y?")
    default_target = float(np.mean(res.y_obs))
    tgt = st.number_input(f"Meta de {yl}", value=default_target, format="%.6g", key=f"m10_tgt_{cm.key}")
    inv = sim_.required_x(res, tgt)
    (st.success if inv["ok"] and not inv["extrap"].is_extrapolation else st.warning)(inv["message"])
    st.caption("Es una aproximación: se despeja X de la recta estimada. No es una garantía y hereda todas las limitaciones del modelo (asociación, no causalidad).")
    C.next_step("Continuar: análisis de sensibilidad ▶", "m11_sensibilidad")
