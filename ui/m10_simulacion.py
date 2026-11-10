"""Módulo 10 · Simulación de escenarios con varias variables a la vez."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import simulation_multi as sim_
from ui import charts
from ui import components as C
from ui.m09_interpretacion import chosen_model

RANGO = {"dentro": "✔ dentro", "leve": "⚠ extrap. leve", "severa": "🛑 extrap. severa"}


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 10", "Simulación de escenarios y estimaciones",
           "Elija un valor para cada variable y vea el resultado estimado, con su incertidumbre y alertas cuando se sale de lo que los datos respaldan.")
    cm, md = chosen_model()
    res = cm.res
    yu, yl = ss.units.get("y", ""), res.y_name
    if ss.cmp.recommendation.status == "no confiable":
        C.callout("Este modelo fue calificado como <b>no confiable</b>: las simulaciones son solo ilustrativas.", "bad")
    lvl = int(round((1 - res.alpha) * 100))
    hib = ss.ctx.y_higher_is_better

    with st.expander("Rangos observados de cada variable", expanded=False):
        C.table(res.x_stats(), 4)

    st.markdown("#### Escenarios pesimista · base · optimista")
    scen = sim_.scenario_values(res, hib)
    rows = []
    for name, vals in scen.items():
        p = res.predict(vals).iloc[0]
        ex = sim_.extrapolation_status(res, vals)
        rows.append({"escenario": name, **{c: vals[c] for c in res.x_names}, f"{yl} estimado": p["y_est"], f"IC {lvl}% inf": p["ic_inf"],
                     f"IC {lvl}% sup": p["ic_sup"], f"IP {lvl}% inf": p["ip_inf"], f"IP {lvl}% sup": p["ip_sup"], "rango": RANGO[ex.status]})
    C.table(pd.DataFrame(rows), 5)
    ss["last_scen"], ss["last_scen_key"] = scen, cm.key
    st.caption(f"Base = cada variable en su mediana. Pesimista / optimista = cada variable en su P10 o P90, según el signo de su coeficiente "
               f"(aquí un {yl} {'menor' if hib else 'mayor'} es peor). IC = intervalo del promedio; IP = intervalo de un caso individual.")

    st.markdown("#### Simulación interactiva")
    base_name = st.selectbox("Escenario de partida", list(scen), index=list(scen).index("base"))
    base = scen[base_name]
    st.caption("Mueva una o varias variables. Las que no toque se quedan en el valor del escenario de partida.")
    new = {}
    cols = st.columns(min(3, res.k))
    for j, c in enumerate(res.x_names):
        lo, hi = res.x_range(c)
        with cols[j % len(cols)]:
            if res.is_binary(c):
                new[c] = float(st.radio(c, [0, 1], index=int(round(base[c])), horizontal=True, key=f"m10_{cm.key}_{base_name}_{c}",
                                        format_func=lambda v: "1 (sí)" if v == 1 else "0 (no)"))
            else:
                span = hi - lo
                new[c] = float(st.slider(c, float(lo - 0.25 * span), float(hi + 0.25 * span), float(base[c]), step=float(span / 200) or None,
                                         key=f"m10_{cm.key}_{base_name}_{c}", format="%.6g", help=f"Observado: de {lo:,.4g} a {hi:,.4g}"))
    sim = sim_.simulate(res, base, new)
    ss["last_sim"], ss["last_sim_key"] = sim, cm.key
    ex = sim["extrap_new"]
    (st.success if ex.status == "dentro" else st.warning if ex.status == "leve" else st.error)(ex.message)
    C.kpis([(f"{yl} · partida", f"{sim['y_base']:,.4g}", base_name), (f"{yl} · simulado", f"{sim['y_new']:,.4g}", yu),
            ("Variación absoluta", f"{sim['dy']:+,.4g}", yu),
            ("Variación porcentual", f"{sim['dy_pct']:+.1f}%" if not np.isnan(sim["dy_pct"]) else "n/a"),
            (f"IP {lvl}% del simulado", f"{sim['new_pi'][0]:,.4g} – {sim['new_pi'][1]:,.4g}", "donde caería un caso individual")])
    C.callout(sim_.interpret_simulation(res, sim), "info", "Interpretación")
    moved = sim["contrib"][sim["contrib"]["cambio_X"].abs() > 1e-12]
    if len(moved):
        st.markdown("**Aporte de cada cambio** (efecto de mover solo esa variable)")
        C.table(moved, 4)

    st.markdown("#### Efecto de una variable (las demás fijas)")
    var = st.selectbox("Variable a recorrer", res.x_names, key=f"m10_var_{cm.key}")
    cv = sim_.partial_curve(res, var, new)
    pts = [dict(name="partida", x=base[var], y=float(res.predict({**new, var: base[var]})["y_est"].iloc[0]), color=charts.BLUE),
           dict(name="simulado", x=new[var], y=sim["y_new"], color=charts.INK)]
    C.chart(charts.partial_effect_chart(cv, var, yl, lvl, pts, res.x_range(var)), "m10_chart")
    C.how_to_read("La línea muestra cómo cambia la estimación al mover **solo** esa variable, dejando las demás en los valores simulados. "
                  "**Banda oscura:** dónde estaría el *promedio* de muchos casos. **Banda clara:** dónde caería *un caso individual*. "
                  "La zona sombreada en ámbar es extrapolación.")

    st.markdown(f"#### ¿Qué valor necesito para alcanzar una meta de {yl}?")
    c1, c2 = st.columns(2)
    tgt = c1.number_input(f"Meta de {yl}", value=float(np.mean(res.y_obs)), format="%.6g", key=f"m10_tgt_{cm.key}")
    lever = c2.selectbox("Variable que puedo mover", res.x_names, key=f"m10_lever_{cm.key}")
    inv = sim_.required_x(res, tgt, lever, new)
    (st.success if inv["ok"] and not inv["extrap"].is_extrapolation else st.warning)(inv["message"])
    st.caption("Es un despeje de la ecuación con las demás variables fijas en los valores simulados. No es una garantía: hereda las limitaciones del modelo.")
    C.next_step("Continuar: análisis de sensibilidad ▶", "m11_sensibilidad")
