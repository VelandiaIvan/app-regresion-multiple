"""Módulo 12 · Informe ejecutivo, recomendaciones y exportación."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core import eda
from core import simulation as sim_
from core.report import build_executive_report
from ui import components as C
from ui import state
from ui.m09_interpretacion import chosen_model


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 12", "Informe ejecutivo y recomendaciones",
           "Un documento que responde la pregunta de negocio con los hallazgos, la calidad del modelo, las estimaciones, los riesgos y acciones concretas. "
           "Todo se genera a partir de los resultados reales de su análisis.")
    cm, md = chosen_model()
    res = cm.res
    cmp = ss.cmp
    scen = ss.get("last_scen") if ss.get("last_scen_key") == cm.key else None
    scen = scen or sim_.scenario_values(res, ss.ctx.y_higher_is_better)
    sim = ss.get("last_sim") if ss.get("last_sim_key") == cm.key else None
    base_x = scen["base"]
    sens = ss.get("last_sens") if ss.get("last_sens_key") == cm.key else None
    if sens is None:
        sens = sim_.sensitivity_table(res, base_x, x_label=res.x_source, y_label=res.y_source)
    srcs = [c for c in dict.fromkeys(m.x_source for m in cmp.models.values()) if c in md.df.columns]
    corr = eda.correlation_table(md.df.loc[md.train_idx], md.y_col, srcs) if srcs else None
    rep = build_executive_report(ctx=ss.ctx, source_name=ss.source_name, shape_raw=ss.loaded.df.shape, plan_records=ss.plan.records(),
                                 change_log=md.log, corr_table=corr, cmp=cmp, chosen_key=cm.key, units=ss.units, sim=sim, sens=sens, scenarios=scen)
    st.markdown(f"##### Modelo incluido en el informe: **{cm.key}** (estado: {cmp.recommendation.status})")
    if cm.key != cmp.recommendation.best_key:
        st.info("El informe usa el modelo que usted eligió, no el recomendado.")
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ Informe (HTML)", rep.to_html().encode("utf-8"), "informe_ejecutivo.html", "text/html", type="primary")
    c2.download_button("⬇ Informe (Markdown)", rep.to_markdown().encode("utf-8"), "informe_ejecutivo.md", "text/markdown")
    sheets = {"comparación": cmp.table, "coeficientes": res.coef_table(), "supuestos": cm.diag.table(), "sensibilidad": sens,
              "bitácora_limpieza": md.log, "plan_limpieza": pd.DataFrame(ss.plan.records())}
    c3.download_button("⬇ Tablas (Excel)", state.to_excel_bytes(sheets), "resultados_regresion.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    t1, t2 = st.tabs(["Vista previa", "Texto (Markdown)"])
    with t1:
        html_doc = rep.to_html()                      # todo el contenido del usuario va escapado (html.escape)
        if hasattr(st, "iframe"):
            st.iframe(html_doc, height=900)
        else:  # Streamlit antiguo
            components.html(html_doc, height=900, scrolling=True)
    with t2:
        st.markdown(rep.to_markdown())
    C.next_step("Aprender los conceptos ▶", "m13_aprendizaje")
