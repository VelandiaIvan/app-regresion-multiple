"""Módulo 8 · Evaluación, validación de supuestos y selección del mejor modelo."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.diagnostics import ICON, LABEL
from ui import charts
from ui import components as C
from ui import state
from ui.m07_modelos import build, model_points

STATUS = {"recomendado": ("ok", "MODELO RECOMENDADO"), "con reservas": ("warn", "RECOMENDADO CON RESERVAS"),
          "no confiable": ("bad", "NO HAY UN MODELO LINEAL SIMPLE CONFIABLE"), "sin candidatos": ("bad", "SIN CANDIDATOS")}


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 8", "Evaluación y selección del mejor modelo",
           "Se comparan todos los modelos con datos de prueba y validación cruzada, se verifican los cuatro supuestos y se elige con un puntaje multicriterio. "
           "No se elige automáticamente el de mayor R².")
    C.need(state.has_data() and bool(ss.y) and bool(ss.x_cands), "Complete los módulos anteriores (datos, Y y X).", "m06_x")
    md, err = state.modeling()
    if md is None:
        st.error(err)
        st.stop()
    if not state.cmp_is_current():
        with st.spinner("Actualizando modelos…"):
            build()
    cmp = ss.cmp
    if not cmp.models:
        st.error("No hay modelos para evaluar.")
        st.stop()
    rec = cmp.recommendation
    kind, title = STATUS[rec.status]
    best = cmp.models.get(rec.best_key) if rec.best_key else None
    body = ""
    if best is not None:
        body = (f"Modelo de referencia: <b>{best.key}</b> · puntaje {best.score:.0f}/100 · R² {best.res.r2:.2f}"
                + (f" · R² en prueba {best.test_metrics['R²_fuera_muestra']:.2f}" if best.test_metrics else "") + ".")
        if rec.status == "no confiable":
            body += " Se muestra solo como referencia: <b>no tome decisiones con él</b>."
    C.callout(body, kind, title)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Por qué**")
        for t in rec.reasons:
            st.markdown(f"- {t}")
    with c2:
        if rec.warnings:
            st.markdown("**Advertencias**")
            for t in rec.warnings:
                st.markdown(f"- ⚠ {t}")
        if rec.note_r2:
            st.info(rec.note_r2)
    if rec.status != "recomendado":
        st.markdown("**Alternativas sugeridas**")
        for a in rec.alternatives:
            st.markdown(f"- {a}")

    st.markdown("#### Comparación de modelos")
    t = cmp.table
    cols = ["ranking", "modelo", "puntaje", "R²", "R²_ajust", "RMSE_entren.", "RMSE_prueba", "MAE_prueba", "R²_prueba", "RMSE_LOOCV", "habilidad", "p_β1",
            "linealidad", "independencia", "homocedasticidad", "normalidad", "influencia", "signo_ok", "riesgo_fuga", "ranking_por_R²", "alertas"]
    st.dataframe(t[cols], hide_index=True, column_config={
        "puntaje": st.column_config.ProgressColumn("Puntaje", min_value=0, max_value=100, format="%.0f"),
        "R²": st.column_config.NumberColumn(format="%.3f"), "R²_ajust": st.column_config.NumberColumn(format="%.3f"),
        "R²_prueba": st.column_config.NumberColumn(format="%.3f"), "habilidad": st.column_config.NumberColumn("Mejora vs promedio", format="%.2f",
                                                                         help="1 − RMSE/RMSE(predecir siempre el promedio). 0 = no mejora nada."),
        "p_β1": st.column_config.NumberColumn(format="%.2g"), "RMSE_entren.": st.column_config.NumberColumn(format="%.4g"),
        "RMSE_prueba": st.column_config.NumberColumn(format="%.4g"), "MAE_prueba": st.column_config.NumberColumn(format="%.4g"),
        "RMSE_LOOCV": st.column_config.NumberColumn(format="%.4g")})
    st.caption("Semáforos: 🟢 cumple · 🟡 reservas · 🔴 no cumple · ⚪ no aplicable. «Mejora vs promedio» promedia la prueba y la validación cruzada. "
               "Si compara muchos modelos con los mismos datos de prueba hay un leve riesgo de optimismo al elegir el mejor.")
    g1, g2 = st.columns(2)
    with g1:
        C.chart(charts.score_chart(t, rec.best_key), "m8_score")
    with g2:
        C.chart(charts.r2_vs_score(t), "m8_r2")
    C.how_to_read("**Puntaje (0–100)** = capacidad predictiva en datos nuevos (40) + cumplimiento de supuestos (20) + coherencia con el negocio (15) + significancia y fuerza (15) "
                  "+ calidad y cantidad de datos (10) − penalizaciones (posible fuga de información, sobreajuste, observaciones muy influyentes). "
                  "Un R² altísimo con riesgo de fuga **pierde** frente a un modelo más modesto pero legítimo.", "¿Cómo se calcula el puntaje?")

    st.markdown("#### Diagnóstico de supuestos")
    keys = list(cmp.models)
    default = ss.chosen if ss.chosen in keys else (rec.best_key or keys[0])
    sel = st.selectbox("Modelo a diagnosticar", keys, index=keys.index(default), key="m8_sel")
    cm = cmp.models[sel]
    d = cm.diag
    kd = {"apto": "ok", "con reservas": "warn", "no confiable": "bad"}[d.verdict]
    C.callout(d.summary, kd, f"Veredicto de supuestos: {d.verdict}")
    cols = st.columns(len(d.checks))
    for c, chk in zip(cols, d.checks):
        c.markdown(f"**{chk.name}**<br>{ICON[chk.severity]} {LABEL[chk.severity]}", unsafe_allow_html=True)
    for chk in d.checks:
        with st.expander(f"{chk.icon} {chk.name} — {LABEL[chk.severity]}", expanded=chk.severity in {"fail", "warn"}):
            st.markdown(chk.explanation)
            if chk.tests:
                st.dataframe(pd.DataFrame([{"prueba": k, "estadístico": v[0], "p-valor": v[1]} for k, v in chk.tests.items()]), hide_index=True)
            if chk.remedy:
                st.markdown(f"**Qué hacer:** {chk.remedy}")
    x_t, y_t = model_points(cm, md)
    g = st.columns(2)
    with g[0]:
        C.chart(charts.residuals_vs_fitted(cm.res), "m8_rvf")
    with g[1]:
        C.chart(charts.residual_hist(cm.res), "m8_rh")
    g = st.columns(2)
    with g[0]:
        C.chart(charts.qq_chart(cm.res), "m8_qq")
    with g[1]:
        C.chart(charts.observed_vs_fitted(cm.res, x_t, y_t), "m8_ovf")
    with st.expander("Gráficos adicionales: dispersión de residuos e influencia"):
        g = st.columns(2)
        with g[0]:
            C.chart(charts.scale_location(cm.res), "m8_sl")
        with g[1]:
            C.chart(charts.cooks_chart(cm.res), "m8_cook")
    C.how_to_read("**Residuos vs. ajustados:** deben ser una nube sin forma alrededor de 0. Curva → falta linealidad; abanico → varianza no constante. "
                  "**Histograma / Q-Q:** los puntos deben seguir la diagonal; desvíos en los extremos indican colas pesadas o asimetría. "
                  "**Observado vs. estimado:** los puntos deben acompañar la diagonal sin curvarse. **Durbin-Watson ≈ 2** indica errores independientes.")
    with st.expander("Desglose del puntaje de este modelo"):
        st.dataframe(pd.DataFrame({"componente": list(cm.parts), "puntos": list(cm.parts.values())}), hide_index=True)
        if cm.flags:
            st.markdown("Alertas: " + "; ".join(cm.flags))

    st.markdown("#### Elegir el modelo para interpretar y simular")
    cur = ss.chosen if ss.chosen in keys else (rec.best_key or keys[0])
    choice = st.selectbox("Modelo elegido", keys, index=keys.index(cur), key="m8_choice",
                          format_func=lambda k: f"{k}{'  ★ recomendado' if k == rec.best_key else ''}")
    ss.chosen = choice
    cc = cmp.models[choice]
    if choice != rec.best_key:
        st.info("Eligió un modelo distinto del recomendado. Revise sus alertas: " + ("; ".join(cc.flags) if cc.flags else "ninguna") + ".")
    if cc.leak == "alta":
        st.error("Este modelo tiene riesgo ALTO de fuga de información: úselo solo si X se conoce antes de necesitar estimar Y.")
    C.next_step("Continuar: interpretar el modelo ▶", "m09_interpretacion")
