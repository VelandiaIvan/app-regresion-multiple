"""Módulo 8 · Evaluación, validación de supuestos y selección del mejor modelo múltiple."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.diagnostics import ICON, LABEL
from ui import charts
from ui import components as C
from ui import state
from ui.m07_modelos import build, short, test_points

STATUS = {"recomendado": ("ok", "MODELO RECOMENDADO"), "con reservas": ("warn", "RECOMENDADO CON RESERVAS"),
          "no confiable": ("bad", "NO HAY UN MODELO LINEAL MÚLTIPLE CONFIABLE"), "sin candidatos": ("bad", "SIN CANDIDATOS")}


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 8", "Evaluación y selección del mejor modelo",
           "Se comparan las combinaciones de 2 o más variables con datos de prueba y validación cruzada, se verifican los supuestos (incluida la "
           "multicolinealidad) y se elige con un puntaje multicriterio. No gana el de mayor R²: ese siempre sería el que tiene más variables.")
    C.need(state.has_data() and bool(ss.y) and len(ss.x_cands) >= 2, "Complete los módulos anteriores (datos, Y y al menos 2 X).", "m06_x")
    md, err = state.modeling()
    if md is None:
        st.error(err)
        st.stop()
    if not state.cmp_is_current():
        with st.spinner("Actualizando modelos…"):
            build()
    cmp = ss.cmp
    if not cmp.models:
        st.error("No hay modelos para evaluar: " + " ".join(cmp.skipped[:3]))
        st.stop()
    rec = cmp.recommendation
    kind, title = STATUS[rec.status]
    best = cmp.models.get(rec.best_key) if rec.best_key else None
    body = ""
    if best is not None:
        body = (f"<b>{best.key}</b> · {best.res.k} variables · puntaje {best.score:.0f}/100 · R² ajustado {best.res.r2_adj:.2f}"
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
    cols = ["ranking", "modelo", "n_variables", "puntaje", "R²_ajust", "R²", "RMSE_entren.", "RMSE_prueba", "R²_prueba", "RMSE_LOOCV", "habilidad",
            "coef_signif", "VIF_máx", "p_F", "linealidad", "independencia", "homocedasticidad", "normalidad", "multicolinealidad", "influencia",
            "signo_ok", "riesgo_fuga", "ranking_por_R²", "alertas"]
    st.dataframe(t[cols], hide_index=True, column_config={
        "puntaje": st.column_config.ProgressColumn("Puntaje", min_value=0, max_value=100, format="%.0f"),
        "n_variables": st.column_config.NumberColumn("N.º X"),
        "R²": st.column_config.NumberColumn(format="%.3f"), "R²_ajust": st.column_config.NumberColumn("R² ajustado", format="%.3f"),
        "R²_prueba": st.column_config.NumberColumn(format="%.3f"),
        "habilidad": st.column_config.NumberColumn("Mejora vs promedio", format="%.2f", help="1 − RMSE/RMSE(predecir siempre el promedio). 0 = no mejora nada."),
        "coef_signif": st.column_config.TextColumn("Coef. signif.", help="Coeficientes significativos / total de variables"),
        "VIF_máx": st.column_config.NumberColumn(format="%.2f"), "p_F": st.column_config.NumberColumn(format="%.2g"),
        "RMSE_entren.": st.column_config.NumberColumn(format="%.4g"), "RMSE_prueba": st.column_config.NumberColumn(format="%.4g"),
        "RMSE_LOOCV": st.column_config.NumberColumn(format="%.4g")})
    st.caption("Semáforos: 🟢 cumple · 🟡 reservas · 🔴 no cumple · ⚪ no aplicable. «Mejora vs promedio» promedia la prueba y la validación cruzada.")
    g1, g2 = st.columns(2)
    with g1:
        C.chart(charts.score_chart(t.assign(modelo=t["modelo"].map(lambda k: short(k, 46))), short(rec.best_key or "", 46)), "m8_score")
    with g2:
        C.chart(charts.r2_vs_score(t.assign(modelo=t["modelo"].map(lambda k: short(k, 46)))), "m8_r2")
    C.how_to_read("**Puntaje (0–100)** = capacidad predictiva en datos nuevos (40) + cumplimiento de supuestos, incluida la multicolinealidad (20) + "
                  "significancia global y de cada coeficiente (15) + coherencia con el negocio (15) + calidad y cantidad de datos (10) − penalizaciones "
                  "(4 puntos por cada variable sin efecto significativo, sobreajuste, VIF > 10, observaciones muy influyentes, posible fuga de información). "
                  "Si dos modelos quedan a menos de 1 punto, **gana el que tiene menos variables**.", "¿Cómo se calcula el puntaje?")

    st.markdown("#### Diagnóstico de supuestos")
    keys = list(cmp.models)
    default = ss.chosen if ss.chosen in keys else (rec.best_key or keys[0])
    sel = st.selectbox("Modelo a diagnosticar", keys, index=keys.index(default), key="m8_sel", format_func=short)
    cm = cmp.models[sel]
    d = cm.diag
    kd = {"apto": "ok", "con reservas": "warn", "no confiable": "bad"}[d.verdict]
    C.callout(d.summary, kd, f"Veredicto de supuestos: {d.verdict}")
    cols_ = st.columns(len(d.checks))
    for c, chk in zip(cols_, d.checks):
        c.markdown(f"**{chk.name}**<br>{ICON[chk.severity]} {LABEL[chk.severity]}", unsafe_allow_html=True)
    for chk in d.checks:
        with st.expander(f"{chk.icon} {chk.name} — {LABEL[chk.severity]}", expanded=chk.severity in {"fail", "warn"}):
            st.markdown(chk.explanation)
            if chk.tests:
                st.dataframe(pd.DataFrame([{"prueba": k, "estadístico": v[0], "p-valor": v[1]} for k, v in chk.tests.items()]), hide_index=True)
            if chk.remedy:
                st.markdown(f"**Qué hacer:** {chk.remedy}")
    y_t, yh_t = test_points(cm, md)
    g = st.columns(2)
    with g[0]:
        C.chart(charts.residuals_vs_fitted(cm.res), "m8_rvf")
    with g[1]:
        C.chart(charts.observed_vs_fitted_multi(cm.res, y_t, yh_t), "m8_ovf")
    g = st.columns(2)
    with g[0]:
        C.chart(charts.qq_chart(cm.res), "m8_qq")
    with g[1]:
        C.chart(charts.residual_hist(cm.res), "m8_rh")
    with st.expander("Gráficos adicionales: dispersión de residuos e influencia"):
        g = st.columns(2)
        with g[0]:
            C.chart(charts.scale_location(cm.res), "m8_sl")
        with g[1]:
            C.chart(charts.cooks_chart(cm.res), "m8_cook")
    C.how_to_read("**Residuos vs. ajustados:** deben ser una nube sin forma alrededor de 0. Curva → falta linealidad; abanico → varianza no constante. "
                  "**Q-Q / histograma:** los puntos deben seguir la diagonal. **VIF:** mide si las X se repiten entre sí (menor que 5 es lo deseable). "
                  "**Durbin-Watson ≈ 2** indica errores independientes (solo aplica si los datos tienen orden en el tiempo).")
    with st.expander("Desglose del puntaje de este modelo"):
        st.dataframe(pd.DataFrame({"componente": list(cm.parts), "puntos": list(cm.parts.values())}), hide_index=True)
        if cm.flags:
            st.markdown("Alertas: " + "; ".join(cm.flags))

    st.markdown("#### Elegir el modelo para interpretar y simular")
    cur = ss.chosen if ss.chosen in keys else (rec.best_key or keys[0])
    choice = st.selectbox("Modelo elegido", keys, index=keys.index(cur), key="m8_choice",
                          format_func=lambda k: f"{short(k)}{'  ★ recomendado' if k == rec.best_key else ''}")
    ss.chosen = choice
    cc = cmp.models[choice]
    if choice != rec.best_key:
        st.info("Eligió un modelo distinto del recomendado. Revise sus alertas: " + ("; ".join(cc.flags) if cc.flags else "ninguna") + ".")
    C.next_step("Continuar: interpretar el modelo ▶", "m09_interpretacion")
