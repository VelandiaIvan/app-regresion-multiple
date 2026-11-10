"""Módulo 9 · Interpretación del modelo múltiple y de cada uno de sus coeficientes."""
from __future__ import annotations

import streamlit as st

from core import interpretation_multi as itp
from ui import charts
from ui import components as C
from ui import state


def chosen_model():
    """Devuelve (MultiModel, ModelingData) del modelo elegido, o detiene la página."""
    ss = st.session_state
    C.need(ss.cmp is not None and bool(ss.chosen), "Primero construya y elija un modelo (módulos 7 y 8).", "m08_evaluacion")
    if not state.cmp_is_current():
        st.warning("La configuración cambió desde la última construcción de modelos. Vaya al módulo 7 para actualizarlos.")
        st.button("Actualizar modelos ▶", on_click=C.goto, args=("m07_modelos",))
        st.stop()
    cm = ss.cmp.models.get(ss.chosen)
    C.need(cm is not None, "El modelo elegido ya no existe. Elija otro en el módulo 8.", "m08_evaluacion")
    return cm, ss.cmp.md


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 9", "Interpretación del modelo y sus coeficientes",
           "Qué significa cada coeficiente en términos del negocio: el efecto de una variable cuando las demás se mantienen constantes. "
           "Versión técnica, en lenguaje sencillo y con ejemplos de sus propios datos.")
    cm, md = chosen_model()
    res = cm.res
    yu = ss.units.get("y", "")
    rec = ss.cmp.recommendation
    if rec.status == "no confiable":
        C.callout("El modelo recomendado fue calificado como <b>no confiable</b>. La siguiente interpretación describe la ecuación ajustada, pero no debe usarse para decidir.", "bad")
    elif cm.key != rec.best_key:
        C.callout("Está interpretando un modelo distinto del recomendado.", "warn")

    st.markdown("#### Ecuación final")
    st.markdown(f"<div class='eq'>{res.equation()}</div>", unsafe_allow_html=True)
    C.callout(itp.plain_language(res, yu), "ok" if res.n_signif == res.k else "warn", "Mensaje central")

    st.markdown("#### Qué variable pesa más")
    c1, c2 = st.columns([3, 2])
    with c1:
        C.chart(charts.coef_importance_chart(itp.importance_table(res)), "m9_imp")
    with c2:
        C.table(itp.importance_table(res), 3)
        st.caption("Los coeficientes originales no son comparables entre sí porque cada variable tiene su unidad; el β estandarizado sí lo es.")

    st.markdown("#### Cada coeficiente, uno por uno")
    items = itp.interpret_all(res, yu)
    for d in items:
        icon = "🟢" if d["significativa"] else "⚪"
        with st.expander(f"{icon} {d['variable']} — relación {d['direccion']}" + ("" if d["significativa"] else " (no significativa)"), expanded=d["significativa"]):
            t1, t2 = st.tabs(["En lenguaje sencillo", "Técnica"])
            t1.markdown(d["efecto"])
            t1.markdown(f"**Magnitud:** {d['magnitud']}")
            t2.markdown(d["tecnica"])
    ii = itp.interpret_intercept(res, yu)
    with st.expander("Intercepto (β0)"):
        t1, t2 = st.tabs(["En lenguaje sencillo", "Técnica"])
        t1.markdown(ii["sencilla"])
        t2.markdown(ii["tecnica"])
        t2.markdown(f"**Viabilidad de la interpretación:** {ii['viabilidad']}")

    st.markdown("#### Ejemplos numéricos con sus datos")
    for ex in itp.numeric_examples(res, yu):
        st.markdown(f"- {ex}")
    st.markdown("##### Cálculo paso a paso (caso típico: cada X en su mediana)")
    med = res.medians()
    p = res.predict(med).iloc[0]
    terms = " + ".join(f"({res.coef(c):,.4g})({med[c]:,.4g})" for c in res.x_names)
    st.markdown(f"Ŷ = {res.b0:,.4g} + {terms} = **{p['y_est_modelo']:,.4g}**")
    st.markdown(f"Estimación: **{p['y_est']:,.4g}** · IC {int(round((1 - res.alpha) * 100))}% del promedio [{p['ic_inf']:,.4g}; {p['ic_sup']:,.4g}] · "
                f"IP de un caso individual [{p['ip_inf']:,.4g}; {p['ip_sup']:,.4g}].")

    st.markdown("#### Qué tan confiable es el modelo")
    for t in itp.explain_metrics(res, cm.test_metrics):
        st.markdown(f"- {t}")
    C.callout("Los coeficientes describen <b>asociaciones</b> observadas en los datos históricos, cada una con las demás variables constantes. Para afirmar que "
              "mover una X <i>causa</i> un cambio en Y se necesita evidencia adicional (experimento, conocimiento del proceso).", "edu", "Correlación no es causalidad")
    C.next_step("Continuar: simular escenarios ▶", "m10_simulacion")
