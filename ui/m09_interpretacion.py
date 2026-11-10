"""Módulo 9 · Interpretación del modelo y sus coeficientes."""
from __future__ import annotations

import numpy as np
import streamlit as st

from core import interpretation as itp
from ui import components as C
from ui import state


def chosen_model():
    """Devuelve (CandidateModel, ModelingData) del modelo elegido, o detiene la página."""
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
           "Qué significan el intercepto y la pendiente en términos del negocio, con su versión técnica y en lenguaje sencillo, y ejemplos con sus propios datos.")
    cm, md = chosen_model()
    res = cm.res
    xu, yu = ss.units.get("x", ""), ss.units.get("y", "")
    rec = ss.cmp.recommendation
    if rec.status == "no confiable":
        C.callout("El modelo recomendado fue calificado como <b>no confiable</b>. La siguiente interpretación describe la recta ajustada, pero no debe usarse para decidir.", "bad")
    elif cm.key != rec.best_key:
        C.callout("Está interpretando un modelo distinto del recomendado.", "warn")
    st.markdown("#### Ecuación final")
    st.markdown(f"<div class='eq'>{res.equation()}</div>", unsafe_allow_html=True)
    if res.form != "lin-lin":
        st.caption(f"Forma funcional **{res.form}** (X: {res.x_transform or 'sin transformar'}; Y: {res.y_transform or 'sin transformar'}). "
                   "La ecuación está en la escala del modelo.")
    st.caption("Escriba las unidades en la barra lateral para enriquecer las frases (p. ej. «millones de pesos», «km»).")

    ii, ss_ = itp.interpret_intercept(res, xu, yu), itp.interpret_slope(res, xu, yu)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Intercepto (β0)")
        t1, t2 = st.tabs(["Técnica", "En lenguaje sencillo"])
        t1.markdown(ii["tecnica"])
        t1.markdown(f"**Viabilidad de la interpretación:** {ii['viabilidad']}")
        t2.markdown(ii["sencilla"])
        C.kpis([("β0", f"{res.b0:,.4g}", f"IC: [{res.ci_b0[0]:,.4g}; {res.ci_b0[1]:,.4g}]")])
    with c2:
        st.markdown("##### Pendiente (β1)")
        t1, t2 = st.tabs(["Técnica", "En lenguaje sencillo"])
        t1.markdown(ss_["tecnica"])
        t1.markdown(f"**Magnitud:** {ss_['magnitud']}")
        t2.markdown(itp.plain_language(res, xu, yu))
        C.kpis([("β1", f"{res.b1:,.4g}", f"IC: [{res.ci_b1[0]:,.4g}; {res.ci_b1[1]:,.4g}]")])
    C.callout(f"<b>Dirección:</b> relación {ss_['direccion']}. <b>Efecto:</b> {ss_['efecto']}", "ok" if ss_["significativa"] else "warn", "Mensaje central")

    st.markdown("#### Ejemplos numéricos con sus datos")
    for ex in itp.numeric_examples(res, xu, yu):
        st.markdown(f"- {ex}")
    st.markdown("##### Cálculo paso a paso")
    x0 = float(np.median(res.x_obs))
    p = res.predict([x0]).iloc[0]
    if res.x_transform is None:
        st.latex(rf"\hat Y=\beta_0+\beta_1X={res.b0:,.4g}+({res.b1:,.4g})\times{x0:,.4g}={res.b0 + res.b1 * x0:,.4g}")
    else:
        st.markdown(f"Con X = {x0:,.4g} (transformada: {res.x_transform}) → Ŷ (escala del modelo) = {p['y_est_modelo']:,.4g}.")
    st.markdown(f"Estimación en la escala original: **{p['y_est']:,.4g}** · IC {int((1 - res.alpha) * 100)}% del promedio [{p['ic_inf']:,.4g}; {p['ic_sup']:,.4g}] · "
                f"IP de una observación [{p['ip_inf']:,.4g}; {p['ip_sup']:,.4g}].")

    st.markdown("#### Qué tan confiable es el modelo")
    for t in itp.explain_metrics(res, cm.test_metrics):
        st.markdown(f"- {t}")
    C.callout("Los coeficientes describen una <b>asociación</b> observada en los datos históricos. Para afirmar que mover X <i>causa</i> un cambio en Y se necesita "
              "evidencia adicional (experimento, conocimiento del proceso, control de otros factores).", "edu", "Correlación no es causalidad")
    C.next_step("Continuar: simular escenarios ▶", "m10_simulacion")
