"""Módulo 7 · Construcción de modelos de regresión lineal simple (uno por candidata)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.comparison import run_comparison
from ui import charts
from ui import components as C
from ui import state


def build() -> None:
    ss = st.session_state
    md, err = state.modeling()
    exp = {k: v["sign"] for k, v in ss.x_meta.items() if v.get("sign")}
    known = {k: v["known"] for k, v in ss.x_meta.items()}
    ss.cmp = run_comparison(md, ss.x_cands, alpha=state.alpha(), include_transforms=ss.include_transforms, expected_sign=exp,
                            known_in_advance=known, leak_cleared=ss.leak_cleared, declared_factors=state.declared_factors())
    ss.cmp_sig = state.cmp_signature()
    if ss.chosen not in ss.cmp.models:
        ss.chosen = ss.cmp.recommendation.best_key


def model_points(cm, md):
    """Observaciones de prueba (en unidades originales) del modelo."""
    if not md.test_idx:
        return None, None
    te = md.df.loc[md.test_idx]
    col = cm.x_col
    if cm.x_transform and col not in te.columns:
        from core.cleaning import TRANSFORMS
        te = te.assign(**{col: TRANSFORMS[cm.x_transform][1](te[cm.x_source].to_numpy(float))})
    te = te.dropna(subset=[col, md.y_source])
    return te[cm.x_source].to_numpy(float), te[md.y_source].to_numpy(float)


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 7", "Modelos de regresión lineal simple",
           "Un modelo independiente por cada variable candidata: Ŷ = β0 + β1·X, estimado por mínimos cuadrados con los datos de entrenamiento.")
    C.need(state.has_data(), "Primero cargue los datos.", "m02_datos")
    C.need(bool(ss.y), "Defina la variable Y.", "m05_y")
    C.need(bool(ss.x_cands), "Seleccione al menos una variable explicativa.", "m06_x")
    md, err = state.modeling()
    if md is None:
        st.error(err)
        st.stop()
    if not state.cmp_is_current():
        with st.spinner("Construyendo y diagnosticando los modelos…"):
            build()
    cmp = ss.cmp
    for s in cmp.skipped:
        st.warning(f"Modelo omitido — {s}")
    if not cmp.models:
        st.error("No se pudo construir ningún modelo. Revise la cantidad de datos completos de cada variable.")
        st.stop()

    st.markdown(f"#### {len(cmp.models)} modelos construidos para estimar «{md.y_source}»"
                + (f" (modelado como {md.y_col}, resultados en la escala original)" if md.y_transform else ""))
    rows = []
    for k, m in cmp.models.items():
        r = m.res
        rows.append({"modelo": k, "n": r.n, "intercepto β0": r.b0, "pendiente β1": r.b1, "EE(β1)": r.se_b1, "t": r.t_b1, "p-valor": r.p_b1,
                     "IC inf β1": r.ci_b1[0], "IC sup β1": r.ci_b1[1], "r": r.r, "R²": r.r2, "R² ajust.": r.r2_adj, "RMSE": r.rmse, "MAE": r.mae,
                     "F": r.F, "p(F)": r.p_F})
    tab = pd.DataFrame(rows).sort_values("R²", ascending=False)
    C.table(tab, 4)
    st.caption("Esta tabla es descriptiva del AJUSTE en entrenamiento. La elección del mejor modelo se hace en el módulo 8, con datos de prueba y supuestos.")

    keys = list(cmp.models)
    default = ss.chosen if ss.chosen in keys else cmp.recommendation.best_key
    key = st.selectbox("Explorar el modelo de", keys, index=keys.index(default) if default in keys else 0, key="m7_sel")
    cm = cmp.models[key]
    r = cm.res
    st.markdown(f"<div class='eq'>{r.equation()}</div>", unsafe_allow_html=True)
    if cm.x_transform or md.y_transform:
        st.caption(f"Forma funcional: **{r.form}**. X transformada: {cm.x_transform or 'no'} · Y transformada: {md.y_transform or 'no'}. "
                   "La ecuación está en la escala del modelo; las estimaciones se devuelven a la escala original.")
    C.kpis([("R²", f"{r.r2:.3f}", f"ajustado {r.r2_adj:.3f}"), ("Correlación r", f"{r.r:+.3f}"), ("RMSE", f"{r.rmse:,.4g}", f"MAE {r.mae:,.4g}"),
            ("F", f"{r.F:,.1f}", f"p = {r.p_F:.2g}"), ("Pendiente β1", f"{r.b1:,.4g}", f"p = {r.p_b1:.2g}")])
    C.table(r.coef_table(), 5)
    st.caption(f"IC al {int((1 - r.alpha) * 100)}%. EE_robusto_HC3: error estándar que no supone varianza constante (útil si hay heterocedasticidad).")

    x_t, y_t = model_points(cm, md)
    t1, t2, t3, t4 = st.tabs(["Recta e intervalos", "Observado vs. estimado", "Residuos", "Tabla de datos del modelo"])
    with t1:
        c1, c2, c3 = st.columns(3)
        ci = c1.toggle("Intervalo de confianza (promedio)", True)
        pi = c2.toggle("Intervalo de predicción (observación)", True)
        tp = c3.toggle("Mostrar datos de prueba", bool(md.test_idx)) if md.test_idx else False
        C.chart(charts.regression_chart(r, x_t if tp else None, y_t if tp else None, show_ci=ci, show_pi=pi), "m7_reg")
        C.how_to_read("La **banda oscura** (intervalo de confianza) dice dónde está la recta *promedio* con 95% de confianza: es estrecha porque promediamos muchos casos. "
                      "La **banda clara** (intervalo de predicción) dice dónde cae *una observación individual*: siempre es más ancha, porque suma la variabilidad natural. "
                      "Los puntos naranja son datos de prueba: si caen dentro de la banda clara en la proporción esperada, el modelo generaliza bien.")
    with t2:
        C.chart(charts.observed_vs_fitted(r, x_t, y_t), "m7_ovf")
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            C.chart(charts.residuals_vs_fitted(r), "m7_rvf")
        with c2:
            C.chart(charts.residual_hist(r), "m7_rh")
    with t4:
        pr = r.predict(r.x_obs)
        d = pd.DataFrame({"fila": r.index, r.x_source: r.x_obs, f"{r.y_source} real": r.y_obs, "estimado": pr["y_est"].to_numpy(),
                          "residuo (escala modelo)": r.resid, "IC inf": pr["ic_inf"], "IC sup": pr["ic_sup"], "IP inf": pr["ip_inf"], "IP sup": pr["ip_sup"],
                          "Cook": r.cooks})
        C.table(d, 5)
        st.download_button("⬇ Descargar tabla (CSV)", d.to_csv(index=False).encode("utf-8-sig"), f"modelo_{key}.csv", "text/csv")

    with st.expander("📐 Fórmulas utilizadas"):
        st.latex(r"\hat Y=\beta_0+\beta_1X,\quad \beta_1=\frac{S_{xy}}{S_{xx}},\quad \beta_0=\bar y-\beta_1\bar x")
        st.latex(r"\hat\sigma^2=\frac{SSE}{n-2},\quad SE(\beta_1)=\frac{\hat\sigma}{\sqrt{S_{xx}}},\quad SE(\beta_0)=\hat\sigma\sqrt{\tfrac1n+\tfrac{\bar x^2}{S_{xx}}}")
        st.latex(r"t=\frac{\beta}{SE(\beta)}\sim t_{n-2},\qquad F=\frac{SSR/1}{SSE/(n-2)}=t^2,\qquad R^2=1-\frac{SSE}{SST},\quad R^2_{aj}=1-(1-R^2)\frac{n-1}{n-2}")
        st.latex(r"RMSE=\sqrt{\tfrac1n\sum e_i^2},\quad MAE=\tfrac1n\sum |e_i|,\quad IC(\beta)=\beta\pm t_{1-\alpha/2,\,n-2}\,SE(\beta)")
        st.latex(r"IC(E[Y|x_0]):\ \hat y_0\pm t\,\hat\sigma\sqrt{\tfrac1n+\tfrac{(x_0-\bar x)^2}{S_{xx}}}\qquad IP(Y_0):\ \hat y_0\pm t\,\hat\sigma\sqrt{1+\tfrac1n+\tfrac{(x_0-\bar x)^2}{S_{xx}}}")
    C.next_step("Continuar: evaluar y elegir el mejor modelo ▶", "m08_evaluacion")
