"""Módulo 7 · Construcción de modelos de regresión lineal MÚLTIPLE (combinaciones de 2 o más variables)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import interpretation_multi as itp
from core.selection import run_multi_comparison
from ui import charts
from ui import components as C
from ui import state


def build() -> None:
    ss = st.session_state
    md, _ = state.modeling()
    exp = {k: v["sign"] for k, v in ss.x_meta.items() if v.get("sign")}
    known = {k: v["known"] for k, v in ss.x_meta.items()}
    ss.cmp = run_multi_comparison(md, ss.x_cands, alpha=state.alpha(), expected_sign=exp, known_in_advance=known,
                                  leak_cleared=ss.leak_cleared, declared_factors=state.declared_factors())
    ss.cmp_sig = state.cmp_signature()
    if ss.chosen not in ss.cmp.models:
        ss.chosen = ss.cmp.recommendation.best_key


def test_points(cm, md):
    """(Y real, Y estimado) del conjunto de prueba, en unidades originales."""
    if not md.test_idx:
        return None, None
    te = md.df.loc[md.test_idx, list(dict.fromkeys(cm.x_cols + [md.y_source]))].dropna()
    if te.empty:
        return None, None
    return te[md.y_source].to_numpy(float), cm.res.y_back(cm.res.predict_model_scale(te[cm.x_cols]))


def short(key: str, n: int = 70) -> str:
    return key if len(key) <= n else key[: n - 1] + "…"


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 7", "Modelos de regresión lineal múltiple",
           "Cada modelo combina DOS O MÁS variables: Ŷ = β0 + β1·X1 + β2·X2 + … La aplicación prueba las combinaciones posibles con los datos de "
           "entrenamiento y conserva las mejores para compararlas.")
    C.need(state.has_data(), "Primero cargue los datos.", "m02_datos")
    C.need(bool(ss.y), "Defina la variable Y.", "m05_y")
    C.need(len(ss.x_cands) >= 2, "Seleccione al menos dos variables explicativas.", "m06_x")
    md, err = state.modeling()
    if md is None:
        st.error(err)
        st.stop()
    if not state.cmp_is_current():
        with st.spinner("Probando combinaciones de variables y diagnosticando los modelos…"):
            build()
    cmp = ss.cmp
    for s in cmp.skipped:
        st.warning(f"Aviso — {s}")
    if not cmp.models:
        st.error("No se pudo construir ningún modelo con 2 o más variables. Revise la selección del módulo 6 y los datos faltantes.")
        st.stop()

    st.markdown(f"#### {len(cmp.models)} modelos finalistas para estimar «{md.y_source}»")
    st.caption(f"Variables en juego: {', '.join(cmp.usable)}. Todos los modelos tienen al menos 2 variables explicativas.")
    rows = []
    for k, m in cmp.models.items():
        r = m.res
        rows.append({"modelo": k, "variables": r.k, "n": r.n, "R²": r.r2, "R² ajust.": r.r2_adj, "RMSE": m.train_metrics["RMSE"],
                     "MAE": m.train_metrics["MAE"], "F": r.F, "p(F)": r.p_F, "coef. significativos": f"{r.n_signif}/{r.k}",
                     "VIF máx.": max(r.vif.values())})
    C.table(pd.DataFrame(rows).sort_values("R² ajust.", ascending=False), 4)
    st.caption("Tabla descriptiva del AJUSTE en entrenamiento. La elección del mejor modelo se hace en el módulo 8, con datos de prueba y supuestos.")

    if cmp.singles is not None and len(cmp.singles):
        best = cmp.models.get(cmp.recommendation.best_key) or next(iter(cmp.models.values()))
        with st.expander("📊 ¿Cuánto se gana al combinar variables? (cada X por sí sola vs. el modelo múltiple)", expanded=True):
            C.chart(charts.singles_vs_multi_chart(cmp.singles, best.res.r2, f"{best.res.k} variables"), "m7_singles")
            top = cmp.singles.iloc[0]
            st.caption(f"La mejor variable por sí sola («{top['variable']}») explica {top['R² sola']:.0%} de Y; "
                       f"el modelo múltiple sugerido explica {best.res.r2:.0%}.")

    keys = list(cmp.models)
    default = ss.chosen if ss.chosen in keys else cmp.recommendation.best_key
    key = st.selectbox("Explorar el modelo", keys, index=keys.index(default) if default in keys else 0, key="m7_sel", format_func=short)
    cm = cmp.models[key]
    r = cm.res
    st.markdown(f"<div class='eq'>{r.equation()}</div>", unsafe_allow_html=True)
    C.kpis([("R² ajustado", f"{r.r2_adj:.3f}", f"R² {r.r2:.3f}"), ("Variables", f"{r.k}", f"{r.n_signif} significativas"),
            ("RMSE", f"{cm.train_metrics['RMSE']:,.4g}", f"MAE {cm.train_metrics['MAE']:,.4g}"),
            ("F global", f"{r.F:,.1f}", f"p = {r.p_F:.2g}"), ("VIF máximo", f"{max(r.vif.values()):.2f}", "menor que 5 es lo deseable")])
    C.table(r.coef_table(), 5)
    st.caption(f"IC al {int(round((1 - r.alpha) * 100))}%. Cada coeficiente es el efecto de su variable con las demás constantes. "
               "β estandarizado: permite comparar el peso de variables con unidades distintas. EE_robusto_HC3: no supone varianza constante.")

    y_t, yh_t = test_points(cm, md)
    t1, t2, t3, t4 = st.tabs(["Peso de cada variable", "Observado vs. estimado", "Residuos", "Tabla de datos del modelo"])
    with t1:
        C.chart(charts.coef_importance_chart(itp.importance_table(r)), "m7_imp")
        C.how_to_read("Barras **verdes**: la variable sube Y. **Ámbar**: la baja. **Gris**: su efecto no es estadísticamente significativo. "
                      "El largo de la barra es el coeficiente estandarizado: cuántas desviaciones estándar se mueve Y por cada desviación estándar de X.")
    with t2:
        C.chart(charts.observed_vs_fitted_multi(r, y_t, yh_t), "m7_ovf")
        C.how_to_read("Con varias X no se puede dibujar «la recta»; este gráfico la reemplaza: cada punto compara lo que pasó (eje vertical) con lo que el "
                      "modelo estimó (horizontal). Mientras más cerca de la diagonal, mejor. Los puntos ámbar son datos de prueba que el modelo no vio.")
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            C.chart(charts.residuals_vs_fitted(r), "m7_rvf")
        with c2:
            C.chart(charts.residual_hist(r), "m7_rh")
    with t4:
        pr = r.predict(pd.DataFrame(r.X, columns=r.x_names))
        d = pd.DataFrame({"fila": r.index, **{c: r.X[:, j] for j, c in enumerate(r.x_names)}, f"{r.y_name} real": r.y_obs,
                          "estimado": pr["y_est"].to_numpy(), "residuo": r.y_obs - pr["y_est"].to_numpy(), "IC inf": pr["ic_inf"],
                          "IC sup": pr["ic_sup"], "IP inf": pr["ip_inf"], "IP sup": pr["ip_sup"], "Cook": r.cooks})
        C.table(d, 5)
        st.download_button("⬇ Descargar tabla (CSV)", d.to_csv(index=False).encode("utf-8-sig"), "modelo_multiple.csv", "text/csv")

    with st.expander("📐 Fórmulas utilizadas"):
        st.latex(r"\hat Y=\beta_0+\beta_1X_1+\beta_2X_2+\dots+\beta_kX_k,\qquad \hat\beta=(X^\top X)^{-1}X^\top y")
        st.latex(r"\hat\sigma^2=\frac{SSE}{n-k-1},\quad SE(\beta_j)=\hat\sigma\sqrt{[(X^\top X)^{-1}]_{jj}},\quad t_j=\frac{\beta_j}{SE(\beta_j)}\sim t_{n-k-1}")
        st.latex(r"F=\frac{SSR/k}{SSE/(n-k-1)},\qquad R^2=1-\frac{SSE}{SST},\quad R^2_{aj}=1-(1-R^2)\frac{n-1}{n-k-1},\quad VIF_j=\frac{1}{1-R_j^2}")
        st.latex(r"IC(E[Y|x_0]):\ \hat y_0\pm t\,\hat\sigma\sqrt{x_0^\top(X^\top X)^{-1}x_0}\qquad IP(Y_0):\ \hat y_0\pm t\,\hat\sigma\sqrt{1+x_0^\top(X^\top X)^{-1}x_0}")
    C.next_step("Continuar: evaluar y elegir el mejor modelo ▶", "m08_evaluacion")
