"""Módulo 3 · Tratamiento y preparación de datos (con bitácora de cambios)."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from core import cleaning, profiling
from core.cleaning import METHOD_LABELS, TRANSFORMS
from ui import charts
from ui import components as C
from ui import state


def _add(kind, column=None, method="", **params):
    st.session_state.plan.add(kind, column, method, **params)
    st.toast("Paso agregado al plan de limpieza", icon="✅")
    st.rerun()


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 3", "Tratamiento y preparación de datos",
           "Cada decisión queda registrada en un plan de limpieza reproducible. Las estadísticas (media, mediana, límites) se calcularán solo con los datos de "
           "entrenamiento al modelar, para evitar fuga de información. La variable a estimar (Y) nunca se imputa.")
    C.need(state.has_data(), "Primero cargue los datos.", "m02_datos")
    raw = ss.loaded.df
    plan = ss.plan
    cls = profiling.classify_columns(raw)
    if ss.y:
        C.callout(f"Variable a estimar: <b>{ss.y}</b> — protegida: no se imputa ni se recorta.", "info")

    tabs = st.tabs(["1 · Faltantes", "2 · Imputación", "3 · Duplicados, inconsistencias y atípicos", "4 · Variables: transformar y codificar",
                    "5 · Bitácora y resultado"])

    # ------------------------------------------------------------------ faltantes
    with tabs[0]:
        ms = cleaning.missing_summary(raw)
        n_miss = ms[ms["n_faltantes"] > 0]
        C.kpis([("Variables con faltantes", f"{len(n_miss)} de {raw.shape[1]}"),
                ("Filas con algún faltante", f"{raw.isna().any(axis=1).mean():.1%}"),
                ("Filas completas", f"{raw.notna().all(axis=1).sum():,}")])
        if n_miss.empty:
            st.success("No hay datos faltantes.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                C.table(ms, 3)
            with c2:
                C.chart(charts.bars(n_miss, "variable", "pct_faltantes", "% de datos faltantes por variable", fmt=".1f", y_title="% faltante"), "m3_missbar")
            C.chart(charts.missing_matrix(raw[n_miss["variable"]]), "m3_missmat")
            co = cleaning.missing_correlation(raw[n_miss["variable"]])
            if not co.empty:
                with st.expander("¿Los faltantes ocurren juntos? (correlación entre indicadores de faltante)"):
                    C.chart(charts.corr_heatmap(co, "Correlación entre faltantes"), "m3_cocorr")
                    st.caption("Valores altos indican que los datos faltan en las mismas filas (p. ej. un sensor que falló). "
                               "Si el faltante depende de otra variable, imputar con un solo valor puede sesgar.")
            C.how_to_read("Si pocas filas (< 5%) tienen faltantes, eliminarlas suele ser lo más honesto. Si faltan muchos datos y no son aleatorios, "
                          "cualquier imputación introduce supuestos: documéntelo en la bitácora.")

    # ------------------------------------------------------------------ imputación
    with tabs[1]:
        ms = cleaning.missing_summary(raw)
        cols = ms.loc[ms["n_faltantes"] > 0, "variable"].tolist()
        if not cols:
            st.success("No hay variables con datos faltantes que imputar.")
        else:
            c1, c2 = st.columns([1, 3])
            sel = c1.selectbox("Variable", cols, key="m3_imp_var")
            adv = cleaning.recommend_imputation(raw, sel, ordered_by=ss.date_col, is_target=(sel == ss.y))
            kind_cls = "bad" if adv.recommended in {"drop_rows", "drop_column"} else "ok"
            with c2:
                C.callout(adv.reason, kind_cls, f"Recomendación para «{sel}»: {METHOD_LABELS.get(adv.recommended, adv.recommended)} "
                                                 f"({adv.pct_missing:.1f}% faltante, tipo {adv.kind})")
            if adv.evaluation is not None and not adv.evaluation.empty:
                st.markdown("**Prueba de métodos:** se ocultan al azar valores conocidos, se imputan y se mide qué tan bien se recuperan "
                            "(media de 8 repeticiones).")
                C.table(adv.evaluation.drop(columns=["clave"]), 4)
                st.caption("«desv_est_conservada_%» cerca de 100 indica que el método no aplana la variabilidad; los métodos de un solo valor (media, mediana, moda) la reducen.")
            if adv.kind == "numérica":
                options = ["mean", "median", "mode", "ffill", "interpolate", "constant", "drop_rows", "drop_column", "none"]
            else:
                options = ["mode", "constant", "drop_rows", "drop_column", "none"]
            default = adv.recommended if adv.recommended in options else options[0]
            m1, m2, m3 = st.columns([2, 1, 1])
            choice = m1.selectbox("Método a aplicar", options, index=options.index(default), format_func=lambda k: METHOD_LABELS.get(k, k), key=f"m3_imp_m_{sel}")
            val = m2.text_input("Valor constante", "Sin dato" if adv.kind != "numérica" else "0", key=f"m3_imp_v_{sel}") if choice == "constant" else None
            if sel == ss.y and choice not in {"drop_rows", "none"}:
                st.error("La variable a estimar (Y) no puede imputarse ni eliminarse como columna. Elija «Eliminar filas».")
            elif m3.button("Agregar al plan", type="primary", key="m3_imp_add"):
                if choice == "none":
                    for i in reversed(range(len(plan.steps))):
                        if plan.steps[i].kind in {"impute", "drop_column"} and plan.steps[i].column == sel:
                            plan.remove(i)
                    st.rerun()
                elif choice == "drop_column":
                    _add("drop_column", sel, "drop_column")
                else:
                    value = float(val) if (choice == "constant" and adv.kind == "numérica") else val
                    _add("impute", sel, choice, value=value) if choice == "constant" else _add("impute", sel, choice)
            if st.button("⚡ Aplicar la recomendación a todas las variables con faltantes", key="m3_imp_all"):
                for c in cols:
                    a = cleaning.recommend_imputation(raw, c, ordered_by=ss.date_col, is_target=(c == ss.y))
                    if a.recommended == "drop_column" and c != ss.y:
                        plan.add("drop_column", c, "drop_column")
                    elif a.recommended in {"mean", "median", "mode", "ffill", "interpolate", "drop_rows"}:
                        plan.add("impute", c, a.recommended)
                    elif a.recommended == "constant":
                        plan.add("impute", c, "constant", value=a.params.get("value", "Sin dato"))
                st.toast("Recomendaciones aplicadas al plan", icon="⚡")
                st.rerun()
            # antes / después
            prev = state.preview()
            if sel in prev.df.columns and any(s.column == sel for s in plan.steps):
                st.markdown("**Antes y después del tratamiento**")
                C.table(cleaning.compare_before_after(raw, prev.df, [sel]), 4)
                if pd.api.types.is_numeric_dtype(raw[sel]):
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=raw[sel].dropna(), name="antes (observados)", marker=dict(color=charts.GRAY_MARK), opacity=0.9))
                    fig.add_trace(go.Histogram(x=prev.df[sel].dropna(), name="después", marker=dict(color=charts.BLUE), opacity=0.55))
                    fig.update_layout(barmode="overlay")
                    C.chart(charts._base(fig, f"Distribución de {sel}: antes vs. después", 300, sel, "frecuencia"), "m3_beforeafter")
                    st.caption("Un pico nuevo en el centro indica que la imputación acumuló muchos valores en un solo punto (reduce la varianza).")

    # ------------------------------------------------------------------ duplicados / inconsistencias / atípicos
    with tabs[2]:
        dup = profiling.duplicates_info(raw)
        st.markdown("##### Duplicados")
        if dup["n_duplicados"]:
            st.warning(f"{dup['n_duplicados']} filas duplicadas exactas.")
            if st.button("Eliminar duplicados (conservar la primera aparición)"):
                _add("drop_duplicates", None, "drop_duplicates")
        else:
            st.caption("Sin duplicados.")
        st.markdown("##### Inconsistencias")
        inc = profiling.find_inconsistencies(raw)
        if inc.empty:
            st.caption("Sin inconsistencias evidentes.")
        else:
            C.table(inc, 3)
            idx = st.selectbox("Aplicar sugerencia a", range(len(inc)), format_func=lambda i: f"{inc.loc[i, 'variable']} — {inc.loc[i, 'problema']}")
            row = inc.loc[idx]
            if st.button("Aplicar sugerencia", key="m3_inc_apply"):
                p, v = row["problema"], row["variable"]
                if p == "negativos improbables":
                    _add("fix_nan", v, "a_faltante", rule="negativos")
                elif p == "valores infinitos":
                    _add("fix_nan", v, "a_faltante", rule="inf")
                elif p == "posible código de dato faltante":
                    _add("fix_nan", v, "a_faltante", rule="valor", value=float(re.search(r"valor (-?\d+)", row["detalle"]).group(1)))
                elif p == "porcentaje fuera de 0–100":
                    _add("fix_nan", v, "a_faltante", rule="rango", min=0, max=100)
                elif p == "categorías duplicadas":
                    _add("normalize_labels", v, "normalizar")
                else:
                    st.info("Esta inconsistencia requiere revisión manual; no hay corrección automática segura.")
        st.markdown("##### Valores atípicos")
        out = profiling.outliers_summary(raw)
        if out.empty:
            st.caption("Sin variables numéricas.")
        else:
            C.table(out[["variable", "atípicos_IQR", "pct_IQR", "atípicos_z>3", "atípicos_MAD", "límite_inferior", "límite_superior"]], 4)
            c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
            v = c1.selectbox("Variable", out["variable"], key="m3_out_var")
            labels = {"flag": "Marcar (columna indicadora)", "winsorize": "Recortar a los límites", "to_nan": "Convertir a faltante (se imputa después)",
                      "remove": "Eliminar filas (solo entrenamiento al modelar)"}
            m = c2.selectbox("Tratamiento", list(labels), format_func=labels.get, key="m3_out_m")
            k = c3.number_input("k del IQR", 1.0, 5.0, 1.5, 0.5, key="m3_out_k")
            if v == ss.y and m in {"winsorize", "to_nan"}:
                st.error("Y está protegida: use «marcar» o «eliminar filas».")
            elif c4.button("Agregar", key="m3_out_add"):
                _add("outlier", v, m, k=k)
            C.how_to_read("**No elimine atípicos por reflejo.** Un pico de ventas por una promoción es información real que el modelo debe conocer. "
                          "Elimine o recorte solo si hay motivos de negocio (error de captura, evento irrepetible). Cuando se pide eliminar filas, "
                          "el modelo solo las quita del conjunto de entrenamiento: la prueba se evalúa con datos tal como llegan en la realidad.")

    # ------------------------------------------------------------------ transformar y codificar
    with tabs[3]:
        st.markdown("##### Variables constantes o con poca variabilidad")
        bad = cls[(cls["constante"]) | (cls["baja_variabilidad"])]
        if bad.empty:
            st.caption("Ninguna.")
        else:
            C.table(bad[["variable", "n_únicos", "constante", "baja_variabilidad", "observaciones"]], 3)
            drop = st.multiselect("Excluir del análisis", bad["variable"].tolist())
            if drop and st.button("Excluir variables seleccionadas"):
                for c in drop:
                    plan.add("drop_column", c, "drop_column")
                st.rerun()
        st.markdown("##### Transformaciones sugeridas (por asimetría)")
        sug = cleaning.suggest_transformations(raw)
        if not sug.empty:
            C.table(sug, 3)
        c1, c2, c3 = st.columns([2, 2, 1])
        num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c]) and not pd.api.types.is_bool_dtype(raw[c])]
        tv = c1.selectbox("Variable a transformar", num_cols, key="m3_tr_var")
        tm = c2.selectbox("Transformación", list(TRANSFORMS), format_func=lambda k: f"{TRANSFORMS[k][0]} — {TRANSFORMS[k][2]}", key="m3_tr_m")
        if c3.button("Crear columna", key="m3_tr_add"):
            _add("transform", tv, tm)
        st.caption("Se crea una columna nueva (la original se conserva). Los modelos con ln/√ de X se evalúan automáticamente en el módulo 7.")
        st.markdown("##### Codificación de variables categóricas")
        cats = cls[(cls["grupo"] == "categórica") & (cls["n_únicos"] >= 2) & (cls["n_únicos"] <= 12)]["variable"].tolist()
        if not cats:
            st.caption("No hay variables categóricas con entre 2 y 12 categorías.")
        else:
            c1, c2, c3 = st.columns([2, 2, 1])
            ev = c1.selectbox("Variable categórica", cats, key="m3_en_var")
            levels = sorted(map(str, raw[ev].dropna().unique()))
            em = c2.selectbox("Método", ["onehot", "binary", "ordinal"],
                              format_func={"onehot": "Variables indicadoras (una por categoría)", "binary": "Binaria (1 = categoría elegida)",
                                           "ordinal": "Ordinal (usted define el orden)"}.get, key="m3_en_m")
            extra = {}
            if em == "binary":
                extra["positive"] = st.selectbox("Categoría = 1", levels, key="m3_en_pos")
            elif em == "ordinal":
                extra["order"] = st.multiselect("Orden de menor a mayor", levels, default=levels, key="m3_en_ord")
            if c3.button("Codificar", key="m3_en_add"):
                _add("encode", ev, em, **extra)
            st.caption("En la regresión, una variable indicadora 0/1 como X mide la diferencia promedio de Y entre dos grupos, con las demás variables constantes.")

    # ------------------------------------------------------------------ bitácora
    with tabs[4]:
        if not len(plan):
            st.info("El plan de limpieza está vacío: los datos se usarán tal como llegaron (las filas sin Y siempre se descartan al modelar).")
        else:
            st.markdown("##### Plan de limpieza (se ejecuta en orden lógico: estructura → atípicos → imputación → transformación → codificación)")
            st.dataframe(pd.DataFrame(plan.records()), hide_index=True)
            c1, c2, c3 = st.columns([1, 1, 3])
            n = c1.number_input("Paso n.º", 1, len(plan), 1, key="m3_del_n")
            if c2.button("Quitar paso"):
                plan.remove(int(n) - 1)
                st.rerun()
            if c3.button("Vaciar plan"):
                plan.clear()
                st.rerun()
            prev = state.preview()
            st.markdown("##### Resultado de aplicar el plan a todos los datos")
            C.kpis([("Filas antes → después", f"{len(raw):,} → {len(prev.df):,}"), ("Columnas antes → después", f"{raw.shape[1]} → {prev.df.shape[1]}"),
                    ("Faltantes antes → después", f"{int(raw.isna().sum().sum()):,} → {int(prev.df.isna().sum().sum()):,}")])
            st.dataframe(prev.log, hide_index=True)
            changed = [c for c in raw.columns if c in prev.df.columns and (raw[c].isna().sum() != prev.df[c].isna().sum())]
            if changed:
                st.markdown("**Comparación antes / después**")
                C.table(cleaning.compare_before_after(raw, prev.df, changed), 4)
            st.markdown("**Vista del conjunto preparado**")
            st.dataframe(prev.df.head(20), hide_index=True)
            st.download_button("⬇ Descargar datos preparados (CSV)", prev.df.to_csv(index=False).encode("utf-8-sig"), "datos_preparados.csv", "text/csv")
        C.how_to_read("El plan es una *receta*, no un archivo modificado: se aplica de nuevo cada vez, así los resultados son reproducibles. "
                      "Al construir los modelos, la receta se ejecuta después de separar entrenamiento y prueba, calculando medias, medianas y límites solo con entrenamiento.",
                      "¿Por qué un plan y no datos ya limpios?")
    C.next_step("Continuar: análisis exploratorio ▶", "m04_eda")
