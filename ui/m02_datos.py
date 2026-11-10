"""Módulo 1 · Carga y exploración inicial de datos (el flujo empieza aquí)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import loader, profiling
from sample_data import generate as G
from ui import charts
from ui import components as C
from ui import state

def _load_example() -> None:
    state.load_sample(st.session_state["sample_sel"])


SEP = {"Automático": None, "Coma ( , )": ",", "Punto y coma ( ; )": ";", "Tabulador": "\t", "Barra ( | )": "|"}


def render() -> None:
    ss = st.session_state
    C.hero("Módulo 1", "Carga de datos", "Cargue un archivo TXT, CSV, XLS o XLSX y la aplicación hace el resto: detecta el formato, "
           "explica de qué tratan los datos y propone qué variable estimar. No necesita describir su negocio.")

    up = st.file_uploader("Archivo de datos", type=["csv", "txt", "tsv", "xls", "xlsx", "xlsm"], key="uploader")
    with st.expander("Opciones avanzadas de lectura"):
        c1, c2, c3 = st.columns(3)
        sep_label = c1.selectbox("Separador (TXT/CSV)", list(SEP))
        enc = c2.selectbox("Codificación (TXT/CSV)", ["Automática", "utf-8", "latin-1", "cp1252"])
        sheets = ss.get("_sheets") or []
        sheet = c3.selectbox("Hoja (Excel)", sheets, disabled=len(sheets) < 2) if sheets else None
    if up is not None:
        sig = (up.name, up.size, sep_label, enc, sheet)
        if ss.get("_file_sig") != sig:
            try:
                res = loader.load_file(up, up.name, sheet=sheet, sep=SEP[sep_label], encoding=None if enc == "Automática" else enc)
                state.set_loaded(res, up.name)
                ss["_sheets"] = res.sheets
                ss["_file_sig"] = sig
                if res.sheets and sheet is None:
                    st.rerun()
            except Exception as exc:
                st.error(f"No se pudo leer el archivo: {exc}")
                st.stop()

    with st.expander(f"🧪 ¿No tiene datos a la mano? Pruebe con un ejemplo ({len(G.DATASETS)} casos sintéticos)", expanded=not state.has_data()):
        key = st.selectbox("Caso de ejemplo", list(G.DATASETS), key="sample_sel")
        st.caption(G.DATASETS[key][2])
        st.button("Cargar ejemplo", on_click=_load_example, type="primary")
        if state.has_data() and "sintéticos" in ss.source_name:
            st.success(f"Cargado: {ss.source_name}")

    if not state.has_data():
        C.callout("Cargue un archivo con una fila por caso y una columna por variable. Para una regresión múltiple necesita una columna numérica "
                  "a estimar (Y) y al menos dos columnas más que la expliquen (X).", "info")
        return

    res = ss.loaded
    df = res.df
    for w in res.warnings:
        st.warning(w)
    prof = profiling.profile(df)
    cls = prof["columns"]

    n_cells = df.shape[0] * df.shape[1]
    C.kpis([("Registros", f"{df.shape[0]:,}"), ("Columnas", f"{df.shape[1]}"),
            ("Duplicados", f"{prof['duplicates']['n_duplicados']:,}", f"{prof['duplicates']['pct']:.1f}% de las filas"),
            ("Celdas faltantes", f"{df.isna().sum().sum() / max(n_cells, 1):.1%}", f"{int(df.isna().sum().sum()):,} celdas"),
            ("Formato", res.fmt, " · ".join(f"{k}: {v}" for k, v in res.meta.items()))])
    st.markdown("#### Primeras cinco filas")
    st.dataframe(prof["head"], hide_index=True)

    n_num, n_cat, n_date = (int((cls["grupo"] == g).sum()) for g in ("numérica", "categórica", "fecha"))
    st.caption(f"Clasificación automática: **{n_num} numéricas**, **{n_cat} categóricas**, **{n_date} fechas**, "
               f"{int(cls['grupo'].isin(['texto', 'identificador']).sum())} texto/identificador.")
    date_cols = cls.loc[cls["grupo"] == "fecha", "variable"].tolist()
    if date_cols:
        idx = date_cols.index(ss.date_col) + 1 if ss.date_col in date_cols else 0
        pick = st.selectbox("Columna de fecha para ordenar en el tiempo (activa pruebas de autocorrelación y la partición temporal)",
                            ["(ninguna)"] + date_cols, index=idx)
        ss.date_col = None if pick == "(ninguna)" else pick

    t1, t2, t3, t4, t5, t6 = st.tabs(["Variables y tipos", "Duplicados", "Valores atípicos", "Inconsistencias", "Estadísticas descriptivas", "Registro de conversiones"])
    with t1:
        C.table(cls[["variable", "dtype", "grupo", "detalle", "n_válidos", "n_faltantes", "pct_faltantes", "n_únicos", "observaciones"]], 3)
        st.caption("Las columnas marcadas como identificador o texto libre se excluyen automáticamente de los candidatos a X e Y.")
    with t2:
        d = prof["duplicates"]
        if d["n_duplicados"]:
            C.callout(f"Hay <b>{d['n_duplicados']}</b> filas duplicadas exactas ({d['pct']:.1f}%). Duplicar observaciones infla artificialmente "
                      "la confianza del modelo. Puede eliminarlas en el módulo 3.", "warn")
            st.dataframe(d["ejemplos"], hide_index=True)
        else:
            st.success("No se detectaron registros duplicados.")
    with t3:
        out = prof["outliers"]
        if out.empty:
            st.info("No hay variables numéricas para evaluar.")
        else:
            C.table(out, 4)
            v = st.selectbox("Ver diagrama de caja de", out["variable"], key="m2_box")
            C.chart(charts.boxplot(df[v], v))
            C.how_to_read("Un valor es atípico si queda fuera de [Q1 − 1,5·RIC; Q3 + 1,5·RIC] (criterio de Tukey), si su z-score supera 3 o si su z-robusto "
                          "(basado en la mediana y el MAD) supera 3,5. **Atípico no significa error**: puede ser un caso real (una promoción, un pico). "
                          "Decida en el módulo 3 si marcarlo, recortarlo o eliminarlo.")
    with t4:
        inc = prof["inconsistencies"]
        if inc.empty:
            st.success("No se detectaron valores inconsistentes evidentes.")
        else:
            C.table(inc, 3)
    with t5:
        C.table(prof["descriptives"], 4)
    with t6:
        if res.conversions:
            st.dataframe(pd.DataFrame(res.conversions), hide_index=True)
        else:
            st.caption("No fue necesario convertir tipos: el archivo ya traía números y fechas reconocibles.")
    C.next_step("Continuar: ver qué dicen sus datos ▶", "m01_contexto")
