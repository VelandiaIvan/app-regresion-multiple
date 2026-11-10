"""Estado compartido de la sesión y cálculos derivados con caché por firma."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from core import loader
from core.cleaning import CleaningPlan, apply_plan
from core.comparison import SplitSpec, prepare_modeling_data
from core.context import BusinessContext
from sample_data import generate as G

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"


def init() -> None:
    ss = st.session_state
    ss.setdefault("ctx", BusinessContext())
    ss.setdefault("loaded", None)
    ss.setdefault("data_id", 0)
    ss.setdefault("plan", CleaningPlan())
    ss.setdefault("y", None)
    ss.setdefault("y_transform", None)
    ss.setdefault("date_col", None)
    ss.setdefault("split", {"method": "aleatoria", "test_size": 0.2, "seed": 42})
    ss.setdefault("x_cands", [])
    ss.setdefault("x_meta", {})
    ss.setdefault("leak_cleared", set())
    ss.setdefault("include_transforms", True)
    ss.setdefault("cmp", None)
    ss.setdefault("cmp_sig", None)
    ss.setdefault("chosen", None)
    ss.setdefault("units", {"x": "", "y": ""})
    ss.setdefault("conf", 95)
    ss.setdefault("_prev_sig", None)
    ss.setdefault("_prev", None)
    ss.setdefault("_md_sig", None)
    ss.setdefault("_md", None)
    ss.setdefault("_md_err", None)
    ss.setdefault("source_name", "")
    ss.setdefault("visited", set())


def alpha() -> float:
    return 1 - st.session_state.get("conf", 95) / 100


def has_data() -> bool:
    return st.session_state.get("loaded") is not None


def reset_analysis() -> None:
    ss = st.session_state
    ss.plan = CleaningPlan()
    ss.y = None
    ss.y_transform = None
    ss.x_cands = []
    ss.x_meta = {}
    ss.leak_cleared = set()
    ss.cmp = None
    ss.cmp_sig = None
    ss.chosen = None
    ss._prev_sig = ss._md_sig = None


def set_loaded(res: loader.LoadResult, name: str) -> None:
    ss = st.session_state
    ss.loaded = res
    ss.data_id += 1
    ss.source_name = name
    ss.date_col = None
    for c in res.df.columns:
        if pd.api.types.is_datetime64_any_dtype(res.df[c]):
            ss.date_col = c
            break
    reset_analysis()


def load_sample(key: str) -> None:
    fname, fn, _ = G.DATASETS[key]
    path = SAMPLE_DIR / fname
    try:
        if not path.exists():
            G.write_all(SAMPLE_DIR)
        res = loader.load_file(path)
    except OSError:  # servidor de solo lectura: se generan los datos en memoria (misma semilla, mismo resultado)
        res = loader.from_dataframe(fn(), fname)
    set_loaded(res, f"{fname} (datos sintéticos de ejemplo)")
    c = G.CONTEXTS[key]
    ss = st.session_state
    ss.ctx = BusinessContext(**{k: v for k, v in c.items() if k not in {"y", "date"}})
    ss.date_col = c["date"] if c["date"] in res.df.columns else ss.date_col
    ss.example_y = c["y"]
    ss.pop("ctx_widgets_loaded", None)
    # los widgets de texto del módulo 1 se enlazan por clave: se sobrescriben para reflejar el ejemplo
    for fld in ("sector", "actividad", "core", "problema", "objetivo_y", "factores", "restricciones", "pregunta"):
        ss[f"ctx_{fld}"] = getattr(ss.ctx, fld)


def preview():
    """Plan aplicado a todos los datos (exploración descriptiva)."""
    ss = st.session_state
    if not has_data():
        return None
    sig = (ss.data_id, ss.plan.signature(), ss.y)
    if ss._prev_sig != sig:
        ss._prev = apply_plan(ss.loaded.df, ss.plan, protect=[ss.y] if ss.y else [])
        ss._prev_sig = sig
    return ss._prev


def split_spec() -> SplitSpec:
    ss = st.session_state
    return SplitSpec(ss.split["method"], ss.split["test_size"], ss.split["seed"], ss.date_col)


def modeling():
    """ModelingData (partición anti-fuga) o (None, mensaje de error)."""
    ss = st.session_state
    if not has_data() or not ss.y:
        return None, "Defina primero los datos y la variable Y."
    sig = (ss.data_id, ss.plan.signature(), ss.y, ss.y_transform, tuple(sorted(ss.split.items())), ss.date_col)
    if ss._md_sig != sig:
        try:
            ss._md = prepare_modeling_data(ss.loaded.df, ss.plan, ss.y, split_spec(), ss.y_transform)
            ss._md_err = None
        except Exception as exc:  # mensaje amigable
            ss._md, ss._md_err = None, str(exc)
        ss._md_sig = sig
    return ss._md, ss._md_err


def declared_factors() -> set:
    """Columnas que el usuario mencionó como factores explicativos en el Módulo 1 (conocimiento de dominio)."""
    from core.context import mention_score
    ss = st.session_state
    if not ss.ctx.factores.strip() or not has_data():
        return set()
    return {c for c in ss.loaded.df.columns if mention_score(c, ss.ctx.factores) >= 0.5}


def cmp_signature():
    ss = st.session_state
    return (ss._md_sig, tuple(ss.x_cands), repr(sorted((k, tuple(v.items())) for k, v in ss.x_meta.items())),
            tuple(sorted(ss.leak_cleared)), ss.include_transforms, ss.conf, ss.ctx.factores)


def cmp_is_current() -> bool:
    ss = st.session_state
    return ss.cmp is not None and ss.cmp_sig == cmp_signature()


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            if d is not None and len(d):
                d.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()
