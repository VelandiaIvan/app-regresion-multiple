"""Componentes visuales reutilizables."""
from __future__ import annotations

import pandas as pd
import streamlit as st

CSS = """
<style>
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--blue:#0f766e;--blue-d:#134e4a;--surface:#fcfcfb}
.block-container{padding-top:2.2rem;max-width:1280px}
h1,h2,h3{letter-spacing:-0.01em}
.hero{background:linear-gradient(120deg,#042f2e 0%,#115e59 55%,#0f766e 100%);color:#fff;padding:22px 26px;border-radius:14px;margin-bottom:18px}
.hero h1{color:#fff;font-size:1.6rem;margin:0 0 4px 0;padding:0}
.hero p{color:#d5f0ea;margin:0;font-size:.95rem}
.hero .tag{display:inline-block;background:rgba(255,255,255,.16);border-radius:20px;padding:2px 10px;font-size:.75rem;margin-bottom:8px}
.callout{padding:12px 16px;border-radius:10px;margin:10px 0;border-left:5px solid var(--blue);background:#ecf7f4;color:var(--ink);font-size:.93rem;line-height:1.5}
.callout b.t{display:block;margin-bottom:2px}
.callout.warn{border-color:#eda100;background:#fff7e0}
.callout.bad{border-color:#d03b3b;background:#fdecec}
.callout.ok{border-color:#0ca30c;background:#eaf7ea}
.callout.edu{border-color:#7c3aed;background:#f3effd}
.kpi{background:var(--surface);border:1px solid var(--grid);border-radius:12px;padding:12px 14px}
.kpi .l{color:var(--ink2);font-size:.78rem;text-transform:uppercase;letter-spacing:.03em}
.kpi .v{font-size:1.5rem;font-weight:600;color:var(--ink);line-height:1.2}
.kpi .s{color:var(--muted);font-size:.78rem}
.eq{font-size:1.25rem;background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:12px 16px;margin:8px 0;font-family:Cambria Math,Georgia,serif}
.pill{display:inline-block;padding:1px 9px;border-radius:12px;font-size:.78rem;font-weight:600}
.pill.ok{background:#dcf3dc;color:#065f06}.pill.warn{background:#ffefc2;color:#7a5200}.pill.bad{background:#fbdcdc;color:#8c1c1c}.pill.na{background:#eceae4;color:#52514e}
[data-testid="stSidebar"]{background:#06302e}
[data-testid="stSidebar"] *{color:#e6f4f1}
[data-testid="stSidebar"] .stRadio label{padding:3px 0}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.2)}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] *{color:#0b0b0b}
[data-testid="stSidebar"] .stButton button{background:#e6f4f1;border:0}
[data-testid="stSidebar"] .stButton button, [data-testid="stSidebar"] .stButton button *{color:#06302e !important;font-weight:600}
div[data-testid="stTabs"] button[aria-selected="true"]{color:var(--blue-d);font-weight:600}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(tag: str, title: str, sub: str) -> None:
    st.markdown(f"<div class='hero'><span class='tag'>{tag}</span><h1>{title}</h1><p>{sub}</p></div>", unsafe_allow_html=True)


def callout(text: str, kind: str = "info", title: str | None = None) -> None:
    t = f"<b class='t'>{title}</b>" if title else ""
    st.markdown(f"<div class='callout {kind}'>{t}{text}</div>", unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "") -> str:
    return f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div><div class='s'>{sub}</div></div>"


def kpis(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        label, value, *rest = it
        c.markdown(kpi(label, value, rest[0] if rest else ""), unsafe_allow_html=True)


def pill(text: str, kind: str) -> str:
    return f"<span class='pill {kind}'>{text}</span>"


def need(cond: bool, msg: str, target: str | None = None) -> None:
    if cond:
        return
    st.warning(msg)
    if target:
        st.button("Ir al módulo requerido ▶", on_click=goto, args=(target,), key=f"need_{target}")
    st.stop()


def goto(page: str) -> None:
    st.session_state["nav"] = page


def next_step(label: str, target: str) -> None:
    st.divider()
    st.button(label, on_click=goto, args=(target,), type="primary", key=f"next_{target}")


def chart(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, key=key)


def table(df: pd.DataFrame, digits: int = 4, **kw) -> None:
    """Tabla con cifras significativas; los números muy pequeños (p-valores) salen en notación científica."""
    cfg = {c: st.column_config.NumberColumn(format=f"%.{digits}g") for c in df.columns if pd.api.types.is_float_dtype(df[c])}
    st.dataframe(df, hide_index=True, column_config=cfg, **kw)


def how_to_read(text: str, title: str = "¿Cómo se interpreta?") -> None:
    with st.expander(f"💡 {title}"):
        st.markdown(text)


def fmt(v: float, d: int = 4) -> str:
    return f"{v:,.{d}g}"
