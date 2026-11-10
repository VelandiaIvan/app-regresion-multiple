"""Gráficos Plotly de la aplicación.

Lineamientos (método dataviz): marcas finas, cuadrícula sólida de línea tenue, un solo eje Y, leyenda siempre que
hay ≥ 2 series, color azul para la serie principal y gris para lo secundario, tinta (no color de serie) para el texto.
Paleta categórica validada (espacios 1–3): azul, naranja, aguamarina. Cada gráfico tiene su tabla gemela en la interfaz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from core.diagnostics import qq_points
from core.regression import SLRResult
from core.simulation import curve

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_DARK, GRAY_MARK, RED = "#184f95", "#c3c2b7", "#e34948"
DIVERGING = [[0, BLUE], [0.5, "#f0efec"], [1, RED]]
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _base(fig: go.Figure, title: str | None = None, height: int = 380, x_title: str | None = None,
          y_title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template=None, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family=FONT, size=12, color=INK2), margin=dict(l=56, r=24, t=48 if title else 24, b=48),
        title=dict(text=title, x=0.01, font=dict(size=14, color=INK)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
        showlegend=legend, hoverlabel=dict(bgcolor="#ffffff", font=dict(family=FONT, color=INK, size=12), bordercolor=BASE),
        hovermode="closest",
    )
    fig.update_xaxes(title=dict(text=x_title, font=dict(color=INK2)), gridcolor=GRID, gridwidth=1, linecolor=BASE,
                     zerolinecolor=BASE, tickfont=dict(color=MUTED), showline=True, ticks="outside", tickcolor=BASE)
    fig.update_yaxes(title=dict(text=y_title, font=dict(color=INK2)), gridcolor=GRID, gridwidth=1, linecolor=BASE,
                     zerolinecolor=BASE, tickfont=dict(color=MUTED))
    return fig


def _lowess(y, x, frac: float) -> np.ndarray:
    """Suavizado LOWESS tolerante a muchos valores repetidos de X (p. ej. días enteros): descarta puntos no finitos."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    x, y = np.asarray(x, float), np.asarray(y, float)
    delta = 0.01 * float(np.ptp(x)) if len(x) > 500 else 0.0          # acelera con muchas observaciones
    with np.errstate(all="ignore"):
        lw = lowess(y, x, frac=frac, delta=delta)
    return lw[np.isfinite(lw).all(axis=1)]


def _pts(x, y, name, color, text=None, size=8, opacity=0.8, hover=None) -> go.Scatter:
    kw = dict(x=x, y=y, mode="markers", name=name, text=text, hovertemplate=hover,
              marker=dict(size=size, color=color, opacity=opacity, line=dict(color=SURFACE, width=1)))
    return go.Scattergl(**kw) if len(x) > 3000 else go.Scatter(**kw)


# ----------------------------------------------------------------------------- AED
def histogram(s: pd.Series, name: str, bins: int | None = None) -> go.Figure:
    x = s.dropna()
    fig = go.Figure(go.Histogram(x=x, nbinsx=bins or None, marker=dict(color=BLUE, line=dict(color=SURFACE, width=1)),
                                 opacity=0.9, name=name, hovertemplate="%{x}<br>frecuencia: %{y}<extra></extra>"))
    for val, lab, col in ((x.mean(), "media", INK), (x.median(), "mediana", ORANGE)):
        fig.add_vline(x=val, line=dict(color=col, width=2), annotation_text=f"{lab} {val:,.4g}", annotation_font=dict(color=INK2, size=11),
                      annotation_position="top")
    return _base(fig, f"Distribución de {name}", 320, name, "frecuencia", legend=False)


def boxplot(s: pd.Series, name: str) -> go.Figure:
    fig = go.Figure(go.Box(x=s.dropna(), name=name, boxpoints="outliers", marker=dict(color=BLUE, size=7, line=dict(color=SURFACE, width=1)),
                           line=dict(color=BLUE_DARK, width=2), fillcolor=_rgba(BLUE, 0.25), orientation="h", boxmean=True))
    return _base(fig, f"Diagrama de caja · {name}", 220, name, None, legend=False)


def bars(df: pd.DataFrame, x: str, y: str, title: str, horizontal: bool = True, highlight: str | None = None,
         fmt: str = ",.4g", height: int | None = None, y_title: str | None = None) -> go.Figure:
    d = df.copy()
    colors = [BLUE if highlight is None or v == highlight else GRAY_MARK for v in d[x]]
    if horizontal:
        d = d.iloc[::-1]
        colors = colors[::-1]
        fig = go.Figure(go.Bar(y=d[x].astype(str), x=d[y], orientation="h", marker=dict(color=colors, cornerradius=4),
                               text=d[y].map(lambda v: format(v, fmt)), textposition="outside", textfont=dict(color=INK2, size=11),
                               hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
        fig.update_yaxes(automargin=True)
        h = height or max(220, 34 * len(d) + 90)
        return _base(fig, title, h, y_title or y, None, legend=False)
    fig = go.Figure(go.Bar(x=d[x].astype(str), y=d[y], marker=dict(color=colors, cornerradius=4),
                           hovertemplate="%{x}: %{y:" + fmt + "}<extra></extra>"))
    return _base(fig, title, height or 320, None, y_title or y, legend=False)


def missing_matrix(df: pd.DataFrame, max_rows: int = 300) -> go.Figure:
    m = df.isna()
    if len(m) > max_rows:
        m = m.iloc[np.linspace(0, len(m) - 1, max_rows).astype(int)]
    fig = go.Figure(go.Heatmap(z=m.astype(int).T.values, x=m.index.astype(str), y=m.columns, showscale=False,
                               colorscale=[[0, GRID], [1, BLUE_DARK]], xgap=0, ygap=2,
                               hovertemplate="fila %{x}<br>%{y}: %{customdata}<extra></extra>",
                               customdata=np.where(m.T.values, "faltante", "dato presente")))
    fig.update_xaxes(showticklabels=False, title="filas (muestra ordenada)")
    return _base(fig, "Mapa de datos faltantes (oscuro = falta)", 60 + 26 * len(m.columns) + 70, legend=False)


def corr_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    z = corr.values
    fig = go.Figure(go.Heatmap(z=z, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=DIVERGING, xgap=2, ygap=2,
                               colorbar=dict(title="r", thickness=12, outlinewidth=0),
                               text=np.round(z, 2) if len(corr) <= 9 else None, texttemplate="%{text}" if len(corr) <= 9 else None,
                               hovertemplate="%{y} × %{x}<br>%{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(tickangle=-35, automargin=True)
    return _base(fig, title, max(340, 46 * len(corr) + 120), legend=False)


def scatter_eda(df: pd.DataFrame, x: str, y: str, loess: bool = True, color_by: str | None = None) -> go.Figure:
    d = df[[x, y] + ([color_by] if color_by else [])].dropna()
    fig = go.Figure()
    if color_by and d[color_by].nunique() <= 3:
        for cat, col in zip(d[color_by].unique(), (BLUE, ORANGE, AQUA)):
            g = d[d[color_by] == cat]
            fig.add_trace(_pts(g[x], g[y], str(cat), col, hover=f"{x}: %{{x:,.4g}}<br>{y}: %{{y:,.4g}}<extra>{cat}</extra>"))
    else:
        fig.add_trace(_pts(d[x], d[y], "observaciones", BLUE, hover=f"{x}: %{{x:,.4g}}<br>{y}: %{{y:,.4g}}<extra></extra>"))
    if len(d) > 5 and d[x].nunique() > 3:
        sl = stats.linregress(d[x], d[y])
        xs = np.linspace(d[x].min(), d[x].max(), 60)
        fig.add_trace(go.Scatter(x=xs, y=sl.intercept + sl.slope * xs, mode="lines", name="recta (MCO)", line=dict(color=INK, width=2),
                                 hoverinfo="skip"))
        if loess and len(d) >= 20:
            lw = _lowess(d[y], d[x], 0.6 if len(d) < 200 else 0.35)
            fig.add_trace(go.Scatter(x=lw[:, 0], y=lw[:, 1], mode="lines", name="tendencia suavizada (LOWESS)",
                                     line=dict(color=ORANGE, width=2), hoverinfo="skip"))
    return _base(fig, f"{y} vs {x}", 400, x, y)


def time_series(d: pd.DataFrame, date_col: str, y: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d[date_col], y=d[y], mode="lines", name=y, line=dict(color=BLUE, width=1.5, shape="linear")))
    if "media_móvil" in d:
        fig.add_trace(go.Scatter(x=d[date_col], y=d["media_móvil"], mode="lines", name="media móvil", line=dict(color=INK, width=2)))
    return _base(fig, f"{y} en el tiempo", 340, None, y)


# ----------------------------------------------------------------------------- modelo
def regression_chart(res: SLRResult, x_test=None, y_test=None, test_label: str = "prueba (no usada para ajustar)",
                     show_ci: bool = True, show_pi: bool = True, pad: float = 0.0, highlight_x: float | None = None,
                     title: str | None = None, level: float | None = None, extra_points: list | None = None) -> go.Figure:
    a = res.alpha if level is None else level
    lvl = int(round((1 - a) * 100))
    cv = curve(res, 160, pad, a)
    fig = go.Figure()
    if show_pi:
        fig.add_trace(go.Scatter(x=np.r_[cv["x"], cv["x"][::-1]], y=np.r_[cv["ip_sup"], cv["ip_inf"][::-1]], fill="toself",
                                 fillcolor=_rgba(BLUE, 0.10), line=dict(width=0), name=f"intervalo de predicción {lvl}% (una observación)",
                                 hoverinfo="skip"))
    if show_ci:
        fig.add_trace(go.Scatter(x=np.r_[cv["x"], cv["x"][::-1]], y=np.r_[cv["ic_sup"], cv["ic_inf"][::-1]], fill="toself",
                                 fillcolor=_rgba(BLUE, 0.30), line=dict(width=0), name=f"intervalo de confianza {lvl}% (promedio)",
                                 hoverinfo="skip"))
    fig.add_trace(_pts(res.x_obs, res.y_obs, f"entrenamiento (n={res.n})", BLUE,
                       hover=f"{res.x_source}: %{{x:,.4g}}<br>{res.y_source}: %{{y:,.4g}}<extra></extra>"))
    if x_test is not None and len(x_test):
        fig.add_trace(_pts(np.asarray(x_test), np.asarray(y_test), test_label, ORANGE,
                           hover=f"{res.x_source}: %{{x:,.4g}}<br>{res.y_source}: %{{y:,.4g}}<extra>prueba</extra>"))
    fig.add_trace(go.Scatter(x=cv["x"], y=cv["y_est"], mode="lines", name="recta estimada", line=dict(color=INK, width=2.5),
                             hovertemplate=f"{res.x_source}: %{{x:,.4g}}<br>Ŷ: %{{y:,.4g}}<extra></extra>"))
    for i, pt in enumerate(sorted(extra_points or [], key=lambda p: p["x"])):
        pos = ("top center", "bottom center")[i % 2]          # alterna arriba/abajo para que las etiquetas no se solapen
        fig.add_trace(go.Scatter(x=[pt["x"]], y=[pt["y"]], mode="markers+text", name=pt["name"], text=[pt["name"]], textposition=pos,
                                 textfont=dict(color=INK, size=11),
                                 marker=dict(size=13, color=pt.get("color", INK), line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"{pt['name']}<br>X: %{{x:,.4g}}<br>Ŷ: %{{y:,.4g}}<extra></extra>"))
    if pad > 0:
        fig.add_vrect(x0=cv["x"].min(), x1=res.x_min, fillcolor=_rgba(ORANGE, 0.07), line_width=0)
        fig.add_vrect(x0=res.x_max, x1=cv["x"].max(), fillcolor=_rgba(ORANGE, 0.07), line_width=0,
                      annotation_text="extrapolación", annotation_position="top right", annotation_font=dict(color=INK2, size=11))
    return _base(fig, title or f"{res.y_source} vs {res.x_source}", 440, res.x_source, res.y_source)


def observed_vs_fitted(res: SLRResult, x_test=None, y_test=None) -> go.Figure:
    est = res.y_back(res.fitted)
    fig = go.Figure()
    lo = float(min(res.y_obs.min(), est.min()))
    hi = float(max(res.y_obs.max(), est.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ajuste perfecto (y = x)", line=dict(color=BASE, width=2), hoverinfo="skip"))
    fig.add_trace(_pts(est, res.y_obs, "entrenamiento", BLUE, hover="estimado: %{x:,.4g}<br>real: %{y:,.4g}<extra></extra>"))
    if x_test is not None and len(x_test):
        yt = res.predict(np.asarray(x_test))["y_est"].to_numpy()
        fig.add_trace(_pts(yt, np.asarray(y_test), "prueba", ORANGE, hover="estimado: %{x:,.4g}<br>real: %{y:,.4g}<extra>prueba</extra>"))
    return _base(fig, "Valores observados vs. estimados", 380, "estimado por el modelo", "observado")


def residuals_vs_fitted(res: SLRResult) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=BASE, width=2))
    fig.add_trace(_pts(res.fitted, res.resid, "residuos", BLUE, hover="ajustado: %{x:,.4g}<br>residuo: %{y:,.4g}<extra></extra>"))
    if res.n >= 20:
        lw = _lowess(res.resid, res.fitted, 0.6)
        fig.add_trace(go.Scatter(x=lw[:, 0], y=lw[:, 1], mode="lines", name="tendencia (LOWESS): debe ser plana", line=dict(color=ORANGE, width=2.5),
                                 hoverinfo="skip"))
    return _base(fig, "Residuos vs. valores ajustados", 360, "valor ajustado (escala del modelo)", "residuo")


def residual_hist(res: SLRResult) -> go.Figure:
    z = res.std_resid
    fig = go.Figure(go.Histogram(x=z, histnorm="probability density", nbinsx=min(30, max(8, res.n // 6)), name="residuos estandarizados",
                                 marker=dict(color=BLUE, line=dict(color=SURFACE, width=1)), opacity=0.9))
    xs = np.linspace(min(z.min(), -3.5), max(z.max(), 3.5), 120)
    fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs), mode="lines", name="normal teórica", line=dict(color=ORANGE, width=2.5), hoverinfo="skip"))
    return _base(fig, "Histograma de residuos", 340, "residuo estandarizado", "densidad")


def qq_chart(res: SLRResult) -> go.Figure:
    th, sm = qq_points(res)
    lo, hi = float(min(th.min(), sm.min())), float(max(th.max(), sm.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="normalidad perfecta", line=dict(color=BASE, width=2), hoverinfo="skip"))
    fig.add_trace(_pts(th, sm, "residuos", BLUE, hover="teórico: %{x:.2f}<br>observado: %{y:.2f}<extra></extra>"))
    return _base(fig, "Gráfico Q-Q normal", 340, "cuantiles teóricos", "cuantiles de los residuos")


def scale_location(res: SLRResult) -> go.Figure:
    fig = go.Figure(_pts(res.fitted, np.sqrt(np.abs(res.std_resid)), "√|residuo estandarizado|", BLUE,
                         hover="ajustado: %{x:,.4g}<br>√|e|: %{y:.3f}<extra></extra>"))
    if res.n >= 20:
        lw = _lowess(np.sqrt(np.abs(res.std_resid)), res.fitted, 0.6)
        fig.add_trace(go.Scatter(x=lw[:, 0], y=lw[:, 1], mode="lines", name="tendencia (debe ser plana)", line=dict(color=ORANGE, width=2.5), hoverinfo="skip"))
    return _base(fig, "Dispersión de residuos (escala-ubicación)", 340, "valor ajustado", "√|residuo estandarizado|")


def cooks_chart(res: SLRResult) -> go.Figure:
    thr = 4 / res.n
    colors = [ORANGE if c > thr else BLUE for c in res.cooks]
    fig = go.Figure(go.Bar(x=np.arange(res.n), y=res.cooks, marker=dict(color=colors), customdata=res.index,
                           hovertemplate="fila %{customdata}<br>Cook: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=thr, line=dict(color=INK2, width=1.5), annotation_text="4/n", annotation_font=dict(color=INK2, size=11))
    return _base(fig, "Influencia: distancia de Cook (naranja = alta)", 300, "observación", "distancia de Cook", legend=False)


# ----------------------------------------------------------------------------- simulación y comparación
def scenario_chart(res: SLRResult, scenarios: dict, sim: dict | None, pad: float = 0.25) -> go.Figure:
    colors = {"pesimista": ORANGE, "base": BLUE, "optimista": AQUA}
    pts = []
    for k, xv in scenarios.items():
        p = res.predict([xv]).iloc[0]
        pts.append(dict(name=k, x=xv, y=p["y_est"], color=colors.get(k, INK2)))
    if sim:
        pts.append(dict(name="simulado", x=sim["x_new"], y=sim["y_new"], color=INK))
    return regression_chart(res, show_ci=True, show_pi=True, pad=pad, extra_points=pts, title="Escenarios sobre la recta estimada")


def sensitivity_chart(tab: pd.DataFrame, y_label: str) -> go.Figure:
    d = tab.iloc[::-1]
    fig = go.Figure(go.Bar(y=d["escenario"], x=d["dif_absoluta"], orientation="h", marker=dict(color=BLUE, cornerradius=4),
                           text=d["dif_absoluta"].map(lambda v: f"{v:+,.4g}"), textposition="outside", textfont=dict(color=INK2, size=11),
                           hovertemplate="%{y}: %{x:+,.4g}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=BASE, width=2))
    fig.update_yaxes(automargin=True)
    return _base(fig, f"Cambio estimado en {y_label} según la variación de X", max(260, 44 * len(d) + 90), f"cambio absoluto en {y_label}", None, legend=False)


def score_chart(table: pd.DataFrame, best_key: str | None) -> go.Figure:
    d = table.sort_values("puntaje", ascending=False).head(12)
    return bars(d, "modelo", "puntaje", "Puntaje multicriterio (0–100)", highlight=best_key, fmt=".0f", y_title="puntaje")


def r2_vs_score(table: pd.DataFrame) -> go.Figure:
    d = table.head(15)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d["modelo"][::-1], x=d["R²"][::-1], orientation="h", name="R² (entrenamiento)", marker=dict(color=GRAY_MARK, cornerradius=4)))
    fig.add_trace(go.Bar(y=d["modelo"][::-1], x=(d["R²_prueba"].fillna(d["R²"]))[::-1], orientation="h", name="R² con datos de prueba",
                         marker=dict(color=BLUE, cornerradius=4)))
    fig.update_layout(barmode="group")
    fig.update_yaxes(automargin=True)
    return _base(fig, "R² en entrenamiento vs. prueba", max(280, 46 * len(d) + 100), "R²", None)
