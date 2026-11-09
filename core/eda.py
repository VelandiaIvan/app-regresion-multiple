"""Análisis exploratorio de datos (AED): distribuciones, correlaciones, linealidad y patrones."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .profiling import outlier_masks


def strength_label(r: float) -> str:
    a = abs(r)
    if np.isnan(a):
        return "no calculable"
    return ("nula" if a < 0.1 else "débil" if a < 0.3 else "moderada" if a < 0.5
            else "fuerte" if a < 0.7 else "muy fuerte")


def direction_label(r: float) -> str:
    if np.isnan(r) or abs(r) < 0.1:
        return "sin dirección clara"
    return "positiva" if r > 0 else "negativa"


# ----------------------------------------------------------------------------
# distribución y frecuencias
# ----------------------------------------------------------------------------
def distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.select_dtypes(include="number").columns:
        x = df[c].dropna()
        if len(x) < 8 or x.nunique() <= 2:
            continue
        sk, ku = float(stats.skew(x)), float(stats.kurtosis(x))
        sample = x.sample(min(len(x), 5000), random_state=0)
        p = float(stats.shapiro(sample).pvalue) if sample.nunique() > 3 else np.nan
        form = ("aprox. simétrica" if abs(sk) < 0.5 else "asimetría moderada" if abs(sk) < 1
                else "asimetría fuerte a la derecha" if sk > 0 else "asimetría fuerte a la izquierda")
        rows.append({"variable": c, "n": len(x), "asimetría": sk, "curtosis_exceso": ku,
                     "Shapiro_p": p, "forma": form,
                     "¿normal (5%)?": "sí" if (not np.isnan(p) and p > 0.05) else "no"})
    return pd.DataFrame(rows)


def categorical_frequencies(df: pd.DataFrame, col: str, top: int = 20) -> pd.DataFrame:
    vc = df[col].value_counts(dropna=False)
    out = pd.DataFrame({"categoría": vc.index.astype(str), "frecuencia": vc.values})
    out["porcentaje"] = out["frecuencia"] / max(len(df), 1) * 100
    out["acumulado_%"] = out["porcentaje"].cumsum()
    return out.head(top)


def category_effect(df: pd.DataFrame, cat: str, y: str, min_group: int = 3) -> dict:
    """Diferencias de Y entre categorías (ANOVA y Kruskal-Wallis)."""
    d = df[[cat, y]].dropna()
    groups = [g[y].to_numpy() for _, g in d.groupby(cat) if len(g) >= min_group]
    table = d.groupby(cat)[y].agg(["count", "mean", "median", "std"]).reset_index().sort_values("mean", ascending=False)
    out = {"table": table, "anova_p": np.nan, "kruskal_p": np.nan, "eta2": np.nan}
    if len(groups) >= 2:
        out["anova_p"] = float(stats.f_oneway(*groups).pvalue)
        out["kruskal_p"] = float(stats.kruskal(*groups).pvalue)
        grand = d[y].mean()
        ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_t = float(((d[y] - grand) ** 2).sum())
        out["eta2"] = ss_b / ss_t if ss_t > 0 else np.nan
    return out


# ----------------------------------------------------------------------------
# linealidad
# ----------------------------------------------------------------------------
def _r2(X: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    sse, sst = float(res @ res), float(((y - y.mean()) ** 2).sum())
    return (1 - sse / sst if sst > 0 else np.nan), sse, beta


def linearity_screen(x: pd.Series, y: pd.Series) -> dict:
    """Compara forma lineal con alternativas (cuadrática, ln x, √x, 1/x) con el mismo Y."""
    d = pd.concat([x, y], axis=1).dropna()
    d.columns = ["x", "y"]
    n = len(d)
    out = {"n": n, "formas": pd.DataFrame(), "mejor": "lineal", "mejora": 0.0, "p_cuadratico": np.nan,
           "no_lineal": False, "mensaje": ""}
    if n < 12 or d["x"].nunique() < 4:
        out["mensaje"] = "Datos insuficientes para evaluar la forma de la relación."
        return out
    xv, yv = d["x"].to_numpy(float), d["y"].to_numpy(float)
    one = np.ones(n)
    forms = {"lineal": np.column_stack([one, xv])}
    forms["cuadrática"] = np.column_stack([one, xv, xv ** 2])
    if (xv > 0).all():
        forms["ln(x)"] = np.column_stack([one, np.log(xv)])
        forms["1/x"] = np.column_stack([one, 1 / xv])
    if (xv >= 0).all():
        forms["√x"] = np.column_stack([one, np.sqrt(xv)])
    rows = []
    for name, X in forms.items():
        r2, sse, _ = _r2(X, yv)
        k = X.shape[1]
        adj = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else np.nan
        rows.append({"forma": name, "R²": r2, "R²_ajustado": adj, "SSE": sse, "k": k})
    tab = pd.DataFrame(rows).sort_values("R²_ajustado", ascending=False).reset_index(drop=True)
    lin_adj = float(tab.loc[tab["forma"] == "lineal", "R²_ajustado"].iloc[0])
    best = tab.iloc[0]
    # prueba F anidada: lineal vs cuadrática
    sse1 = float(tab.loc[tab["forma"] == "lineal", "SSE"].iloc[0])
    sse2 = float(tab.loc[tab["forma"] == "cuadrática", "SSE"].iloc[0])
    if sse2 > 0 and n > 3:
        F = (sse1 - sse2) / (sse2 / (n - 3))
        out["p_cuadratico"] = float(1 - stats.f.cdf(F, 1, n - 3))
    gain = float(best["R²_ajustado"] - lin_adj)
    out.update({"formas": tab.drop(columns=["SSE", "k"]), "mejor": best["forma"], "mejora": gain})
    out["no_lineal"] = bool(best["forma"] != "lineal" and gain > 0.03 and (out["p_cuadratico"] < 0.01 or best["forma"] in {"ln(x)", "√x", "1/x"}))
    if out["no_lineal"]:
        out["mensaje"] = (f"La forma «{best['forma']}» explica {gain * 100:.1f} puntos más de R² ajustado que la recta: "
                          "hay indicios de curvatura. Considere transformar X o usar un modelo no lineal.")
    else:
        out["mensaje"] = "Ninguna forma alternativa mejora de forma relevante a la recta: el supuesto de linealidad es razonable."
    return out


# ----------------------------------------------------------------------------
# correlaciones
# ----------------------------------------------------------------------------
def pair_stats(x: pd.Series, y: pd.Series) -> dict:
    d = pd.concat([x, y], axis=1).dropna()
    n = len(d)
    out = {"n": n, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    if n < 4 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return out
    r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
    rho, ps = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    out.update({"pearson_r": float(r), "pearson_p": float(p), "spearman_rho": float(rho), "spearman_p": float(ps)})
    return out


def correlation_table(df: pd.DataFrame, target: str, candidates: list[str] | None = None) -> pd.DataFrame:
    """Correlación de cada variable numérica con `target` (Pearson + Spearman + chequeos de forma)."""
    if candidates is None:
        candidates = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])
                      and not pd.api.types.is_bool_dtype(df[c])]
    rows = []
    y = df[target]
    for c in candidates:
        ps = pair_stats(df[c], y)
        if ps["n"] < 4:
            continue
        r, rho = ps["pearson_r"], ps["spearman_rho"]
        lin = linearity_screen(df[c], y)
        # sensibilidad a atípicos: r sin los atípicos IQR de ambas variables
        d = pd.concat([df[c], y], axis=1).dropna()
        keep = ~(outlier_masks(d.iloc[:, 0])["iqr"] | outlier_masks(d.iloc[:, 1])["iqr"])
        r_clean = float(stats.pearsonr(d.loc[keep].iloc[:, 0], d.loc[keep].iloc[:, 1])[0]) if keep.sum() > 8 and d.loc[keep].iloc[:, 0].nunique() > 1 else np.nan
        gap = abs(rho) - abs(r)
        notes = []
        if lin["no_lineal"]:
            notes.append(f"posible curvatura ({lin['mejor']})")
        if gap > 0.12:
            notes.append("monótona no lineal (|ρ| > |r|)")
        if not np.isnan(r_clean) and abs(r - r_clean) > 0.15:
            notes.append("r sensible a atípicos")
        if abs(r) > 0.98:
            notes.append("⚠ casi perfecta: ¿fuga/identidad contable?")
        rows.append({
            "variable": c, "n": ps["n"], "pearson_r": r, "pearson_p": ps["pearson_p"],
            "spearman_rho": rho, "spearman_p": ps["spearman_p"], "r²": r * r,
            "dirección": direction_label(r), "intensidad": strength_label(r),
            "r_sin_atípicos": r_clean, "forma_sugerida": lin["mejor"], "alertas": "; ".join(notes),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("pearson_r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True) if not out.empty else out


def correlation_matrix(df: pd.DataFrame, method: str = "pearson", cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or [c for c in df.select_dtypes(include="number").columns if not pd.api.types.is_bool_dtype(df[c])]
    cols = [c for c in cols if df[c].nunique() > 1]
    return df[cols].corr(method=method)


def describe_relationship(row: pd.Series, x_name: str, y_name: str) -> str:
    r, rho = row["pearson_r"], row["spearman_rho"]
    txt = (f"Relación {row['dirección']} {row['intensidad']} entre «{x_name}» y «{y_name}» "
           f"(Pearson r = {r:.2f}; Spearman ρ = {rho:.2f}; n = {int(row['n'])}). ")
    if row["dirección"] == "positiva":
        txt += "Cuando X sube, Y tiende a subir. "
    elif row["dirección"] == "negativa":
        txt += "Cuando X sube, Y tiende a bajar. "
    if row["pearson_p"] < 0.05:
        txt += "La asociación es estadísticamente distinta de cero. "
    else:
        txt += "No se puede descartar que la asociación sea producto del azar. "
    if row["alertas"]:
        txt += "Alertas: " + row["alertas"] + ". "
    txt += "Recuerde: asociación no implica causalidad."
    return txt


# ----------------------------------------------------------------------------
# tendencias y patrones temporales
# ----------------------------------------------------------------------------
def time_patterns(df: pd.DataFrame, date_col: str, y: str) -> dict:
    d = df[[date_col, y]].dropna().sort_values(date_col)
    out = {"n": len(d), "ok": False}
    if len(d) < 12:
        return out
    t = (d[date_col] - d[date_col].min()).dt.days.to_numpy(float)
    yy = d[y].to_numpy(float)
    lr = stats.linregress(t, yy)
    span_days = float(t.max() - t.min())
    per_year = lr.slope * 365.25
    out.update({"ok": True, "span_días": span_days, "pendiente_por_año": float(per_year), "p_tendencia": float(lr.pvalue),
                "r2_tendencia": float(lr.rvalue ** 2), "autocorr_lag1": float(pd.Series(yy).autocorr(1)),
                "serie": d.assign(media_móvil=d[y].rolling(max(3, len(d) // 12), min_periods=2).mean())})
    if span_days >= 700:
        m = d.groupby(d[date_col].dt.month)[y].mean()
        grand = yy.mean()
        ss_b = float(sum(((d[date_col].dt.month == mo).sum()) * (v - grand) ** 2 for mo, v in m.items()))
        ss_t = float(((yy - grand) ** 2).sum())
        out["estacionalidad_mensual"] = m
        out["fuerza_estacional"] = ss_b / ss_t if ss_t > 0 else np.nan
    return out
