r"""Regresión lineal simple por mínimos cuadrados ordinarios (MCO), con fórmulas explícitas.

Modelo:      Ŷ = β0 + β1·X
Pendiente:   β1 = Sxy / Sxx                  con Sxx = Σ(xi − x̄)²,  Sxy = Σ(xi − x̄)(yi − ȳ)
Intercepto:  β0 = ȳ − β1·x̄
Varianza:    σ̂² = SSE / (n − 2)              (error estándar residual σ̂ = √σ̂²)
Errores est: SE(β1) = σ̂ / √Sxx ;  SE(β0) = σ̂ · √(1/n + x̄²/Sxx)
t y p:       t = β / SE(β) ~ t(n−2)
F:           F = (SSR/1) / (SSE/(n−2))  (= t² en regresión simple)
IC coef.:    β ± t(1−α/2, n−2) · SE(β)
Respuesta media en x0:   Ŷ0 ± t · σ̂ · √(1/n + (x0 − x̄)²/Sxx)       → intervalo de confianza
Nueva observación en x0: Ŷ0 ± t · σ̂ · √(1 + 1/n + (x0 − x̄)²/Sxx)   → intervalo de predicción
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .cleaning import TRANSFORMS


class RegressionError(ValueError):
    pass


def forward(kind: str | None, x):
    if not kind:
        return np.asarray(x, float)
    return np.asarray(TRANSFORMS[kind][1](np.asarray(x, float)), float)


def inverse_point(kind: str | None, v, sigma2: float = 0.0, smear: float = 1.0):
    """Devuelve a la escala original un valor estimado (con corrección por retransformación)."""
    v = np.asarray(v, float)
    if not kind:
        return v
    if kind == "ln":
        return np.exp(v) * smear
    if kind == "ln1p":
        return np.expm1(v) * smear
    if kind == "sqrt":
        return np.clip(v, 0, None) ** 2 + sigma2
    if kind == "sq":
        return np.sqrt(np.clip(v, 0, None))
    if kind == "inv":
        with np.errstate(divide="ignore"):
            return 1.0 / v
    raise RegressionError(f"Transformación desconocida: {kind}")


def inverse_bounds(kind: str | None, lo, hi):
    """Los intervalos se transforman de forma monótona (sin factor de corrección)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if not kind:
        return lo, hi
    a, b = inverse_point(kind, lo), inverse_point(kind, hi)
    return (np.minimum(a, b), np.maximum(a, b)) if kind == "inv" else (a, b)


@dataclass
class SLRResult:
    x_name: str
    y_name: str
    n: int
    b0: float
    b1: float
    se_b0: float
    se_b1: float
    t_b0: float
    t_b1: float
    p_b0: float
    p_b1: float
    ci_b0: tuple
    ci_b1: tuple
    r: float
    r2: float
    r2_adj: float
    rmse: float
    mae: float
    sigma: float
    df_resid: int
    F: float
    p_F: float
    sse: float
    ssr: float
    sst: float
    x_mean: float
    y_mean: float
    sxx: float
    alpha: float
    x: np.ndarray
    y: np.ndarray
    fitted: np.ndarray
    resid: np.ndarray
    leverage: np.ndarray
    cooks: np.ndarray
    std_resid: np.ndarray
    index: np.ndarray
    rmse_loocv: float
    se_b1_hc3: float
    p_b1_hc3: float
    x_transform: str | None = None
    y_transform: str | None = None
    x_source: str = ""
    y_source: str = ""
    x_raw: np.ndarray | None = None
    y_raw: np.ndarray | None = None
    smear: float = 1.0
    extra: dict = field(default_factory=dict)

    # ----- rangos observados (en unidades originales de X) -----
    @property
    def x_obs(self) -> np.ndarray:
        return self.x_raw if self.x_raw is not None else self.x

    @property
    def y_obs(self) -> np.ndarray:
        return self.y_raw if self.y_raw is not None else self.y

    @property
    def x_min(self) -> float:
        return float(np.min(self.x_obs))

    @property
    def x_max(self) -> float:
        return float(np.max(self.x_obs))

    @property
    def y_min(self) -> float:
        return float(np.min(self.y_obs))

    @property
    def y_max(self) -> float:
        return float(np.max(self.y_obs))

    @property
    def form(self) -> str:
        """Forma funcional: lineal-lineal, log-lineal, lineal-log o log-log."""
        xl = self.x_transform in {"ln", "ln1p"}
        yl = self.y_transform in {"ln", "ln1p"}
        return {(False, False): "lin-lin", (False, True): "log-lin", (True, False): "lin-log", (True, True): "log-log"}[(xl, yl)]

    # ----- predicción -----
    def predict(self, x_new, alpha: float | None = None) -> pd.DataFrame:
        """Estimación puntual, intervalo de confianza (media) e intervalo de predicción (observación).

        `x_new` va en las unidades ORIGINALES de X; el resultado queda en las unidades originales de Y.
        """
        a = self.alpha if alpha is None else alpha
        x0 = np.atleast_1d(np.asarray(x_new, float))
        xt = forward(self.x_transform, x0)
        tcrit = stats.t.ppf(1 - a / 2, self.df_resid)
        yh = self.b0 + self.b1 * xt
        h = 1 / self.n + (xt - self.x_mean) ** 2 / self.sxx
        se_mean, se_pred = self.sigma * np.sqrt(h), self.sigma * np.sqrt(1 + h)
        sig2 = self.sigma ** 2
        ci_lo, ci_hi = inverse_bounds(self.y_transform, yh - tcrit * se_mean, yh + tcrit * se_mean)
        pi_lo, pi_hi = inverse_bounds(self.y_transform, yh - tcrit * se_pred, yh + tcrit * se_pred)
        return pd.DataFrame({
            "x": x0, "y_est": inverse_point(self.y_transform, yh, sig2, self.smear),
            "ic_inf": ci_lo, "ic_sup": ci_hi, "ip_inf": pi_lo, "ip_sup": pi_hi,
            "y_est_modelo": yh, "se_media": se_mean, "se_predicción": se_pred,
        })

    def y_back(self, yhat_model):
        return inverse_point(self.y_transform, yhat_model, self.sigma ** 2, self.smear)

    # ----- presentación -----
    def equation(self, digits: int = 4) -> str:
        xl = self.x_name
        yl = self.y_name
        sign = "+" if self.b1 >= 0 else "−"
        return f"Ŷ = {self.b0:,.{digits}g} {sign} {abs(self.b1):,.{digits}g}·X      [Y = {yl}; X = {xl}]"

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"coeficiente": "Intercepto (β0)", "estimación": self.b0, "error_estándar": self.se_b0, "t": self.t_b0,
             "p_valor": self.p_b0, "IC_inf": self.ci_b0[0], "IC_sup": self.ci_b0[1], "EE_robusto_HC3": np.nan, "p_robusto": np.nan},
            {"coeficiente": "Pendiente (β1)", "estimación": self.b1, "error_estándar": self.se_b1, "t": self.t_b1,
             "p_valor": self.p_b1, "IC_inf": self.ci_b1[0], "IC_sup": self.ci_b1[1],
             "EE_robusto_HC3": self.se_b1_hc3, "p_robusto": self.p_b1_hc3},
        ])

    def summary_dict(self) -> dict:
        return {"n": self.n, "β0": self.b0, "β1": self.b1, "r": self.r, "R²": self.r2, "R² ajustado": self.r2_adj,
                "RMSE": self.rmse, "MAE": self.mae, "σ̂ (error estándar residual)": self.sigma,
                "F": self.F, "p(F)": self.p_F, "p(β1)": self.p_b1, "RMSE LOOCV": self.rmse_loocv}


def fit_slr(x, y, *, x_name: str = "X", y_name: str = "Y", alpha: float = 0.05,
            x_transform: str | None = None, y_transform: str | None = None,
            x_source: str | None = None, y_source: str | None = None,
            x_raw=None, y_raw=None, index=None) -> SLRResult:
    """Ajusta la regresión lineal simple. Elimina pares con faltantes (casos completos)."""
    xs = pd.Series(np.asarray(x, float) if not isinstance(x, pd.Series) else x.astype(float).to_numpy())
    ys = pd.Series(np.asarray(y, float) if not isinstance(y, pd.Series) else y.astype(float).to_numpy())
    if index is None:
        index = x.index.to_numpy() if isinstance(x, pd.Series) else np.arange(len(xs))
    index = np.asarray(index)
    ok = np.isfinite(xs.to_numpy()) & np.isfinite(ys.to_numpy())
    xv, yv, idx = xs.to_numpy()[ok], ys.to_numpy()[ok], index[ok]
    xr = np.asarray(x_raw, float)[ok] if x_raw is not None else None
    yr = np.asarray(y_raw, float)[ok] if y_raw is not None else None
    n = len(xv)
    if n < 3:
        raise RegressionError(f"Se necesitan al menos 3 observaciones completas (hay {n}).")
    xbar, ybar = xv.mean(), yv.mean()
    sxx = float(((xv - xbar) ** 2).sum())
    if sxx <= 0:
        raise RegressionError("X es constante: no hay variación para estimar una pendiente.")
    sxy = float(((xv - xbar) * (yv - ybar)).sum())
    b1 = sxy / sxx
    b0 = ybar - b1 * xbar
    fitted = b0 + b1 * xv
    resid = yv - fitted
    sse = float(resid @ resid)
    sst = float(((yv - ybar) ** 2).sum())
    ssr = sst - sse
    dfr = n - 2
    sigma2 = sse / dfr if dfr > 0 else np.nan
    sigma = float(np.sqrt(sigma2))
    se_b1 = sigma / np.sqrt(sxx)
    se_b0 = sigma * np.sqrt(1 / n + xbar ** 2 / sxx)
    t_b1 = b1 / se_b1 if se_b1 > 0 else np.inf * np.sign(b1)
    t_b0 = b0 / se_b0 if se_b0 > 0 else np.inf * np.sign(b0)
    p_b1 = float(2 * stats.t.sf(abs(t_b1), dfr))
    p_b0 = float(2 * stats.t.sf(abs(t_b0), dfr))
    tc = stats.t.ppf(1 - alpha / 2, dfr)
    r2 = ssr / sst if sst > 0 else np.nan
    r = float(np.sign(b1) * np.sqrt(max(r2, 0))) if sst > 0 else np.nan
    r2_adj = 1 - (1 - r2) * (n - 1) / dfr if dfr > 0 else np.nan
    F = ssr / sigma2 if sigma2 > 0 else np.inf
    p_F = float(stats.f.sf(F, 1, dfr)) if np.isfinite(F) else 0.0
    h = 1 / n + (xv - xbar) ** 2 / sxx
    cooks = (resid ** 2 / (2 * sigma2)) * h / (1 - h) ** 2 if sigma2 > 0 else np.zeros(n)
    std_resid = resid / (sigma * np.sqrt(1 - h)) if sigma > 0 else np.zeros(n)
    loo = resid / (1 - h)                                                  # PRESS: validación cruzada «uno fuera»
    rmse_loocv = float(np.sqrt(np.mean(loo ** 2)))
    # error estándar robusto HC3 de la pendiente
    se_hc3 = float(np.sqrt((((xv - xbar) ** 2) * (resid / (1 - h)) ** 2).sum()) / sxx)
    p_hc3 = float(2 * stats.t.sf(abs(b1 / se_hc3), dfr)) if se_hc3 > 0 else 0.0
    smear = float(np.mean(np.exp(resid))) if y_transform in {"ln", "ln1p"} else 1.0
    return SLRResult(
        x_name=x_name, y_name=y_name, n=n, b0=float(b0), b1=float(b1), se_b0=float(se_b0), se_b1=float(se_b1),
        t_b0=float(t_b0), t_b1=float(t_b1), p_b0=p_b0, p_b1=p_b1,
        ci_b0=(float(b0 - tc * se_b0), float(b0 + tc * se_b0)), ci_b1=(float(b1 - tc * se_b1), float(b1 + tc * se_b1)),
        r=r, r2=float(r2), r2_adj=float(r2_adj), rmse=float(np.sqrt(sse / n)), mae=float(np.mean(np.abs(resid))),
        sigma=sigma, df_resid=dfr, F=float(F), p_F=p_F, sse=sse, ssr=float(ssr), sst=sst,
        x_mean=float(xbar), y_mean=float(ybar), sxx=sxx, alpha=alpha, x=xv, y=yv, fitted=fitted, resid=resid,
        leverage=h, cooks=cooks, std_resid=std_resid, index=idx, rmse_loocv=rmse_loocv,
        se_b1_hc3=se_hc3, p_b1_hc3=p_hc3, x_transform=x_transform, y_transform=y_transform,
        x_source=x_source or x_name, y_source=y_source or y_name, x_raw=xr, y_raw=yr, smear=smear,
    )


def prediction_metrics(y_true, y_pred, y_baseline: float | None = None) -> dict:
    """Métricas de error. `y_baseline` (media de entrenamiento) define el R² fuera de muestra y la «habilidad»."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    n = len(yt)
    if n == 0:
        return {"n": 0}
    err = yt - yp
    rmse = float(np.sqrt(np.mean(err ** 2)))
    out = {"n": n, "RMSE": rmse, "MAE": float(np.mean(np.abs(err))), "sesgo": float(np.mean(err))}
    base = float(np.mean(yt)) if y_baseline is None else y_baseline
    sst = float(((yt - base) ** 2).sum())
    out["R²_fuera_muestra"] = 1 - float((err ** 2).sum()) / sst if sst > 0 else np.nan
    rmse_base = float(np.sqrt(np.mean((yt - base) ** 2)))
    out["RMSE_referencia"] = rmse_base
    out["habilidad"] = 1 - rmse / rmse_base if rmse_base > 0 else np.nan     # mejora vs. predecir siempre la media
    if (yt > 0).all():
        out["MAPE_%"] = float(np.mean(np.abs(err / yt)) * 100)
    return out
