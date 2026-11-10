r"""Regresión lineal MÚLTIPLE por mínimos cuadrados ordinarios (MCO), en forma matricial.

Modelo:        Ŷ = β0 + β1·X1 + β2·X2 + … + βk·Xk          (k ≥ 2 variables explicativas)
Coeficientes:  β = (XᵀX)⁻¹ Xᵀy                              (X incluye la columna de unos)
Varianza:      σ̂² = SSE / (n − k − 1)
Errores est.:  SE(βj) = σ̂ · √[(XᵀX)⁻¹]jj
t y p:         tj = βj / SE(βj) ~ t(n−k−1)                  (efecto de Xj MANTENIENDO constantes las demás)
F global:      F = (SSR/k) / (SSE/(n−k−1))
R² ajustado:   1 − (1 − R²)(n − 1)/(n − k − 1)
VIF:           VIFj = 1 / (1 − R²j)                         (R²j: Xj explicada por las otras X)
Respuesta media en x0:   Ŷ0 ± t · σ̂ · √(x0ᵀ(XᵀX)⁻¹x0)        → intervalo de confianza
Nueva observación en x0: Ŷ0 ± t · σ̂ · √(1 + x0ᵀ(XᵀX)⁻¹x0)    → intervalo de predicción
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .regression import RegressionError, inverse_bounds, inverse_point


@dataclass
class MLRResult:
    x_names: list
    y_name: str
    n: int
    k: int
    beta: np.ndarray            # [β0, β1, …, βk]
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    ci: np.ndarray              # (k+1, 2)
    se_hc3: np.ndarray
    p_hc3: np.ndarray
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
    alpha: float
    X: np.ndarray               # (n, k) valores observados de las X
    y: np.ndarray               # Y en la escala del modelo
    y_raw: np.ndarray           # Y en su escala original
    fitted: np.ndarray
    resid: np.ndarray
    leverage: np.ndarray
    cooks: np.ndarray
    std_resid: np.ndarray
    index: np.ndarray
    rmse_loocv: float
    xtx_inv: np.ndarray
    vif: dict
    std_beta: dict
    y_transform: str | None = None
    smear: float = 1.0
    extra: dict = field(default_factory=dict)

    # ----- accesos cómodos -----
    @property
    def y_source(self) -> str:
        return self.y_name

    @property
    def y_obs(self) -> np.ndarray:
        return self.y_raw

    @property
    def y_mean(self) -> float:
        return float(np.mean(self.y))

    @property
    def b0(self) -> float:
        return float(self.beta[0])

    def coef(self, name: str) -> float:
        return float(self.beta[1 + self.x_names.index(name)])

    def p_of(self, name: str) -> float:
        return float(self.p[1 + self.x_names.index(name)])

    def ci_of(self, name: str) -> tuple:
        lo, hi = self.ci[1 + self.x_names.index(name)]
        return float(lo), float(hi)

    def x_values(self, name: str) -> np.ndarray:
        return self.X[:, self.x_names.index(name)]

    def x_range(self, name: str) -> tuple:
        v = self.x_values(name)
        return float(v.min()), float(v.max())

    def x_stats(self) -> pd.DataFrame:
        rows = []
        for j, nm in enumerate(self.x_names):
            v = self.X[:, j]
            q = np.quantile(v, [0.10, 0.5, 0.90])
            rows.append({"variable": nm, "mínimo": float(v.min()), "P10": float(q[0]), "mediana": float(q[1]), "media": float(v.mean()),
                         "P90": float(q[2]), "máximo": float(v.max()), "desv_est": float(v.std(ddof=1))})
        return pd.DataFrame(rows)

    def medians(self) -> dict:
        return {nm: float(np.median(self.X[:, j])) for j, nm in enumerate(self.x_names)}

    def is_binary(self, name: str) -> bool:
        return bool(set(np.unique(self.x_values(name)).tolist()) <= {0.0, 1.0})

    @property
    def n_signif(self) -> int:
        return int((self.p[1:] < self.alpha).sum())

    # ----- predicción -----
    def _matrix(self, x_new) -> np.ndarray:
        if isinstance(x_new, dict):
            x_new = pd.DataFrame([x_new])
        if isinstance(x_new, pd.DataFrame):
            missing = [c for c in self.x_names if c not in x_new.columns]
            if missing:
                raise RegressionError("Faltan variables para estimar: " + ", ".join(missing))
            arr = x_new[self.x_names].to_numpy(float)
        else:
            arr = np.atleast_2d(np.asarray(x_new, float))
        if arr.shape[1] != self.k:
            raise RegressionError(f"Se esperaban {self.k} variables y llegaron {arr.shape[1]}.")
        return arr

    def predict_model_scale(self, x_new) -> np.ndarray:
        arr = self._matrix(x_new)
        return self.beta[0] + arr @ self.beta[1:]

    def predict(self, x_new, alpha: float | None = None) -> pd.DataFrame:
        """Estimación puntual, intervalo de confianza (media) e intervalo de predicción (observación), en la escala original de Y."""
        a = self.alpha if alpha is None else alpha
        arr = self._matrix(x_new)
        Xd = np.column_stack([np.ones(len(arr)), arr])
        yh = Xd @ self.beta
        h = np.einsum("ij,jk,ik->i", Xd, self.xtx_inv, Xd)
        tcrit = stats.t.ppf(1 - a / 2, self.df_resid)
        se_mean, se_pred = self.sigma * np.sqrt(np.clip(h, 0, None)), self.sigma * np.sqrt(1 + np.clip(h, 0, None))
        ci_lo, ci_hi = inverse_bounds(self.y_transform, yh - tcrit * se_mean, yh + tcrit * se_mean)
        pi_lo, pi_hi = inverse_bounds(self.y_transform, yh - tcrit * se_pred, yh + tcrit * se_pred)
        return pd.DataFrame({"y_est": self.y_back(yh), "ic_inf": ci_lo, "ic_sup": ci_hi, "ip_inf": pi_lo, "ip_sup": pi_hi,
                             "y_est_modelo": yh, "se_media": se_mean, "se_predicción": se_pred})

    def y_back(self, yhat_model):
        return inverse_point(self.y_transform, yhat_model, self.sigma ** 2, self.smear)

    # ----- presentación -----
    def equation(self, digits: int = 4) -> str:
        terms = [f"{self.beta[0]:,.{digits}g}"]
        for j, nm in enumerate(self.x_names):
            b = self.beta[j + 1]
            terms.append(f"{'+' if b >= 0 else '−'} {abs(b):,.{digits}g}·{nm}")
        lhs = "Ŷ" if not self.y_transform else f"{self.y_transform}(Ŷ)"
        return f"{lhs} = " + " ".join(terms) + f"      [Y = {self.y_name}]"

    def coef_table(self) -> pd.DataFrame:
        names = ["Intercepto (β0)"] + [f"{nm} (β{j + 1})" for j, nm in enumerate(self.x_names)]
        vif = [np.nan] + [self.vif.get(nm, np.nan) for nm in self.x_names]
        sb = [np.nan] + [self.std_beta.get(nm, np.nan) for nm in self.x_names]
        return pd.DataFrame({
            "coeficiente": names, "estimación": self.beta, "error_estándar": self.se, "t": self.t, "p_valor": self.p,
            "IC_inf": self.ci[:, 0], "IC_sup": self.ci[:, 1], "significativo": ["sí" if v < self.alpha else "no" for v in self.p],
            "beta_estandarizado": sb, "VIF": vif, "EE_robusto_HC3": self.se_hc3, "p_robusto": self.p_hc3,
        })

    def summary_dict(self) -> dict:
        return {"n": self.n, "variables": self.k, "R²": self.r2, "R² ajustado": self.r2_adj, "RMSE": self.rmse, "MAE": self.mae,
                "σ̂ (error estándar residual)": self.sigma, "F": self.F, "p(F)": self.p_F, "RMSE LOOCV": self.rmse_loocv}


def vif_values(X: np.ndarray, names: list) -> dict:
    """Factor de inflación de la varianza de cada X (1 = sin solapamiento con las demás)."""
    if X.shape[1] < 2:
        return {names[0]: 1.0} if names else {}
    with np.errstate(all="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    try:
        d = np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        d = np.full(X.shape[1], np.inf)
    return {nm: float(v) if np.isfinite(v) and v > 0 else float("inf") for nm, v in zip(names, d)}


def fit_mlr(X: pd.DataFrame, y, *, y_name: str = "Y", alpha: float = 0.05, y_transform: str | None = None,
            y_raw=None, index=None) -> MLRResult:
    """Ajusta la regresión múltiple con las columnas de `X`. Usa casos completos."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X, float))
        X.columns = [f"X{j + 1}" for j in range(X.shape[1])]
    names = [str(c) for c in X.columns]
    k = len(names)
    if k < 1:
        raise RegressionError("Se necesita al menos una variable explicativa.")
    Xv = X.to_numpy(float)
    yv = np.asarray(y, float)
    yr = np.asarray(y_raw, float) if y_raw is not None else yv.copy()
    idx = np.asarray(index) if index is not None else (X.index.to_numpy() if isinstance(X, pd.DataFrame) else np.arange(len(yv)))
    ok = np.isfinite(Xv).all(axis=1) & np.isfinite(yv) & np.isfinite(yr)
    Xv, yv, yr, idx = Xv[ok], yv[ok], yr[ok], idx[ok]
    n = len(yv)
    if n < k + 3:
        raise RegressionError(f"Se necesitan al menos {k + 3} observaciones completas para {k} variables (hay {n}).")
    const = [nm for j, nm in enumerate(names) if np.ptp(Xv[:, j]) == 0]
    if const:
        raise RegressionError("Variable(s) constante(s), sin variación para estimar su efecto: " + ", ".join(const))
    Xd = np.column_stack([np.ones(n), Xv])
    if np.linalg.matrix_rank(Xd) < k + 1:
        raise RegressionError("Multicolinealidad perfecta: alguna variable es combinación exacta de otras.")
    beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
    xtx_inv = np.linalg.inv(Xd.T @ Xd)
    fitted = Xd @ beta
    resid = yv - fitted
    sse = float(resid @ resid)
    ybar = float(yv.mean())
    sst = float(((yv - ybar) ** 2).sum())
    ssr = sst - sse
    dfr = n - k - 1
    sigma2 = sse / dfr
    sigma = float(np.sqrt(sigma2))
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.inf * np.sign(beta))
    p = 2 * stats.t.sf(np.abs(t), dfr)
    tc = stats.t.ppf(1 - alpha / 2, dfr)
    ci = np.column_stack([beta - tc * se, beta + tc * se])
    r2 = ssr / sst if sst > 0 else np.nan
    r2_adj = 1 - (1 - r2) * (n - 1) / dfr
    F = (ssr / k) / sigma2 if sigma2 > 0 else np.inf
    p_F = float(stats.f.sf(F, k, dfr)) if np.isfinite(F) else 0.0
    h = np.clip(np.einsum("ij,jk,ik->i", Xd, xtx_inv, Xd), 0, 1 - 1e-12)
    cooks = (resid ** 2 / ((k + 1) * sigma2)) * h / (1 - h) ** 2 if sigma2 > 0 else np.zeros(n)
    std_resid = resid / (sigma * np.sqrt(1 - h)) if sigma > 0 else np.zeros(n)
    loo = resid / (1 - h)                                                   # PRESS: validación cruzada «uno fuera»
    rmse_loocv = float(np.sqrt(np.mean(loo ** 2)))
    meat = Xd.T @ (Xd * (loo ** 2)[:, None])                                # HC3
    cov_hc3 = xtx_inv @ meat @ xtx_inv
    se_hc3 = np.sqrt(np.clip(np.diag(cov_hc3), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        p_hc3 = np.where(se_hc3 > 0, 2 * stats.t.sf(np.abs(beta / se_hc3), dfr), 0.0)
    sd_y = float(yv.std(ddof=1))
    std_beta = {nm: float(beta[j + 1] * Xv[:, j].std(ddof=1) / sd_y) if sd_y > 0 else np.nan for j, nm in enumerate(names)}
    smear = float(np.mean(np.exp(resid))) if y_transform in {"ln", "ln1p"} else 1.0
    return MLRResult(
        x_names=names, y_name=y_name, n=n, k=k, beta=beta, se=se, t=t, p=p, ci=ci, se_hc3=se_hc3, p_hc3=p_hc3,
        r2=float(r2), r2_adj=float(r2_adj), rmse=float(np.sqrt(sse / n)), mae=float(np.mean(np.abs(resid))), sigma=sigma,
        df_resid=dfr, F=float(F), p_F=p_F, sse=sse, ssr=float(ssr), sst=sst, alpha=alpha, X=Xv, y=yv, y_raw=yr,
        fitted=fitted, resid=resid, leverage=h, cooks=cooks, std_resid=std_resid, index=idx, rmse_loocv=rmse_loocv,
        xtx_inv=xtx_inv, vif=vif_values(Xv, names), std_beta=std_beta, y_transform=y_transform, smear=smear,
    )


def quick_loocv(Xv: np.ndarray, yv: np.ndarray) -> tuple[float, float]:
    """(RMSE de validación cruzada «uno fuera», R² ajustado) sin construir todo el resultado: sirve para explorar subconjuntos."""
    n, k = Xv.shape
    Xd = np.column_stack([np.ones(n), Xv])
    q, r = np.linalg.qr(Xd)
    if np.min(np.abs(np.diag(r))) < 1e-10 * max(np.max(np.abs(np.diag(r))), 1e-300):
        return np.inf, -np.inf
    h = np.clip((q ** 2).sum(axis=1), 0, 1 - 1e-12)
    resid = yv - q @ (q.T @ yv)
    sst = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / sst if sst > 0 else 0.0
    dfr = n - k - 1
    r2_adj = 1 - (1 - r2) * (n - 1) / dfr if dfr > 0 else -np.inf
    return float(np.sqrt(np.mean((resid / (1 - h)) ** 2))), float(r2_adj)
