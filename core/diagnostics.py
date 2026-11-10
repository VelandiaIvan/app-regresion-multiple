"""Diagnóstico de los supuestos de la regresión lineal simple.

Supuestos: (1) linealidad, (2) independencia de los errores, (3) homocedasticidad, (4) normalidad
de los residuos. Además se revisan observaciones influyentes (distancia de Cook).
Cada verificación devuelve un semáforo: ok / warn / fail / na.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .regression import SLRResult, fit_slr

ICON = {"ok": "🟢", "warn": "🟡", "fail": "🔴", "na": "⚪"}
LABEL = {"ok": "Cumple", "warn": "Reservas", "fail": "No cumple", "na": "No aplicable"}


@dataclass
class Check:
    key: str
    name: str
    severity: str                # ok | warn | fail | na
    tests: dict                  # nombre de prueba -> (estadístico, p)
    explanation: str
    remedy: str = ""

    @property
    def icon(self) -> str:
        return ICON[self.severity]


@dataclass
class Diagnostics:
    checks: list = field(default_factory=list)
    verdict: str = "apto"        # apto | con reservas | no confiable
    summary: str = ""

    def get(self, key: str) -> Check:
        return next(c for c in self.checks if c.key == key)

    def table(self) -> pd.DataFrame:
        rows = []
        for c in self.checks:
            tests = "; ".join(f"{k}: {v[0]:.3g}" + (f" (p={v[1]:.3g})" if v[1] is not None and not np.isnan(v[1]) else "")
                              for k, v in c.tests.items())
            rows.append({"supuesto": c.name, "resultado": f"{c.icon} {LABEL[c.severity]}", "pruebas": tests,
                         "interpretación": c.explanation, "qué hacer": c.remedy})
        return pd.DataFrame(rows)


def _lm_test(resid2: np.ndarray, X: np.ndarray, df: int) -> tuple[float, float]:
    """Estadístico LM = n·R² de la regresión auxiliar de e² sobre X."""
    n = len(resid2)
    beta, *_ = np.linalg.lstsq(X, resid2, rcond=None)
    fit = X @ beta
    sst = float(((resid2 - resid2.mean()) ** 2).sum())
    r2 = 1 - float(((resid2 - fit) ** 2).sum()) / sst if sst > 0 else 0.0
    lm = n * r2
    return float(lm), float(stats.chi2.sf(lm, df))


# ----------------------------------------------------------------------------
def check_linearity(res: SLRResult) -> Check:
    n = res.n
    x = res.x - res.x_mean
    if n < 10 or np.ptp(res.x) == 0:
        return Check("linealidad", "Linealidad", "na", {}, "Muy pocas observaciones para evaluar la forma de la relación.")
    X1 = np.column_stack([np.ones(n), x])
    X2 = np.column_stack([np.ones(n), x, x ** 2])
    b1, *_ = np.linalg.lstsq(X1, res.y, rcond=None)
    b2, *_ = np.linalg.lstsq(X2, res.y, rcond=None)
    sse1 = float(((res.y - X1 @ b1) ** 2).sum())
    sse2 = float(((res.y - X2 @ b2) ** 2).sum())
    F = (sse1 - sse2) / (sse2 / (n - 3)) if sse2 > 0 else 0.0
    p_quad = float(stats.f.sf(F, 1, n - 3))
    gain = (sse1 - sse2) / res.sst if res.sst > 0 else 0.0
    # prueba de rachas sobre los residuos ordenados por X
    order = np.argsort(res.x, kind="stable")
    s = res.resid[order] > 0
    n1, n2 = int(s.sum()), int((~s).sum())
    runs = 1 + int((s[1:] != s[:-1]).sum())
    p_runs = np.nan
    if n1 > 0 and n2 > 0:
        mu = 2 * n1 * n2 / (n1 + n2) + 1
        var = (mu - 1) * (mu - 2) / (n1 + n2 - 1)
        z = (runs - mu) / np.sqrt(var) if var > 0 else 0.0
        p_runs = float(2 * stats.norm.sf(abs(z)))
    tests = {"término cuadrático (F)": (F, p_quad), "rachas de signos": (float(runs), p_runs)}
    if (p_quad < 0.01 and gain > 0.02) or (not np.isnan(p_runs) and p_runs < 0.01 and gain > 0.01):
        sev, txt = "fail", (f"Los residuos muestran un patrón curvo: agregar X² reduce el error en {gain * 100:.1f}% de la variación total. "
                            "La recta no captura la forma de la relación.")
    elif p_quad < 0.05 or (not np.isnan(p_runs) and p_runs < 0.05):
        sev, txt = "warn", "Hay señales leves de curvatura; revise el gráfico de residuos contra valores ajustados."
    else:
        sev, txt = "ok", "No se detecta curvatura sistemática: una recta es una descripción razonable."
    remedy = ("Transformar X (ln, √, 1/x), transformar Y, o usar un modelo no lineal / polinómico."
              if sev != "ok" else "")
    return Check("linealidad", "Linealidad", sev, tests, txt, remedy)


def check_independence(res: SLRResult, ordered: bool) -> Check:
    if not ordered:
        return Check("independencia", "Independencia de los errores", "na", {},
                     "Los datos no tienen un orden temporal declarado, por lo que la autocorrelación no se puede probar. "
                     "Verifique que no haya mediciones repetidas del mismo sujeto, lugar o periodo.",
                     "Si las filas son una serie de tiempo, indique la columna de fecha para activar la prueba.")
    e = res.resid
    n = len(e)
    if n < 12:
        return Check("independencia", "Independencia de los errores", "na", {}, "Muy pocas observaciones.")
    dw = float(((e[1:] - e[:-1]) ** 2).sum() / (e @ e))
    # Breusch-Godfrey orden 1: e_t ~ 1 + x_t + e_{t-1}
    X = np.column_stack([np.ones(n - 1), res.x[1:], e[:-1]])
    beta, *_ = np.linalg.lstsq(X, e[1:], rcond=None)
    fit = X @ beta
    r2 = 1 - float(((e[1:] - fit) ** 2).sum()) / float(((e[1:] - e[1:].mean()) ** 2).sum())
    lm = (n - 1) * r2
    p_bg = float(stats.chi2.sf(lm, 1))
    rho = float(np.corrcoef(e[1:], e[:-1])[0, 1])
    tests = {"Durbin-Watson": (dw, np.nan), "Breusch-Godfrey (orden 1)": (lm, p_bg), "autocorrelación lag 1": (rho, np.nan)}
    if p_bg < 0.01 or dw < 1.2 or dw > 2.8:
        sev, txt = "fail", (f"Los errores consecutivos están correlacionados (DW = {dw:.2f}, ρ = {rho:.2f}). "
                            "Los errores estándar y p-valores resultan demasiado optimistas.")
    elif p_bg < 0.05 or dw < 1.5 or dw > 2.5:
        sev, txt = "warn", f"Indicios leves de autocorrelación (DW = {dw:.2f})."
    else:
        sev, txt = "ok", f"Sin autocorrelación relevante (DW = {dw:.2f}, cercano a 2)."
    remedy = ("Use un modelo de series de tiempo (rezagos de Y, errores AR/ARIMA), incluya la tendencia o trabaje con diferencias."
              if sev != "ok" else "")
    return Check("independencia", "Independencia de los errores", sev, tests, txt, remedy)


def check_homoscedasticity(res: SLRResult) -> Check:
    n = res.n
    if n < 10:
        return Check("homocedasticidad", "Homocedasticidad", "na", {}, "Muy pocas observaciones.")
    e2 = res.resid ** 2
    xs = res.x - res.x_mean
    lm_bp, p_bp = _lm_test(e2, np.column_stack([np.ones(n), xs]), 1)
    lm_w, p_w = _lm_test(e2, np.column_stack([np.ones(n), xs, xs ** 2]), 2)
    order = np.argsort(res.x)
    k = max(n // 3, 3)
    sd_lo, sd_hi = float(res.resid[order[:k]].std(ddof=1)), float(res.resid[order[-k:]].std(ddof=1))
    ratio = sd_hi / sd_lo if sd_lo > 0 else np.inf
    tests = {"Breusch-Pagan": (lm_bp, p_bp), "White": (lm_w, p_w), "razón de desv. est. (tercio alto/bajo)": (ratio, np.nan)}
    if min(p_bp, p_w) < 0.01:
        sev, txt = "fail", (f"La dispersión de los residuos cambia con X (desv. est. del tercio alto ≈ {ratio:.2f}× la del bajo). "
                            "Los intervalos de predicción dejan de ser confiables: serán demasiado anchos en una zona y estrechos en otra.")
    elif min(p_bp, p_w) < 0.05:
        sev, txt = "warn", "Hay señales moderadas de varianza no constante."
    else:
        sev, txt = "ok", "La dispersión de los errores es aproximadamente constante a lo largo de X."
    remedy = ("Transformar Y (p. ej. logaritmo), usar mínimos cuadrados ponderados o interpretar con errores estándar robustos (HC3, ya calculados)."
              if sev != "ok" else "")
    return Check("homocedasticidad", "Homocedasticidad", sev, tests, txt, remedy)


def check_normality(res: SLRResult) -> Check:
    e = res.resid
    n = len(e)
    if n < 8:
        return Check("normalidad", "Normalidad de los residuos", "na", {}, "Muy pocas observaciones.")
    sk, ku = float(stats.skew(e)), float(stats.kurtosis(e))
    jb, p_jb = stats.jarque_bera(e)
    tests = {"Jarque-Bera": (float(jb), float(p_jb)), "asimetría": (sk, np.nan), "curtosis (exceso)": (ku, np.nan)}
    if n <= 5000:
        sw = stats.shapiro(e)
        tests = {"Shapiro-Wilk": (float(sw.statistic), float(sw.pvalue)), **tests}
        p_main = float(sw.pvalue)
    else:
        p_main = float(p_jb)
    mild = abs(sk) < 0.8 and abs(ku) < 2
    if p_main >= 0.05:
        sev, txt = "ok", "Los residuos son compatibles con una distribución normal."
    elif mild or (n > 200 and abs(sk) < 1 and abs(ku) < 3):
        sev, txt = "warn", (f"Se rechaza normalidad estricta (p = {p_main:.3g}) pero la desviación es leve (asimetría {sk:.2f}, curtosis {ku:.2f})"
                            + ("; con muestra grande el teorema central del límite protege la inferencia sobre los coeficientes." if n > 100 else "."))
    else:
        sev, txt = "fail", (f"Residuos claramente no normales (asimetría {sk:.2f}, curtosis {ku:.2f}). Los intervalos de predicción no son confiables.")
    remedy = ("Transformar Y (ln/√), revisar atípicos o usar regresión robusta; los intervalos de predicción dependen de este supuesto."
              if sev != "ok" else "")
    return Check("normalidad", "Normalidad de los residuos", sev, tests, txt, remedy)


def check_influence(res: SLRResult) -> Check:
    n = res.n
    thr = 4 / n
    infl = res.cooks > thr
    n_inf = int(infl.sum())
    dmax = float(res.cooks.max())
    change = np.nan
    if 0 < n_inf <= max(1, int(0.1 * n)) and n - n_inf >= 5:
        try:
            r2 = fit_slr(res.x[~infl], res.y[~infl])
            change = (r2.b1 - res.b1) / abs(res.b1) * 100 if res.b1 else np.nan
        except Exception:
            pass
    tests = {"Cook máx.": (dmax, np.nan), f"obs. con Cook > 4/n": (float(n_inf), np.nan)}
    if not np.isnan(change):
        tests["cambio % de la pendiente sin ellas"] = (change, np.nan)
    if dmax > 1 or (not np.isnan(change) and abs(change) > 30):
        sev, txt = "fail", (f"Pocas observaciones determinan la recta (Cook máx. {dmax:.2f}"
                            + (f"; quitarlas cambia la pendiente {change:+.0f}%" if not np.isnan(change) else "") + ").")
    elif n_inf / n > 0.08 or (not np.isnan(change) and abs(change) > 15):
        sev, txt = "warn", (f"{n_inf} observaciones con influencia alta"
                            + (f"; la pendiente cambia {change:+.0f}% sin ellas." if not np.isnan(change) else "."))
    else:
        sev, txt = "ok", "Ninguna observación domina el ajuste."
    remedy = ("Verifique si son errores de captura o casos especiales; compare el modelo con y sin ellas o use regresión robusta."
              if sev != "ok" else "")
    return Check("influencia", "Observaciones influyentes", sev, tests, txt, remedy)


def run_diagnostics(res: SLRResult, ordered: bool = False) -> Diagnostics:
    checks = [check_linearity(res), check_independence(res, ordered), check_homoscedasticity(res),
              check_normality(res), check_influence(res)]
    d = Diagnostics(checks=checks)
    core = [c for c in checks if c.key in {"linealidad", "independencia", "homocedasticidad", "normalidad"} and c.severity != "na"]
    fails = [c for c in core if c.severity == "fail"]
    warns = [c for c in core if c.severity == "warn"]
    infl_fail = d.get("influencia").severity == "fail"
    if d.get("linealidad").severity == "fail" or len(fails) >= 2:
        d.verdict = "no confiable"
    elif fails or infl_fail or len(warns) >= 2:
        d.verdict = "con reservas"
    else:
        d.verdict = "apto"
    names = lambda cs: ", ".join(c.name.lower() for c in cs)
    d.summary = {
        "apto": "Los supuestos principales se cumplen razonablemente; la inferencia es confiable.",
        "con reservas": f"Úselo con cautela. Incumplimientos o reservas en: {names(fails + warns)}"
                        + ("; observaciones influyentes" if infl_fail else "") + ".",
        "no confiable": f"Los supuestos no se sostienen ({names(fails)}). Las estimaciones puntuales pueden orientar, "
                        "pero los intervalos y p-valores no son confiables.",
    }[d.verdict]
    return d


def qq_points(res: SLRResult) -> tuple[np.ndarray, np.ndarray]:
    """Cuantiles teóricos normales vs. residuos estandarizados ordenados."""
    (osm, osr), _ = stats.probplot(res.std_resid, dist="norm")
    return np.asarray(osm), np.asarray(osr)


# ============================================================================
# Regresión MÚLTIPLE
# ============================================================================
def _runs_p(signs: np.ndarray) -> tuple[int, float]:
    n1, n2 = int(signs.sum()), int((~signs).sum())
    runs = 1 + int((signs[1:] != signs[:-1]).sum())
    if n1 == 0 or n2 == 0:
        return runs, np.nan
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = (mu - 1) * (mu - 2) / (n1 + n2 - 1)
    z = (runs - mu) / np.sqrt(var) if var > 0 else 0.0
    return runs, float(2 * stats.norm.sf(abs(z)))


def m_check_linearity(res) -> Check:
    """Prueba RESET de Ramsey (¿mejora el modelo al agregar Ŷ²?) y rachas de los residuos ordenados por Ŷ."""
    n, k = res.n, res.k
    if n < k + 10:
        return Check("linealidad", "Linealidad", "na", {}, "Muy pocas observaciones para evaluar la forma de la relación.")
    Xd = np.column_stack([np.ones(n), res.X])
    f = res.fitted - res.fitted.mean()
    X2 = np.column_stack([Xd, f ** 2])
    b2, *_ = np.linalg.lstsq(X2, res.y, rcond=None)
    sse2 = float(((res.y - X2 @ b2) ** 2).sum())
    df2 = n - k - 2
    F = (res.sse - sse2) / (sse2 / df2) if sse2 > 0 and df2 > 0 else 0.0
    p_reset = float(stats.f.sf(max(F, 0), 1, df2)) if df2 > 0 else np.nan
    gain = (res.sse - sse2) / res.sst if res.sst > 0 else 0.0
    order = np.argsort(res.fitted, kind="stable")
    runs, p_runs = _runs_p(res.resid[order] > 0)
    tests = {"RESET de Ramsey (F)": (float(F), p_reset), "rachas de signos": (float(runs), p_runs)}
    if (p_reset < 0.01 and gain > 0.02) or (not np.isnan(p_runs) and p_runs < 0.01 and gain > 0.01):
        sev, txt = "fail", (f"Los residuos muestran un patrón curvo: agregar un término cuadrático reduce el error en {gain * 100:.1f}% de la "
                            "variación total. La combinación lineal de las X no captura la forma de la relación.")
    elif p_reset < 0.05 or (not np.isnan(p_runs) and p_runs < 0.05):
        sev, txt = "warn", "Hay señales leves de curvatura; revise el gráfico de residuos contra valores ajustados."
    else:
        sev, txt = "ok", "No se detecta curvatura sistemática: la forma lineal es una descripción razonable."
    remedy = "Transformar alguna X (ln, √), transformar Y o agregar términos cuadráticos / interacciones." if sev != "ok" else ""
    return Check("linealidad", "Linealidad", sev, tests, txt, remedy)


def m_check_independence(res, ordered: bool) -> Check:
    name = "Independencia de los errores"
    if not ordered:
        return Check("independencia", name, "na", {},
                     "Los datos no tienen un orden temporal declarado, por lo que la autocorrelación no se puede probar. "
                     "Verifique que no haya mediciones repetidas del mismo sujeto, lugar o periodo.",
                     "Si las filas son una serie de tiempo, indique la columna de fecha para activar la prueba.")
    e = res.resid
    n = len(e)
    if n < res.k + 12:
        return Check("independencia", name, "na", {}, "Muy pocas observaciones.")
    dw = float(((e[1:] - e[:-1]) ** 2).sum() / (e @ e))
    X = np.column_stack([np.ones(n - 1), res.X[1:], e[:-1]])
    beta, *_ = np.linalg.lstsq(X, e[1:], rcond=None)
    fit = X @ beta
    sst = float(((e[1:] - e[1:].mean()) ** 2).sum())
    r2 = 1 - float(((e[1:] - fit) ** 2).sum()) / sst if sst > 0 else 0.0
    lm = (n - 1) * r2
    p_bg = float(stats.chi2.sf(lm, 1))
    rho = float(np.corrcoef(e[1:], e[:-1])[0, 1])
    tests = {"Durbin-Watson": (dw, np.nan), "Breusch-Godfrey (orden 1)": (lm, p_bg), "autocorrelación lag 1": (rho, np.nan)}
    if p_bg < 0.01 or dw < 1.2 or dw > 2.8:
        sev, txt = "fail", (f"Los errores consecutivos están correlacionados (DW = {dw:.2f}, ρ = {rho:.2f}). "
                            "Los errores estándar y p-valores resultan demasiado optimistas.")
    elif p_bg < 0.05 or dw < 1.5 or dw > 2.5:
        sev, txt = "warn", f"Indicios leves de autocorrelación (DW = {dw:.2f})."
    else:
        sev, txt = "ok", f"Sin autocorrelación relevante (DW = {dw:.2f}, cercano a 2)."
    remedy = ("Use un modelo de series de tiempo (rezagos de Y, errores AR/ARIMA), incluya la tendencia o trabaje con diferencias."
              if sev != "ok" else "")
    return Check("independencia", name, sev, tests, txt, remedy)


def m_check_homoscedasticity(res) -> Check:
    n, k = res.n, res.k
    if n < k + 10:
        return Check("homocedasticidad", "Homocedasticidad", "na", {}, "Muy pocas observaciones.")
    e2 = res.resid ** 2
    lm_bp, p_bp = _lm_test(e2, np.column_stack([np.ones(n), res.X]), k)
    f = res.fitted - res.fitted.mean()
    lm_w, p_w = _lm_test(e2, np.column_stack([np.ones(n), f, f ** 2]), 2)
    order = np.argsort(res.fitted)
    g = max(n // 3, 3)
    sd_lo, sd_hi = float(res.resid[order[:g]].std(ddof=1)), float(res.resid[order[-g:]].std(ddof=1))
    ratio = sd_hi / sd_lo if sd_lo > 0 else np.inf
    tests = {"Breusch-Pagan": (lm_bp, p_bp), "White (sobre Ŷ y Ŷ²)": (lm_w, p_w), "razón de desv. est. (tercio alto/bajo de Ŷ)": (ratio, np.nan)}
    if min(p_bp, p_w) < 0.01:
        sev, txt = "fail", (f"La dispersión de los residuos cambia con el nivel estimado (desv. est. del tercio alto ≈ {ratio:.2f}× la del bajo). "
                            "Los intervalos de predicción dejan de ser confiables.")
    elif min(p_bp, p_w) < 0.05:
        sev, txt = "warn", "Hay señales moderadas de varianza no constante."
    else:
        sev, txt = "ok", "La dispersión de los errores es aproximadamente constante."
    remedy = ("Transformar Y (p. ej. logaritmo), usar mínimos cuadrados ponderados o interpretar con errores estándar robustos (HC3, ya calculados)."
              if sev != "ok" else "")
    return Check("homocedasticidad", "Homocedasticidad", sev, tests, txt, remedy)


def m_check_multicollinearity(res) -> Check:
    name = "Multicolinealidad (VIF)"
    vif = res.vif
    worst = max(vif, key=lambda c: vif[c])
    vmax = vif[worst]
    tests = {f"VIF de {c}": (float(v), np.nan) for c, v in vif.items()}
    if vmax > 10:
        sev, txt = "fail", (f"«{worst}» se solapa casi por completo con las otras X (VIF = {vmax:.1f} > 10). Los coeficientes individuales se vuelven "
                            "inestables: pueden cambiar de signo o perder significancia aunque el modelo prediga bien.")
    elif vmax > 5:
        sev, txt = "warn", f"«{worst}» comparte bastante información con las otras X (VIF = {vmax:.1f}, entre 5 y 10). Interprete sus coeficientes con cautela."
    else:
        sev, txt = "ok", f"Las X aportan información distinta entre sí (VIF máximo = {vmax:.2f})."
    remedy = "Retire una de las variables redundantes, combínelas en un índice o elija el modelo con menos variables." if sev != "ok" else ""
    return Check("multicolinealidad", name, sev, tests, txt, remedy)


def m_check_influence(res) -> Check:
    from .multiple import fit_mlr
    n = res.n
    thr = 4 / n
    infl = res.cooks > thr
    n_inf = int(infl.sum())
    dmax = float(res.cooks.max())
    change = np.nan
    if 0 < n_inf <= max(1, int(0.1 * n)) and n - n_inf >= res.k + 5:
        try:
            r2 = fit_mlr(pd.DataFrame(res.X[~infl], columns=res.x_names), res.y[~infl])
            base = np.abs(res.beta[1:])
            rel = np.where(base > 0, (r2.beta[1:] - res.beta[1:]) / np.where(base > 0, base, 1) * 100, 0.0)
            sig = res.p[1:] < res.alpha                                   # solo importan los coeficientes que se interpretan
            pool = rel[sig] if sig.any() else rel
            change = float(pool[np.argmax(np.abs(pool))])
        except Exception:
            pass
    tests = {"Cook máx.": (dmax, np.nan), "obs. con Cook > 4/n": (float(n_inf), np.nan)}
    if not np.isnan(change):
        tests["mayor cambio % de un coeficiente sin ellas"] = (change, np.nan)
    if dmax > 1 or (not np.isnan(change) and abs(change) > 30):
        sev, txt = "fail", (f"Pocas observaciones determinan el ajuste (Cook máx. {dmax:.2f}"
                            + (f"; quitarlas cambia un coeficiente {change:+.0f}%" if not np.isnan(change) else "") + ").")
    elif n_inf / n > 0.08 or (not np.isnan(change) and abs(change) > 15):
        sev, txt = "warn", (f"{n_inf} observaciones con influencia alta"
                            + (f"; algún coeficiente cambia {change:+.0f}% sin ellas." if not np.isnan(change) else "."))
    else:
        sev, txt = "ok", "Ninguna observación domina el ajuste."
    remedy = ("Verifique si son errores de captura o casos especiales; compare el modelo con y sin ellas o use regresión robusta."
              if sev != "ok" else "")
    return Check("influencia", "Observaciones influyentes", sev, tests, txt, remedy)


def run_diagnostics_multi(res, ordered: bool = False) -> Diagnostics:
    checks = [m_check_linearity(res), m_check_independence(res, ordered), m_check_homoscedasticity(res),
              check_normality(res), m_check_multicollinearity(res), m_check_influence(res)]
    d = Diagnostics(checks=checks)
    core = [c for c in checks if c.key in {"linealidad", "independencia", "homocedasticidad", "normalidad"} and c.severity != "na"]
    fails = [c for c in core if c.severity == "fail"]
    warns = [c for c in core if c.severity == "warn"]
    extra_fail = [c for c in checks if c.key in {"influencia", "multicolinealidad"} and c.severity == "fail"]
    if d.get("linealidad").severity == "fail" or len(fails) >= 2:
        d.verdict = "no confiable"
    elif fails or extra_fail or len(warns) >= 2:
        d.verdict = "con reservas"
    else:
        d.verdict = "apto"
    names = lambda cs: ", ".join(c.name.lower() for c in cs)
    d.summary = {
        "apto": "Los supuestos principales se cumplen razonablemente; la inferencia es confiable.",
        "con reservas": f"Úselo con cautela. Incumplimientos o reservas en: {names(fails + warns + extra_fail)}.",
        "no confiable": f"Los supuestos no se sostienen ({names(fails)}). Las estimaciones puntuales pueden orientar, "
                        "pero los intervalos y p-valores no son confiables.",
    }[d.verdict]
    return d
