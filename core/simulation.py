"""Estimación, simulación de escenarios y análisis de sensibilidad sobre un modelo ajustado."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .regression import SLRResult, forward, inverse_point


@dataclass
class Extrapolation:
    status: str            # dentro | leve | severa
    distance_pct: float    # distancia fuera del rango, como % de la amplitud observada
    message: str

    @property
    def is_extrapolation(self) -> bool:
        return self.status != "dentro"


def extrapolation_status(res: SLRResult, x: float) -> Extrapolation:
    lo, hi = res.x_min, res.x_max
    rng = hi - lo
    if lo <= x <= hi:
        return Extrapolation("dentro", 0.0, f"X = {x:,.4g} está dentro del rango observado [{lo:,.4g}; {hi:,.4g}]: es una interpolación.")
    dist = (lo - x) if x < lo else (x - hi)
    pct = dist / rng * 100 if rng > 0 else np.inf
    side = "por debajo del mínimo" if x < lo else "por encima del máximo"
    if pct <= 10:
        return Extrapolation("leve", pct, f"⚠ EXTRAPOLACIÓN LEVE: X = {x:,.4g} está {side} observado ({pct:.1f}% de la amplitud). "
                             "La estimación es menos confiable; asuma que la relación sigue siendo lineal fuera de lo observado.")
    return Extrapolation("severa", pct, f"🛑 EXTRAPOLACIÓN SEVERA: X = {x:,.4g} está {side} observado por {pct:.0f}% de la amplitud. "
                         "Los datos no respaldan este valor; la recta puede no cumplirse allí. No use la estimación para decidir.")


def effective_sign(res: SLRResult) -> int:
    p = res.predict([res.x_min, res.x_max])["y_est"].to_numpy()
    return int(np.sign(p[1] - p[0]))


def scenario_values(res: SLRResult, y_higher_is_better: bool = True, low_q: float = 0.10, high_q: float = 0.90,
                    custom: dict | None = None) -> dict:
    """Valores de X para los escenarios pesimista / base / optimista.

    Por defecto usan el rango observado (P10, mediana, P90). «Pesimista» es el X que produce el peor Y
    según el sentido del negocio (si Y más alto es mejor, pesimista = el X que lo hace menor).
    """
    x = res.x_obs
    lo, mid, hi = np.quantile(x, [low_q, 0.5, high_q])
    s = effective_sign(res) or 1
    good_is_high_x = (s > 0) == y_higher_is_better
    pes, opt = (lo, hi) if good_is_high_x else (hi, lo)
    out = {"pesimista": float(pes), "base": float(mid), "optimista": float(opt)}
    if custom:
        out.update({k: float(v) for k, v in custom.items() if v is not None})
    return out


def _pct(new: float, base: float) -> float:
    return float((new - base) / abs(base) * 100) if abs(base) > 1e-12 else np.nan


def simulate(res: SLRResult, x_base: float, x_new: float, alpha: float | None = None) -> dict:
    """Compara el escenario base con el simulado."""
    pb = res.predict([x_base], alpha).iloc[0]
    pn = res.predict([x_new], alpha).iloc[0]
    return {
        "x_base": float(x_base), "x_new": float(x_new), "dx": float(x_new - x_base), "dx_pct": _pct(x_new, x_base),
        "y_base": float(pb["y_est"]), "y_new": float(pn["y_est"]), "dy": float(pn["y_est"] - pb["y_est"]),
        "dy_pct": _pct(pn["y_est"], pb["y_est"]),
        "base_ci": (float(pb["ic_inf"]), float(pb["ic_sup"])), "base_pi": (float(pb["ip_inf"]), float(pb["ip_sup"])),
        "new_ci": (float(pn["ic_inf"]), float(pn["ic_sup"])), "new_pi": (float(pn["ip_inf"]), float(pn["ip_sup"])),
        "extrap_base": extrapolation_status(res, x_base), "extrap_new": extrapolation_status(res, x_new),
    }


def interpret_simulation(res: SLRResult, sim: dict, x_label: str, y_label: str, alpha: float | None = None) -> str:
    a = res.alpha if alpha is None else alpha
    lvl = int(round((1 - a) * 100))
    direction = "aumenta" if sim["dx"] > 0 else "disminuye" if sim["dx"] < 0 else "no cambia"
    txt = (f"Si {x_label} {direction} de {sim['x_base']:,.4g} a {sim['x_new']:,.4g}, el modelo estima que {y_label} pasa de "
           f"{sim['y_base']:,.4g} a {sim['y_new']:,.4g} (diferencia {sim['dy']:+,.4g}"
           + (f"; {sim['dy_pct']:+.1f}%" if not np.isnan(sim["dy_pct"]) else "") + "). ")
    txt += (f"Una observación individual con ese X caería entre {sim['new_pi'][0]:,.4g} y {sim['new_pi'][1]:,.4g} "
            f"con {lvl}% de probabilidad (intervalo de predicción); el promedio de muchas observaciones con ese X estaría entre "
            f"{sim['new_ci'][0]:,.4g} y {sim['new_ci'][1]:,.4g} (intervalo de confianza). ")
    if sim["extrap_new"].is_extrapolation:
        txt += sim["extrap_new"].message + " "
    txt += "Esto describe una asociación observada; no demuestra que cambiar X cause el cambio en Y."
    return txt


def sensitivity_table(res: SLRResult, x_base: float, changes=(5, 10, -5, -10), custom=(), alpha: float | None = None,
                      x_label: str = "X", y_label: str = "Y") -> pd.DataFrame:
    """Efecto sobre Y de variaciones porcentuales de X."""
    pivot = x_base if abs(x_base) > 1e-12 else float(np.mean(np.abs(res.x_obs)))
    rows = []
    for ch in list(changes) + [c for c in custom if c is not None]:
        x_new = x_base + pivot * ch / 100
        sim = simulate(res, x_base, x_new, alpha)
        name = f"{ch:+g}%" + ("" if ch in changes else " (personalizado)")
        rows.append({
            "escenario": name, "x_inicial": sim["x_base"], "x_nuevo": sim["x_new"], "y_inicial": sim["y_base"],
            "y_nuevo": sim["y_new"], "dif_absoluta": sim["dy"], "var_%": sim["dy_pct"],
            "ip_inf": sim["new_pi"][0], "ip_sup": sim["new_pi"][1],
            "extrapolación": sim["extrap_new"].status,
            "interpretación": _row_text(sim, ch, x_label, y_label),
        })
    return pd.DataFrame(rows)


def _row_text(sim: dict, ch: float, x_label: str, y_label: str) -> str:
    verb = "aumentar" if ch > 0 else "reducir"
    yv = "sube" if sim["dy"] > 0 else "baja" if sim["dy"] < 0 else "se mantiene"
    pct = f" ({sim['dy_pct']:+.1f}%)" if not np.isnan(sim["dy_pct"]) else ""
    t = f"Al {verb} {x_label} {abs(ch):g}%, {y_label} estimado {yv} {abs(sim['dy']):,.4g}{pct}."
    if sim["extrap_new"].is_extrapolation:
        t += " ⚠ Fuera del rango observado."
    return t


def required_x(res: SLRResult, y_target: float) -> dict:
    """«Regresión inversa»: X que, según la recta, produce un Y objetivo (aproximación; no es causal)."""
    if abs(res.b1) < 1e-15:
        return {"ok": False, "message": "La pendiente es ~0: ningún X cambia Y."}
    yt = float(forward(res.y_transform, [y_target])[0])
    if not np.isfinite(yt):
        return {"ok": False, "message": "El objetivo está fuera del dominio de la transformación de Y."}
    xt = (yt - res.b0) / res.b1
    x = float(inverse_point(res.x_transform, xt)) if res.x_transform else float(xt)
    if not np.isfinite(x):
        return {"ok": False, "message": "No existe un X real que produzca ese objetivo."}
    ex = extrapolation_status(res, x)
    return {"ok": True, "x": x, "extrap": ex,
            "message": f"Según la recta, Y ≈ {y_target:,.4g} corresponde a X ≈ {x:,.4g}. " + ex.message}


def curve(res: SLRResult, n: int = 120, pad: float = 0.0, alpha: float | None = None) -> pd.DataFrame:
    """Recta con bandas de confianza y de predicción sobre una malla de X (puede extenderse fuera del rango)."""
    lo, hi = res.x_min, res.x_max
    ext = (hi - lo) * pad
    a, b = lo - ext, hi + ext
    if res.x_transform in {"ln", "sqrt", "ln1p", "inv"}:
        a = max(a, lo * 0.5 if lo > 0 else 1e-9)
    grid = np.linspace(a, b, n)
    out = res.predict(grid, alpha)
    out["extrapolación"] = (out["x"] < lo) | (out["x"] > hi)
    return out
