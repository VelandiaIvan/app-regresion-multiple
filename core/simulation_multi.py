"""Estimación, escenarios y sensibilidad sobre un modelo de regresión MÚLTIPLE (varias X a la vez)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .multiple import MLRResult
from .regression import forward
from .simulation import Extrapolation

ORDER = {"dentro": 0, "leve": 1, "severa": 2}


def extrapolation_var(res: MLRResult, name: str, x: float) -> Extrapolation:
    lo, hi = res.x_range(name)
    rng = hi - lo
    if lo <= x <= hi:
        return Extrapolation("dentro", 0.0, f"«{name}» = {x:,.4g} está dentro de lo observado [{lo:,.4g}; {hi:,.4g}].")
    dist = (lo - x) if x < lo else (x - hi)
    pct = dist / rng * 100 if rng > 0 else np.inf
    side = "por debajo del mínimo" if x < lo else "por encima del máximo"
    if pct <= 10:
        return Extrapolation("leve", pct, f"⚠ «{name}» = {x:,.4g} está {side} observado ({pct:.1f}% de su amplitud): extrapolación leve.")
    return Extrapolation("severa", pct, f"🛑 «{name}» = {x:,.4g} está {side} observado por {pct:.0f}% de su amplitud: extrapolación severa.")


def extrapolation_status(res: MLRResult, values: dict) -> Extrapolation:
    """Peor caso entre las variables, más una revisión de combinaciones inusuales (palanca/leverage)."""
    per = [extrapolation_var(res, c, float(values[c])) for c in res.x_names]
    worst = max(per, key=lambda e: (ORDER[e.status], e.distance_pct))
    if worst.status != "dentro":
        msgs = " ".join(e.message for e in per if e.status != "dentro")
        tail = (" La estimación es menos confiable." if worst.status == "leve"
                else " Los datos no respaldan este escenario; no use la estimación para decidir.")
        return Extrapolation(worst.status, worst.distance_pct, msgs + tail)
    x0 = np.r_[1.0, [float(values[c]) for c in res.x_names]]
    h0 = float(x0 @ res.xtx_inv @ x0)
    if h0 > 2 * float(res.leverage.max()):
        return Extrapolation("leve", 0.0, "⚠ Cada valor está dentro de su rango, pero la COMBINACIÓN es inusual: casi no hay casos parecidos "
                             "en los datos (extrapolación oculta). Tome la estimación con cautela.")
    return Extrapolation("dentro", 0.0, "Todos los valores están dentro de lo observado: es una interpolación respaldada por los datos.")


def scenario_values(res: MLRResult, y_higher_is_better: bool = True, low_q: float = 0.10, high_q: float = 0.90) -> dict:
    """Escenarios pesimista / base / optimista: cada X en su P10, mediana o P90 según el signo de su coeficiente.

    «Pesimista» es la combinación de valores observados que produce el PEOR resultado para el negocio.
    """
    out = {"pesimista": {}, "base": {}, "optimista": {}}
    for c in res.x_names:
        v = res.x_values(c)
        lo, mid, hi = (float(q) for q in np.quantile(v, [low_q, 0.5, high_q]))
        if res.is_binary(c):
            lo, hi = float(v.min()), float(v.max())
            mid = float(round(np.mean(v)))
        good_is_high = (res.coef(c) > 0) == y_higher_is_better
        out["pesimista"][c], out["optimista"][c] = (lo, hi) if good_is_high else (hi, lo)
        out["base"][c] = mid
    return out


def _pct(new: float, base: float) -> float:
    return float((new - base) / abs(base) * 100) if abs(base) > 1e-12 else np.nan


def simulate(res: MLRResult, base: dict, new: dict, alpha: float | None = None) -> dict:
    """Compara el escenario base con el simulado y descompone el cambio por variable."""
    p = res.predict(pd.DataFrame([base, new]), alpha)
    pb, pn = p.iloc[0], p.iloc[1]
    contrib = []
    for c in res.x_names:
        dx = float(new[c]) - float(base[c])
        step = dict(base)
        step[c] = new[c]
        dy = float(res.predict(step, alpha)["y_est"].iloc[0] - pb["y_est"])          # efecto de mover SOLO esta variable
        contrib.append({"variable": c, "base": float(base[c]), "simulado": float(new[c]), "cambio_X": dx, "efecto_en_Y": dy})
    return {
        "base": {c: float(base[c]) for c in res.x_names}, "new": {c: float(new[c]) for c in res.x_names},
        "y_base": float(pb["y_est"]), "y_new": float(pn["y_est"]), "dy": float(pn["y_est"] - pb["y_est"]),
        "dy_pct": _pct(pn["y_est"], pb["y_est"]),
        "base_ci": (float(pb["ic_inf"]), float(pb["ic_sup"])), "base_pi": (float(pb["ip_inf"]), float(pb["ip_sup"])),
        "new_ci": (float(pn["ic_inf"]), float(pn["ic_sup"])), "new_pi": (float(pn["ip_inf"]), float(pn["ip_sup"])),
        "extrap_base": extrapolation_status(res, base), "extrap_new": extrapolation_status(res, new),
        "contrib": pd.DataFrame(contrib),
    }


def interpret_simulation(res: MLRResult, sim: dict, alpha: float | None = None) -> str:
    a = res.alpha if alpha is None else alpha
    lvl = int(round((1 - a) * 100))
    moved = sim["contrib"][sim["contrib"]["cambio_X"].abs() > 1e-12]
    y = res.y_name
    if moved.empty:
        txt = f"No cambió ninguna variable: el «{y}» estimado se mantiene en {sim['y_base']:,.4g}. "
    else:
        parts = ", ".join(f"«{r['variable']}» de {r['base']:,.4g} a {r['simulado']:,.4g}" for _, r in moved.iterrows())
        txt = (f"Al cambiar {parts}, el modelo estima que «{y}» pasa de {sim['y_base']:,.4g} a {sim['y_new']:,.4g} "
               f"(diferencia {sim['dy']:+,.4g}" + (f"; {sim['dy_pct']:+.1f}%" if not np.isnan(sim["dy_pct"]) else "") + "). ")
        if len(moved) > 1:
            top = moved.reindex(moved["efecto_en_Y"].abs().sort_values(ascending=False).index).iloc[0]
            txt += f"El cambio que más pesa es el de «{top['variable']}» ({top['efecto_en_Y']:+,.4g}). "
    txt += (f"Un caso individual con esos valores caería entre {sim['new_pi'][0]:,.4g} y {sim['new_pi'][1]:,.4g} con {lvl}% de probabilidad "
            f"(intervalo de predicción); el promedio de muchos casos así estaría entre {sim['new_ci'][0]:,.4g} y {sim['new_ci'][1]:,.4g} "
            "(intervalo de confianza). ")
    if sim["extrap_new"].is_extrapolation:
        txt += sim["extrap_new"].message + " "
    txt += "Esto describe una asociación observada; no demuestra que cambiar las X cause el cambio en Y."
    return txt


def sensitivity_table(res: MLRResult, base: dict, changes=(5, 10, -5, -10), variables: list | None = None,
                      alpha: float | None = None) -> pd.DataFrame:
    """Efecto sobre Y de mover UNA variable a la vez (las demás quedan en el punto de partida)."""
    variables = variables or list(res.x_names)
    y0 = float(res.predict(base, alpha)["y_est"].iloc[0])
    rows = []
    for c in variables:
        x0 = float(base[c])
        pivot = x0 if abs(x0) > 1e-12 else float(np.mean(np.abs(res.x_values(c)))) or 1.0
        for ch in changes:
            if ch is None:
                continue
            new = dict(base)
            new[c] = x0 + abs(pivot) * ch / 100
            p = res.predict(new, alpha).iloc[0]
            dy = float(p["y_est"] - y0)
            ex = extrapolation_var(res, c, new[c])
            verb = "aumentar" if ch > 0 else "reducir"
            yv = "sube" if dy > 0 else "baja" if dy < 0 else "se mantiene"
            pct = _pct(p["y_est"], y0)
            rows.append({
                "variable": c, "escenario": f"{ch:+g}%", "x_inicial": x0, "x_nuevo": float(new[c]), "y_inicial": y0,
                "y_nuevo": float(p["y_est"]), "dif_absoluta": dy, "var_%": pct, "ip_inf": float(p["ip_inf"]), "ip_sup": float(p["ip_sup"]),
                "extrapolación": ex.status,
                "interpretación": (f"Al {verb} «{c}» {abs(ch):g}% (las demás iguales), «{res.y_name}» estimado {yv} {abs(dy):,.4g}"
                                   + (f" ({pct:+.1f}%)" if not np.isnan(pct) else "") + "."
                                   + (" ⚠ Fuera del rango observado." if ex.is_extrapolation else "")),
            })
    return pd.DataFrame(rows)


def tornado_table(res: MLRResult, base: dict, pct: float = 10.0) -> pd.DataFrame:
    """Qué variable mueve más a Y ante un mismo ±pct% (ordenadas por impacto)."""
    t = sensitivity_table(res, base, changes=(pct, -pct))
    if t.empty:
        return t
    up = t[t["escenario"].str.startswith("+")].set_index("variable")["dif_absoluta"]
    dn = t[t["escenario"].str.startswith("-")].set_index("variable")["dif_absoluta"]
    out = pd.DataFrame({"variable": up.index, f"+{pct:g}%": up.values, f"-{pct:g}%": dn.reindex(up.index).values})
    out["impacto"] = out[[f"+{pct:g}%", f"-{pct:g}%"]].abs().max(axis=1)
    return out.sort_values("impacto", ascending=False).reset_index(drop=True)


def required_x(res: MLRResult, y_target: float, name: str, base: dict) -> dict:
    """Valor de UNA variable que, con las demás fijas en `base`, produce el Y objetivo (despeje de la ecuación)."""
    b = res.coef(name)
    if abs(b) < 1e-15:
        return {"ok": False, "message": f"El coeficiente de «{name}» es ~0: moverla no cambia Y."}
    yt = float(forward(res.y_transform, [y_target])[0])
    if not np.isfinite(yt):
        return {"ok": False, "message": "La meta está fuera del dominio de la transformación de Y."}
    if res.y_transform in {"ln", "ln1p"} and res.smear > 0:
        yt -= float(np.log(res.smear))                                    # deshace la corrección de retransformación
    rest = res.b0 + sum(res.coef(c) * float(base[c]) for c in res.x_names if c != name)
    x = float((yt - rest) / b)
    ex = extrapolation_var(res, name, x)
    return {"ok": True, "x": x, "extrap": ex,
            "message": f"Con las demás variables fijas, «{res.y_name}» ≈ {y_target:,.4g} corresponde a «{name}» ≈ {x:,.4g}. " + ex.message}


def partial_curve(res: MLRResult, name: str, base: dict, n: int = 80, pad: float = 0.15, alpha: float | None = None) -> pd.DataFrame:
    """Efecto parcial: Ŷ al mover solo `name` (las demás fijas en `base`), con bandas de confianza y de predicción."""
    lo, hi = res.x_range(name)
    ext = (hi - lo) * pad
    grid = np.linspace(lo - ext, hi + ext, n)
    frame = pd.DataFrame({c: np.full(n, float(base[c])) for c in res.x_names})
    frame[name] = grid
    out = res.predict(frame, alpha)
    out.insert(0, "x", grid)
    out["extrapolación"] = (grid < lo) | (grid > hi)
    return out
