"""Interpretación de un modelo de regresión múltiple: cada coeficiente es un efecto PARCIAL (con las demás X constantes)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .multiple import MLRResult

CETERIS = "manteniendo constantes las demás variables"


def _u(unit: str) -> str:
    return f" {unit}" if unit else ""


def _fmt(v: float) -> str:
    return f"{v:,.4g}"


def _is_log(res: MLRResult) -> bool:
    return res.y_transform in {"ln", "ln1p"}


def interpret_intercept(res: MLRResult, y_unit: str = "") -> dict:
    y = res.y_name
    lvl = int(round((1 - res.alpha) * 100))
    lo, hi = res.ci[0]
    sig = "estadísticamente distinto de cero" if res.p[0] < res.alpha else "NO distinguible de cero estadísticamente"
    in_range = all(res.x_range(c)[0] <= 0 <= res.x_range(c)[1] for c in res.x_names)
    tecnica = (f"β0 = {_fmt(res.beta[0])} (EE = {_fmt(res.se[0])}; IC {lvl}%: [{_fmt(lo)}; {_fmt(hi)}]; p = {res.p[0]:.3g}) es {sig}. "
               f"Es el valor que el modelo asigna a {'ln(' + y + ')' if _is_log(res) else y} cuando TODAS las X valen 0.")
    if in_range:
        viab = "El valor 0 está dentro del rango observado de todas las X: la interpretación literal es válida."
        sencilla = f"Si todas las variables valieran 0, el modelo esperaría un «{y}» cercano a {_fmt(res.y_back(res.beta[0]))}{_u(y_unit)} (nivel base)."
    else:
        fuera = [c for c in res.x_names if not (res.x_range(c)[0] <= 0 <= res.x_range(c)[1])]
        viab = ("En los datos nunca ocurre que todas las X valgan 0 (" + ", ".join(fuera[:3]) + ("…" if len(fuera) > 3 else "")
                + " no incluyen el 0): interpretar β0 literalmente sería una extrapolación. Trátelo como un parámetro de ajuste.")
        sencilla = "Es el punto de partida matemático de la ecuación; por sí solo no tiene lectura de negocio porque las variables nunca valen 0 a la vez."
    return {"tecnica": tecnica, "viabilidad": viab, "sencilla": sencilla, "en_rango": bool(in_range)}


def interpret_coefficient(res: MLRResult, name: str, y_unit: str = "") -> dict:
    """Lectura de un coeficiente: efecto de +1 unidad de X sobre Y con las demás X constantes."""
    y = res.y_name
    b = res.coef(name)
    lo, hi = res.ci_of(name)
    pv = res.p_of(name)
    lvl = int(round((1 - res.alpha) * 100))
    sig = pv < res.alpha
    binary = res.is_binary(name)
    change = f"Cuando «{name}» = 1 (frente a 0)" if binary else f"Por cada unidad adicional de «{name}»"
    if _is_log(res):
        pct = (np.exp(b) - 1) * 100
        efecto = f"{change}, «{y}» cambia aproximadamente {pct:+.2f}% (IC {lvl}%: {(np.exp(lo) - 1) * 100:+.2f}% a {(np.exp(hi) - 1) * 100:+.2f}%), {CETERIS}."
    elif res.y_transform:
        efecto = f"{change}, «{y}» {'sube' if b > 0 else 'baja'} (el tamaño depende del nivel, por la transformación «{res.y_transform}» de Y), {CETERIS}."
    else:
        efecto = f"{change}, «{y}» cambia en promedio {b:+,.4g}{_u(y_unit)} (IC {lvl}%: {lo:+,.4g} a {hi:+,.4g}), {CETERIS}."
    sd = float(np.std(res.x_values(name), ddof=1))
    magnitud = (f"Un cambio típico de «{name}» (1 desviación estándar = {_fmt(sd)}) mueve «{y}» en {b * sd:+,.4g}"
                + (" en la escala del modelo" if res.y_transform else "") + f"; coeficiente estandarizado β* = {res.std_beta[name]:+.2f}.")
    j = 1 + res.x_names.index(name)
    tecnica = (f"β = {_fmt(b)} (EE = {_fmt(res.se[j])}; t = {res.t[j]:.2f}; p = {pv:.3g}; IC {lvl}%: [{_fmt(lo)}; {_fmt(hi)}]; VIF = {res.vif[name]:.2f}). "
               + (f"Es significativa al {int(round(res.alpha * 100))}%." if sig else
                  f"NO es significativa al {int(round(res.alpha * 100))}%: una vez consideradas las otras variables, no hay evidencia de que aporte (el IC incluye 0)."))
    return {"variable": name, "efecto": efecto, "tecnica": tecnica, "magnitud": magnitud, "significativa": bool(sig),
            "direccion": "positiva" if b > 0 else "negativa", "coef": b, "p": pv, "std_beta": res.std_beta[name]}


def interpret_all(res: MLRResult, y_unit: str = "") -> list[dict]:
    """Coeficientes ordenados por importancia relativa (|β estandarizado|)."""
    items = [interpret_coefficient(res, c, y_unit) for c in res.x_names]
    return sorted(items, key=lambda d: -abs(d["std_beta"]) if np.isfinite(d["std_beta"]) else 0)


def importance_table(res: MLRResult) -> pd.DataFrame:
    rows = [{"variable": c, "coeficiente": res.coef(c), "β estandarizado": res.std_beta[c], "p_valor": res.p_of(c),
             "significativa": "sí" if res.p_of(c) < res.alpha else "no", "VIF": res.vif[c]} for c in res.x_names]
    t = pd.DataFrame(rows)
    return t.reindex(t["β estandarizado"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def plain_language(res: MLRResult, y_unit: str = "") -> str:
    items = interpret_all(res, y_unit)
    sig = [d for d in items if d["significativa"]]
    r2 = res.r2_adj
    quality = ("explican muy poco" if r2 < 0.1 else "explican una parte pequeña" if r2 < 0.3 else "explican una parte moderada"
               if r2 < 0.5 else "explican una parte importante" if r2 < 0.8 else "explican casi todo")
    txt = (f"En palabras simples: juntas, las {res.k} variables {quality} de lo que le pasa a «{res.y_name}» "
           f"(R² ajustado = {r2:.0%}); el resto depende de otros factores o del azar. ")
    if sig:
        top = sig[0]
        txt += f"La que más pesa es «{top['variable']}»: {top['efecto']} "
        if len(sig) > 1:
            txt += "También aportan " + ", ".join(f"«{d['variable']}» ({'sube' if d['coef'] > 0 else 'baja'} Y)" for d in sig[1:]) + ". "
    not_sig = [d["variable"] for d in items if not d["significativa"]]
    if not_sig:
        txt += "No se pudo confirmar un efecto propio de " + ", ".join(f"«{v}»" for v in not_sig) + ". "
    txt += f"Una estimación típicamente se equivoca en ±{_typical_error(res):,.4g}{_u(y_unit)}."
    return txt


def _typical_error(res: MLRResult) -> float:
    """Error típico (RMSE) en la escala ORIGINAL de Y."""
    return float(np.sqrt(np.mean((res.y_obs - res.y_back(res.fitted)) ** 2)))


def numeric_examples(res: MLRResult, y_unit: str = "") -> list[str]:
    out = []
    lo = {c: float(np.quantile(res.x_values(c), 0.25)) for c in res.x_names}
    mid = res.medians()
    hi = {c: float(np.quantile(res.x_values(c), 0.75)) for c in res.x_names}
    p = res.predict(pd.DataFrame([lo, mid, hi]))
    out.append(f"Un caso con todas las X en su valor típico (mediana) tiene un «{res.y_name}» estimado de {_fmt(p['y_est'][1])}{_u(y_unit)}. "
               f"Con todas en su cuartil bajo: {_fmt(p['y_est'][0])}; con todas en su cuartil alto: {_fmt(p['y_est'][2])}.")
    order = np.argsort(np.abs(res.std_resid))
    typical, worst = int(order[len(order) // 2]), int(np.argmax(np.abs(res.std_resid)))
    est_all = res.y_back(res.fitted)
    for tag, i in (("Observación típica", typical), ("Observación peor ajustada", worst)):
        xs = ", ".join(f"{c} = {_fmt(res.X[i, j])}" for j, c in enumerate(res.x_names))
        yo, est = float(res.y_obs[i]), float(est_all[i])
        err = f"; error = {_fmt(yo - est)}{_u(y_unit)}" + (f" ({(yo - est) / yo * 100:+.1f}% del valor real)" if yo else "")
        out.append(f"{tag} (fila {res.index[i]}): {xs}. Y real = {_fmt(yo)}, Y estimado = {_fmt(est)}{err}.")
    return out


def explain_metrics(res: MLRResult, test_metrics: dict | None = None) -> list[str]:
    out = [f"R² = {res.r2:.3f} y R² ajustado = {res.r2_adj:.3f}: el modelo explica {res.r2_adj:.1%} de la variación de Y. En regresión múltiple se usa el "
           "AJUSTADO, porque el R² simple siempre sube al agregar variables, aunque no sirvan.",
           f"Error típico de estimación (RMSE, en unidades de Y) = {_fmt(_typical_error(res))}.",
           f"F = {res.F:,.2f} (p = {res.p_F:.3g}): contrasta si el conjunto de variables explica más que usar solo el promedio.",
           f"Coeficientes significativos: {res.n_signif} de {res.k}. VIF máximo = {max(res.vif.values()):.2f} (menor que 5 es lo deseable)."]
    if test_metrics:
        out.append(f"Con datos de prueba (no usados para ajustar): RMSE = {_fmt(test_metrics['RMSE'])}, MAE = {_fmt(test_metrics['MAE'])}, "
                   f"R² fuera de muestra = {test_metrics['R²_fuera_muestra']:.3f}.")
    return out
