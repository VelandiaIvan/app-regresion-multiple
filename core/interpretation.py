"""Interpretación del intercepto y la pendiente: versión técnica y en lenguaje sencillo, con ejemplos numéricos reales."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .regression import SLRResult


def _u(unit: str) -> str:
    return f" {unit}" if unit else ""


def _fmt(v: float) -> str:
    return f"{v:,.4g}"


def interpret_intercept(res: SLRResult, x_unit: str = "", y_unit: str = "") -> dict:
    x, y = res.x_name if res.x_transform is None else res.x_source, res.y_source
    sig = "estadísticamente distinto de cero" if res.p_b0 < res.alpha else "NO distinguible de cero estadísticamente"
    in_range = res.x_min <= 0 <= res.x_max
    if res.x_transform in {"ln", "ln1p", "sqrt", "inv"}:
        zero_txt = f"el punto donde la transformación de «{x}» vale 0"
    else:
        zero_txt = f"«{x}» = 0"
    tecnica = (f"β0 = {_fmt(res.b0)} (EE = {_fmt(res.se_b0)}; IC {int((1 - res.alpha) * 100)}%: [{_fmt(res.ci_b0[0])}; {_fmt(res.ci_b0[1])}]; "
               f"p = {res.p_b0:.3g}) es {sig}. Es el valor que la recta asigna a "
               f"{'ln(' + y + ')' if res.y_transform in {'ln', 'ln1p'} else y} cuando {zero_txt}.")
    if res.x_transform is None and in_range:
        viab = f"El valor 0 está dentro del rango observado de «{x}» [{_fmt(res.x_min)}; {_fmt(res.x_max)}]: la interpretación literal es válida."
    elif res.x_transform is None:
        viab = (f"El rango observado de «{x}» es [{_fmt(res.x_min)}; {_fmt(res.x_max)}] y NO incluye 0. Interpretar β0 como "
                f"«valor de {y} cuando {x} = 0» sería una extrapolación; trátelo como un parámetro de ajuste de la recta.")
    else:
        viab = "Con una transformación de X el intercepto es un parámetro de ajuste; interprete mejor los valores estimados dentro del rango observado."
    if res.form == "lin-lin" and res.x_transform is None:
        sencilla = (f"Si «{x}» fuera 0, el modelo esperaría que «{y}» valiera cerca de {_fmt(res.b0)}{_u(y_unit)}. "
                    + ("En muchos negocios esto se parece a un «nivel base» o costo fijo (lo que ocurre aun sin el factor)." if in_range or res.b0 > 0 else "")
                    + (" Pero cuidado: " + viab if not in_range else ""))
    else:
        sencilla = "Es el punto de partida matemático de la curva; lo útil es leer los valores estimados dentro del rango observado."
    return {"tecnica": tecnica, "viabilidad": viab, "sencilla": sencilla, "en_rango": bool(in_range and res.x_transform is None)}


def interpret_slope(res: SLRResult, x_unit: str = "", y_unit: str = "") -> dict:
    x, y = res.x_source, res.y_source
    lvl = int((1 - res.alpha) * 100)
    sig = res.p_b1 < res.alpha
    sentido = "positiva (X y Y se mueven en el mismo sentido)" if res.b1 > 0 else "negativa (cuando X sube, Y baja)"
    form = res.form
    ci = res.ci_b1
    if form == "lin-lin":
        eff = f"Por cada unidad adicional de «{x}»{_u(x_unit)}, «{y}» cambia en promedio {res.b1:+,.4g}{_u(y_unit)}"
        eff_ci = f"(IC {lvl}%: {ci[0]:+,.4g} a {ci[1]:+,.4g})"
        mag = f"Un cambio de 1 desviación estándar de X ({np.std(res.x_obs, ddof=1):,.4g}) mueve Y en {res.b1 * np.std(res.x_obs, ddof=1):+,.4g}."
    elif form == "log-lin":
        pct = (np.exp(res.b1) - 1) * 100
        eff = f"Por cada unidad adicional de «{x}»{_u(x_unit)}, «{y}» cambia aproximadamente {pct:+.2f}%"
        eff_ci = f"(IC {lvl}%: {(np.exp(ci[0]) - 1) * 100:+.2f}% a {(np.exp(ci[1]) - 1) * 100:+.2f}%)"
        mag = "El efecto es porcentual y compuesto: cada unidad extra multiplica Y por e^β1."
    elif form == "lin-log":
        d1, d10 = res.b1 * np.log(1.01), res.b1 * np.log(1.10)
        eff = f"Un aumento de 1% en «{x}» cambia «{y}» en promedio {d1:+,.4g}{_u(y_unit)} (un aumento de 10%: {d10:+,.4g})"
        eff_ci = f"(IC {lvl}% de β1: {ci[0]:+,.4g} a {ci[1]:+,.4g})"
        mag = "Rendimientos decrecientes: el mismo % de aumento aporta lo mismo, pero cada unidad extra de X aporta cada vez menos."
    else:  # log-log
        e = (1.01 ** res.b1 - 1) * 100
        eff = f"Elasticidad: un aumento de 1% en «{x}» cambia «{y}» aproximadamente {e:+.3f}%"
        eff_ci = f"(IC {lvl}% de la elasticidad: {ci[0]:+.3f} a {ci[1]:+.3f})"
        mag = ("Elasticidad > 1: Y reacciona más que proporcionalmente a X." if abs(res.b1) > 1 else
               "Elasticidad < 1: Y reacciona menos que proporcionalmente a X (efecto «inelástico»).")
    if res.x_transform in {"sqrt", "inv", "sq"} or res.y_transform in {"sqrt", "inv", "sq"}:
        eff = f"El efecto de «{x}» sobre «{y}» no es constante (modelo con transformación «{res.x_transform or res.y_transform}»)"
        eff_ci = "(vea los ejemplos numéricos)"
        mag = "El efecto marginal depende del nivel de X."
    tecnica = (f"β1 = {_fmt(res.b1)} (EE = {_fmt(res.se_b1)}; t = {res.t_b1:.2f}; p = {res.p_b1:.3g}; IC {lvl}%: [{_fmt(ci[0])}; {_fmt(ci[1])}]). "
               f"Relación {sentido}. " + (f"Es estadísticamente significativa al {int(res.alpha * 100)}%." if sig else
                                          f"NO es significativa al {int(res.alpha * 100)}%: no hay evidencia suficiente de que la pendiente sea distinta de 0 (el IC incluye 0)."))
    return {"efecto": f"{eff} {eff_ci}.", "tecnica": tecnica, "magnitud": mag, "significativa": bool(sig),
            "direccion": "positiva" if res.b1 > 0 else "negativa"}


def plain_language(res: SLRResult, x_unit: str = "", y_unit: str = "") -> str:
    s = interpret_slope(res, x_unit, y_unit)
    r2 = res.r2
    quality = ("explica muy poco" if r2 < 0.1 else "explica una parte pequeña" if r2 < 0.3 else "explica una parte moderada"
               if r2 < 0.5 else "explica una parte importante" if r2 < 0.8 else "explica casi todo")
    txt = f"En palabras simples: {s['efecto']} "
    txt += (f"Sin embargo, «{res.x_source}» {quality} de lo que le pasa a «{res.y_source}» (R² = {r2:.0%}); "
            f"el resto depende de otros factores o del azar. Una estimación típicamente se equivoca en ±{res.rmse:,.4g}{_u(y_unit)}.")
    if not s["significativa"]:
        txt += " Además, con estos datos no podemos descartar que la relación sea casualidad."
    return txt


def numeric_examples(res: SLRResult, x_unit: str = "", y_unit: str = "", df: pd.DataFrame | None = None) -> list[str]:
    """Ejemplos con datos reales: percentiles de X y observaciones reales comparadas con su estimación."""
    out = []
    q25, q50, q75 = np.quantile(res.x_obs, [0.25, 0.5, 0.75])
    p = res.predict([q25, q50, q75])
    out.append(f"En el cuartil bajo de «{res.x_source}» (X = {_fmt(q25)}{_u(x_unit)}) el modelo estima Y = {_fmt(p['y_est'][0])}{_u(y_unit)}; "
               f"en la mediana (X = {_fmt(q50)}) Y = {_fmt(p['y_est'][1])}; en el cuartil alto (X = {_fmt(q75)}) Y = {_fmt(p['y_est'][2])}. "
               f"Pasar de la zona baja a la alta cambia Y en {_fmt(p['y_est'][2] - p['y_est'][0])}{_u(y_unit)}.")
    # observaciones reales: la de mayor y menor residuo estandarizado, y una típica
    order = np.argsort(np.abs(res.std_resid))
    typical = order[len(order) // 2]
    worst = int(np.argmax(np.abs(res.std_resid)))
    for tag, i in (("Observación típica", typical), ("Observación peor ajustada", worst)):
        xo, yo = res.x_obs[i], res.y_obs[i]
        est = float(res.predict([xo])["y_est"][0])
        out.append(f"{tag} (fila {res.index[i]}): X = {_fmt(xo)}{_u(x_unit)}, Y real = {_fmt(yo)}, Y estimado = {_fmt(est)}; "
                   f"error = {_fmt(yo - est)}{_u(y_unit)} ({(yo - est) / yo * 100:+.1f}% del valor real)." if yo else
                   f"{tag} (fila {res.index[i]}): X = {_fmt(xo)}, Y real = {_fmt(yo)}, Y estimado = {_fmt(est)}.")
    return out


def explain_metrics(res: SLRResult, test_metrics: dict | None = None) -> list[str]:
    out = [f"R² = {res.r2:.3f}: la recta explica {res.r2:.1%} de la variación de Y en los datos de ajuste (R² ajustado {res.r2_adj:.3f}).",
           f"RMSE = {_fmt(res.rmse)}: el error típico de estimación (misma unidad de Y). MAE = {_fmt(res.mae)}: error absoluto promedio.",
           f"F = {res.F:,.2f} (p = {res.p_F:.3g}): en regresión simple equivale a t² del coeficiente de X; contrasta si el modelo explica más que el promedio.",
           f"Correlación r = {res.r:+.3f}: intensidad y signo de la relación lineal."]
    if test_metrics:
        out.append(f"Con datos de prueba (no usados para ajustar): RMSE = {_fmt(test_metrics['RMSE'])}, MAE = {_fmt(test_metrics['MAE'])}, "
                   f"R² fuera de muestra = {test_metrics['R²_fuera_muestra']:.3f}.")
    return out
