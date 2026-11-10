"""Informe ejecutivo del modelo de regresión MÚLTIPLE (exportable a Markdown y HTML)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import interpretation_multi as itp
from .comparison import ComparisonResult
from .context import BusinessContext
from .report import Report
from .selection import MultiModel


def recommendations(ctx: BusinessContext, cm: MultiModel | None, cmp: ComparisonResult, units: dict) -> list[str]:
    """Recomendaciones accionables derivadas de los resultados (no genéricas)."""
    rec = cmp.recommendation
    if cm is None:
        return ["No tome decisiones con estos datos: reúna más observaciones o variables y vuelva a modelar."] + rec.alternatives
    res, yu = cm.res, units.get("y", "")
    out = []
    if rec.status == "no confiable" or (cm.key != rec.best_key and cm.diag.verdict == "no confiable"):
        out.append("Use este modelo solo como referencia exploratoria: sus supuestos o su capacidad predictiva no justifican decisiones de alto impacto.")
        return out + rec.alternatives
    items = itp.interpret_all(res, yu)
    sig = [d for d in items if d["significativa"]]
    if sig:
        out.append(f"Concentre la gestión en «{sig[0]['variable']}», la variable con mayor peso relativo: {sig[0]['efecto']}")
        for d in sig[1:3]:
            out.append(f"Considere también «{d['variable']}»: {d['efecto']}")
    not_sig = [d["variable"] for d in items if not d["significativa"]]
    if not_sig:
        out.append("No base decisiones en " + ", ".join(f"«{v}»" for v in not_sig) + ": con estos datos no se confirma que tengan un efecto propio.")
    rng = "; ".join(f"{c} [{res.x_range(c)[0]:,.4g}; {res.x_range(c)[1]:,.4g}]" for c in res.x_names)
    out.append(f"Mantenga las decisiones dentro de los rangos observados ({rng}); fuera de ellos el modelo extrapola.")
    err = float(np.sqrt(np.mean((res.y_obs - res.y_back(res.fitted)) ** 2)))
    out.append(f"Al comunicar estimaciones entregue RANGOS (intervalo de predicción) y no un solo número: el error típico es ±{err:,.4g}{(' ' + yu) if yu else ''}.")
    if ctx.restricciones.strip():
        out.append("Contraste la recomendación con las restricciones declaradas (" + ctx.restricciones.strip()[:160] + "): el modelo no las incorpora automáticamente.")
    out.append("Los coeficientes son ASOCIACIONES observadas, cada una con las demás variables constantes. Antes de invertir para mover una "
               "palanca, valide el efecto con un piloto o prueba controlada y compare su costo contra el beneficio estimado.")
    if cm.diag.get("multicolinealidad").severity != "ok":
        out.append("Hay variables que se solapan entre sí: no interprete sus coeficientes por separado; úselo para estimar, no para atribuir efectos.")
    if cm.diag.get("homocedasticidad").severity != "ok":
        out.append("La precisión no es uniforme: sea más cauto en los niveles donde el error se dispersa más (use los errores estándar robustos).")
    if cm.diag.get("independencia").severity in {"warn", "fail"}:
        out.append("Los errores están autocorrelacionados (típico en series de tiempo): incorpore rezagos o un modelo de series antes de proyectar a futuro.")
    if res.r2_adj < 0.5:
        out.append(f"El modelo explica {res.r2_adj:.0%} de Y: busque variables adicionales relevantes para mejorar la precisión.")
    out.append("Reentrene el modelo periódicamente y compare el error real contra el estimado; si sube, los datos o el contexto cambiaron.")
    return out


def build_executive_report(*, ctx: BusinessContext, source_name: str, shape_raw: tuple, change_log: pd.DataFrame,
                           corr_table: pd.DataFrame | None, cmp: ComparisonResult, chosen_key: str, units: dict,
                           understanding: dict | None = None, sim: dict | None = None, sens: pd.DataFrame | None = None,
                           scenarios: dict | None = None) -> Report:
    cm = cmp.models.get(chosen_key)
    rec = cmp.recommendation
    r = Report("Informe ejecutivo · Modelo de regresión lineal múltiple")
    r.h2("1. Resumen ejecutivo")
    if cm is None:
        r.callout("No fue posible construir un modelo con los datos disponibles.", "bad")
        return r
    res, md = cm.res, cmp.md
    yu = units.get("y", "")
    kind = {"recomendado": "ok", "con reservas": "warn", "no confiable": "bad"}.get(rec.status, "info")
    if ctx.pregunta.strip():
        r.p(f"Pregunta de negocio: {ctx.pregunta.strip()}")
    estado = rec.status.upper() if cm.key == rec.best_key else "ELEGIDO POR EL USUARIO (no es el recomendado)"
    r.callout(f"Modelo para estimar «{res.y_name}» con {res.k} variables ({', '.join(res.x_names)}). Estado: {estado}. "
              f"R² ajustado = {res.r2_adj:.2f}; coeficientes significativos: {res.n_signif} de {res.k}.", kind)
    r.p(itp.plain_language(res, yu))
    r.p("Ecuación: " + res.equation())

    r.h2("2. Los datos y el objetivo")
    if understanding:
        r.bullets([understanding["tema"], understanding["unidad"], understanding["tamaño"]] + list(understanding["calidad"]))
    extra = [f"Variable a estimar (Y): {md.y_source}"]
    if ctx.sector.strip():
        extra.append(f"Sector: {ctx.sector}")
    if ctx.problema.strip():
        extra.append(f"Problema: {ctx.problema}")
    if ctx.restricciones.strip():
        extra.append(f"Restricciones: {ctx.restricciones}")
    r.bullets(extra)

    r.h2("3. Preparación de los datos")
    r.p(f"Fuente: {source_name}. {shape_raw[0]:,} registros × {shape_raw[1]} columnas. Tras la preparación quedaron {md.n_with_y:,} registros con Y válida: "
        f"{len(md.train_idx):,} para entrenamiento y {len(md.test_idx):,} para prueba (partición {md.spec.method}, semilla {md.spec.seed}).")
    r.bullets(md.warnings)
    if len(change_log):
        r.h3("Bitácora de cambios (trazabilidad)")
        r.table(change_log)

    r.h2("4. Relación de cada variable con Y")
    if corr_table is not None and len(corr_table):
        r.table(corr_table.head(10)[["variable", "n", "pearson_r", "spearman_rho", "dirección", "intensidad", "alertas"]], 3)
    if cmp.singles is not None and len(cmp.singles):
        best_single = cmp.singles.iloc[0]
        r.p(f"Por sí sola, la mejor variable («{best_single['variable']}») explicaría {best_single['R² sola']:.0%} de Y; "
            f"el modelo múltiple elegido explica {res.r2:.0%} (R² ajustado {res.r2_adj:.0%}).")

    r.h2("5. Selección del modelo")
    r.bullets(rec.reasons)
    if rec.note_r2:
        r.p(rec.note_r2)
    cols = ["ranking", "modelo", "n_variables", "puntaje", "R²_ajust", "R²_prueba", "RMSE_prueba", "coef_signif", "VIF_máx", "alertas"]
    r.table(cmp.table[cols].head(8), 3)
    r.p("Se compararon combinaciones de 2 o más variables. No se elige por el R² más alto: el puntaje combina capacidad predictiva con datos nuevos, "
        "supuestos, significancia de los coeficientes y coherencia con el negocio, y penaliza variables que no aportan.")

    r.h2("6. Calidad estadística")
    r.bullets(itp.explain_metrics(res, cm.test_metrics))
    r.table(res.coef_table()[["coeficiente", "estimación", "error_estándar", "t", "p_valor", "IC_inf", "IC_sup", "significativo", "VIF"]], 4)
    r.table(cm.diag.table()[["supuesto", "resultado", "interpretación"]])
    r.callout(cm.diag.summary, {"apto": "ok", "con reservas": "warn", "no confiable": "bad"}[cm.diag.verdict])

    r.h2("7. Interpretación de los coeficientes")
    ii = itp.interpret_intercept(res, yu)
    r.bullets([d["efecto"] + ("" if d["significativa"] else " (No significativo.)") for d in itp.interpret_all(res, yu)]
              + [ii["sencilla"]])
    r.h3("Importancia relativa")
    r.table(itp.importance_table(res), 3)
    r.h3("Ejemplos con los datos")
    r.bullets(itp.numeric_examples(res, yu))

    r.h2("8. Estimaciones y simulaciones")
    if scenarios:
        rows = []
        for name, vals in scenarios.items():
            p = res.predict(vals).iloc[0]
            rows.append({"escenario": name, **{c: vals[c] for c in res.x_names}, "Y estimado": p["y_est"], "IC inf": p["ic_inf"],
                         "IC sup": p["ic_sup"], "IP inf": p["ip_inf"], "IP sup": p["ip_sup"]})
        r.table(pd.DataFrame(rows), 4)
        r.p("IC = intervalo de confianza del promedio; IP = intervalo de predicción de un caso individual (siempre más ancho).")
    if sim:
        r.p(f"Simulación: «{res.y_name}» estimado pasa de {sim['y_base']:,.4g} a {sim['y_new']:,.4g} "
            f"({sim['dy']:+,.4g}" + (f"; {sim['dy_pct']:+.1f}%" if not np.isnan(sim["dy_pct"]) else "") + ").")
        moved = sim["contrib"][sim["contrib"]["cambio_X"].abs() > 1e-12]
        if len(moved):
            r.table(moved, 4)
    if sens is not None and len(sens):
        r.h3("Análisis de sensibilidad (una variable a la vez)")
        r.table(sens[["variable", "escenario", "x_inicial", "x_nuevo", "y_inicial", "y_nuevo", "dif_absoluta", "var_%", "extrapolación"]], 4)

    r.h2("9. Riesgos y limitaciones")
    lim = list(rec.warnings) + [f"{c.name}: {c.explanation}" for c in cm.diag.checks if c.severity in {"warn", "fail"}]
    lim += ["Los coeficientes son efectos parciales: valen si las demás variables se mantienen constantes, lo que no siempre ocurre en la práctica.",
            "Puede haber variables relevantes que no están en los datos (sesgo por variable omitida).",
            "Asociación no es causalidad.", "Los intervalos suponen que los supuestos del modelo se cumplen."]
    if len(cmp.models) > 3:
        lim.append(f"Se compararon {len(cmp.models)} combinaciones con los mismos datos de prueba: existe un leve riesgo de optimismo al elegir la mejor.")
    r.bullets(lim)

    r.h2("10. Recomendaciones accionables")
    r.bullets(recommendations(ctx, cm, cmp, units))

    r.h2("Anexo · Metodología y fórmulas")
    r.bullets(["Modelo: Ŷ = β0 + β1·X1 + … + βk·Xk estimado por mínimos cuadrados ordinarios: β = (XᵀX)⁻¹Xᵀy.",
               "Error estándar residual: σ̂ = √(SSE/(n−k−1)). t = β/EE(β). F = (SSR/k)/(SSE/(n−k−1)). R² ajustado = 1 − (1−R²)(n−1)/(n−k−1).",
               "Multicolinealidad: VIF = 1/(1 − R²j), donde R²j es el R² de la variable j explicada por las demás X.",
               "Intervalo de confianza de la respuesta media: Ŷ0 ± t·σ̂·√(x0ᵀ(XᵀX)⁻¹x0). Intervalo de predicción: Ŷ0 ± t·σ̂·√(1 + x0ᵀ(XᵀX)⁻¹x0).",
               "Validación: partición entrenamiento/prueba sin fuga (estadísticas de limpieza solo con entrenamiento) y validación cruzada dejando uno fuera (PRESS).",
               "Selección: todas las combinaciones de 2 o más variables (o paso a paso si hay más de 9), puntaje multicriterio "
               "(predicción 40, supuestos 20, inferencia 15, negocio 15, calidad de datos 10, menos penalizaciones) y preferencia por el modelo más sencillo."])
    return r
