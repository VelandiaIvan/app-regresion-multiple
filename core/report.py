"""Informe ejecutivo (Módulo 12): construye un documento estructurado exportable a Markdown y HTML."""
from __future__ import annotations

import html
from datetime import date

import numpy as np
import pandas as pd

from . import interpretation as itp
from .comparison import ComparisonResult, CandidateModel
from .context import BusinessContext
from .simulation import scenario_values, sensitivity_table, simulate


class Report:
    def __init__(self, title: str):
        self.title = title
        self.blocks: list[tuple] = []

    def h2(self, t): self.blocks.append(("h2", t)); return self
    def h3(self, t): self.blocks.append(("h3", t)); return self
    def p(self, t): self.blocks.append(("p", t)); return self
    def bullets(self, items): self.blocks.append(("ul", [str(i) for i in items if str(i).strip()])); return self
    def callout(self, t, kind="info"): self.blocks.append(("callout", t, kind)); return self

    def table(self, df: pd.DataFrame, digits: int = 4):
        d = df.copy()
        for c in d.columns:
            if pd.api.types.is_float_dtype(d[c]):
                d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{digits}g}")
        self.blocks.append(("table", d))
        return self

    # ---- salida ----
    def to_markdown(self) -> str:
        out = [f"# {self.title}", f"*Generado el {date.today():%d/%m/%Y}*", ""]
        for b in self.blocks:
            k = b[0]
            if k == "h2": out += [f"## {b[1]}", ""]
            elif k == "h3": out += [f"### {b[1]}", ""]
            elif k == "p": out += [b[1], ""]
            elif k == "ul": out += [f"- {i}" for i in b[1]] + [""]
            elif k == "callout": out += [f"> **{ {'warn': 'Atención', 'bad': 'Alerta', 'ok': 'OK', 'info': 'Nota'}[b[2]] }:** {b[1]}", ""]
            elif k == "table":
                d = b[1]
                out += ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
                out += ["| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in d.itertuples(index=False)] + [""]
        return "\n".join(out)

    def to_html(self) -> str:
        css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:32px auto;padding:0 20px;color:#1f2937;line-height:1.55}"
               "h1{color:#0f3d5e;border-bottom:3px solid #0f3d5e;padding-bottom:8px}h2{color:#0f3d5e;margin-top:32px}"
               "table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}th{background:#0f3d5e;color:#fff;text-align:left}"
               "th,td{padding:6px 8px;border:1px solid #d1d5db}tr:nth-child(even){background:#f3f6f9}"
               ".c{padding:10px 14px;border-left:4px solid #3b82f6;background:#eff6ff;margin:12px 0}"
               ".c.warn{border-color:#f59e0b;background:#fffbeb}.c.bad{border-color:#dc2626;background:#fef2f2}.c.ok{border-color:#16a34a;background:#f0fdf4}"
               ".m{color:#6b7280;font-size:12px}")
        e = html.escape
        out = [f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>{e(self.title)}</title><style>{css}</style></head><body>",
               f"<h1>{e(self.title)}</h1><p class='m'>Generado el {date.today():%d/%m/%Y}</p>"]
        for b in self.blocks:
            k = b[0]
            if k in {"h2", "h3"}: out.append(f"<{k}>{e(b[1])}</{k}>")
            elif k == "p": out.append(f"<p>{e(b[1])}</p>")
            elif k == "ul": out.append("<ul>" + "".join(f"<li>{e(i)}</li>" for i in b[1]) + "</ul>")
            elif k == "callout": out.append(f"<div class='c {b[2]}'>{e(b[1])}</div>")
            elif k == "table":
                d = b[1]
                out.append("<table><tr>" + "".join(f"<th>{e(str(c))}</th>" for c in d.columns) + "</tr>" +
                           "".join("<tr>" + "".join(f"<td>{e(str(v))}</td>" for v in r) + "</tr>" for r in d.itertuples(index=False)) + "</table>")
        out.append("</body></html>")
        return "".join(out)


def recommendations(ctx: BusinessContext, cm: CandidateModel | None, cmp: ComparisonResult, units: dict) -> list[str]:
    """Recomendaciones accionables derivadas de los resultados (no genéricas)."""
    rec = cmp.recommendation
    if cm is None:
        return ["No tome decisiones con estos datos: reúna más observaciones o variables y vuelva a modelar."] + rec.alternatives
    res, xu, yu = cm.res, units.get("x", ""), units.get("y", "")
    out = []
    if rec.status == "no confiable" or cm.key != rec.best_key and cm.diag.verdict == "no confiable":
        out.append("Use este modelo solo como referencia exploratoria: sus supuestos o su capacidad predictiva no justifican decisiones de alto impacto.")
        out += rec.alternatives
        return out
    s = itp.interpret_slope(res, xu, yu)
    out.append(f"Planifique con la relación estimada: {s['efecto']}")
    lo, hi = res.x_min, res.x_max
    out.append(f"Mantenga las decisiones dentro del rango observado de «{res.x_source}» [{lo:,.4g}; {hi:,.4g}]; fuera de él el modelo extrapola y deja de estar respaldado por los datos.")
    out.append(f"Al comunicar estimaciones, entregue RANGOS (intervalo de predicción) y no un solo número: el error típico es ±{res.rmse:,.4g}{(' ' + yu) if yu else ''}.")
    if res.b1 != 0 and ctx.restricciones.strip():
        out.append("Contraste la recomendación con las restricciones declaradas (" + ctx.restricciones.strip()[:160] + "): el modelo no las incorpora automáticamente.")
    out.append("La relación es una ASOCIACIÓN observada. Antes de invertir recursos para mover X, valide el efecto con un piloto o prueba controlada (A/B) y compare el costo de mover X contra el beneficio estimado en Y.")
    if cm.diag.get("homocedasticidad").severity != "ok":
        out.append("La dispersión del error cambia con X: la precisión no es uniforme; sea más cauto en la zona de mayor dispersión.")
    if cm.diag.get("independencia").severity in {"warn", "fail"}:
        out.append("Los errores están autocorrelacionados (típico en series de tiempo): incorpore rezagos o un modelo de series antes de proyectar a futuro.")
    if res.r2 < 0.5:
        out.append(f"X explica solo {res.r2:.0%} de Y: explore variables adicionales (regresión múltiple) para mejorar la precisión.")
    out.append("Reentrene el modelo periódicamente y monitoree el error real contra el estimado; si el error sube, los datos o el contexto cambiaron.")
    return out


def build_executive_report(*, ctx: BusinessContext, source_name: str, shape_raw: tuple, plan_records: list, change_log: pd.DataFrame,
                           corr_table: pd.DataFrame | None, cmp: ComparisonResult, chosen_key: str, units: dict,
                           sim: dict | None = None, sens: pd.DataFrame | None = None, scenarios: dict | None = None) -> Report:
    cm = cmp.models.get(chosen_key)
    rec = cmp.recommendation
    r = Report("Informe ejecutivo · Modelo de regresión lineal simple")
    r.h2("1. Resumen ejecutivo")
    if cm is None:
        r.callout("No fue posible construir un modelo con los datos disponibles.", "bad")
        return r
    res, md = cm.res, cmp.md
    kind = {"recomendado": "ok", "con reservas": "warn", "no confiable": "bad"}.get(rec.status, "info")
    if ctx.pregunta.strip():
        r.p(f"Pregunta de negocio: {ctx.pregunta.strip()}")
    s = itp.interpret_slope(res, units.get("x", ""), units.get("y", ""))
    r.callout(f"Estado del modelo «{cm.key}»: {rec.status.upper()}. {s['efecto']} R² = {res.r2:.2f}; "
              f"el error típico es ±{res.rmse:,.4g}. {rec.reasons[0] if cm.key == rec.best_key else ''}", kind)
    r.p("Ecuación: " + res.equation())

    r.h2("2. Contexto y objetivo del análisis")
    r.bullets([f"Sector: {ctx.sector or '—'}", f"Actividad: {ctx.actividad or '—'}", f"Core del negocio: {ctx.core or '—'}",
               f"Problema: {ctx.problema or '—'}", f"Variable a estimar (Y): {md.y_source}",
               f"Factores considerados: {ctx.factores or '—'}", f"Restricciones: {ctx.restricciones or 'no declaradas'}"])

    r.h2("3. Datos y preparación")
    r.p(f"Fuente: {source_name}. {shape_raw[0]:,} registros × {shape_raw[1]} columnas. Tras la preparación quedaron {md.n_with_y:,} registros con Y válida: "
        f"{len(md.train_idx):,} para entrenamiento y {len(md.test_idx):,} para prueba (partición {md.spec.method}, semilla {md.spec.seed}).")
    r.bullets(md.warnings)
    if len(change_log):
        r.h3("Bitácora de cambios (trazabilidad)")
        r.table(change_log.drop(columns=[c for c in ["método"] if False], errors="ignore"))

    r.h2("4. Relaciones identificadas")
    if corr_table is not None and len(corr_table):
        t = corr_table.head(8)[["variable", "n", "pearson_r", "spearman_rho", "dirección", "intensidad", "alertas"]]
        r.table(t, 3)
    r.p("Correlación no implica causalidad: los resultados describen asociaciones en los datos observados.")

    r.h2("5. Variable más relevante y modelo recomendado")
    r.bullets(rec.reasons)
    if rec.note_r2:
        r.p(rec.note_r2)
    cols = ["ranking", "modelo", "R²", "R²_prueba", "RMSE_prueba", "p_β1", "puntaje", "alertas"]
    r.table(cmp.table[cols].head(6), 3)

    r.h2("6. Calidad estadística")
    r.bullets(itp.explain_metrics(res, cm.test_metrics))
    r.table(res.coef_table(), 4)
    r.table(cm.diag.table()[["supuesto", "resultado", "interpretación"]])
    r.callout(cm.diag.summary, {"apto": "ok", "con reservas": "warn", "no confiable": "bad"}[cm.diag.verdict])

    r.h2("7. Interpretación de coeficientes")
    ii, ss = itp.interpret_intercept(res, units.get("x", ""), units.get("y", "")), itp.interpret_slope(res, units.get("x", ""), units.get("y", ""))
    r.bullets([ii["tecnica"], ii["viabilidad"], ss["tecnica"], ss["magnitud"], itp.plain_language(res, units.get("x", ""), units.get("y", ""))])
    r.h3("Ejemplos con los datos")
    r.bullets(itp.numeric_examples(res, units.get("x", ""), units.get("y", "")))

    r.h2("8. Estimaciones y simulaciones")
    if scenarios:
        rows = []
        for k, xv in scenarios.items():
            p = res.predict([xv]).iloc[0]
            rows.append({"escenario": k, "X": xv, "Y estimado": p["y_est"], "IC inf": p["ic_inf"], "IC sup": p["ic_sup"],
                         "IP inf": p["ip_inf"], "IP sup": p["ip_sup"]})
        r.table(pd.DataFrame(rows), 4)
        r.p("IC = intervalo de confianza del promedio; IP = intervalo de predicción de una observación individual (siempre más ancho).")
    if sim:
        r.p(f"Simulación: X de {sim['x_base']:,.4g} a {sim['x_new']:,.4g} → Y estimado de {sim['y_base']:,.4g} a {sim['y_new']:,.4g} "
            f"({sim['dy']:+,.4g}" + (f"; {sim['dy_pct']:+.1f}%" if not np.isnan(sim['dy_pct']) else "") + ").")
    if sens is not None and len(sens):
        r.h3("Análisis de sensibilidad")
        r.table(sens[["escenario", "x_inicial", "x_nuevo", "y_inicial", "y_nuevo", "dif_absoluta", "var_%", "extrapolación"]], 4)

    r.h2("9. Riesgos y limitaciones")
    lim = list(rec.warnings) + [f"{c.name}: {c.explanation}" for c in cm.diag.checks if c.severity in {"warn", "fail"}]
    lim += ["Regresión simple: una sola variable explicativa; omite el efecto de otros factores (sesgo por variable omitida).",
            "Asociación no es causalidad.", "Los intervalos suponen que los supuestos del modelo se cumplen."]
    if len(cmp.models) > 3:
        lim.append(f"Se compararon {len(cmp.models)} modelos con los mismos datos de prueba: existe un leve riesgo de optimismo al elegir el mejor (selección múltiple).")
    r.bullets(lim)

    r.h2("10. Recomendaciones accionables")
    r.bullets(recommendations(ctx, cm, cmp, units))

    r.h2("Anexo · Metodología y fórmulas")
    r.bullets(["Modelo: Ŷ = β0 + β1·X estimado por mínimos cuadrados ordinarios. β1 = Sxy/Sxx; β0 = ȳ − β1·x̄.",
               "Error estándar residual: σ̂ = √(SSE/(n−2)). t = β/EE(β). F = (SSR/1)/(SSE/(n−2)).",
               "Intervalo de confianza de la respuesta media: Ŷ0 ± t·σ̂·√(1/n + (x0−x̄)²/Sxx).",
               "Intervalo de predicción: Ŷ0 ± t·σ̂·√(1 + 1/n + (x0−x̄)²/Sxx).",
               "Validación: partición entrenamiento/prueba sin fuga (estadísticas de limpieza solo con entrenamiento) y validación cruzada dejando uno fuera (PRESS).",
               "Selección: puntaje multicriterio (predicción 40, supuestos 20, negocio 15, inferencia 15, calidad de datos 10, menos penalizaciones); no se elige por R²."])
    return r
