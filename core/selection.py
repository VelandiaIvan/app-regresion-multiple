"""Selección del mejor modelo de regresión MÚLTIPLE (siempre con 2 o más variables explicativas).

Cómo se eligen las combinaciones:
  1. Se apartan las variables que no estarán disponibles al decidir o con riesgo ALTO de fuga de información.
  2. Con hasta 9 variables se exploran TODAS las combinaciones de 2 o más; con más, se usa selección paso a paso
     (hacia adelante desde la mejor pareja y hacia atrás desde el modelo completo).
  3. Cada combinación se evalúa con validación cruzada «dejando uno fuera» (solo datos de entrenamiento).
  4. Las finalistas se ajustan completas: diagnóstico de supuestos, datos de prueba y puntaje multicriterio.
  5. Ante puntajes casi iguales gana el modelo con MENOS variables (parsimonia).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .comparison import ComparisonResult, ModelingData, Recommendation
from .context import leakage_risk
from .diagnostics import Diagnostics, run_diagnostics_multi
from .multiple import MLRResult, fit_mlr, quick_loocv
from .regression import RegressionError, prediction_metrics

MIN_VARS = 2
EXHAUSTIVE_LIMIT = 9
SEP = " + "


@dataclass
class MultiModel:
    key: str
    x_cols: list
    res: MLRResult
    diag: Diagnostics
    train_metrics: dict
    test_metrics: dict | None
    skill_loocv: float
    skill_test: float | None
    skill: float
    sign_ok: bool | None
    sign_issues: list
    leak: str
    leak_msg: str
    excluded: str | None
    coverage: float
    score: float = 0.0
    parts: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)

    @property
    def n_vars(self) -> int:
        return len(self.x_cols)

    @property
    def all_significant(self) -> bool:
        return self.res.n_signif == self.res.k


def model_key(cols: list) -> str:
    return SEP.join(cols)


# ----------------------------------------------------------------------------
# filtro de variables y exploración de combinaciones
# ----------------------------------------------------------------------------
def screen_variables(md: ModelingData, x_cols: list, known_in_advance: dict, leak_cleared: set,
                     declared_factors: set) -> tuple[list, dict, list]:
    """Devuelve (variables utilizables, riesgo de fuga por variable, mensajes de las apartadas)."""
    usable, leaks, dropped = [], {}, []
    tr = md.df.loc[md.train_idx]
    for c in dict.fromkeys(x_cols):
        if c not in md.df.columns or not pd.api.types.is_numeric_dtype(md.df[c]):
            dropped.append(f"{c}: no es numérica (codifíquela en el módulo de preparación).")
            continue
        if c in {md.y_col, md.y_source}:
            continue
        pair = tr[[c, md.y_source]].dropna()
        if len(pair) < 8 or pair[c].nunique() < 2:
            dropped.append(f"{c}: sin datos suficientes o sin variación en entrenamiento.")
            continue
        r = float(np.corrcoef(pair[c].to_numpy(float), pair[md.y_source].to_numpy(float))[0, 1])
        src = md.transforms[c]["source"] if c in md.transforms else c
        leak, msg = leakage_risk(src, md.y_source, r, declared_factor=src in declared_factors)
        if leak != "ninguna" and (c in leak_cleared or src in leak_cleared):
            leak, msg = "ninguna", ""
        if known_in_advance.get(c) is False:
            dropped.append(f"{c}: no estará disponible en el momento de decidir/estimar (fuga temporal).")
            continue
        if leak == "alta":
            dropped.append(f"{c}: apartada por riesgo ALTO de fuga de información (|r| = {abs(r):.3f} con Y). {msg}")
            continue
        usable.append(c)
        leaks[c] = (leak, msg)
    return usable, leaks, dropped


def _candidate_subsets(Xv: np.ndarray, yv: np.ndarray, names: list, max_finalists: int) -> list[list]:
    p = len(names)
    scored: dict[tuple, float] = {}

    def score(idx: tuple) -> float:
        if idx not in scored:
            scored[idx] = quick_loocv(Xv[:, idx], yv)[0]
        return scored[idx]

    if p <= EXHAUSTIVE_LIMIT:
        for size in range(MIN_VARS, p + 1):
            for idx in combinations(range(p), size):
                score(idx)
    else:
        for idx in combinations(range(p), MIN_VARS):                      # todas las parejas
            score(idx)
        cur = min((i for i in scored if len(i) == MIN_VARS), key=lambda i: scored[i])
        while len(cur) < p:                                               # hacia adelante
            options = [tuple(sorted(cur + (j,))) for j in range(p) if j not in cur]
            nxt = min(options, key=score)
            if score(nxt) > score(cur) * 1.02 and len(cur) >= 4:
                break
            cur = nxt
        cur = tuple(range(p))                                             # hacia atrás
        score(cur)
        while len(cur) > MIN_VARS:
            options = [tuple(j for j in cur if j != d) for d in cur]
            cur = min(options, key=score)
    finite = {i: s for i, s in scored.items() if np.isfinite(s)}
    if not finite:
        return []
    ranked = sorted(finite, key=lambda i: finite[i])
    finalists = ranked[:max_finalists]
    for size in sorted({len(i) for i in finite}):                         # el mejor de cada tamaño siempre compite
        best_size = min((i for i in finite if len(i) == size), key=lambda i: finite[i])
        if best_size not in finalists:
            finalists.append(best_size)
    return [[names[j] for j in idx] for idx in finalists[:max_finalists + 6]]


# ----------------------------------------------------------------------------
# construcción y puntaje de un modelo
# ----------------------------------------------------------------------------
def build_multi(md: ModelingData, cols: list, *, alpha: float, expected_sign: dict, leaks: dict) -> MultiModel:
    df = md.df
    keep = list(dict.fromkeys(cols + [md.y_col, md.y_source]))
    tr = df.loc[md.train_idx, keep].dropna()
    if len(tr) < max(8, len(cols) + 5):
        raise RegressionError(f"Solo {len(tr)} observaciones completas de entrenamiento para {len(cols)} variables.")
    res = fit_mlr(tr[cols], tr[md.y_col], y_name=md.y_source, alpha=alpha, y_transform=md.y_transform,
                  y_raw=tr[md.y_source], index=tr.index.to_numpy())
    diag = run_diagnostics_multi(res, ordered=md.ordered)
    base = float(np.mean(res.y_obs))
    tm = prediction_metrics(res.y_obs, res.y_back(res.fitted), y_baseline=base)
    test_m, skill_test = None, None
    if md.test_idx:
        te = df.loc[md.test_idx, keep].dropna()
        if len(te) >= 5:
            yhat = res.y_back(res.predict_model_scale(te[cols]))
            test_m = prediction_metrics(te[md.y_source].to_numpy(float), yhat, y_baseline=base)
            skill_test = test_m.get("habilidad")
    n = res.n
    base_loo = float(np.sqrt(np.mean(((res.y - res.y_mean) * n / (n - 1)) ** 2)))
    skill_loo = 1 - res.rmse_loocv / base_loo if base_loo > 0 else np.nan
    vals = [v for v in (skill_test, skill_loo) if v is not None and np.isfinite(v)]
    skill = float(np.mean(vals)) if vals else np.nan
    issues = [c for c in cols if expected_sign.get(c) and np.sign(res.coef(c)) != expected_sign[c]]
    declared = [c for c in cols if expected_sign.get(c)]
    sign_ok = None if not declared else (len(issues) == 0)
    med = [c for c in cols if leaks.get(c, ("ninguna", ""))[0] == "media"]
    leak = "media" if med else "ninguna"
    leak_msg = "; ".join(f"«{c}»: {leaks[c][1]}" for c in med)
    coverage = int(df[keep].dropna().shape[0]) / max(len(df), 1)
    return MultiModel(key=model_key(cols), x_cols=list(cols), res=res, diag=diag, train_metrics=tm, test_metrics=test_m,
                      skill_loocv=skill_loo, skill_test=skill_test, skill=skill, sign_ok=sign_ok, sign_issues=issues,
                      leak=leak, leak_msg=leak_msg, excluded=None, coverage=coverage)


def score_multi(m: MultiModel) -> None:
    res, diag = m.res, m.diag
    sev_pts = {"ok": 1.0, "warn": 0.5, "fail": 0.0}
    weights = {"linealidad": 6, "homocedasticidad": 4, "normalidad": 3, "independencia": 4, "multicolinealidad": 3}
    applicable = {key: w for key, w in weights.items() if diag.get(key).severity != "na"}
    wsum = sum(applicable.values()) or 1
    assump = sum(w * sev_pts[diag.get(key).severity] for key, w in applicable.items()) / wsum * 20
    pred = 40 * float(np.clip(m.skill if np.isfinite(m.skill) else 0, 0, 1))
    share_sig = res.n_signif / res.k
    infer = (5 if res.p_F < res.alpha else 0) + 5 * share_sig + 5 * float(np.clip(res.r2_adj / 0.5, 0, 1))
    biz = {True: 15, None: 9, False: 0}[m.sign_ok]
    per_param = res.n / (res.k + 1)
    n_pts = 5 if per_param >= 15 else 2.5 if per_param >= 8 else 0
    dq = 5 * float(np.clip(m.coverage / 0.9, 0, 1)) + n_pts
    pen, flags = 0.0, []
    not_sig = [c for c in m.x_cols if res.p_of(c) >= res.alpha]
    if not_sig:
        pen += 4 * len(not_sig)
        flags.append("sin efecto significativo: " + ", ".join(not_sig))
    if m.leak == "media":
        pen += 8
        flags.append("alguna X podría conocerse después de Y")
    if m.test_metrics and m.train_metrics.get("RMSE", 0) > 0:
        ratio = m.test_metrics["RMSE"] / m.train_metrics["RMSE"]
        pen += 10 * float(np.clip((ratio - 1.3) / 0.7, 0, 1))             # gradual: sin saltos bruscos alrededor de un umbral
        if ratio > 1.5:
            flags.append(f"error en prueba {ratio:.1f}× el de entrenamiento (posible sobreajuste)")
    if diag.get("influencia").severity == "fail":
        pen += 5
        flags.append("observaciones muy influyentes")
    if diag.get("multicolinealidad").severity == "fail":
        pen += 6
        flags.append("multicolinealidad alta (VIF > 10)")
    if m.sign_ok is False:
        flags.append("signo contrario al esperado: " + ", ".join(m.sign_issues))
    if per_param < 8:
        flags.append("pocas observaciones por parámetro")
    m.parts = {"predictivo (40)": pred, "supuestos (20)": assump, "inferencia (15)": infer, "negocio (15)": biz,
               "calidad de datos (10)": dq, "penalizaciones": -pen}
    m.score = float(np.clip(pred + assump + infer + biz + dq - pen, 0, 100))
    m.flags = flags


# ----------------------------------------------------------------------------
# tablas
# ----------------------------------------------------------------------------
def multi_table(models: dict) -> pd.DataFrame:
    rows = []
    for key, m in models.items():
        r, d = m.res, m.diag
        tm = m.test_metrics or {}
        rows.append({
            "modelo": key, "n_variables": r.k, "n_entrenamiento": r.n, "n_prueba": tm.get("n", 0),
            "R²": r.r2, "R²_ajust": r.r2_adj, "RMSE_entren.": m.train_metrics["RMSE"], "MAE_entren.": m.train_metrics["MAE"],
            "RMSE_prueba": tm.get("RMSE", np.nan), "MAE_prueba": tm.get("MAE", np.nan), "R²_prueba": tm.get("R²_fuera_muestra", np.nan),
            "RMSE_LOOCV": r.rmse_loocv, "habilidad": m.skill, "p_F": r.p_F, "coef_signif": f"{r.n_signif}/{r.k}",
            "VIF_máx": max(r.vif.values()), "linealidad": d.get("linealidad").icon, "independencia": d.get("independencia").icon,
            "homocedasticidad": d.get("homocedasticidad").icon, "normalidad": d.get("normalidad").icon,
            "multicolinealidad": d.get("multicolinealidad").icon, "influencia": d.get("influencia").icon,
            "signo_ok": {True: "sí", False: "no", None: "—"}[m.sign_ok], "riesgo_fuga": m.leak,
            "puntaje": m.score, "alertas": "; ".join(m.flags),
        })
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.sort_values(["puntaje", "n_variables"], ascending=[False, True]).reset_index(drop=True)
    t.insert(0, "ranking", np.arange(1, len(t) + 1))
    t["ranking_por_R²"] = t["R²"].rank(ascending=False, method="min").astype(int)
    return t


def singles_table(md: ModelingData, cols: list) -> pd.DataFrame:
    """Referencia: qué lograría cada variable POR SÍ SOLA (regresión simple), para ver cuánto aporta combinarlas."""
    rows = []
    tr = md.df.loc[md.train_idx]
    for c in cols:
        d = tr[[c, md.y_col]].dropna()
        if len(d) < 5 or d[c].nunique() < 2:
            continue
        loo, r2a = quick_loocv(d[[c]].to_numpy(float), d[md.y_col].to_numpy(float))
        r = float(np.corrcoef(d[c].to_numpy(float), d[md.y_col].to_numpy(float))[0, 1])
        rows.append({"variable": c, "r con Y": r, "R² sola": r * r, "R² ajust. sola": r2a, "RMSE LOOCV sola (escala modelo)": loo})
    out = pd.DataFrame(rows)
    return out.sort_values("R² sola", ascending=False).reset_index(drop=True) if not out.empty else out


# ----------------------------------------------------------------------------
# recomendación
# ----------------------------------------------------------------------------
def _alternatives(best: MultiModel | None, models: dict) -> list[str]:
    out: list[str] = []
    ref = [best] if best else list(models.values())
    sev = lambda key: [m.diag.get(key).severity for m in ref]
    if any(s in {"warn", "fail"} for s in sev("linealidad")):
        out.append("Relación curva: en el módulo de preparación transforme alguna X (ln, √) o agregue un término cuadrático, y vuelva a modelar.")
    if any(s in {"warn", "fail"} for s in sev("homocedasticidad")):
        out.append("Varianza no constante: apóyese en los errores estándar robustos (HC3) o modele ln(Y).")
    if any(s == "fail" for s in sev("independencia")):
        out.append("Errores autocorrelacionados: incluya rezagos de Y o use un modelo de series de tiempo.")
    if any(s in {"warn", "fail"} for s in sev("normalidad")):
        out.append("Residuos no normales: revise atípicos o transforme Y; no confíe en los intervalos de predicción.")
    if any(s in {"warn", "fail"} for s in sev("multicolinealidad")):
        out.append("Variables redundantes: retire una de las X muy correlacionadas entre sí o use el modelo con menos variables.")
    if any(s in {"warn", "fail"} for s in sev("influencia")):
        out.append("Observaciones muy influyentes: valide esos registros y compare el modelo con y sin ellos.")
    if ref and all(m.res.r2_adj < 0.3 for m in ref):
        out.append("Las variables disponibles explican poco de Y: agregue variables relevantes (incluidas categóricas codificadas) o segmente los datos.")
    if ref and any(m.res.n / (m.res.k + 1) < 10 for m in ref):
        out.append("Pocas observaciones por variable: reúna más datos o use menos variables (regla práctica: 10–15 observaciones por parámetro).")
    if not out:
        out.append("Para mayor precisión, evalúe interacciones entre variables o variables adicionales del negocio.")
    return out


def recommend_multi(models: dict, dropped: list, alpha: float) -> Recommendation:
    if not models:
        return Recommendation(None, "sin candidatos",
                              ["No se pudo construir ningún modelo con 2 o más variables explicativas."], list(dropped),
                              ["Seleccione al menos 2 variables numéricas utilizables como X (puede codificar categóricas en el módulo de preparación)."])
    strict = {k: m for k, m in models.items() if m.res.p_F < alpha and m.sign_ok is not False and m.all_significant
              and m.diag.get("multicolinealidad").severity != "fail" and m.leak == "ninguna"}
    relaxed = {k: m for k, m in models.items() if m.res.p_F < alpha and m.sign_ok is not False}
    leak_forced = False
    if strict and relaxed:
        # Un modelo «limpio» solo se prefiere si no sacrifica demasiado: si los que usan una variable dudosa son mucho
        # mejores, se recomienda ese con reservas y se le pide al usuario confirmar que la variable se conoce antes.
        best_relaxed = max(relaxed.values(), key=lambda m: m.score)
        if best_relaxed.leak != "ninguna" and best_relaxed.score > max(m.score for m in strict.values()) + 10:
            strict, leak_forced = {}, True
    pool = strict or relaxed or models
    top = max(m.score for m in pool.values())
    near = [m for m in pool.values() if m.score >= top - 1.0]              # empate técnico → gana el más sencillo
    best = min(near, key=lambda m: (m.n_vars, -m.score))
    r = best.res
    reasons, warns = [], []
    reasons.append(f"Mayor puntaje multicriterio ({best.score:.0f}/100) entre {len(models)} combinaciones finalistas de 2 o más variables"
                   + (" (ante puntajes casi iguales se prefirió el modelo con menos variables)." if len(near) > 1 else "."))
    reasons.append(f"Usa {r.k} variables: {', '.join(best.x_cols)}. Coeficientes significativos: {r.n_signif} de {r.k}.")
    if best.skill_test is not None and np.isfinite(best.skill_test):
        reasons.append(f"Con datos que no usó para ajustarse, reduce el error de predicción en {best.skill_test * 100:.0f}% "
                       f"frente a predecir siempre el promedio (RMSE de prueba {best.test_metrics['RMSE']:,.4g}).")
    reasons.append(f"Validación cruzada (dejando uno fuera): mejora de {best.skill_loocv * 100:.0f}% sobre el promedio.")
    reasons.append(f"El modelo en conjunto es significativo (F = {r.F:,.1f}; p = {r.p_F:.3g}); R² ajustado = {r.r2_adj:.2f}.")
    if best.sign_ok:
        reasons.append("Los signos de los coeficientes coinciden con lo esperado por el negocio.")
    reasons.append(f"Supuestos: {best.diag.verdict} — {best.diag.summary}")
    status = "recomendado"
    if not strict:
        if leak_forced:
            status = "con reservas"
            warns.append("El mejor modelo usa una variable que podría conocerse DESPUÉS de Y. Los modelos sin ella son mucho menos precisos; "
                         "confirme en el módulo 6 que esa variable está disponible antes de estimar, o retírela.")
        elif relaxed:
            status = "con reservas"
            warns.append("Ninguna combinación tiene TODOS sus coeficientes significativos y sin multicolinealidad alta; "
                         "se recomienda la mejor disponible, pero interprete con cautela las variables no significativas.")
        else:
            status = "no confiable"
            warns.append("Ningún modelo cumplió los filtros mínimos (significancia global y signos esperados); se muestra el mejor solo como referencia.")
    skill_ok = np.isfinite(best.skill) and best.skill >= 0.5
    if best.diag.verdict == "no confiable":
        if best.diag.get("linealidad").severity == "fail" or not skill_ok:
            status = "no confiable"
            warns.append("Los supuestos del modelo no se sostienen: no use los intervalos ni p-valores para decidir.")
        else:
            status = "con reservas" if status == "recomendado" else status
            failed = [c.name.lower() for c in best.diag.checks if c.severity == "fail" and c.key != "influencia"]
            warns.append("Predice bien con datos nuevos, pero la INFERENCIA no es confiable (incumple: " + ", ".join(failed) +
                         "). Úselo para estimaciones puntuales, no para afirmar significancia.")
    elif best.diag.verdict == "con reservas" and status == "recomendado":
        status = "con reservas"
    if np.isfinite(best.skill) and best.skill < 0.10:
        status = "no confiable"
        warns.append("El modelo casi no mejora a predecir el promedio: capacidad predictiva insuficiente.")
    if r.r2_adj < 0.10:
        status = "no confiable"
        warns.append(f"R² ajustado = {r.r2_adj:.2f}: las variables explican muy poco del comportamiento de Y.")
    if best.leak == "media":
        warns.append(best.leak_msg)
    warns += [d for d in dropped if "fuga" in d]
    if best.test_metrics and best.train_metrics["RMSE"] > 0 and best.test_metrics["RMSE"] / best.train_metrics["RMSE"] > 1.5:
        warns.append("El error en prueba es mucho mayor que en entrenamiento: cautela con el sobreajuste o con cambios en el tiempo.")
    top_r2 = max(models.values(), key=lambda m: m.res.r2)
    note = ""
    if top_r2.key != best.key:
        why = []
        extra = [c for c in top_r2.x_cols if top_r2.res.p_of(c) >= alpha]
        if extra:
            why.append("incluye variables sin efecto significativo (" + ", ".join(extra) + ")")
        if np.isfinite(top_r2.skill) and np.isfinite(best.skill) and top_r2.skill < best.skill - 0.005:
            why.append(f"predice peor con datos nuevos (mejora {top_r2.skill * 100:.0f}% vs {best.skill * 100:.0f}%)")
        if top_r2.diag.get("multicolinealidad").severity in {"warn", "fail"} and best.diag.get("multicolinealidad").severity == "ok":
            why.append("sus variables son redundantes entre sí (VIF alto)")
        if top_r2.sign_ok is False:
            why.append("tiene signos contrarios a lo esperado por el negocio")
        if not why:
            why.append(f"la mejora es mínima y no compensa la complejidad adicional (puntaje {top_r2.score:.0f} vs {best.score:.0f})")
        note = (f"El modelo con mayor R² es «{top_r2.key}» (R² = {top_r2.res.r2:.3f}; el R² nunca baja al agregar variables), "
                "pero no fue el elegido: " + "; ".join(why) + ".")
    alts = _alternatives(best, models)
    return Recommendation(best.key, status, reasons, warns, alts if status != "recomendado" else alts[:2], note)


def run_multi_comparison(md: ModelingData, x_cols: list, *, alpha: float = 0.05, expected_sign: dict | None = None,
                         known_in_advance: dict | None = None, leak_cleared: set | None = None,
                         declared_factors: set | None = None, max_finalists: int = 8) -> ComparisonResult:
    expected_sign = expected_sign or {}
    usable, leaks, dropped = screen_variables(md, x_cols, known_in_advance or {}, leak_cleared or set(), declared_factors or set())
    models: dict[str, MultiModel] = {}
    skipped: list[str] = list(dropped)
    singles = singles_table(md, usable)
    if len(usable) >= MIN_VARS:
        tr = md.df.loc[md.train_idx, usable + [md.y_col]].dropna()
        if len(tr) >= len(usable) + 5:
            subsets = _candidate_subsets(tr[usable].to_numpy(float), tr[md.y_col].to_numpy(float), usable, max_finalists)
            clean = [c for c in usable if leaks[c][0] == "ninguna"]       # también compiten modelos SIN variables sospechosas
            if MIN_VARS <= len(clean) < len(usable):
                extra = _candidate_subsets(tr[clean].to_numpy(float), tr[md.y_col].to_numpy(float), clean, 4)
                subsets += [s for s in extra if s not in subsets]
        else:                                                             # muy pocos casos completos en común: solo parejas
            subsets = [list(c) for c in combinations(usable, MIN_VARS)][:max_finalists * 2]
            skipped.append("Pocas filas completas en todas las variables a la vez: solo se evaluaron parejas de variables.")
        for cols in subsets:
            try:
                m = build_multi(md, cols, alpha=alpha, expected_sign=expected_sign, leaks=leaks)
            except (RegressionError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
                skipped.append(f"{model_key(cols)}: {exc}")
                continue
            score_multi(m)
            models[m.key] = m
    else:
        skipped.append(f"Quedaron {len(usable)} variable(s) utilizable(s): la regresión múltiple necesita al menos {MIN_VARS}.")
    table = multi_table(models)
    rec = recommend_multi(models, dropped, alpha)
    out = ComparisonResult(models=models, table=table, recommendation=rec, skipped=skipped, md=md)
    out.singles = singles
    out.usable = usable
    return out
