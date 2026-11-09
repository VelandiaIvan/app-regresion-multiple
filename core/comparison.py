"""Construcción de un modelo por variable candidata, validación con datos de prueba, ranking y recomendación.

Principios:
  * La partición entrenamiento/prueba se hace ANTES de calcular estadísticas de limpieza (sin fuga).
  * Los modelos se ajustan solo con entrenamiento; la prueba se usa únicamente para evaluar.
  * El ranking NO es por R²: combina capacidad predictiva, supuestos, significancia, coherencia con el
    negocio y calidad de datos, y penaliza la posible fuga de información.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cleaning import FIT_KINDS, STRUCTURAL_KINDS, TRANSFORMS, CleaningPlan, apply_plan
from .context import leakage_risk
from .diagnostics import Diagnostics, run_diagnostics
from .regression import RegressionError, SLRResult, fit_slr, prediction_metrics


@dataclass
class SplitSpec:
    method: str = "aleatoria"          # aleatoria | temporal
    test_size: float = 0.2
    seed: int = 42
    date_col: str | None = None


@dataclass
class ModelingData:
    df: pd.DataFrame
    train_idx: list
    test_idx: list
    y_col: str
    y_source: str
    y_transform: str | None
    transforms: dict
    log: pd.DataFrame
    ordered: bool
    spec: SplitSpec
    warnings: list = field(default_factory=list)
    n_raw: int = 0
    n_with_y: int = 0


@dataclass
class CandidateModel:
    key: str
    x_col: str
    x_source: str
    x_transform: str | None
    res: SLRResult
    diag: Diagnostics
    train_metrics: dict
    test_metrics: dict | None
    skill_loocv: float
    skill_test: float | None
    skill: float
    effective_sign: int
    sign_ok: bool | None
    leak: str
    leak_msg: str
    excluded: str | None
    coverage: float
    score: float = 0.0
    parts: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)


@dataclass
class Recommendation:
    best_key: str | None
    status: str                  # recomendado | con reservas | no confiable | sin candidatos
    reasons: list
    warnings: list
    alternatives: list
    note_r2: str = ""


@dataclass
class ComparisonResult:
    models: dict
    table: pd.DataFrame
    recommendation: Recommendation
    skipped: list
    md: ModelingData


# ----------------------------------------------------------------------------
# partición y preparación
# ----------------------------------------------------------------------------
def make_split(d: pd.DataFrame, spec: SplitSpec) -> tuple[list, list, list]:
    """Devuelve (train_idx, test_idx, avisos)."""
    warns: list[str] = []
    n = len(d)
    n_test = int(round(n * spec.test_size))
    idx = list(d.index)
    if spec.date_col and spec.date_col in d.columns:
        idx = list(d[spec.date_col].sort_values(kind="stable", na_position="first").index)
    if n < 25 or n_test < 8:
        warns.append(f"Con {n} observaciones no hay suficientes datos para reservar un conjunto de prueba (mín. 8). "
                     "Se evalúa con validación cruzada «dejando uno fuera» (LOOCV), calculada solo con los datos de ajuste.")
        return idx, [], warns
    if spec.method == "temporal":
        if not spec.date_col:
            warns.append("Partición temporal sin columna de fecha: se usa el orden de las filas como orden temporal.")
        return idx[:-n_test], idx[-n_test:], warns
    rng = np.random.default_rng(spec.seed)
    perm = rng.permutation(n)
    pos = {i: k for k, i in enumerate(idx)}
    test_set = {idx[k] for k in np.sort(perm[:n_test])}
    train = [i for i in idx if i not in test_set]
    test = [i for i in idx if i in test_set]
    if spec.date_col:
        warns.append("Partición aleatoria sobre una serie con fecha: puede ser optimista porque la prueba se «entremezcla» con "
                     "el entrenamiento. Para estimar el futuro use partición temporal.")
    return train, test, warns


def prepare_modeling_data(raw: pd.DataFrame, plan: CleaningPlan, y: str, spec: SplitSpec,
                          y_transform: str | None = None) -> ModelingData:
    """Aplica el plan de forma anti-fuga: estructura → quitar Y faltante → partición → pasos con ajuste."""
    if y not in raw.columns:
        raise RegressionError(f"La variable «{y}» no existe en los datos.")
    warns: list[str] = []
    s1 = apply_plan(raw, plan.subset(STRUCTURAL_KINDS), protect=[y])
    d1 = s1.df
    d1 = d1.assign(**{y: pd.to_numeric(d1[y], errors="coerce")})
    n_raw = len(d1)
    miss_y = int(d1[y].isna().sum())
    d1 = d1[d1[y].notna()]
    if miss_y:
        warns.append(f"{miss_y} filas sin valor de Y se descartaron (la variable a estimar nunca se imputa).")
    train, test, w = make_split(d1, spec)
    warns += w
    fit_plan = CleaningPlan(list(plan.subset(FIT_KINDS).steps))
    if y_transform:
        fit_plan.add("transform", y, y_transform, note="Transformación de la variable a estimar")
    s2 = apply_plan(d1, fit_plan, fit_index=train, protect=[y])
    df2 = s2.df
    y_col = y
    if y_transform:
        y_col = f"{TRANSFORMS[y_transform][0]}({y})"
        bad = int(df2[y_col].isna().sum())
        if bad:
            warns.append(f"{bad} filas quedaron fuera del dominio de {y_transform}(Y) y se excluyen.")
            df2 = df2[df2[y_col].notna()]
    train = [i for i in train if i in df2.index]
    test = [i for i in test if i in df2.index]
    if spec.date_col and spec.date_col in df2.columns:
        train = list(df2.loc[train, spec.date_col].sort_values(kind="stable").index)
        test = list(df2.loc[test, spec.date_col].sort_values(kind="stable").index)
    ordered = bool(spec.date_col) or spec.method == "temporal"
    log = pd.concat([s1.log, s2.log], ignore_index=True)
    return ModelingData(df=df2, train_idx=train, test_idx=test, y_col=y_col, y_source=y, y_transform=y_transform,
                        transforms=s2.transforms, log=log, ordered=ordered, spec=spec, warnings=warns,
                        n_raw=n_raw, n_with_y=len(d1))


# ----------------------------------------------------------------------------
# construcción de candidatos
# ----------------------------------------------------------------------------
def _variants(md: ModelingData, x: str, include_transforms: bool) -> list[tuple[str, str, str | None]]:
    """(columna_modelo, fuente, transformación) para una X: original y variantes ln/√ si el dominio lo permite."""
    src = md.transforms[x]["source"] if x in md.transforms else x
    kind = md.transforms[x]["kind"] if x in md.transforms else None
    out = [(x, src, kind)]
    s = md.df[src] if src in md.df.columns else None
    if include_transforms and x not in md.transforms and s is not None and s.nunique() > 3:
        v = s.dropna()
        if len(v) and (v > 0).all():
            out.append((f"{TRANSFORMS['ln'][0]}({x})", x, "ln"))
        if len(v) and (v >= 0).all():
            out.append((f"{TRANSFORMS['sqrt'][0]}({x})", x, "sqrt"))
    return out


def _effective_sign(res: SLRResult) -> int:
    lo, hi = res.x_min, res.x_max
    if hi <= lo:
        return 0
    p = res.predict([lo, hi])["y_est"].to_numpy()
    d = p[1] - p[0]
    return int(np.sign(d)) if np.isfinite(d) else 0


def build_candidate(md: ModelingData, col: str, src: str, kind: str | None, *,
                    alpha: float, expected_sign: dict, known_in_advance: dict, leak_cleared: set) -> CandidateModel:
    df = md.df.copy()
    if kind and col not in df.columns:
        df[col] = TRANSFORMS[kind][1](df[src].to_numpy(float))
    keep = list(dict.fromkeys([col, src, md.y_col, md.y_source]))        # sin columnas repetidas
    tr = df.loc[md.train_idx, keep].dropna(subset=[col, md.y_col])
    if len(tr) < 8:
        raise RegressionError(f"Solo {len(tr)} observaciones completas de entrenamiento (mínimo 8).")
    res = fit_slr(tr[col], tr[md.y_col], x_name=src if not kind else col, y_name=md.y_source, alpha=alpha,
                  x_transform=kind, y_transform=md.y_transform, x_source=src, y_source=md.y_source,
                  x_raw=tr[src], y_raw=tr[md.y_source], index=tr.index.to_numpy())
    diag = run_diagnostics(res, ordered=md.ordered)
    tm = prediction_metrics(res.y_obs, res.y_back(res.fitted), y_baseline=float(np.mean(res.y_obs)))
    test_m, skill_test = None, None
    if md.test_idx:
        te = df.loc[md.test_idx, keep].dropna(subset=[col, md.y_col])
        if len(te) >= 5:
            yhat = res.y_back(res.b0 + res.b1 * te[col].to_numpy(float))
            test_m = prediction_metrics(te[md.y_source].to_numpy(float), yhat, y_baseline=float(np.mean(res.y_obs)))
            skill_test = test_m.get("habilidad")
    n = res.n
    base_loo = float(np.sqrt(np.mean(((res.y - res.y_mean) * n / (n - 1)) ** 2)))
    skill_loo = 1 - res.rmse_loocv / base_loo if base_loo > 0 else np.nan
    sk_vals = [v for v in (skill_test, skill_loo) if v is not None and np.isfinite(v)]
    skill = float(np.mean(sk_vals)) if sk_vals else np.nan
    eff = _effective_sign(res)
    exp = expected_sign.get(src)
    sign_ok = None if not exp else bool(eff == exp)
    leak, msg = leakage_risk(src, md.y_source, float(np.corrcoef(res.x_obs, res.y_obs)[0, 1]) if res.n > 2 else np.nan)
    if leak != "ninguna" and src in leak_cleared:
        leak, msg = "ninguna", "Usted confirmó que esta variable se conoce antes de estimar Y."
    excluded = None
    if known_in_advance.get(src) is False:
        excluded = "No estará disponible en el momento de decidir/estimar (fuga temporal)."
    n_complete = int(df[[col, md.y_col]].dropna().shape[0])
    coverage = n_complete / max(len(df), 1)
    return CandidateModel(key=col, x_col=col, x_source=src, x_transform=kind, res=res, diag=diag, train_metrics=tm,
                          test_metrics=test_m, skill_loocv=skill_loo, skill_test=skill_test, skill=skill,
                          effective_sign=eff, sign_ok=sign_ok, leak=leak, leak_msg=msg, excluded=excluded,
                          coverage=coverage)


def score_model(m: CandidateModel, alpha: float) -> None:
    res, diag = m.res, m.diag
    sev_pts = {"ok": 1.0, "warn": 0.5, "fail": 0.0}
    weights = {"linealidad": 7, "homocedasticidad": 5, "normalidad": 3, "independencia": 5}
    applicable = {k: w for k, w in weights.items() if diag.get(k).severity != "na"}
    wsum = sum(applicable.values()) or 1
    assump = sum(w * sev_pts[diag.get(k).severity] for k, w in applicable.items()) / wsum * 20
    pred = 40 * float(np.clip(m.skill if np.isfinite(m.skill) else 0, 0, 1))
    infer = (10 if res.p_b1 < res.alpha else 0) + 5 * float(np.clip(res.r2_adj / 0.5, 0, 1))
    biz = {True: 15, None: 9, False: 0}[m.sign_ok]
    n_pts = 5 if res.n >= 30 else 2.5 if res.n >= 15 else 0
    dq = 5 * float(np.clip(m.coverage / 0.9, 0, 1)) + n_pts
    pen = 0.0
    flags = []
    if m.leak == "alta":
        pen += 45
        flags.append("posible fuga de información")
    elif m.leak == "media":
        pen += 12
        flags.append("X y Y parecen ambas resultados")
    if m.test_metrics and m.train_metrics.get("RMSE", 0) > 0:
        ratio = m.test_metrics["RMSE"] / m.train_metrics["RMSE"]
        if ratio > 1.5:
            pen += 10
            flags.append(f"error en prueba {ratio:.1f}× el de entrenamiento (posible sobreajuste o cambio de régimen)")
    if diag.get("influencia").severity == "fail":
        pen += 5
        flags.append("observaciones muy influyentes")
    if m.sign_ok is False:
        flags.append("signo contrario al esperado por el negocio")
    if res.p_b1 >= res.alpha:
        flags.append("pendiente no significativa")
    m.parts = {"predictivo (40)": pred, "inferencia (15)": infer, "supuestos (20)": assump,
               "negocio (15)": biz, "calidad de datos (10)": dq, "penalizaciones": -pen}
    m.score = float(np.clip(pred + infer + assump + biz + dq - pen, 0, 100))
    m.flags = flags


def run_comparison(md: ModelingData, x_cols: list[str], *, alpha: float = 0.05, include_transforms: bool = True,
                   expected_sign: dict | None = None, known_in_advance: dict | None = None,
                   leak_cleared: set | None = None) -> ComparisonResult:
    expected_sign = expected_sign or {}
    known_in_advance = known_in_advance or {}
    leak_cleared = leak_cleared or set()
    models: dict[str, CandidateModel] = {}
    skipped: list[str] = []
    for x in x_cols:
        for col, src, kind in _variants(md, x, include_transforms):
            try:
                cm = build_candidate(md, col, src, kind, alpha=alpha, expected_sign=expected_sign,
                                     known_in_advance=known_in_advance, leak_cleared=leak_cleared)
            except (RegressionError, KeyError, ValueError) as exc:
                skipped.append(f"{col}: {exc}")
                continue
            score_model(cm, alpha)
            models[cm.key] = cm
    table = comparison_table(models)
    rec = recommend(models, md, alpha)
    return ComparisonResult(models=models, table=table, recommendation=rec, skipped=skipped, md=md)


# ----------------------------------------------------------------------------
# tabla y recomendación
# ----------------------------------------------------------------------------
def comparison_table(models: dict[str, CandidateModel]) -> pd.DataFrame:
    rows = []
    for k, m in models.items():
        r, d = m.res, m.diag
        tm = m.test_metrics or {}
        rows.append({
            "modelo": k, "variable_X": m.x_source, "transformación": m.x_transform or "—",
            "n_entrenamiento": r.n, "n_prueba": tm.get("n", 0), "β1 (pendiente)": r.b1, "p_β1": r.p_b1,
            "R²": r.r2, "R²_ajust": r.r2_adj, "RMSE_entren.": m.train_metrics["RMSE"], "MAE_entren.": m.train_metrics["MAE"],
            "RMSE_prueba": tm.get("RMSE", np.nan), "MAE_prueba": tm.get("MAE", np.nan),
            "R²_prueba": tm.get("R²_fuera_muestra", np.nan), "RMSE_LOOCV": r.rmse_loocv,
            "habilidad": m.skill, "linealidad": d.get("linealidad").icon, "independencia": d.get("independencia").icon,
            "homocedasticidad": d.get("homocedasticidad").icon, "normalidad": d.get("normalidad").icon,
            "influencia": d.get("influencia").icon,
            "signo_ok": {True: "sí", False: "no", None: "—"}[m.sign_ok], "riesgo_fuga": m.leak,
            "excluido": m.excluded or "", "puntaje": m.score, "alertas": "; ".join(m.flags),
        })
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.sort_values("puntaje", ascending=False).reset_index(drop=True)
    t.insert(0, "ranking", np.arange(1, len(t) + 1))
    t["ranking_por_R²"] = t["R²"].rank(ascending=False, method="min").astype(int)
    return t


def suggest_alternatives(models: dict[str, CandidateModel], best: CandidateModel | None, md: ModelingData) -> list[str]:
    out: list[str] = []
    ref = [best] if best else list(models.values())
    sev = lambda key: [m.diag.get(key).severity for m in ref]
    if any(s in {"warn", "fail"} for s in sev("linealidad")):
        out.append("Relación curva: pruebe transformaciones de X (ln, √, 1/x), un polinomio de grado 2 o un modelo no lineal (p. ej. rendimientos decrecientes).")
    if any(s in {"warn", "fail"} for s in sev("homocedasticidad")):
        out.append("Varianza no constante: transforme Y con logaritmo, use mínimos cuadrados ponderados o apóyese en errores estándar robustos (HC3).")
    if any(s == "fail" for s in sev("independencia")):
        out.append("Errores autocorrelacionados: use un modelo de series de tiempo (regresión con errores AR, rezagos de Y o ARIMA) en vez de MCO simple.")
    if any(s in {"warn", "fail"} for s in sev("normalidad")):
        out.append("Residuos no normales: revise atípicos, transforme Y o use regresión robusta (Huber / Theil–Sen); evite confiar en intervalos de predicción.")
    if any(s in {"warn", "fail"} for s in sev("influencia")):
        out.append("Observaciones muy influyentes: valide esos registros y compare el modelo con y sin ellos, o use regresión robusta.")
    if ref and all(m.res.r2 < 0.3 for m in ref):
        out.append("Ninguna variable por sí sola explica bien Y: pase a regresión múltiple (varias X a la vez), agregue variables relevantes o segmente por categorías.")
    if ref and any(m.res.n < 30 for m in ref):
        out.append("Pocas observaciones: reúna más datos antes de tomar decisiones; los intervalos serán muy anchos.")
    if not out:
        out.append("Si requiere mayor precisión, evalúe regresión múltiple incorporando una segunda variable explicativa.")
    return out


def recommend(models: dict[str, CandidateModel], md: ModelingData, alpha: float) -> Recommendation:
    if not models:
        return Recommendation(None, "sin candidatos", ["No se pudo construir ningún modelo con los datos y variables elegidos."],
                              [], ["Revise la selección de variables, los faltantes y el número de observaciones."])
    eligible = {k: m for k, m in models.items()
                if m.excluded is None and m.leak != "alta" and m.res.p_b1 < alpha and m.sign_ok is not False}
    pool = eligible if eligible else {k: m for k, m in models.items() if m.excluded is None}
    best = max(pool.values(), key=lambda m: m.score) if pool else None
    if best is None:
        return Recommendation(None, "sin candidatos", ["Todas las variables fueron excluidas por no estar disponibles al decidir."],
                              [], suggest_alternatives(models, None, md))
    warns, reasons = [], []
    r = best.res
    reasons.append(f"Mayor puntaje multicriterio ({best.score:.0f}/100) entre {len(eligible) or len(pool)} modelos elegibles.")
    if best.skill_test is not None and np.isfinite(best.skill_test):
        reasons.append(f"Con datos que no usó para ajustarse, reduce el error de predicción en {best.skill_test * 100:.0f}% "
                       f"frente a predecir siempre el promedio (RMSE de prueba {best.test_metrics['RMSE']:,.4g}).")
    reasons.append(f"Validación cruzada (dejando uno fuera): mejora de {best.skill_loocv * 100:.0f}% sobre el promedio.")
    reasons.append(f"Pendiente significativa (p = {r.p_b1:.3g}); R² = {r.r2:.2f}.")
    if best.sign_ok:
        reasons.append("El signo de la relación coincide con lo esperado por el negocio.")
    reasons.append(f"Supuestos: {best.diag.verdict} — {best.diag.summary}")
    status = "recomendado"
    if not eligible:
        warns.append("Ningún modelo cumplió los filtros mínimos (significancia, signo esperado, sin fuga evidente); "
                     "se muestra el mejor entre los restantes solo como referencia.")
        status = "no confiable"
    skill_ok = best.skill is not None and np.isfinite(best.skill) and best.skill >= 0.5
    if best.diag.verdict == "no confiable":
        structural = best.diag.get("linealidad").severity == "fail"
        if structural or not skill_ok:
            status = "no confiable"
            warns.append("Los supuestos del modelo no se sostienen: no use los intervalos ni p-valores para decidir.")
        else:
            if status == "recomendado":
                status = "con reservas"
            failed = [c.name.lower() for c in best.diag.checks if c.severity == "fail" and c.key != "influencia"]
            warns.append("Predice bien con datos nuevos, pero la INFERENCIA no es confiable (incumple: " + ", ".join(failed) +
                         "). Los p-valores y los intervalos de confianza/predicción son demasiado optimistas; úselo para "
                         "estimaciones puntuales, no para afirmar significancia.")
    elif best.diag.verdict == "con reservas" and status == "recomendado":
        status = "con reservas"
    if best.skill is not None and np.isfinite(best.skill) and best.skill < 0.10:
        status = "no confiable"
        warns.append("El modelo casi no mejora a predecir el promedio: capacidad predictiva insuficiente.")
    if r.r2 < 0.10:
        status = "no confiable"
        warns.append(f"R² = {r.r2:.2f}: la variable explica muy poco del comportamiento de Y.")
    if best.leak == "media":
        warns.append(best.leak_msg)
    seen_src: set[str] = set()
    for s in sorted((m for m in models.values() if m.leak == "alta" and m.excluded is None), key=lambda m: -m.res.r2):
        if s.x_source in seen_src:                       # una advertencia por variable (no por cada transformación)
            continue
        seen_src.add(s.x_source)
        warns.append(f"«{s.x_source}» tiene correlación casi perfecta con Y (R² = {s.res.r2:.2f}) y fue apartada por riesgo de fuga: "
                     "úsela solo si se conoce ANTES de necesitar estimar Y.")
    if best.test_metrics and best.train_metrics["RMSE"] > 0 and best.test_metrics["RMSE"] / best.train_metrics["RMSE"] > 1.5:
        warns.append("El error en prueba es mucho mayor que en entrenamiento: cautela con el sobreajuste o con cambios en el tiempo.")
    top_r2 = max(models.values(), key=lambda m: m.res.r2)
    note = ""
    if top_r2.key != best.key:
        why = []
        if top_r2.leak != "ninguna":
            why.append("riesgo de fuga de información")
        if top_r2.excluded:
            why.append("no estará disponible al decidir")
        if top_r2.res.p_b1 >= alpha:
            why.append("pendiente no significativa")
        if top_r2.sign_ok is False:
            why.append("signo contrario al esperado por el negocio")
        sk_t, sk_b = top_r2.skill, best.skill
        if np.isfinite(sk_t) and np.isfinite(sk_b) and sk_t < sk_b - 0.005:
            why.append(f"predice peor con datos nuevos (mejora {sk_t * 100:.0f}% vs {sk_b * 100:.0f}%)")
        worse = [c.name.lower() for c in top_r2.diag.checks if c.key != "influencia"
                 and {"ok": 0, "warn": 1, "fail": 2, "na": 0}[c.severity] > {"ok": 0, "warn": 1, "fail": 2, "na": 0}[best.diag.get(c.key).severity]]
        if worse:
            why.append("incumple más los supuestos (" + ", ".join(worse) + ")")
        if not why:
            why.append(f"la diferencia de desempeño es mínima y se prefirió el de mayor puntaje global ({best.score:.0f} vs {top_r2.score:.0f})")
        note = (f"El modelo con mayor R² es «{top_r2.key}» (R² = {top_r2.res.r2:.2f}), pero no fue el elegido: "
                + "; ".join(why) + ".")
    alternatives = suggest_alternatives(models, best, md) if status != "recomendado" else suggest_alternatives(models, best, md)[:2]
    return Recommendation(best.key, status, reasons, warns, alternatives, note)
