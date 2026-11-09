"""Tratamiento y preparación de datos con trazabilidad.

Diseño anti-fuga: los pasos se guardan en un `CleaningPlan` (receta). Al aplicarlo se puede
indicar `fit_index` (filas de entrenamiento): las estadísticas (media, mediana, moda, límites de
atípicos, categorías) se calculan SOLO con esas filas y luego se aplican a todo el conjunto.
La variable a estimar (Y) está protegida: nunca se imputa ni se recorta.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .loader import is_text
from .profiling import iqr_bounds, strip_accents

PRIORITY = {"drop_duplicates": 0, "drop_column": 1, "fix_nan": 2, "normalize_labels": 3,
            "outlier": 4, "impute": 5, "transform": 6, "encode": 7}
STRUCTURAL_KINDS = {"drop_duplicates", "drop_column", "fix_nan", "normalize_labels"}
FIT_KINDS = {"outlier", "impute", "transform", "encode"}

TRANSFORMS = {
    "ln": ("ln", lambda x: np.where(x > 0, np.log(np.where(x > 0, x, 1)), np.nan), "requiere valores > 0"),
    "ln1p": ("ln1p", lambda x: np.where(x > -1, np.log1p(np.where(x > -1, x, 0)), np.nan), "requiere valores > -1"),
    "sqrt": ("raíz", lambda x: np.where(x >= 0, np.sqrt(np.where(x >= 0, x, 0)), np.nan), "requiere valores ≥ 0"),
    "sq": ("cuadrado", lambda x: x ** 2, "amplifica valores grandes"),
    "inv": ("inverso", lambda x: np.where(x != 0, 1.0 / np.where(x != 0, x, 1), np.nan), "requiere valores ≠ 0"),
}


@dataclass
class Step:
    kind: str
    column: str | None
    method: str
    params: dict = field(default_factory=dict)
    note: str = ""


class CleaningPlan:
    """Receta de limpieza ordenada y serializable."""

    def __init__(self, steps: list[Step] | None = None):
        self.steps: list[Step] = list(steps or [])

    def add(self, kind: str, column: str | None = None, method: str = "", note: str = "", **params) -> "CleaningPlan":
        if kind not in PRIORITY:
            raise ValueError(f"Tipo de paso desconocido: {kind}")
        # reemplaza un paso equivalente sobre la misma columna para evitar duplicados
        # (transformaciones y codificaciones con métodos distintos sí pueden coexistir)
        def same(s: Step) -> bool:
            if s.kind != kind or s.column != column or s.params.get("rule") != params.get("rule"):
                return False
            return s.method == method if kind in {"transform", "encode"} else True
        self.steps = [s for s in self.steps if not same(s)]
        self.steps.append(Step(kind, column, method, params, note))
        return self

    def remove(self, i: int) -> None:
        if 0 <= i < len(self.steps):
            self.steps.pop(i)

    def clear(self) -> None:
        self.steps = []

    def subset(self, kinds: set[str]) -> "CleaningPlan":
        return CleaningPlan([s for s in self.steps if s.kind in kinds])

    def records(self) -> list[dict]:
        return [{"n": i + 1, "tipo": s.kind, "variable": s.column, "método": s.method,
                 "parámetros": json.dumps(s.params, ensure_ascii=False, default=str), "nota": s.note}
                for i, s in enumerate(self.steps)]

    def signature(self) -> str:
        return json.dumps([(s.kind, s.column, s.method, s.params) for s in self.steps], default=str, sort_keys=True)

    def __len__(self) -> int:
        return len(self.steps)


@dataclass
class ApplyResult:
    df: pd.DataFrame
    log: pd.DataFrame
    transforms: dict          # columna nueva -> {"source", "kind"}
    fitted: dict


# ----------------------------------------------------------------------------
# diagnóstico de faltantes
# ----------------------------------------------------------------------------
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for c in df.columns:
        m = int(df[c].isna().sum())
        rows.append({"variable": c, "tipo": _kind(df[c]), "n_faltantes": m,
                     "pct_faltantes": m / max(n, 1) * 100, "n_completos": n - m})
    out = pd.DataFrame(rows).sort_values("pct_faltantes", ascending=False).reset_index(drop=True)
    return out


def missing_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna()


def missing_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlación entre indicadores de faltante (¿faltan juntos?)."""
    m = df.isna().astype(float)
    m = m.loc[:, m.std() > 0]
    return m.corr() if m.shape[1] > 1 else pd.DataFrame()


def rows_missing_distribution(df: pd.DataFrame) -> pd.DataFrame:
    k = df.isna().sum(axis=1)
    vc = k.value_counts().sort_index()
    return pd.DataFrame({"campos_faltantes_por_fila": vc.index, "filas": vc.values,
                         "pct": vc.values / max(len(df), 1) * 100})


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "fecha"
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return "numérica"
    return "categórica"


# ----------------------------------------------------------------------------
# recomendación y evaluación de imputación
# ----------------------------------------------------------------------------
@dataclass
class ImputeAdvice:
    column: str
    kind: str
    pct_missing: float
    recommended: str
    reason: str
    evaluation: pd.DataFrame | None = None
    params: dict = field(default_factory=dict)


METHOD_LABELS = {
    "mean": "Media", "median": "Mediana", "mode": "Moda", "ffill": "Último valor conocido (ffill)",
    "interpolate": "Interpolación lineal", "constant": "Constante", "drop_rows": "Eliminar filas",
    "drop_column": "Eliminar variable", "none": "No tratar",
}


def _impute_series(s: pd.Series, method: str, fit_values: pd.Series | None = None, value=None) -> pd.Series:
    ref = s if fit_values is None else fit_values
    ref = ref.dropna()
    if method == "mean":
        return s.fillna(ref.mean())
    if method == "median":
        return s.fillna(ref.median())
    if method == "mode":
        mode = ref.mode()
        return s.fillna(mode.iloc[0]) if len(mode) else s
    if method == "constant":
        return s.fillna(value)
    if method == "ffill":
        return s.ffill()
    if method == "bfill":
        return s.bfill()
    if method == "interpolate":
        return s.interpolate(limit_direction="both")
    raise ValueError(method)


def evaluate_imputation_methods(s: pd.Series, methods: list[str], frac: float = 0.15, n_rep: int = 8,
                                seed: int = 0) -> pd.DataFrame:
    """Evalúa métodos enmascarando valores conocidos al azar y midiendo qué tan bien se recuperan."""
    rng = np.random.default_rng(seed)
    known = s.dropna()
    if len(known) < 20:
        return pd.DataFrame()
    k = max(5, int(len(known) * frac))
    is_num = pd.api.types.is_numeric_dtype(s)
    acc: dict[str, list] = {m: [] for m in methods}
    for _ in range(n_rep):
        idx = rng.choice(known.index.to_numpy(), size=k, replace=False)
        masked = s.copy()
        truth = masked.loc[idx].copy()
        masked.loc[idx] = np.nan
        for m in methods:
            try:
                filled = _impute_series(masked, m)
            except Exception:
                continue
            est = filled.loc[idx]
            if is_num:
                err = (est - truth).astype(float)
                sd_after = float(filled.std())
                acc[m].append((float(np.sqrt(np.nanmean(err ** 2))), float(np.nanmean(err.abs())),
                               sd_after, float(s.dropna().std())))
            else:
                acc[m].append((float((est == truth).mean()), np.nan, np.nan, np.nan))
    rows = []
    for m, vals in acc.items():
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        if is_num:
            rows.append({"método": METHOD_LABELS.get(m, m), "clave": m, "RMSE": arr[:, 0].mean(),
                         "MAE": arr[:, 1].mean(),
                         "desv_est_conservada_%": (arr[:, 2] / arr[:, 3]).mean() * 100})
        else:
            rows.append({"método": METHOD_LABELS.get(m, m), "clave": m, "exactitud": arr[:, 0].mean()})
    return pd.DataFrame(rows)


def recommend_imputation(df: pd.DataFrame, col: str, ordered_by: str | None = None,
                         is_target: bool = False, seed: int = 0) -> ImputeAdvice:
    """Propone y justifica un método de imputación para una variable."""
    s = df[col]
    pct = float(s.isna().mean() * 100)
    kind = _kind(s)
    if pct == 0:
        return ImputeAdvice(col, kind, 0.0, "none", "No tiene datos faltantes.")
    if is_target:
        return ImputeAdvice(col, kind, pct, "drop_rows",
                            "Es la variable que se quiere estimar: no se imputa. Rellenarla fabricaría la respuesta, "
                            "sesgaría la pendiente y haría trampa en la validación. Se descartan esas filas.")
    if pct > 60:
        return ImputeAdvice(col, kind, pct, "drop_column",
                            f"Falta el {pct:.0f}% de los datos: imputar significaría inventar la mayor parte de la variable. "
                            "Se recomienda excluirla o conseguir los datos.")
    if kind == "fecha":
        return ImputeAdvice(col, kind, pct, "drop_rows",
                            "Una fecha faltante no se puede inferir de forma confiable; se recomienda eliminar esas filas.")
    note_reg = (" No se usa imputación por regresión sobre otras variables porque inflaría artificialmente la relación "
                "que luego se quiere medir.")
    if kind == "categórica":
        ev = evaluate_imputation_methods(s, ["mode"]) if s.notna().sum() >= 20 else None
        top = s.value_counts(normalize=True)
        top_share = float(top.iloc[0]) if len(top) else 0
        if top_share >= 0.5:
            return ImputeAdvice(col, kind, pct, "mode",
                                f"La categoría más frecuente concentra el {top_share:.0%}; la moda es razonable y simple.",
                                ev)
        return ImputeAdvice(col, kind, pct, "constant",
                            "Ninguna categoría domina; es más honesto crear la categoría «Sin dato» que asignar una al azar.",
                            ev, {"value": "Sin dato"})
    # numérica
    x = s.dropna()
    skew = float(stats.skew(x)) if len(x) > 2 else 0.0
    lo, hi = iqr_bounds(x)
    out_share = float(((x < lo) | (x > hi)).mean())
    ordered = False
    if ordered_by and ordered_by in df.columns:
        xs = df.sort_values(ordered_by)[col]
        ac = xs.dropna().autocorr(lag=1) if xs.notna().sum() > 10 else 0
        ordered = bool(ac is not None and ac > 0.7)
    methods = ["mean", "median"] + (["mode"] if x.nunique() <= 15 else []) + (["ffill", "interpolate"] if ordered else [])
    ev = evaluate_imputation_methods(s, methods, seed=seed)
    skewed = abs(skew) > 1 or out_share > 0.05
    base_msg = (f"Asimetría {skew:.2f}, {out_share:.1%} de atípicos → distribución "
                f"{'sesgada o con colas: la media se distorsiona' if skewed else 'aproximadamente simétrica'}.")
    rec, reason = ("median" if skewed else "mean"), ""
    if ev is not None and not ev.empty:
        metric = "MAE" if skewed else "RMSE"
        ev = ev.sort_values(metric).reset_index(drop=True)
        best = ev.iloc[0]
        cand = ev[ev[metric] <= best[metric] * 1.03]            # empate técnico → preferir el más simple
        order = ["median", "mean", "mode", "interpolate", "ffill"]
        rec = sorted(cand["clave"], key=lambda k: order.index(k) if k in order else 9)[0]
        reason = (f"{base_msg} En una prueba enmascarando valores conocidos, «{METHOD_LABELS[rec]}» obtuvo "
                  f"{metric}={float(ev.loc[ev['clave'] == rec, metric].iloc[0]):,.4g} (criterio: {metric} porque "
                  f"{'la distribución es sesgada y el error absoluto es más robusto' if skewed else 'la distribución es simétrica'}). ")
    else:
        reason = f"{base_msg} Pocos datos para evaluar métodos; se aplica la regla por forma de la distribución. "
    if rec in {"mean", "median", "mode"}:
        reason += "Esta imputación reduce la varianza (acerca valores a un centro), por eso conviene que el % de faltantes sea bajo."
    if ordered and rec in {"ffill", "interpolate"}:
        reason += " La serie tiene alta autocorrelación, por eso un método basado en el orden temporal es apropiado."
    reason += note_reg
    return ImputeAdvice(col, kind, pct, rec, reason.strip(), ev)


def suggest_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Sugiere transformaciones por asimetría y dominio."""
    rows = []
    for c in df.select_dtypes(include="number").columns:
        x = df[c].dropna()
        if x.nunique() <= 5 or pd.api.types.is_bool_dtype(df[c]):
            continue
        sk = float(stats.skew(x)) if len(x) > 2 else 0.0
        sug, why = "ninguna", "Distribución razonablemente simétrica."
        if sk > 1:
            if x.min() > 0:
                sug, why = "ln", f"Asimetría positiva fuerte ({sk:.2f}) y valores positivos: el logaritmo la reduce."
            elif x.min() >= 0:
                sug, why = "ln1p", f"Asimetría positiva fuerte ({sk:.2f}) con ceros: ln(1+x)."
            else:
                sug, why = "sqrt", f"Asimetría positiva ({sk:.2f}) pero hay negativos: desplace el origen antes de transformar."
        elif sk < -1:
            sug, why = "ninguna", f"Asimetría negativa ({sk:.2f}): no hay una transformación estándar simple; revise atípicos."
        rows.append({"variable": c, "asimetría": sk, "mín": float(x.min()), "sugerencia": sug, "razón": why})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# aplicación del plan
# ----------------------------------------------------------------------------
def apply_plan(df: pd.DataFrame, plan: CleaningPlan, fit_index=None, protect=()) -> ApplyResult:
    """Aplica el plan. `fit_index`: filas con las que se calculan estadísticas (entrenamiento)."""
    work = df.copy()
    fit = set(work.index) if fit_index is None else set(fit_index) & set(work.index)
    restrict_removal = fit_index is not None
    protect = set(protect)
    log: list[dict] = []
    transforms: dict = {}
    fitted: dict = {}

    def note(step: Step, affected, detail: str):
        log.append({"tipo": step.kind, "variable": step.column or "—", "método": METHOD_LABELS.get(step.method, step.method),
                    "filas_afectadas": affected, "detalle": detail})

    ordered = sorted(enumerate(plan.steps), key=lambda t: (PRIORITY[t[1].kind], t[0]))
    for _, st in ordered:
        col = st.column
        if st.kind == "drop_duplicates":
            mask = work.duplicated(keep="first")
            work = work[~mask]
            fit &= set(work.index)
            note(st, int(mask.sum()), "Se conserva la primera aparición de cada fila repetida.")
            continue
        if col is not None and col not in work.columns:
            note(st, 0, f"Variable «{col}» no existe en los datos; paso omitido.")
            continue
        if st.kind == "drop_column":
            if col in protect:
                note(st, 0, "Variable protegida (Y); no se elimina.")
                continue
            work = work.drop(columns=[col])
            note(st, 0, "Variable eliminada.")
        elif st.kind == "fix_nan":
            s = work[col]
            rule = st.params.get("rule")
            if rule == "negativos":
                m = s < 0
            elif rule == "inf":
                m = pd.Series(np.isinf(s.astype(float)), index=s.index)
            elif rule == "valor":
                m = s == st.params["value"]
            elif rule == "rango":
                m = (s < st.params.get("min", -np.inf)) | (s > st.params.get("max", np.inf))
            else:
                m = pd.Series(False, index=s.index)
            work[col] = s.mask(m)
            note(st, int(m.sum()), f"Valores inconsistentes ({rule}) convertidos a faltante.")
        elif st.kind == "normalize_labels":
            s = work[col]
            key = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
            key = key.map(lambda v: strip_accents(v.lower()) if isinstance(v, str) else v)
            canon = s.astype("string").str.strip().groupby(key).agg(lambda g: g.value_counts().index[0])
            new = key.map(canon)
            changed = int((new.fillna("§") != s.astype("string").fillna("§")).sum())
            work[col] = new.where(s.notna())
            note(st, changed, "Etiquetas unificadas (mayúsculas, espacios y tildes).")
        elif st.kind == "outlier":
            s = work[col].astype(float)
            k = float(st.params.get("k", 1.5))
            lo, hi = iqr_bounds(s.loc[list(fit & set(s.index))], k)
            fitted[f"outlier:{col}"] = (lo, hi)
            m = (s < lo) | (s > hi)
            n_out = int(m.sum())
            if st.method == "winsorize":
                if col in protect:
                    note(st, 0, "Y protegida: no se recorta (use «marcar» o «eliminar»).")
                else:
                    work[col] = s.clip(lo, hi)
                    note(st, n_out, f"Valores llevados a los límites [{lo:,.4g}; {hi:,.4g}].")
            elif st.method == "to_nan":
                if col in protect:
                    note(st, 0, "Y protegida: no se convierte a faltante.")
                else:
                    work[col] = s.mask(m)
                    note(st, n_out, "Atípicos convertidos a faltante (se imputan después).")
            elif st.method == "remove":
                drop = m & (work.index.isin(fit) if restrict_removal else True)
                work = work[~drop]
                fit &= set(work.index)
                note(st, int(drop.sum()), "Filas eliminadas" + (" (solo entrenamiento; prueba intacta)." if restrict_removal else "."))
            elif st.method == "flag":
                work[f"{col}__atípico"] = m
                note(st, n_out, f"Columna indicadora «{col}__atípico» creada.")
        elif st.kind == "impute":
            s = work[col]
            miss = int(s.isna().sum())
            if st.method == "drop_rows":
                drop = s.isna()
                work = work[~drop]
                fit &= set(work.index)
                note(st, miss, "Filas con faltante eliminadas (análisis de casos completos).")
            elif col in protect:
                note(st, 0, "Y protegida: nunca se imputa. Sus filas faltantes se descartan al modelar.")
            else:
                fit_vals = s.loc[list(fit & set(s.index))]
                if st.method in {"mean", "median"} and not pd.api.types.is_numeric_dtype(s):
                    note(st, 0, "Método numérico en variable no numérica; omitido.")
                    continue
                work[col] = _impute_series(s, st.method, fit_vals, st.params.get("value"))
                if st.method in {"mean", "median", "mode"}:
                    fitted[f"impute:{col}"] = float(fit_vals.mean() if st.method == "mean" else
                                                    fit_vals.median() if st.method == "median" else
                                                    (fit_vals.mode().iloc[0] if len(fit_vals.mode()) else np.nan)) \
                        if pd.api.types.is_numeric_dtype(s) else (fit_vals.mode().iloc[0] if len(fit_vals.mode()) else None)
                note(st, miss, f"Estadístico calculado con {len(fit_vals.dropna()):,} filas de ajuste.")
        elif st.kind == "transform":
            label, fn, _ = TRANSFORMS[st.method]
            new = f"{label}({col})"
            vals = fn(work[col].astype(float).to_numpy())
            work[new] = vals
            transforms[new] = {"source": col, "kind": st.method}
            note(st, int(np.isnan(vals).sum() - work[col].isna().sum()), f"Nueva columna «{new}». Invalidó valores fuera del dominio.")
        elif st.kind == "encode":
            s = work[col]
            cats = list(s.loc[list(fit & set(s.index))].dropna().unique())
            if st.method == "binary":
                pos = st.params.get("positive") or (sorted(map(str, cats))[0] if cats else None)
                new = f"{col}={pos}"
                work[new] = np.where(s.isna(), np.nan, (s.astype(str) == str(pos)).astype(float))
                note(st, int(s.notna().sum()), f"1 si «{pos}», 0 en otro caso → «{new}».")
            elif st.method == "onehot":
                drop_first = bool(st.params.get("drop_first", False))
                levels = sorted(map(str, cats))
                levels = levels[1:] if drop_first and len(levels) > 1 else levels
                for lv in levels:
                    work[f"{col}={lv}"] = np.where(s.isna(), np.nan, (s.astype(str) == lv).astype(float))
                note(st, int(s.notna().sum()), f"{len(levels)} variables indicadoras creadas.")
            elif st.method == "ordinal":
                order = [str(o) for o in st.params.get("order", sorted(map(str, cats)))]
                mapping = {o: i for i, o in enumerate(order)}
                new = f"{col}#ord"
                work[new] = s.astype(str).map(mapping).where(s.notna())
                note(st, int(s.notna().sum()), f"Orden asumido: {' < '.join(order)} → «{new}».")
    return ApplyResult(df=work, log=pd.DataFrame(log), transforms=transforms, fitted=fitted)


def compare_before_after(before: pd.DataFrame, after: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Compara la distribución de cada variable antes y después del tratamiento."""
    cols = columns or [c for c in before.columns if c in after.columns]
    rows = []
    for c in cols:
        b, a = before[c], after[c]
        row = {"variable": c, "faltantes_antes": int(b.isna().sum()), "faltantes_después": int(a.isna().sum()),
               "filas_antes": len(b), "filas_después": len(a)}
        if pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
            for name, fn in (("media", np.nanmean), ("mediana", np.nanmedian), ("desv_est", lambda v: np.nanstd(v, ddof=1))):
                bv, av = b.dropna().to_numpy(float), a.dropna().to_numpy(float)
                row[f"{name}_antes"] = fn(bv) if len(bv) > 1 else np.nan
                row[f"{name}_después"] = fn(av) if len(av) > 1 else np.nan
            if b.notna().sum() > 5 and a.notna().sum() > 5:
                row["KS_p_valor"] = float(stats.ks_2samp(b.dropna(), a.dropna()).pvalue)
        rows.append(row)
    return pd.DataFrame(rows)
