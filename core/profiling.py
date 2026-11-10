"""Perfilado de datos: tipos, duplicados, atípicos, inconsistencias y estadísticas descriptivas."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import stats

from .loader import is_text

ID_HINTS = {"id", "codigo", "cod", "nit", "cedula", "documento", "folio", "consecutivo",
            "identificador", "uuid", "llave", "key", "index", "indice", "radicado"}
NONNEG_HINTS = {"precio", "costo", "coste", "venta", "ventas", "cantidad", "edad", "area", "peso",
                "tiempo", "distancia", "consumo", "unidades", "ingreso", "ingresos", "gasto", "gastos",
                "duracion", "volumen", "habitaciones", "dias", "horas", "salario", "valor", "demanda",
                "produccion", "visitas", "trafico", "stock", "inventario", "temperatura_k"}
PCT_HINTS = {"pct", "porcentaje", "percent", "porc"}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def name_tokens(name: str) -> list[str]:
    """Separa un nombre de columna en tokens (snake_case, camelCase, espacios)."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    s = strip_accents(s.lower())
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


# ----------------------------------------------------------------------------
# clasificación de variables
# ----------------------------------------------------------------------------
def classify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clasifica cada columna en numérica / categórica / fecha y señala particularidades."""
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        nn = s.dropna()
        n_unique = int(nn.nunique())
        uniq_ratio = n_unique / max(len(nn), 1)
        toks = set(name_tokens(col))
        notes: list[str] = []
        usable_y = usable_x = False

        if pd.api.types.is_datetime64_any_dtype(s):
            grupo, detalle = "fecha", "fecha/hora"
        elif pd.api.types.is_bool_dtype(s):
            grupo, detalle, usable_x = "categórica", "booleana", True
        elif pd.api.types.is_numeric_dtype(s):
            is_int = bool((nn % 1 == 0).all()) if len(nn) else False
            if n_unique <= 2:
                grupo, detalle, usable_x = "categórica", "binaria numérica (0/1)", True
            elif toks & ID_HINTS and uniq_ratio >= 0.98:
                grupo, detalle = "identificador", "identificador numérico"
                notes.append("No es una magnitud medible: se excluye de los modelos.")
            elif is_int and uniq_ratio == 1 and len(nn) > 5 and (nn.is_monotonic_increasing or nn.is_monotonic_decreasing):
                grupo, detalle = "numérica", "índice secuencial"
                notes.append("Secuencia 1..n: puede representar tendencia temporal, no una causa en sí misma.")
                usable_x = True
            elif is_int and n_unique <= 10 and n >= 50:
                grupo, detalle, usable_x, usable_y = "numérica", "discreta (pocos valores; ¿ordinal/categórica?)", True, False
                notes.append("Pocos valores distintos: valide si es ordinal o categórica.")
            else:
                grupo = "numérica"
                detalle = "discreta (conteo)" if is_int else "continua"
                usable_x = True
                usable_y = n_unique >= 10
        else:
            if uniq_ratio > 0.9 and n_unique > 20:
                grupo, detalle = "texto", "texto libre / identificador"
                notes.append("Casi todos los valores son distintos: no sirve como categoría.")
            else:
                grupo = "categórica"
                detalle = "nominal" if n_unique <= 50 else "nominal de alta cardinalidad"
                if n_unique > 50:
                    notes.append("Alta cardinalidad: considere agrupar categorías.")

        miss = int(s.isna().sum())
        constant = n_unique <= 1
        low_var = False
        if grupo in {"numérica"} and n_unique > 1:
            sd, mu = float(nn.std()), float(abs(nn.mean()))
            top_share = float(nn.value_counts(normalize=True).iloc[0])
            low_var = (mu > 0 and sd / mu < 1e-3) or top_share >= 0.95
            if low_var:
                notes.append("Variabilidad muy baja: aporta poca información al modelo.")
        if constant:
            notes.append("Variable constante: no puede explicar nada.")
            usable_x = usable_y = False
        if miss / max(n, 1) > 0.4:
            notes.append(f"{miss / n:.0%} de datos faltantes.")
        rows.append({
            "variable": col, "dtype": str(s.dtype), "grupo": grupo, "detalle": detalle,
            "n_válidos": int(len(nn)), "n_faltantes": miss, "pct_faltantes": miss / max(n, 1) * 100,
            "n_únicos": n_unique, "constante": constant, "baja_variabilidad": low_var,
            "usable_como_Y": bool(usable_y and not constant), "usable_como_X": bool(usable_x and not constant),
            "observaciones": " ".join(notes),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# duplicados
# ----------------------------------------------------------------------------
def duplicates_info(df: pd.DataFrame) -> dict:
    mask = df.duplicated(keep="first")
    all_mask = df.duplicated(keep=False)
    return {
        "n_duplicados": int(mask.sum()),
        "pct": float(mask.mean() * 100) if len(df) else 0.0,
        "n_filas_involucradas": int(all_mask.sum()),
        "ejemplos": df[all_mask].sort_values(list(df.columns[:3])).head(10) if all_mask.any() else df.head(0),
        "mask": mask,
    }


# ----------------------------------------------------------------------------
# atípicos
# ----------------------------------------------------------------------------
def iqr_bounds(x: pd.Series, k: float = 1.5) -> tuple[float, float]:
    x = x.dropna()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def outlier_masks(x: pd.Series, k: float = 1.5) -> dict[str, pd.Series]:
    """Máscaras de atípicos por tres criterios: Tukey (IQR), z-score y z-robusto (MAD)."""
    x = x.astype(float)
    lo, hi = iqr_bounds(x, k)
    m_iqr = (x < lo) | (x > hi)
    mu, sd = x.mean(), x.std()
    m_z = ((x - mu).abs() / sd > 3) if sd and sd > 0 else pd.Series(False, index=x.index)
    med = x.median()
    mad = (x - med).abs().median()
    m_mad = (0.6745 * (x - med).abs() / mad > 3.5) if mad and mad > 0 else pd.Series(False, index=x.index)
    return {"iqr": m_iqr.fillna(False), "z": m_z.fillna(False), "mad": m_mad.fillna(False)}


def outliers_summary(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include="number").columns:
        x = df[col]
        if x.dropna().nunique() <= 2:
            continue
        masks = outlier_masks(x, k)
        lo, hi = iqr_bounds(x, k)
        n = int(x.notna().sum())
        rows.append({
            "variable": col, "límite_inferior": lo, "límite_superior": hi,
            "atípicos_IQR": int(masks["iqr"].sum()), "pct_IQR": masks["iqr"].sum() / max(n, 1) * 100,
            "atípicos_z>3": int(masks["z"].sum()), "atípicos_MAD": int(masks["mad"].sum()),
            "mín": float(x.min()), "máx": float(x.max()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# inconsistencias
# ----------------------------------------------------------------------------
def find_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta valores inconsistentes o sospechosos. Cada hallazgo incluye una sugerencia."""
    rows = []
    for col in df.columns:
        s = df[col]
        toks = set(name_tokens(col))
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            x = s.dropna()
            inf = int(np.isinf(x).sum()) if len(x) else 0
            if inf:
                rows.append((col, "valores infinitos", inf, "Contiene ±inf.", "Convertir a faltante."))
            x = x[np.isfinite(x)]
            if toks & NONNEG_HINTS and (x < 0).any():
                rows.append((col, "negativos improbables", int((x < 0).sum()),
                             f"El nombre sugiere magnitud no negativa; mínimo observado {x.min():,.4g}.",
                             "Revisar; si es error, convertir a faltante."))
            if toks & PCT_HINTS and ((x < 0) | (x > 100)).any():
                rows.append((col, "porcentaje fuera de 0–100", int(((x < 0) | (x > 100)).sum()),
                             f"Rango observado {x.min():,.4g} a {x.max():,.4g}.", "Revisar escala (¿proporción 0–1?) o errores."))
            for sentinel in (-999, -9999, 9999, 99999, 999999):
                share = float((x == sentinel).mean()) if len(x) else 0
                if share > 0.005 and x.nunique() > 10:
                    rows.append((col, "posible código de dato faltante", int((x == sentinel).sum()),
                                 f"El valor {sentinel} se repite ({share:.1%}).", f"Convertir {sentinel} a faltante."))
            if len(x) > 20 and (x == 0).mean() > 0.5:
                rows.append((col, "exceso de ceros", int((x == 0).sum()),
                             f"{(x == 0).mean():.0%} de los valores son 0.", "Evaluar si el 0 es real o ausencia de dato."))
        elif is_text(s):
            nn = s.dropna().astype(str)
            if nn.empty:
                continue
            norm = nn.map(lambda v: strip_accents(v.strip().lower()))
            if nn.nunique() > norm.nunique():
                rows.append((col, "categorías duplicadas", int(nn.nunique() - norm.nunique()),
                             "Etiquetas que difieren solo en mayúsculas, espacios o tildes.", "Normalizar etiquetas."))
            num_like = pd.to_numeric(nn, errors="coerce").notna().mean()
            if 0.05 < num_like < 0.95:
                rows.append((col, "tipo mixto (texto y números)", int(pd.to_numeric(nn, errors="coerce").notna().sum()),
                             f"{num_like:.0%} de los valores parecen números.", "Depurar valores no numéricos."))
    return pd.DataFrame(rows, columns=["variable", "problema", "n_afectados", "detalle", "sugerencia"])


# ----------------------------------------------------------------------------
# descriptivas
# ----------------------------------------------------------------------------
def descriptives(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.select_dtypes(include="number").columns if not pd.api.types.is_bool_dtype(df[c])]
    rows = []
    for c in cols:
        x = df[c].dropna()
        if x.empty:
            continue
        mean = float(x.mean())
        rows.append({
            "variable": c, "n": int(len(x)), "media": mean, "desv_est": float(x.std()) if len(x) > 1 else np.nan,
            "mín": float(x.min()), "Q1": float(x.quantile(.25)), "mediana": float(x.median()),
            "Q3": float(x.quantile(.75)), "máx": float(x.max()),
            "CV_%": float(x.std() / abs(mean) * 100) if mean else np.nan,
            "asimetría": float(stats.skew(x, bias=False)) if len(x) > 2 and x.nunique() > 1 else np.nan,
            "curtosis_exceso": float(stats.kurtosis(x, bias=False)) if len(x) > 3 and x.nunique() > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def profile(df: pd.DataFrame) -> dict:
    """Perfil completo (Módulo 2) en un solo diccionario."""
    return {
        "shape": df.shape,
        "head": df.head(5),
        "columns": classify_columns(df),
        "duplicates": duplicates_info(df),
        "outliers": outliers_summary(df),
        "inconsistencies": find_inconsistencies(df),
        "descriptives": descriptives(df),
        "memory_kb": float(df.memory_usage(deep=True).sum() / 1024),
    }
