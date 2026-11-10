"""Entendimiento automático del archivo cargado: «¿de qué tratan estos datos?».

TRANSPARENCIA: no usa un modelo de lenguaje. Lee los nombres de las columnas (con un diccionario de conceptos de negocio),
los tipos de dato y estadísticas simples, y arma una explicación en palabras. Es una ayuda para orientarse; el usuario
puede corregirla.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .context import CONCEPT_OF, DRIVER_CONCEPTS, OUTCOME_CONCEPTS, SYNONYM_GROUPS, column_concepts

# Descripción legible de cada grupo de sinónimos de core.context.SYNONYM_GROUPS (mismo orden)
CONCEPT_LABEL = [
    "ventas o ingresos", "costos o gastos", "precios o tarifas", "cantidades o volúmenes", "tiempos o duraciones",
    "distancias o recorridos", "clientes o usuarios", "publicidad o inversión comercial", "demanda o consumo",
    "personal", "área o tamaño", "clima", "descuentos", "rendimiento o productividad", "satisfacción o calidad",
    "utilidad o margen", "edad o antigüedad", "inventario", "energía", "insumos", "peso o carga", "nivel o categoría",
    "montos de crédito",
]
assert len(CONCEPT_LABEL) == len(SYNONYM_GROUPS)


def _fmt(v: float) -> str:
    return f"{v:,.4g}"


def describe_column(df: pd.DataFrame, row: pd.Series) -> dict:
    """Qué parece medir una columna, con evidencia."""
    c = row["variable"]
    s = df[c]
    grupo = row["grupo"]
    ints = [k for k in column_concepts(c) if isinstance(k, int)]
    concept = CONCEPT_LABEL[ints[0]] if ints else ""
    if grupo == "fecha":
        v = s.dropna()
        mide = "una fecha: ordena los registros en el tiempo"
        detalle = f"de {v.min():%d/%m/%Y} a {v.max():%d/%m/%Y}" if len(v) else ""
        papel = "orden temporal"
    elif grupo == "numérica":
        v = pd.to_numeric(s, errors="coerce").dropna()
        uniq = set(v.unique().tolist())
        if uniq <= {0, 1, 0.0, 1.0}:
            mide = "un indicador sí/no (1 = sí, 0 = no)"
        elif len(v) and (v == v.round()).all() and (v >= 0).all():
            mide = "un conteo o cantidad entera"
        elif len(v) and v.min() >= 0 and v.max() <= 1:
            mide = "una proporción entre 0 y 1"
        elif len(v) and v.min() >= 0 and v.max() <= 100 and ("pct" in c.lower() or "porc" in c.lower() or "%" in c):
            mide = "un porcentaje"
        else:
            mide = "una medida numérica continua"
        if concept:
            mide += f" relacionada con {concept}"
        detalle = f"va de {_fmt(v.min())} a {_fmt(v.max())}; valor típico (mediana) {_fmt(v.median())}" if len(v) else ""
        if ints and ints[0] in OUTCOME_CONCEPTS:
            papel = "posible resultado (Y)"
        elif ints and ints[0] in DRIVER_CONCEPTS:
            papel = "posible factor (X)"
        else:
            papel = "medida"
    elif grupo == "categórica":
        vc = s.dropna().astype(str).value_counts()
        mide = f"una clasificación en {len(vc)} grupos" + (f" relacionada con {concept}" if concept else "")
        detalle = "p. ej. " + ", ".join(f"{k} ({v})" for k, v in vc.head(3).items())
        papel = "segmento (codifíquela para usarla como X)"
    elif grupo == "identificador":
        mide, detalle, papel = "un código que identifica cada registro", f"{row['n_únicos']} valores distintos", "no se usa en el modelo"
    else:
        mide, detalle, papel = "texto libre", f"{row['n_únicos']} valores distintos", "no se usa en el modelo"
    return {"variable": c, "tipo": grupo, "qué parece medir": mide, "valores": detalle, "papel probable": papel,
            "faltantes_%": float(row["pct_faltantes"])}


def describe_dataset(df: pd.DataFrame, classes: pd.DataFrame, date_col: str | None = None, source_name: str = "") -> dict:
    """Resumen en lenguaje natural de la tabla: tema, unidad de análisis, tamaño, calidad y papel de cada columna."""
    cols = pd.DataFrame([describe_column(df, r) for _, r in classes.iterrows()])
    n, p = df.shape
    counts: dict[int, list] = {}
    for c in df.columns:
        for k in column_concepts(str(c)):
            if isinstance(k, int):
                counts.setdefault(k, []).append(str(c))
    themes = sorted(counts, key=lambda k: -len(counts[k]))[:3]
    if themes:
        tema = "Los datos parecen tratar sobre " + _join([CONCEPT_LABEL[k] for k in themes]) + "."
        evidencia = "; ".join(f"{CONCEPT_LABEL[k]}: {', '.join(counts[k][:3])}" for k in themes)
    else:
        nums = classes.loc[classes["grupo"] == "numérica", "variable"].astype(str).tolist()
        tema = ("No reconocí conceptos de negocio en los nombres de las columnas; la tabla relaciona "
                + _join(nums[:6]) + ("…" if len(nums) > 6 else "") + ".") if nums else "No reconocí el tema de la tabla."
        evidencia = ""
    ids = classes.loc[classes["grupo"] == "identificador", "variable"].astype(str).tolist()
    if date_col and date_col in df.columns:
        d = df[date_col].dropna().sort_values()
        step = d.diff().dt.days.dropna()
        freq = ""
        if len(step):
            med = float(step.median())
            freq = ("diaria" if med <= 1.5 else "semanal" if med <= 8 else "mensual" if med <= 32 else "trimestral" if med <= 95 else "anual")
        repeated = d.duplicated().mean() if len(d) else 0
        if repeated < 0.2 and freq:
            unidad = f"Cada fila parece ser un periodo (serie de tiempo {freq}, columna «{date_col}»)."
        else:
            unidad = f"Cada fila parece ser un registro o transacción con fecha («{date_col}»)."
    elif ids:
        unidad = f"Cada fila parece ser un caso distinto identificado por «{ids[0]}» (un cliente, producto, pedido, etc.)."
    else:
        unidad = "Cada fila parece ser una observación o caso independiente (un periodo, un cliente, una operación…)."
    g = classes["grupo"].value_counts()
    tam = (f"La tabla tiene {n:,} filas y {p} columnas: {int(g.get('numérica', 0))} numéricas, {int(g.get('categórica', 0))} categóricas"
           + (f", {int(g.get('fecha', 0))} de fecha" if g.get("fecha", 0) else "") + ".")
    miss = float(df.isna().sum().sum() / max(n * p, 1))
    dup = int(df.duplicated().sum())
    calidad = []
    calidad.append("No hay celdas vacías." if miss == 0 else f"{miss:.1%} de las celdas están vacías (se tratan en el módulo de preparación).")
    if dup:
        calidad.append(f"Hay {dup} filas repetidas exactas.")
    n_num = int(g.get("numérica", 0))
    if n < 30:
        calidad.append(f"Son pocas observaciones ({n}): los resultados tendrán bastante incertidumbre.")
    if n_num < 3:
        calidad.append("Hay menos de 3 columnas numéricas: la regresión múltiple necesita una Y y al menos 2 X numéricas "
                       "(puede convertir categorías en variables 0/1 en el módulo de preparación).")
    return {"tema": tema, "evidencia": evidencia, "unidad": unidad, "tamaño": tam, "calidad": calidad, "columnas": cols,
            "n_numéricas": n_num}


def _join(items: list[str]) -> str:
    items = [str(i) for i in items if str(i)]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]


def explain_target(rank: pd.DataFrame, expl: dict | None = None) -> dict:
    """Explica por qué se sugiere la primera candidata a Y y cuáles serían sus posibles X."""
    if rank is None or rank.empty:
        return {"y": None, "texto": "No hay columnas numéricas continuas que puedan ser la variable a estimar.", "x": [], "alternativas": []}
    top = rank.iloc[0]
    y = top["variable"]
    reasons = [r for r in str(top["razones"]).split("; ") if r]
    alts = rank.iloc[1:3]
    close = alts[alts["puntaje"] >= top["puntaje"] - 8]["variable"].tolist()
    xs = (expl or {}).get(y, (np.nan, []))[1]
    texto = f"La variable que más parece ser el RESULTADO a estimar es «{y}»"
    texto += " porque " + _join(reasons) + "." if reasons else "."
    if close:
        texto += (f" También podría ser {_join(['«' + c + '»' for c in close])}: tienen un puntaje muy parecido, "
                  "así que confirme cuál es la que usted quiere estimar.")
    return {"y": y, "texto": texto, "x": xs, "alternativas": close, "advertencias": str(top.get("advertencias", ""))}


__all__ = ["describe_dataset", "describe_column", "explain_target", "CONCEPT_LABEL", "CONCEPT_OF"]
