"""Contexto de negocio → recomendación de variable dependiente (Y) y de candidatas a X.

IMPORTANTE (transparencia): este módulo NO usa un modelo de lenguaje. Cruza el texto libre del usuario
con los nombres de columna mediante normalización, raíces simples y una lista amplia de sinónimos
multisectoriales (que sólo sirve para «reconocer» conceptos; nunca restringe el tipo de negocio).
Es una heurística explicable: cada puntaje trae sus razones y la decisión final es del usuario.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profiling import name_tokens, strip_accents

STOPWORDS = set("""
de la el los las un una unos unas y o u en por para con sin sobre entre del al se su sus que como mas menos muy es son
ser fue han hay este esta estos estas ese esa eso lo le les mi nuestro nuestra nuestros tambien cada todo toda todos
quiero queremos necesito necesitamos estimar predecir pronosticar explicar saber conocer analizar entender
empresa negocio compania organizacion cuanto cuanta cuantos cuantas cual cuales cuando donde porque ante hacia desde
the of and to for per vs
""".split())

# Grupos de sinónimos (raíces ya normalizadas, en singular)
SYNONYM_GROUPS: list[set[str]] = [
    {"venta", "ingreso", "facturacion", "revenue", "sale", "recaudo", "ticket", "facturado"},
    {"costo", "gasto", "egreso", "cost", "expense", "erogacion", "coste"},
    {"precio", "tarifa", "price", "canon", "avaluo", "valor", "cotizacion"},
    {"cantidad", "unidad", "volumen", "qty", "unit", "produccion", "output", "lote"},
    {"tiempo", "duracion", "demora", "plazo", "hora", "time", "lead", "espera", "tardanza"},
    {"distancia", "km", "kilometro", "recorrido", "ruta", "distance", "trayecto"},
    {"cliente", "usuario", "consumidor", "comprador", "customer", "visita", "trafico", "afluencia", "paciente", "estudiante"},
    {"publicidad", "marketing", "promocion", "pauta", "anuncio", "advertising", "campana", "inversion"},
    {"demanda", "consumo", "uso", "demand", "consumption", "pedido"},
    {"empleado", "operario", "trabajador", "personal", "colaborador", "staff", "funcionario", "nomina"},
    {"area", "tamano", "superficie", "metro", "m2", "size", "extension"},
    {"temperatura", "clima", "lluvia", "precipitacion", "humedad", "temp"},
    {"descuento", "rebaja", "oferta", "discount"},
    {"rendimiento", "productividad", "cosecha", "yield", "eficiencia"},
    {"satisfaccion", "nps", "calificacion", "rating", "calidad", "quality"},
    {"utilidad", "margen", "rentabilidad", "ganancia", "profit", "ebitda", "beneficio"},
    {"edad", "antiguedad", "age", "vejez"},
    {"inventario", "stock", "existencia"},
    {"energia", "kwh", "electricidad", "potencia"},
    {"fertilizante", "abono", "insumo"},
    {"peso", "carga", "tonelada", "kg", "weight"},
    {"estrato", "nivel", "categoria", "segmento"},
    {"monto", "desembolso", "desembolsado", "aprobado", "cuota", "prestamo", "credito"},
]
CONCEPT_OF = {tok: i for i, g in enumerate(SYNONYM_GROUPS) for tok in g}
# Conceptos que típicamente son «resultados» del negocio (candidatos a Y)
OUTCOME_CONCEPTS = {0, 1, 2, 3, 4, 8, 13, 14, 15, 17, 22}
# Conceptos que típicamente son «palancas» o factores (candidatos a X)
DRIVER_CONCEPTS = {5, 6, 7, 9, 10, 11, 12, 16, 18, 19, 20, 21}
UNIT_WORDS = {"millon", "mil", "miles", "pct", "porcentaje", "percent", "cop", "usd", "total", "num", "numero",
              "nro", "id", "c", "x", "valor", "mensual", "semanal", "anual", "diario", "neto", "bruto"}


def stem(tok: str) -> str:
    t = strip_accents(tok.lower())
    if t.endswith("ciones"):
        t = t[:-6] + "cion"
    elif t.endswith("iones"):
        t = t[:-5] + "ion"
    elif len(t) > 5 and t.endswith("es"):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s"):
        t = t[:-1]
    return t


def tokens(text: str) -> list[str]:
    toks = re.split(r"[^a-z0-9]+", strip_accents(str(text).lower()))
    return [stem(t) for t in toks if len(t) >= 2 and t not in STOPWORDS and not t.isdigit()]


def concepts(toks: list[str]) -> set:
    return {CONCEPT_OF.get(t, t) for t in toks if t not in UNIT_WORDS}


def column_concepts(col: str) -> set:
    # los dígitos del nombre (rangos «0_10», años) no son conceptos
    return concepts([stem(t) for t in name_tokens(col) if not t.isdigit()])


@dataclass
class BusinessContext:
    sector: str = ""
    actividad: str = ""
    core: str = ""
    problema: str = ""
    objetivo_y: str = ""        # qué quiere estimar
    factores: str = ""          # qué cree que lo explica
    restricciones: str = ""
    pregunta: str = ""
    y_higher_is_better: bool = True

    def full_text(self) -> str:
        return " ".join([self.sector, self.actividad, self.core, self.problema, self.objetivo_y,
                         self.factores, self.restricciones, self.pregunta])

    def target_text(self) -> str:
        # La pregunta de negocio suele mencionar también los factores («¿la espera explica la satisfacción?»):
        # solo se usa para identificar Y cuando el usuario no declaró explícitamente el resultado a estimar.
        return self.objetivo_y if self.objetivo_y.strip() else self.pregunta

    def filled_fields(self) -> int:
        return sum(bool(v.strip()) for v in (self.sector, self.actividad, self.core, self.problema,
                                             self.objetivo_y, self.factores, self.restricciones, self.pregunta))

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def mention_score(col: str, text: str) -> float:
    """0–1: qué tanto del nombre de la columna aparece (por concepto) en el texto del usuario."""
    if not text.strip():
        return 0.0
    norm_text = " ".join(strip_accents(text.lower()).replace("_", " ").split())
    norm_col = " ".join(strip_accents(col.lower()).replace("_", " ").split())
    if norm_col and re.search(rf"(?<![a-z0-9]){re.escape(norm_col)}(?![a-z0-9])", norm_text):
        return 1.0
    cc = column_concepts(col)
    tc = concepts(tokens(text))
    if not cc:
        return 0.0
    concept_match = len(cc & tc) / len(cc)
    # coincidencia LITERAL de las palabras de la columna con las del texto: desempata sinónimos
    # (si el usuario escribió «ventas», «ventas_millones» debe superar a «ingresos_netos_millones»)
    lit = {stem(t) for t in name_tokens(col) if not t.isdigit()} - UNIT_WORDS
    literal_match = len(lit & set(tokens(text))) / len(lit) if lit else 0.0
    return 0.5 * concept_match + 0.5 * literal_match if concept_match > 0 else 0.5 * literal_match


def interpret_context(ctx: BusinessContext, columns: list[str]) -> dict:
    """Resume lo que la aplicación «entendió» del contexto (para que el usuario lo valide)."""
    full = tokens(ctx.full_text())
    freq = pd.Series(full).value_counts() if full else pd.Series(dtype=int)
    t_cols = [c for c in columns if mention_score(c, ctx.target_text()) >= 0.5]
    f_cols = [c for c in columns if mention_score(c, ctx.factores) >= 0.5]
    constraints = [s.strip() for s in re.split(r"[.;\n]", ctx.restricciones) if s.strip()]
    missing = [name for name, v in (("sector", ctx.sector), ("actividad", ctx.actividad), ("objetivo (Y)", ctx.objetivo_y),
                                    ("factores posibles", ctx.factores), ("pregunta de negocio", ctx.pregunta)) if not v.strip()]
    return {
        "conceptos_clave": list(freq.head(10).index),
        "columnas_para_Y": t_cols, "columnas_para_X": f_cols, "restricciones": constraints,
        "campos_sin_completar": missing,
        "resumen": (f"Negocio: {ctx.sector or '—'} · {ctx.actividad or '—'}. "
                    f"Resultado a estimar: {ctx.objetivo_y or '—'}. Factores supuestos: {ctx.factores or '—'}."),
    }


# ----------------------------------------------------------------------------
# riesgo de fuga de información
# ----------------------------------------------------------------------------
def leakage_risk(x_name: str, y_name: str, r: float, declared_factor: bool = False) -> tuple[str, str]:
    """('alta'|'media'|'ninguna', explicación). Se evalúa con r de entrenamiento.

    `declared_factor`: el usuario declaró explícitamente X como factor explicativo (Módulo 1). Eso exime del riesgo MEDIO
    (conocimiento de dominio: p. ej. el ingreso del cliente es un dato de entrada, no un resultado), pero nunca del ALTO.
    """
    if np.isnan(r):
        return "ninguna", ""
    a = abs(r)
    cx, cy = column_concepts(x_name), column_concepts(y_name)
    shared = cx & cy
    if a >= 0.98:
        return "alta", (f"|r| = {a:.3f}: la relación es casi perfecta. Suele indicar que X es una forma contable, derivada "
                        "o consecuencia de Y (fuga de información), no una causa.")
    if shared and a >= 0.9:
        return "alta", ("Comparte concepto con Y (nombres equivalentes) y la correlación es muy alta: probablemente mide lo mismo "
                        "en otra escala o definición.")
    if declared_factor:
        return "ninguna", ""
    if shared and a >= 0.6:
        return "media", ("Comparte concepto con Y (p. ej. monto y cuota, ventas e ingresos) y la correlación es alta: puede calcularse "
                         "DESPUÉS de conocer Y. Verifique que se conozca antes de necesitar estimarlo.")
    # «resultado» solo si TODOS los conceptos informativos del nombre lo son: «ingresos_netos» sí, «ingreso_mensual_cliente» no
    # (ahí «cliente» es una característica del solicitante, conocida antes de decidir).
    out_x = bool(cx) and all(isinstance(c, int) and c in OUTCOME_CONCEPTS for c in cx)
    out_y = bool(cy) and all(isinstance(c, int) and c in OUTCOME_CONCEPTS for c in cy)
    if out_x and out_y and a >= 0.8:
        return "media", ("X y Y parecen ambos RESULTADOS del negocio (p. ej. ingresos y costos). Verifique que X se conozca ANTES "
                         "de necesitar estimar Y; si no, el modelo no se podrá usar para decidir.")
    return "ninguna", ""


# ----------------------------------------------------------------------------
# ranking de Y
# ----------------------------------------------------------------------------
def rank_target_candidates(df: pd.DataFrame, classes: pd.DataFrame, ctx: BusinessContext) -> pd.DataFrame:
    num = classes[classes["usable_como_Y"]]["variable"].tolist()
    if not num:
        return pd.DataFrame(columns=["variable", "puntaje", "razones", "advertencias"])
    corr = df[num].corr(method="spearman") if len(num) > 1 else pd.DataFrame()
    has_text = bool(ctx.target_text().strip())
    rows = []
    for c in num:
        cls = classes.set_index("variable").loc[c]
        m_t = mention_score(c, ctx.target_text())
        m_all = mention_score(c, ctx.full_text())
        cc = column_concepts(c)
        outcome = float(any(isinstance(k, int) and k in OUTCOME_CONCEPTS for k in cc))
        driver = float(any(isinstance(k, int) and k in DRIVER_CONCEPTS for k in cc))
        completeness = 1 - cls["pct_faltantes"] / 100
        continuous = 1.0 if cls["n_únicos"] >= 15 else 0.6
        quality = completeness * continuous * (0.5 if cls["baja_variabilidad"] else 1.0)
        conn = 0.0
        if not corr.empty and c in corr.columns:
            conn = float(corr[c].drop(c).abs().nlargest(3).mean()) if len(corr) > 1 else 0.0
        if has_text:
            score = 45 * m_t + 8 * m_all + 14 * outcome - 5 * driver * (1 - outcome) + 14 * quality + 14 * conn
        else:
            score = 25 * outcome - 8 * driver * (1 - outcome) + 30 * quality + 40 * conn
        reasons, warns = [], []
        if m_t >= 0.5:
            reasons.append("coincide con el resultado que usted describió")
        if outcome:
            reasons.append("concepto típico de resultado de negocio (ventas, costos, tiempos, consumo…)")
        if driver and not outcome:
            warns.append("suena más a factor explicativo que a resultado")
        reasons.append(f"{completeness:.0%} de datos completos")
        if conn > 0.4:
            reasons.append(f"se asocia con otras variables (|ρ| medio de las 3 mayores: {conn:.2f})")
        if cls["pct_faltantes"] > 20:
            warns.append(f"{cls['pct_faltantes']:.0f}% faltante")
        if cls["baja_variabilidad"]:
            warns.append("poca variabilidad")
        if cls["detalle"].startswith("discreta"):
            warns.append("pocos valores distintos")
        rows.append({"variable": c, "puntaje": float(np.clip(score, 0, 100)), "razones": "; ".join(reasons),
                     "advertencias": "; ".join(warns)})
    return pd.DataFrame(rows).sort_values("puntaje", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# ranking de X (prefiltro, antes de construir un modelo por candidata)
# ----------------------------------------------------------------------------
def rank_predictor_candidates(df: pd.DataFrame, y: str, classes: pd.DataFrame, ctx: BusinessContext,
                              corr_table: pd.DataFrame, expected_sign: dict | None = None) -> pd.DataFrame:
    expected_sign = expected_sign or {}
    cls = classes.set_index("variable")
    rows = []
    for _, r in corr_table.iterrows():
        c = r["variable"]
        if c not in cls.index or not cls.loc[c, "usable_como_X"]:
            continue
        strength = max(abs(r["pearson_r"]), abs(r["spearman_rho"]) * 0.9)
        p = r["pearson_p"]
        sig = min(-np.log10(max(p, 1e-12)) / 4, 1.0) if p < 0.05 else 0.0
        completeness = 1 - cls.loc[c, "pct_faltantes"] / 100
        size_ok = min(r["n"] / 30, 1.0)
        mention = mention_score(c, ctx.factores)
        cc = column_concepts(c)
        driver = any(isinstance(k, int) and k in DRIVER_CONCEPTS for k in cc)
        biz = min(1.0, 0.7 * mention + 0.3 * driver)
        exp = expected_sign.get(c)
        coherent = 0.5 if not exp else (1.0 if np.sign(r["pearson_r"]) == exp else 0.0)
        lin = 0.5 if "curvatura" in r["alertas"] else 1.0
        leak, leak_msg = leakage_risk(c, y, r["pearson_r"], declared_factor=mention >= 0.5)
        score = (35 * strength + 15 * sig + 15 * completeness * size_ok + 15 * biz + 10 * coherent + 10 * lin)
        score -= {"alta": 40, "media": 15, "ninguna": 0}[leak]
        pos, neg = [], []
        if strength >= 0.5:
            pos.append(f"relación {r['dirección']} {r['intensidad']}")
        if p < 0.05:
            pos.append("estadísticamente significativa")
        if mention >= 0.5:
            pos.append("usted la mencionó como factor")
        elif driver:
            pos.append("concepto típico de palanca/factor")
        if exp and coherent == 1.0:
            pos.append("signo coherente con lo esperado")
        if exp and coherent == 0.0:
            neg.append("signo contrario a lo esperado")
        if completeness < 0.8:
            neg.append(f"{(1 - completeness):.0%} faltante")
        if r["n"] < 30:
            neg.append("pocas observaciones")
        if leak != "ninguna":
            neg.append(f"riesgo de fuga {leak}: {leak_msg}")
        if r["alertas"]:
            neg.append(r["alertas"])
        rows.append({"variable": c, "n": int(r["n"]), "pearson_r": r["pearson_r"], "spearman_rho": r["spearman_rho"],
                     "p_valor": p, "riesgo_fuga": leak, "puntaje_preliminar": float(np.clip(score, 0, 100)),
                     "a_favor": "; ".join(pos), "alertas": "; ".join(neg)})
    out = pd.DataFrame(rows)
    return out.sort_values("puntaje_preliminar", ascending=False).reset_index(drop=True) if not out.empty else out
