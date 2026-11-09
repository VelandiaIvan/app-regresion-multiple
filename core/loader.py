"""Carga de archivos TXT / CSV / XLS / XLSX con detección automática de formato.

No depende de nombres de columnas fijos: detecta codificación, separador, fila de
encabezado, números con coma decimal y fechas. Cada conversión queda registrada
(`LoadResult.conversions`) para conservar trazabilidad.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".tsv", ".xls", ".xlsx", ".xlsm"}
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
NA_TOKENS = ["", "NA", "N/A", "n/a", "na", "NaN", "nan", "NULL", "null", "None", "none",
             "s/d", "S/D", "sin dato", "SIN DATO", "?", "--", "#N/A", "#DIV/0!", "#VALUE!"]

DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%Y-%m", "%m/%Y", "%d.%m.%Y",
)
_DATE_SHAPE = re.compile(r"^\d{1,4}[-/.]\d{1,4}([-/.]\d{1,4})?([ T]\d{1,2}:\d{2}(:\d{2})?)?$")


@dataclass
class LoadResult:
    df: pd.DataFrame
    filename: str
    fmt: str
    meta: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    conversions: list = field(default_factory=list)
    sheets: list = field(default_factory=list)


# ----------------------------------------------------------------------------
# utilidades de bajo nivel
# ----------------------------------------------------------------------------
def is_text(s: pd.Series) -> bool:
    """True para columnas de texto (compatible con pandas 2 y 3)."""
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _to_bytes(source) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "getvalue"):
        return source.getvalue()
    if hasattr(source, "read"):
        return source.read()
    raise TypeError("Fuente de datos no soportada")


def _decode(raw: bytes) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1 (con reemplazos)"


def detect_separator(text: str) -> str:
    """Elige el separador cuyo conteo por línea es más consistente."""
    lines = [ln for ln in text.splitlines()[:40] if ln.strip()]
    if not lines:
        return ","
    best, best_score = None, 0.0
    for cand in (";", "\t", "|", ","):
        counts = [ln.count(cand) for ln in lines]
        mode = max(set(counts), key=counts.count)
        if mode == 0:
            continue
        consistency = counts.count(mode) / len(counts)
        score = consistency * 10 + min(mode, 30) * 0.05
        if consistency >= 0.8 and score > best_score:
            best, best_score = cand, score
    if best:
        return best
    try:
        return csv.Sniffer().sniff("\n".join(lines[:10])).delimiter
    except csv.Error:
        pass
    if all(re.search(r"\s{2,}|\t", ln) for ln in lines[:10]):
        return r"\s{2,}"
    return ","


def clean_column_names(cols) -> list[str]:
    out, seen = [], {}
    for i, c in enumerate(cols):
        name = re.sub(r"\s+", " ", str(c)).strip()
        if not name or name.lower().startswith("unnamed") or name.lower() == "nan":
            name = f"columna_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        out.append(name)
    return out


def _detect_header_row(raw: pd.DataFrame, max_scan: int = 15) -> int:
    scan = raw.head(max_scan)
    if scan.empty:
        return 0
    counts = scan.notna().sum(axis=1)
    max_count = counts.max()
    for i in range(len(scan)):
        row = scan.iloc[i]
        if counts.iloc[i] >= 0.6 * max_count and counts.iloc[i] >= 2:
            vals = row.dropna()
            n_str = sum(isinstance(v, str) and not _looks_numeric(v) for v in vals)
            if n_str >= 0.6 * len(vals):
                return i
    return 0


def _looks_numeric(v) -> bool:
    try:
        float(str(v).replace(",", "."))
        return True
    except ValueError:
        return False


# ----------------------------------------------------------------------------
# conversión de tipos
# ----------------------------------------------------------------------------
_US_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_EU_THOUSANDS = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")
_DEC_COMMA = re.compile(r"^-?\d+,\d+$")
_PLAIN = re.compile(r"^-?\d+(\.\d+)?$")


def _parse_localized_number(s: pd.Series) -> pd.Series | None:
    """Convierte cadenas numéricas con separadores regionales. None si no aplica."""
    txt = s.astype("string").str.strip()
    txt = txt.str.replace(r"[\$€£\s]", "", regex=True).str.replace("%", "", regex=False)
    nn = txt.dropna()
    nn = nn[nn != ""]
    if nn.empty:
        return None

    def conv(v: str):
        if _PLAIN.match(v):
            return float(v)
        if _US_THOUSANDS.match(v):
            return float(v.replace(",", ""))
        if _EU_THOUSANDS.match(v):
            return float(v.replace(".", "").replace(",", "."))
        if _DEC_COMMA.match(v):
            return float(v.replace(",", "."))
        return np.nan

    parsed = nn.map(conv).astype(float)
    if parsed.notna().mean() < 0.95:
        return None
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[parsed.index] = parsed
    return out


def _parse_dates(s: pd.Series) -> tuple[pd.Series, str] | None:
    nn = s.dropna().astype(str).str.strip()
    nn = nn[nn != ""]
    if len(nn) < 3:
        return None
    sample = nn.head(300)
    if sample.map(lambda v: bool(_DATE_SHAPE.match(v))).mean() < 0.9:
        return None
    best, best_rate = None, 0.0
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(nn, format=fmt, errors="coerce")
        rate = parsed.notna().mean()
        if rate > best_rate:
            best, best_rate = fmt, rate
        if rate >= 0.999:
            break
    if best is None or best_rate < 0.95:
        return None
    out = pd.to_datetime(s.astype("string").str.strip(), format=best, errors="coerce")
    return out, best


def coerce_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Detecta y convierte columnas numéricas (incluida coma decimal) y fechas."""
    df = df.copy()
    log: list[dict] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            continue
        if is_text(s):
            df[col] = s.map(lambda v: v.strip() if isinstance(v, str) else v)
            s = df[col]
        if s.notna().sum() == 0:
            continue
        # 1) numérico directo (p. ej. objetos de Excel)
        conv = pd.to_numeric(s, errors="coerce")
        if conv.notna().sum() >= 0.95 * s.notna().sum():
            df[col] = conv
            log.append({"variable": col, "conversión": "texto/objeto → número", "detalle": "conversión directa"})
            continue
        # 2) número con formato regional
        loc = _parse_localized_number(s)
        if loc is not None:
            df[col] = loc
            log.append({"variable": col, "conversión": "texto → número",
                        "detalle": "formato regional (coma decimal / separador de miles / símbolos)"})
            continue
        # 3) fecha
        d = _parse_dates(s)
        if d is not None:
            df[col] = d[0]
            log.append({"variable": col, "conversión": "texto → fecha", "detalle": f"formato {d[1]}"})
    return df, log


# ----------------------------------------------------------------------------
# lectores
# ----------------------------------------------------------------------------
def _finalize(df: pd.DataFrame, res: LoadResult) -> LoadResult:
    n0, c0 = df.shape
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.shape != (n0, c0):
        res.warnings.append(f"Se eliminaron {n0 - df.shape[0]} filas y {c0 - df.shape[1]} columnas totalmente vacías.")
    df.columns = clean_column_names(df.columns)
    df, conv = coerce_types(df)
    df = df.reset_index(drop=True)
    res.df, res.conversions = df, conv
    if len(df) < 20:
        res.warnings.append("El archivo tiene menos de 20 filas: cualquier modelo será poco confiable.")
    if df.shape[1] < 2:
        res.warnings.append("Se detectó una sola columna; revise el separador o la hoja seleccionada.")
    return res


def load_file(source, filename: str | None = None, sheet: str | int | None = None,
              sep: str | None = None, encoding: str | None = None) -> LoadResult:
    """Carga un archivo (ruta, bytes o archivo subido) y devuelve un `LoadResult`."""
    if filename is None:
        filename = Path(source).name if isinstance(source, (str, Path)) else getattr(source, "name", "datos")
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Formato '{ext}' no soportado. Use: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
    raw = _to_bytes(source)
    if not raw:
        raise ValueError("El archivo está vacío.")

    if ext in {".xls", ".xlsx", ".xlsm"}:
        return _load_excel(raw, filename, ext, sheet)
    return _load_text(raw, filename, ext, sep, encoding)


def _load_text(raw: bytes, filename: str, ext: str, sep: str | None, encoding: str | None) -> LoadResult:
    if encoding:
        text, enc = raw.decode(encoding, errors="replace"), encoding
    else:
        text, enc = _decode(raw)
    res = LoadResult(df=pd.DataFrame(), filename=filename, fmt=ext.lstrip(".").upper())
    sep_used = sep or ("\t" if ext == ".tsv" else detect_separator(text))
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep_used, engine="python" if len(sep_used) > 1 else "c",
                         na_values=NA_TOKENS, keep_default_na=True, skipinitialspace=True,
                         on_bad_lines="warn")
    except Exception as exc:  # pragma: no cover - mensaje amigable
        raise ValueError(f"No se pudo leer el archivo de texto: {exc}") from exc
    res.meta = {"codificación": enc, "separador": "tabulador" if sep_used == "\t" else sep_used}
    return _finalize(df, res)


def _load_excel(raw: bytes, filename: str, ext: str, sheet) -> LoadResult:
    engine = "xlrd" if ext == ".xls" else "openpyxl"
    try:
        xl = pd.ExcelFile(io.BytesIO(raw), engine=engine)
    except Exception as exc:
        raise ValueError(f"No se pudo abrir el Excel ({engine}): {exc}") from exc
    sheets = list(xl.sheet_names)
    res = LoadResult(df=pd.DataFrame(), filename=filename, fmt=ext.lstrip(".").upper(), sheets=sheets)

    candidates = [sheet] if sheet not in (None, "") else sheets
    chosen, raw_df = None, None
    for name in candidates:
        tmp = xl.parse(name, header=None)
        tmp = tmp.dropna(how="all").dropna(axis=1, how="all")
        if tmp.shape[0] >= 2 and tmp.shape[1] >= 1:
            chosen, raw_df = name, tmp
            break
    if raw_df is None:
        raise ValueError("Ninguna hoja contiene datos tabulares.")
    orig_rows = raw_df.index.to_numpy()          # numeración real de filas de la hoja (base 0)
    raw_df = raw_df.reset_index(drop=True)
    h = _detect_header_row(raw_df)
    header = raw_df.iloc[h].tolist()
    body = raw_df.iloc[h + 1:].copy()
    body.columns = clean_column_names(header)
    body = body.reset_index(drop=True)
    res.meta = {"hoja": chosen, "fila_encabezado": int(orig_rows[h]) + 1}
    if len(sheets) > 1 and sheet in (None, ""):
        res.warnings.append(f"El libro tiene {len(sheets)} hojas; se cargó '{chosen}'. Puede elegir otra en la barra superior.")
    return _finalize(body, res)


def from_dataframe(df: pd.DataFrame, name: str = "datos") -> LoadResult:
    """Empaqueta un DataFrame existente (p. ej. datos de ejemplo)."""
    res = LoadResult(df=pd.DataFrame(), filename=name, fmt="DataFrame")
    return _finalize(df.copy(), res)
