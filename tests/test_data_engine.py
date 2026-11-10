"""Pruebas del motor de datos: carga, perfilado, limpieza y AED."""
import numpy as np
import pandas as pd
import pytest

from core import cleaning, eda, loader, profiling
from core.cleaning import CleaningPlan, apply_plan


# ----------------------------------------------------------------------------- carga
def test_load_csv_txt_excel(sample_dir):
    csv = loader.load_file(sample_dir / "ventas_retail.csv")
    assert csv.df.shape[0] == 243 and pd.api.types.is_datetime64_any_dtype(csv.df["semana"])

    txt = loader.load_file(sample_dir / "consumo_energia.txt")          # ; y coma decimal
    assert txt.meta["separador"] == ";"
    assert pd.api.types.is_float_dtype(txt.df["consumo_kwh"])
    assert 20000 < txt.df["consumo_kwh"].median() < 200000

    xls = loader.load_file(sample_dir / "entregas_logistica.xlsx")      # encabezado desplazado
    assert list(xls.df.columns)[:2] == ["distancia_km", "peso_kg"]
    assert xls.meta["fila_encabezado"] == 4 and len(xls.df) == 300


def test_load_rejects_bad_extension():
    with pytest.raises(ValueError):
        loader.load_file(b"abc", filename="datos.pdf")


def test_decimal_comma_and_thousands():
    s = pd.Series(["1.234,50", "2.000,00", "15,5", "7"])
    out = loader._parse_localized_number(s)
    assert out.tolist() == [1234.5, 2000.0, 15.5, 7.0]


def test_legacy_xls_if_available():
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "ESTUDIO" / "EJECUCION PRESUPUESTAL ACTUALIZADA.xls"
    if not p.exists():
        pytest.skip("archivo .xls de ejemplo no disponible")
    res = loader.load_file(p)
    assert res.fmt == "XLS" and res.df.shape[0] > 3


# ----------------------------------------------------------------------------- perfilado
def test_profile_detects_issues(retail):
    prof = profiling.profile(retail)
    cols = prof["columns"].set_index("variable")
    assert cols.loc["id_registro", "grupo"] in {"identificador", "numérica"}
    assert cols.loc["ciudad", "grupo"] == "categórica"
    assert cols.loc["semana", "grupo"] == "fecha"
    assert prof["duplicates"]["n_duplicados"] == 3
    out = prof["outliers"].set_index("variable")
    assert out.loc["ventas_millones", "atípicos_IQR"] >= 3
    inc = prof["inconsistencies"]
    assert (inc["problema"] == "categorías duplicadas").any()


def test_constant_and_low_variance_detection():
    df = pd.DataFrame({"a": [1] * 60, "b": np.r_[np.ones(58), [2, 3]], "c": np.arange(60.0)})
    c = profiling.classify_columns(df).set_index("variable")
    assert c.loc["a", "constante"] and not c.loc["a", "usable_como_X"]
    assert c.loc["b", "baja_variabilidad"]


# ----------------------------------------------------------------------------- limpieza
def test_target_never_imputed(retail):
    df = retail.copy()
    df.loc[:5, "ventas_millones"] = np.nan
    plan = CleaningPlan().add("impute", "ventas_millones", "mean")
    res = apply_plan(df, plan, protect=["ventas_millones"])
    assert res.df["ventas_millones"].isna().sum() == 6
    assert "protegida" in res.log.iloc[0]["detalle"]


def test_imputation_uses_only_fit_rows():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0, np.nan, np.nan]})
    plan = CleaningPlan().add("impute", "x", "mean")
    res = apply_plan(df, plan, fit_index=[0, 1, 2, 3])        # entrenamiento = filas 0..3
    assert res.df.loc[5, "x"] == pytest.approx(2.5)           # NO usa el 100 de la fila de prueba


def test_outlier_removal_only_touches_train_when_fit_index_given():
    x = np.r_[np.arange(30.0), [500.0, 600.0]]          # filas 30 y 31 son atípicas
    df = pd.DataFrame({"x": x})
    plan = CleaningPlan().add("outlier", "x", "remove")
    # atípicos en el conjunto de PRUEBA: no se eliminan (la prueba debe reflejar la realidad)
    res_test_side = apply_plan(df, plan, fit_index=list(range(0, 30)))
    assert len(res_test_side.df) == 32
    # atípicos dentro del conjunto de entrenamiento: sí se eliminan
    res_train_side = apply_plan(df, plan, fit_index=list(range(0, 32)))
    assert len(res_train_side.df) == 30 and not (res_train_side.df["x"] > 100).any()
    # sin partición (exploración): se eliminan de todos los datos
    assert len(apply_plan(df, plan).df) == 30


def test_transform_and_encode_are_traceable(manufactura):
    plan = (CleaningPlan().add("transform", "unidades_producidas", "ln")
            .add("encode", "turno", "onehot", drop_first=True))
    res = apply_plan(manufactura, plan)
    assert "ln(unidades_producidas)" in res.df.columns
    assert res.transforms["ln(unidades_producidas)"]["source"] == "unidades_producidas"
    assert {"turno=Mañana", "turno=Noche"} <= set(res.df.columns) or {"turno=Noche", "turno=Tarde"} <= set(res.df.columns)
    assert len(res.log) == 2


def test_normalize_labels_and_duplicates(retail):
    plan = CleaningPlan().add("normalize_labels", "ciudad").add("drop_duplicates")
    res = apply_plan(retail, plan)
    assert len(res.df) == 240
    assert res.df["ciudad"].nunique() == 4


def test_imputation_advice_and_evaluation(retail):
    adv = cleaning.recommend_imputation(retail, "trafico_clientes", ordered_by="semana")
    assert adv.recommended in {"mean", "median", "mode", "ffill", "interpolate"}
    assert adv.evaluation is not None and not adv.evaluation.empty
    tgt = cleaning.recommend_imputation(retail, "ventas_millones", is_target=True)
    assert tgt.recommended == "none" or tgt.recommended == "drop_rows"
    df = retail.copy()
    df.loc[:2, "ventas_millones"] = np.nan
    assert cleaning.recommend_imputation(df, "ventas_millones", is_target=True).recommended == "drop_rows"


def test_before_after_comparison(retail):
    plan = CleaningPlan().add("impute", "trafico_clientes", "median")
    res = apply_plan(retail, plan)
    cmp = cleaning.compare_before_after(retail, res.df, ["trafico_clientes"])
    assert cmp.loc[0, "faltantes_después"] == 0 and cmp.loc[0, "faltantes_antes"] > 0


# ----------------------------------------------------------------------------- AED
def test_correlation_flags_leakage(retail):
    tab = eda.correlation_table(retail, "ventas_millones")
    row = tab.set_index("variable").loc["ingresos_netos_millones"]
    assert row["pearson_r"] > 0.98 and "fuga" in row["alertas"]
    assert tab.set_index("variable").loc["inversion_publicidad_millones", "pearson_r"] > 0.5


def test_linearity_screen_detects_curvature():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.uniform(0, 10, 200))
    y = pd.Series((x - 5) ** 2 + rng.normal(0, 1, 200))
    out = eda.linearity_screen(x, y)
    assert out["no_lineal"] and out["mejor"] == "cuadrática"
    y_lin = 3 + 2 * x + pd.Series(rng.normal(0, 1, 200))
    assert not eda.linearity_screen(x, y_lin)["no_lineal"]


def test_time_patterns_detect_trend():
    d = pd.DataFrame({"f": pd.date_range("2020-01-01", periods=60, freq="MS")})
    d["y"] = np.arange(60) * 2.0 + np.random.default_rng(1).normal(0, 1, 60)
    out = eda.time_patterns(d, "f", "y")
    assert out["ok"] and out["p_tendencia"] < 0.001 and out["pendiente_por_año"] > 0
