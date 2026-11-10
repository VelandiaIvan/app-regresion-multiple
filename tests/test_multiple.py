"""Regresión MÚLTIPLE: exactitud frente a statsmodels, diagnósticos, selección con 2+ variables y entendimiento automático."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

from core import context, profiling, understanding
from core import interpretation_multi as itp
from core import simulation_multi as sim_
from core.cleaning import CleaningPlan
from core.comparison import SplitSpec, prepare_modeling_data
from core.context import BusinessContext
from core.diagnostics import run_diagnostics_multi
from core.multiple import fit_mlr
from core.regression import RegressionError
from core.selection import run_multi_comparison

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def xy():
    rng = np.random.default_rng(1)
    n = 150
    X = pd.DataFrame({"a": rng.normal(10, 2, n), "b": rng.normal(5, 1, n), "c": rng.normal(0, 3, n)})
    y = 4 + 2 * X.a - 3 * X.b + 0.5 * X.c + rng.normal(0, 1.5, n)
    return X, y


@pytest.fixture(scope="module")
def restaurante():
    return pd.read_csv(ROOT / "sample_data" / "ventas_restaurante.csv", sep=";", decimal=",")


def test_matches_statsmodels(xy):
    X, y = xy
    r = fit_mlr(X, y)
    o = sm.OLS(y, sm.add_constant(X)).fit()
    inf = OLSInfluence(o)
    assert np.allclose(r.beta, o.params) and np.allclose(r.se, o.bse) and np.allclose(r.p, o.pvalues)
    assert np.allclose(r.ci, o.conf_int().to_numpy())
    assert r.r2 == pytest.approx(o.rsquared) and r.r2_adj == pytest.approx(o.rsquared_adj)
    assert r.F == pytest.approx(o.fvalue) and r.p_F == pytest.approx(o.f_pvalue)
    assert np.allclose(r.se_hc3, o.get_robustcov_results("HC3").bse)
    assert np.allclose(r.cooks, inf.cooks_distance[0]) and np.allclose(r.leverage, inf.hat_matrix_diag)
    assert r.rmse_loocv == pytest.approx(np.sqrt(np.mean(inf.resid_press ** 2)))
    Xc = sm.add_constant(X).to_numpy()
    assert np.allclose([r.vif[c] for c in X], [variance_inflation_factor(Xc, j + 1) for j in range(3)])


def test_prediction_intervals_match_statsmodels(xy):
    X, y = xy
    r = fit_mlr(X, y)
    o = sm.OLS(y, sm.add_constant(X)).fit()
    x0 = pd.DataFrame({"a": [9.0, 14], "b": [5.0, 2], "c": [1.0, -4]})
    ref = o.get_prediction(sm.add_constant(x0, has_constant="add")).summary_frame(alpha=0.05)
    p = r.predict(x0)
    assert np.allclose(p["y_est"], ref["mean"])
    assert np.allclose(p[["ic_inf", "ic_sup"]], ref[["mean_ci_lower", "mean_ci_upper"]])
    assert np.allclose(p[["ip_inf", "ip_sup"]], ref[["obs_ci_lower", "obs_ci_upper"]])
    assert r.predict({"a": 9.0, "b": 5.0, "c": 1.0})["y_est"].iloc[0] == pytest.approx(ref["mean"].iloc[0])


def test_degenerate_inputs(xy):
    X, y = xy
    with pytest.raises(RegressionError):
        fit_mlr(X.assign(d=X.a * 2), y)                    # combinación lineal exacta
    with pytest.raises(RegressionError):
        fit_mlr(X.assign(k=1.0), y)                        # constante
    with pytest.raises(RegressionError):
        fit_mlr(X.head(4), y.head(4))                      # muy pocas filas


def test_multicollinearity_is_flagged(xy):
    X, y = xy
    rng = np.random.default_rng(2)
    X2 = X.assign(a_casi=X.a + rng.normal(0, 0.05, len(X)))
    d = run_diagnostics_multi(fit_mlr(X2, y))
    assert d.get("multicolinealidad").severity == "fail"
    assert run_diagnostics_multi(fit_mlr(X, y)).get("multicolinealidad").severity == "ok"


def test_clean_data_is_apto_and_curved_fails(xy):
    X, y = xy
    assert run_diagnostics_multi(fit_mlr(X, y)).verdict == "apto"
    y_curvo = 4 + 0.8 * X.a ** 2 - 3 * X.b + np.random.default_rng(3).normal(0, 1, len(X))
    assert run_diagnostics_multi(fit_mlr(X, y_curvo)).get("linealidad").severity == "fail"


def _compare(df, y, **kw):
    md = prepare_modeling_data(df, CleaningPlan(), y, SplitSpec())
    xs = [c for c in md.df.columns if c != y and pd.api.types.is_numeric_dtype(md.df[c])]
    return run_multi_comparison(md, xs, **kw)


def test_selection_keeps_real_variables_and_drops_noise():
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({f"x{j}": rng.normal(0, 1, n) for j in range(1, 6)})
    df["y"] = 10 + 3 * df.x1 - 2 * df.x2 + 1.5 * df.x3 + rng.normal(0, 1, n)        # x4 y x5 son ruido
    cmp = _compare(df, "y")
    best = cmp.models[cmp.recommendation.best_key]
    assert set(best.x_cols) == {"x1", "x2", "x3"}
    assert cmp.recommendation.status == "recomendado"
    assert all(m.res.k >= 2 for m in cmp.models.values())
    assert cmp.table.iloc[0]["modelo"] == best.key
    full = max(cmp.models.values(), key=lambda m: m.res.k)
    assert full.res.r2 >= best.res.r2 and "R² nunca baja" in cmp.recommendation.note_r2


def test_selection_never_returns_single_variable_models():
    rng = np.random.default_rng(8)
    n = 120
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    df["y"] = 5 + 4 * df.x1 + rng.normal(0, 1, n)                                    # solo x1 importa
    cmp = _compare(df, "y")
    assert list(cmp.models) == ["x1 + x2"]
    assert cmp.recommendation.status == "con reservas"                               # x2 no es significativa: se advierte
    one = run_multi_comparison(prepare_modeling_data(df, CleaningPlan(), "y", SplitSpec()), ["x1"])
    assert not one.models and one.recommendation.status == "sin candidatos"


def test_stepwise_path_with_many_variables():
    rng = np.random.default_rng(9)
    n = 400
    df = pd.DataFrame({f"x{j}": rng.normal(0, 1, n) for j in range(1, 13)})          # 12 > límite exhaustivo
    df["y"] = 2 * df.x1 + 2 * df.x5 - 2 * df.x9 + rng.normal(0, 1, n)
    cmp = _compare(df, "y")
    assert set(cmp.models[cmp.recommendation.best_key].x_cols) == {"x1", "x5", "x9"}


def test_expected_sign_and_unavailable_variables():
    rng = np.random.default_rng(10)
    n = 200
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n), "x3": rng.normal(0, 1, n)})
    df["y"] = 3 * df.x1 + 2 * df.x2 + 2 * df.x3 + rng.normal(0, 1, n)
    cmp = _compare(df, "y", expected_sign={"x1": -1})
    assert all(m.sign_ok is False for m in cmp.models.values() if "x1" in m.x_cols)
    assert "x1" not in cmp.models[cmp.recommendation.best_key].x_cols
    cmp2 = _compare(df, "y", known_in_advance={"x3": False})
    assert all("x3" not in m.x_cols for m in cmp2.models.values())


def test_restaurant_file_target_and_model(restaurante):
    """Archivo real del usuario (precio del plato, platos, gaseosas, cervezas): sin texto, Y = Bowls."""
    cls = profiling.classify_columns(restaurante)
    rank = context.rank_target_candidates(restaurante, cls, BusinessContext())
    assert rank.iloc[0]["variable"] == "Bowls"
    assert "mejor se explican" in rank.iloc[0]["razones"]
    price = rank.set_index("variable").loc["Bowl Price"]
    assert "precio" in price["advertencias"]
    cmp = _compare(restaurante, "Bowls")
    best = cmp.models[cmp.recommendation.best_key]
    assert best.res.k >= 2 and "Bowl Price" in best.x_cols and best.res.coef("Bowl Price") < 0
    assert best.leak == "ninguna" and best.res.r2_adj > 0.7
    assert context.leakage_risk("Bowl Price", "Bowls", -0.71)[0] == "ninguna"         # precio ≠ cantidad
    assert context.leakage_risk("cuota_mensual_millones", "monto_aprobado_millones", 0.8)[0] == "media"


def test_understanding_describes_dataset(restaurante):
    info = understanding.describe_dataset(restaurante, profiling.classify_columns(restaurante))
    assert "precios" in info["tema"] and "186 filas y 4 columnas" in info["tamaño"]
    cols = info["columnas"].set_index("variable")
    assert "conteo" in cols.loc["Bowls", "qué parece medir"] and "precios" in cols.loc["Bowl Price", "qué parece medir"]
    rank = context.rank_target_candidates(restaurante, profiling.classify_columns(restaurante), BusinessContext())
    exp = understanding.explain_target(rank, context.explainability(restaurante, list(restaurante.columns)))
    assert exp["y"] == "Bowls" and "Bowl Price" in exp["x"] and "porque" in exp["texto"]


def test_interpretation_and_simulation(restaurante):
    cmp = _compare(restaurante, "Bowls")
    res = cmp.models[cmp.recommendation.best_key].res
    items = itp.interpret_all(res)
    assert all("manteniendo constantes" in d["efecto"] for d in items)
    assert abs(items[0]["std_beta"]) >= abs(items[-1]["std_beta"])
    assert "R² ajustado" in itp.plain_language(res)
    scen = sim_.scenario_values(res, True)
    y = {k: float(res.predict(v)["y_est"].iloc[0]) for k, v in scen.items()}
    assert y["pesimista"] < y["base"] < y["optimista"]
    new = {**scen["base"], "Bowl Price": scen["base"]["Bowl Price"] + 1}
    sim = sim_.simulate(res, scen["base"], new)
    assert sim["dy"] == pytest.approx(res.coef("Bowl Price")) and sim["extrap_new"].status == "dentro"
    far = {**scen["base"], "Bowl Price": res.x_range("Bowl Price")[1] * 3}
    assert sim_.extrapolation_status(res, far).status == "severa"
    inv = sim_.required_x(res, y["base"] + 20, "Bowl Price", scen["base"])
    assert float(res.predict({**scen["base"], "Bowl Price": inv["x"]})["y_est"].iloc[0]) == pytest.approx(y["base"] + 20)
    tor = sim_.tornado_table(res, scen["base"])
    assert len(tor) == res.k and tor["impacto"].is_monotonic_decreasing
    sens = sim_.sensitivity_table(res, scen["base"])
    assert len(sens) == res.k * 4 and (sens["interpretación"].str.contains("las demás iguales")).all()
