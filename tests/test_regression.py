"""Validación del motor de regresión contra statsmodels y comportamiento de los diagnósticos."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import OLSInfluence

from core.diagnostics import run_diagnostics
from core.regression import RegressionError, fit_slr, prediction_metrics


def _data(n=120, b0=5.0, b1=2.5, sd=3.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(10, 60, n)
    return x, b0 + b1 * x + rng.normal(0, sd, n)


def test_matches_statsmodels_exactly():
    x, y = _data()
    res = fit_slr(x, y)
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    assert res.b0 == pytest.approx(ols.params[0], rel=1e-10)
    assert res.b1 == pytest.approx(ols.params[1], rel=1e-10)
    assert res.se_b0 == pytest.approx(ols.bse[0], rel=1e-9)
    assert res.se_b1 == pytest.approx(ols.bse[1], rel=1e-9)
    assert res.t_b1 == pytest.approx(ols.tvalues[1], rel=1e-9)
    assert res.p_b1 == pytest.approx(ols.pvalues[1], rel=1e-6, abs=1e-30)
    assert res.r2 == pytest.approx(ols.rsquared, rel=1e-10)
    assert res.r2_adj == pytest.approx(ols.rsquared_adj, rel=1e-10)
    assert res.F == pytest.approx(ols.fvalue, rel=1e-9)
    assert res.F == pytest.approx(res.t_b1 ** 2, rel=1e-9)               # F = t² en regresión simple
    assert np.allclose(res.ci_b1, ols.conf_int()[1], rtol=1e-9)
    assert np.allclose(res.ci_b0, ols.conf_int()[0], rtol=1e-9)
    assert res.rmse == pytest.approx(np.sqrt(ols.mse_resid * (len(x) - 2) / len(x)), rel=1e-9)
    assert res.r == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-10)


def test_prediction_and_confidence_intervals_match_statsmodels():
    x, y = _data()
    res = fit_slr(x, y)
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    new = np.array([15.0, 35.0, 55.0])
    sf = ols.get_prediction(sm.add_constant(new, has_constant="add")).summary_frame(alpha=0.05)
    pr = res.predict(new)
    assert np.allclose(pr["y_est"], sf["mean"], rtol=1e-9)
    assert np.allclose(pr["ic_inf"], sf["mean_ci_lower"], rtol=1e-9)
    assert np.allclose(pr["ic_sup"], sf["mean_ci_upper"], rtol=1e-9)
    assert np.allclose(pr["ip_inf"], sf["obs_ci_lower"], rtol=1e-9)
    assert np.allclose(pr["ip_sup"], sf["obs_ci_upper"], rtol=1e-9)
    assert (pr["ip_sup"] - pr["ip_inf"] > pr["ic_sup"] - pr["ic_inf"]).all()    # IP siempre más ancho que IC


def test_influence_and_loocv_match_statsmodels():
    x, y = _data(60, seed=3)
    res = fit_slr(x, y)
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    infl = OLSInfluence(ols)
    assert np.allclose(res.cooks, infl.cooks_distance[0], rtol=1e-8)
    assert np.allclose(res.leverage, infl.hat_matrix_diag, rtol=1e-9)
    assert res.rmse_loocv == pytest.approx(np.sqrt(np.mean(infl.resid_press ** 2)), rel=1e-9)
    # error estándar robusto HC3
    assert res.se_b1_hc3 == pytest.approx(ols.get_robustcov_results("HC3").bse[1], rel=1e-8)


def test_diagnostic_tests_agree_with_statsmodels():
    rng = np.random.default_rng(5)
    x = rng.uniform(1, 50, 200)
    y = 3 + 1.2 * x + rng.normal(0, 1, 200) * (0.5 + 0.1 * x)             # heterocedástico
    res = fit_slr(x, y)
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    d = run_diagnostics(res)
    bp = het_breuschpagan(ols.resid, ols.model.exog)
    w = het_white(ols.resid, ols.model.exog)
    chk = d.get("homocedasticidad")
    assert chk.tests["Breusch-Pagan"][0] == pytest.approx(bp[0], rel=1e-6)
    assert chk.tests["White"][0] == pytest.approx(w[0], rel=1e-6)
    assert chk.severity == "fail"


def test_durbin_watson_and_autocorrelation_detected():
    rng = np.random.default_rng(2)
    n = 150
    x = np.arange(n, dtype=float)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.8 * e[i - 1] + rng.normal()
    y = 10 + 0.5 * x + e
    res = fit_slr(x, y)
    d = run_diagnostics(res, ordered=True)
    assert d.get("independencia").tests["Durbin-Watson"][0] == pytest.approx(durbin_watson(res.resid), rel=1e-9)
    assert d.get("independencia").severity == "fail"
    assert run_diagnostics(res, ordered=False).get("independencia").severity == "na"


def test_clean_linear_data_is_apto():
    x, y = _data(200, sd=2.0, seed=11)
    d = run_diagnostics(fit_slr(x, y), ordered=False)
    assert d.verdict in {"apto", "con reservas"} and d.get("linealidad").severity in {"ok", "warn"}


def test_curved_data_fails_linearity():
    rng = np.random.default_rng(9)
    x = rng.uniform(0, 10, 200)
    y = 2 + (x - 5) ** 2 + rng.normal(0, 1, 200)
    d = run_diagnostics(fit_slr(x, y))
    assert d.get("linealidad").severity == "fail" and d.verdict == "no confiable"


def test_non_normal_and_influential_data():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 10, 80)
    y = 1 + 2 * x + rng.exponential(3, 80)
    assert run_diagnostics(fit_slr(x, y)).get("normalidad").severity in {"warn", "fail"}
    x2 = np.r_[rng.uniform(0, 10, 30), [60.0]]
    y2 = np.r_[1 + 2 * x2[:-1] + rng.normal(0, 1, 30), [-40.0]]
    assert run_diagnostics(fit_slr(x2, y2)).get("influencia").severity in {"warn", "fail"}


def test_log_transform_back_transformation():
    rng = np.random.default_rng(1)
    x = rng.uniform(1, 100, 300)
    y = np.exp(0.5 + 0.03 * x + rng.normal(0, 0.2, 300))
    res = fit_slr(x, np.log(y), y_transform="ln", y_raw=y)
    pr = res.predict([50.0]).iloc[0]
    true_mean = np.exp(0.5 + 0.03 * 50 + 0.2 ** 2 / 2)
    assert pr["y_est"] == pytest.approx(true_mean, rel=0.08)               # factor de Duan
    assert pr["ip_inf"] < pr["y_est"] < pr["ip_sup"] and pr["ip_inf"] > 0
    assert res.form == "log-lin"


def test_errors_on_degenerate_input():
    with pytest.raises(RegressionError):
        fit_slr([1, 1, 1, 1], [1, 2, 3, 4])
    with pytest.raises(RegressionError):
        fit_slr([1, 2], [1, 2])


def test_prediction_metrics_skill():
    y = np.array([10, 12, 14, 16.0])
    m = prediction_metrics(y, y, y_baseline=13.0)
    assert m["RMSE"] == 0 and m["habilidad"] == pytest.approx(1.0)
    m2 = prediction_metrics(y, np.full(4, 13.0), y_baseline=13.0)
    assert m2["habilidad"] == pytest.approx(0.0, abs=1e-12)
