"""Pruebas de extremo a extremo con datasets de sectores distintos (todos sintéticos)."""
import numpy as np
import pandas as pd
import pytest

from core import cleaning, comparison, context, eda, profiling, simulation
from core.cleaning import CleaningPlan
from core.comparison import SplitSpec, prepare_modeling_data, run_comparison
from sample_data import generate as G


def _ctx(key) -> context.BusinessContext:
    c = G.CONTEXTS[key]
    return context.BusinessContext(**{k: v for k, v in c.items() if k not in {"y", "date"}})


def _run(key, x_cols, *, y_transform=None, temporal=False, plan=None, **kw):
    spec_ctx = G.CONTEXTS[key]
    df = G.DATASETS[key][1]()
    plan = plan or CleaningPlan()
    spec = SplitSpec("temporal" if temporal else "aleatoria", 0.2, 42, spec_ctx["date"])
    md = prepare_modeling_data(df, plan, spec_ctx["y"], spec, y_transform=y_transform)
    return df, md, run_comparison(md, x_cols, **kw)


# ----------------------------------------------------------------------------- contexto → Y
@pytest.mark.parametrize("key", list(G.DATASETS))
def test_target_recommendation_matches_business_goal(key):
    df = G.DATASETS[key][1]()
    classes = profiling.classify_columns(df)
    ranking = context.rank_target_candidates(df, classes, _ctx(key))
    # la sugerencia nº 1 debe ser la variable que el usuario describió (no un sinónimo derivado, p. ej. ingresos_netos vs ventas)
    assert ranking.iloc[0]["variable"] == G.CONTEXTS[key]["y"], ranking.head(4)[["variable", "puntaje", "razones"]]


# ----------------------------------------------------------------------------- retail: fuga
def test_retail_leakage_is_detected_and_not_recommended():
    key = "Retail · ventas semanales (con fuga de información)"
    plan = (CleaningPlan().add("drop_duplicates").add("normalize_labels", "ciudad")
            .add("impute", "trafico_clientes", "median").add("impute", "descuento_pct", "median"))
    _, md, cmp = _run(key, ["inversion_publicidad_millones", "trafico_clientes", "descuento_pct",
                            "temperatura_c", "ingresos_netos_millones"], plan=plan, include_transforms=False)
    leak = cmp.models["ingresos_netos_millones"]
    assert leak.leak == "alta"
    assert cmp.recommendation.best_key != "ingresos_netos_millones"
    assert cmp.recommendation.best_key in {"inversion_publicidad_millones", "trafico_clientes"}
    assert any("ingresos_netos_millones" in w for w in cmp.recommendation.warnings)
    # el modelo con mayor R² es el de la fuga, y la app explica por qué no lo eligió
    assert "ingresos_netos_millones" in cmp.recommendation.note_r2
    assert len(md.train_idx) + len(md.test_idx) == 240


# ----------------------------------------------------------------------------- logística: heterocedasticidad
def test_logistics_heteroscedasticity_flagged():
    key = "Logística · tiempo de entrega (heterocedástico)"
    _, md, cmp = _run(key, ["distancia_km", "peso_kg", "experiencia_conductor_anios"], include_transforms=False)
    best = cmp.models[cmp.recommendation.best_key]
    assert best.x_source == "distancia_km"
    assert best.diag.get("homocedasticidad").severity in {"warn", "fail"}
    assert any("pondera" in a or "logaritmo" in a for a in cmp.recommendation.alternatives)
    assert best.res.b1 > 0 and best.res.p_b1 < 1e-6


# ----------------------------------------------------------------------------- energía: no linealidad + autocorrelación
def test_energy_nonlinearity_and_autocorrelation():
    key = "Servicios · consumo de energía (serie con autocorrelación)"
    _, md, cmp = _run(key, ["temperatura_media_c", "ocupacion_pct", "tarifa_cop_kwh"], temporal=True, include_transforms=False)
    temp = cmp.models["temperatura_media_c"]
    assert temp.res.r2 < 0.2                                          # la U no se captura con una recta
    assert md.ordered
    sev = [m.diag.get("independencia").severity for m in cmp.models.values()]
    assert any(s in {"warn", "fail"} for s in sev)
    assert cmp.recommendation.best_key in {"ocupacion_pct", "tarifa_cop_kwh"}
    # predice bien, pero la inferencia es optimista por la autocorrelación → «con reservas», no «no confiable»
    best = cmp.models[cmp.recommendation.best_key]
    if best.diag.get("independencia").severity == "fail" and best.skill >= 0.5 and best.diag.get("linealidad").severity != "fail":
        assert cmp.recommendation.status == "con reservas"
        assert any("INFERENCIA" in w for w in cmp.recommendation.warnings)
    assert len(md.test_idx) == 19 and max(md.train_idx) < min(md.test_idx)      # partición temporal sin mezcla


# ----------------------------------------------------------------------------- inmobiliario: log-log
def test_real_estate_log_log_improves_fit():
    key = "Inmobiliario · precio de vivienda (log-log)"
    _, md_lin, cmp_lin = _run(key, ["area_m2"], include_transforms=False)
    _, md_log, cmp_log = _run(key, ["area_m2"], y_transform="ln", include_transforms=True)
    r2_lin = cmp_lin.models["area_m2"].test_metrics["R²_fuera_muestra"]
    best = cmp_log.models[cmp_log.recommendation.best_key]
    assert best.res.form == "log-log" and best.x_transform == "ln"
    assert 0.8 < best.res.b1 < 1.1                                   # elasticidad ≈ 0,95 (valor verdadero)
    assert best.test_metrics["R²_fuera_muestra"] >= r2_lin - 0.02    # métrica en la escala original de Y
    assert best.res.predict([100.0])["y_est"].iloc[0] > 0


# ----------------------------------------------------------------------------- manufactura: costo fijo e identidad contable
def test_manufacturing_slope_and_derived_variable_warning():
    key = "Manufactura · costo total (costo fijo + variable)"
    _, md, cmp = _run(key, ["unidades_producidas", "horas_maquina", "operarios", "temperatura_planta_c",
                            "ingreso_total_millones"], include_transforms=False)
    u = cmp.models["unidades_producidas"].res
    assert 0.55 < u.b1 < 0.70 and u.ci_b1[0] < 0.62 < u.ci_b1[1]
    assert cmp.models["ingreso_total_millones"].leak in {"media", "alta"}
    assert cmp.models["temperatura_planta_c"].res.p_b1 > 0.01       # variable sin relación real
    assert cmp.recommendation.best_key in {"unidades_producidas", "horas_maquina"}


# ----------------------------------------------------------------------------- agro: curvatura
def test_agro_detects_curvature_and_sqrt_variant_helps():
    key = "Agro · rendimiento por fertilizante (curvo)"
    _, md, cmp = _run(key, ["fertilizante_kg_ha"], include_transforms=True)
    raw = cmp.models["fertilizante_kg_ha"]
    sq = [m for m in cmp.models.values() if m.x_transform == "sqrt"][0]
    assert raw.diag.get("linealidad").severity in {"warn", "fail"}
    assert sq.res.r2 > raw.res.r2 + 0.02
    # una recta (aun transformada) no captura la saturación → no debe presentarse como modelo confiable
    assert cmp.recommendation.status == "no confiable"
    top = max(cmp.models.values(), key=lambda m: m.res.r2)
    if top.key != cmp.recommendation.best_key:
        assert "peor con datos nuevos" not in cmp.recommendation.note_r2 or top.skill < cmp.models[cmp.recommendation.best_key].skill
    assert any("curva" in a.lower() or "no lineal" in a.lower() for a in
               comparison.suggest_alternatives(cmp.models, raw, md))


# ----------------------------------------------------------------------------- sin relación
def test_no_relationship_is_not_reliable():
    key = "Servicio al cliente · satisfacción (sin relación)"
    _, md, cmp = _run(key, ["tiempo_espera_min", "antiguedad_cliente_meses", "edad"], include_transforms=False)
    assert cmp.recommendation.status == "no confiable"
    assert cmp.recommendation.warnings
    assert all(m.res.r2 < 0.2 for m in cmp.models.values())


# ----------------------------------------------------------------------------- anti-fuga en la preparación
def test_pipeline_has_no_leakage_between_train_and_test():
    df = G.retail_ventas()
    df.loc[df.index[-5:], "trafico_clientes"] = np.nan
    plan = CleaningPlan().add("impute", "trafico_clientes", "mean")
    spec = SplitSpec("aleatoria", 0.2, 7, None)
    md = prepare_modeling_data(df, plan, "ventas_millones", spec)
    tr_mean = df.loc[md.train_idx, "trafico_clientes"].mean()          # media SOLO de entrenamiento (con faltantes ignorados)
    filled_test = [i for i in md.test_idx if np.isnan(df.loc[i, "trafico_clientes"])]
    filled_train = [i for i in md.train_idx if np.isnan(df.loc[i, "trafico_clientes"])]
    for i in filled_test + filled_train:
        assert md.df.loc[i, "trafico_clientes"] == pytest.approx(tr_mean)
    assert set(md.train_idx).isdisjoint(md.test_idx)


def test_missing_target_rows_dropped_not_imputed():
    df = G.manufactura_costos()
    df.loc[:9, "costo_total_millones"] = np.nan
    plan = CleaningPlan().add("impute", "costo_total_millones", "mean")
    md = prepare_modeling_data(df, plan, "costo_total_millones", SplitSpec())
    assert len(md.train_idx) + len(md.test_idx) == 140
    assert any("sin valor de Y" in w for w in md.warnings)


def test_small_sample_uses_loocv_only():
    df = G.servicio_satisfaccion().head(20)
    md = prepare_modeling_data(df, CleaningPlan(), "satisfaccion_0_10", SplitSpec())
    assert md.test_idx == [] and any("LOOCV" in w for w in md.warnings)
    cmp = run_comparison(md, ["tiempo_espera_min"], include_transforms=False)
    assert cmp.models["tiempo_espera_min"].test_metrics is None


# ----------------------------------------------------------------------------- simulación
def test_simulation_scenarios_sensitivity_and_extrapolation():
    key = "Manufactura · costo total (costo fijo + variable)"
    _, md, cmp = _run(key, ["unidades_producidas"], include_transforms=False)
    res = cmp.models["unidades_producidas"].res
    sc = simulation.scenario_values(res, y_higher_is_better=False)       # costo: menos es mejor
    assert sc["pesimista"] > sc["base"] > sc["optimista"]
    sc2 = simulation.scenario_values(res, y_higher_is_better=True)
    assert sc2["pesimista"] < sc2["optimista"]
    sim = simulation.simulate(res, sc["base"], sc["base"] * 1.1)
    assert sim["dy"] == pytest.approx(res.b1 * sc["base"] * 0.1, rel=1e-9)
    assert not sim["extrap_new"].is_extrapolation
    far = simulation.simulate(res, sc["base"], res.x_max * 2)
    assert far["extrap_new"].status == "severa"
    mild = simulation.extrapolation_status(res, res.x_max * 1.02)
    assert mild.status == "leve"
    tab = simulation.sensitivity_table(res, sc["base"], custom=[15])
    assert list(tab["escenario"])[:4] == ["+5%", "+10%", "-5%", "-10%"] and len(tab) == 5
    assert tab.loc[0, "dif_absoluta"] == pytest.approx(res.b1 * sc["base"] * 0.05, rel=1e-9)
    inv = simulation.required_x(res, res.predict([sc["base"]])["y_est"].iloc[0])
    assert inv["x"] == pytest.approx(sc["base"], rel=1e-6)
    cv = simulation.curve(res, pad=0.2)
    assert cv["extrapolación"].any() and (cv["ip_sup"] > cv["ic_sup"]).all()


# ============================================================================ bases de 1000 registros (salud y banca)
SALUD = "Salud · costo de hospitalización (1000 registros)"
BANCA = "Banca · monto de crédito aprobado (1000 registros)"


@pytest.mark.parametrize("key,fname,cols", [(SALUD, "atencion_hospitalaria.csv", 12), (BANCA, "cartera_credito.csv", 14)])
def test_new_1000_record_datasets_shape_and_files(key, fname, cols, sample_dir):
    from core import loader
    df = G.DATASETS[key][1]()
    assert df.shape == (1000, cols) and int(df.duplicated().sum()) == 10
    assert (df.isna().mean() > 0).sum() >= 3                                   # varios faltantes realistas
    res = loader.load_file(sample_dir / fname)                                 # el archivo escrito se lee igual
    assert res.df.shape == (1000, cols) and res.meta["codificación"] == "utf-8-sig"
    assert pd.api.types.is_datetime64_any_dtype(res.df.iloc[:, 1])
    pd.testing.assert_frame_equal(G.DATASETS[key][1](), df)                    # reproducible (semilla fija)


def test_leakage_rule_declared_factor_exempts_medium_but_never_high():
    from core.context import leakage_risk
    # «cuota» comparte concepto con «monto» y correlaciona 0,74 → riesgo medio salvo que el usuario la declare como factor
    assert leakage_risk("cuota_mensual_millones", "monto_aprobado_millones", 0.74)[0] == "media"
    assert leakage_risk("cuota_mensual_millones", "monto_aprobado_millones", 0.74, declared_factor=True)[0] == "ninguna"
    # nunca se exime la correlación casi perfecta
    assert leakage_risk("monto_desembolsado_millones", "monto_aprobado_millones", 0.999, declared_factor=True)[0] == "alta"
    # el ingreso del cliente (factor declarado) no se confunde con un resultado del negocio
    assert leakage_risk("ingreso_mensual_millones", "monto_aprobado_millones", 0.83, declared_factor=True)[0] == "ninguna"
    # ingresos netos vs ventas (ambos resultados) sigue siendo fuga
    assert leakage_risk("ingresos_netos_millones", "ventas_millones", 0.995)[0] == "alta"


def _decl(key, xs):
    return {x for x in xs if context.mention_score(x, G.CONTEXTS[key]["factores"]) >= 0.5}


def test_health_cost_leak_and_log_transform_helps():
    xs = ["dias_estancia", "num_procedimientos", "edad", "nivel_complejidad", "costo_facturado_eps_millones"]
    _, md, cmp = _run(SALUD, xs, include_transforms=False, declared_factors=_decl(SALUD, xs))
    assert cmp.models["costo_facturado_eps_millones"].leak == "alta"
    assert cmp.recommendation.best_key == "dias_estancia"
    assert cmp.models["dias_estancia"].res.b1 > 0 and cmp.models["dias_estancia"].res.p_b1 < 1e-30
    assert cmp.models["dias_estancia"].diag.get("homocedasticidad").severity == "fail"        # costo multiplicativo
    # con ln(Y) los residuos mejoran y el modelo deja de ser «no confiable»
    xs2 = ["dias_estancia", "num_procedimientos", "edad", "nivel_complejidad"]
    _, _, cmp_ln = _run(SALUD, xs2, y_transform="ln", include_transforms=False)
    assert cmp_ln.models["dias_estancia"].diag.get("normalidad").severity in {"ok", "warn"}
    assert cmp_ln.recommendation.status in {"con reservas", "recomendado"}


def test_health_negative_stay_is_detected_and_fixable():
    from core import profiling
    df = G.DATASETS[SALUD][1]()
    inc = profiling.find_inconsistencies(df)
    assert ((inc["variable"] == "dias_estancia") & (inc["problema"] == "negativos improbables")).any()
    res = cleaning.apply_plan(df, CleaningPlan().add("fix_nan", "dias_estancia", "a_faltante", rule="negativos"))
    assert (res.df["dias_estancia"].dropna() >= 0).all() and res.df["dias_estancia"].isna().sum() >= 2


def test_bank_income_is_best_legit_predictor_and_leaks_are_flagged():
    xs = ["ingreso_mensual_millones", "score_crediticio", "deuda_vigente_millones", "plazo_meses",
          "monto_desembolsado_millones", "cuota_mensual_millones"]
    _, md, cmp = _run(BANCA, xs, include_transforms=False, declared_factors=_decl(BANCA, xs))
    assert cmp.models["monto_desembolsado_millones"].leak == "alta"
    assert cmp.models["cuota_mensual_millones"].leak == "media"                 # se calcula después del monto
    assert cmp.models["ingreso_mensual_millones"].leak == "ninguna"             # factor declarado por el usuario
    assert cmp.recommendation.best_key == "ingreso_mensual_millones"
    assert "monto_desembolsado_millones" in cmp.recommendation.note_r2          # mayor R², pero descartado por fuga


def test_bank_cleaning_improves_reliability():
    xs = ["ingreso_mensual_millones", "score_crediticio", "deuda_vigente_millones", "plazo_meses"]
    _, _, raw = _run(BANCA, xs, include_transforms=False)
    assert raw.recommendation.status == "no confiable"                          # datos sucios: ingresos con error, negativos, duplicados
    plan = (CleaningPlan().add("drop_duplicates").add("fix_nan", "ingreso_mensual_millones", "a_faltante", rule="negativos")
            .add("outlier", "ingreso_mensual_millones", "remove", k=3).add("impute", "ingreso_mensual_millones", "median")
            .add("impute", "score_crediticio", "median").add("impute", "deuda_vigente_millones", "median"))
    _, md, clean = _run(BANCA, xs, include_transforms=False, plan=plan)
    assert clean.recommendation.status in {"con reservas", "recomendado"}
    best = clean.models[clean.recommendation.best_key]
    assert best.x_source == "ingreso_mensual_millones" and best.res.r2 > 0.5
    assert best.test_metrics["R²_fuera_muestra"] > 0.6
    # la «solución» ingenua (atípicos → faltante → mediana) destruye la relación legítima de los ingresos altos
    naive = (CleaningPlan().add("drop_duplicates").add("fix_nan", "ingreso_mensual_millones", "a_faltante", rule="negativos")
             .add("outlier", "ingreso_mensual_millones", "to_nan", k=3).add("impute", "ingreso_mensual_millones", "median")
             .add("impute", "score_crediticio", "median").add("impute", "deuda_vigente_millones", "median"))
    _, _, nv = _run(BANCA, xs, include_transforms=False, plan=naive)
    assert nv.models["ingreso_mensual_millones"].res.r2 < best.res.r2 - 0.1
