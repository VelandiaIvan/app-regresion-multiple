"""Prueba de humo de la interfaz: recorre los 13 módulos con cada dataset de ejemplo sin excepciones.

AppTest no admite el radio de navegación de app.py (su valor difiere del texto mostrado), así que cada
módulo se ejecuta con un script mínimo que reutiliza `render()` de la página; la navegación real se verifica en el navegador.
"""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import loader
from sample_data import generate as G

ROOT = str(Path(__file__).resolve().parents[1])
# El flujo empieza por los datos (m02) y sigue con el entendimiento automático (m01).
ORDER = ["m02_datos", "m01_contexto", "m03_preparacion", "m04_eda", "m05_y", "m06_x", "m07_modelos", "m08_evaluacion",
         "m09_interpretacion", "m10_simulacion", "m11_sensibilidad", "m12_informe", "m13_aprendizaje"]

SCRIPT = f'''
import sys, importlib
sys.path.insert(0, r"{ROOT}")
import streamlit as st
from ui import state
state.init()
importlib.import_module("ui." + st.session_state["_page"]).render()
'''


def _app() -> AppTest:
    at = AppTest.from_string(SCRIPT, default_timeout=180)
    at.session_state["_page"] = "m02_datos"
    at.run()
    return at


def _go(at: AppTest, page: str) -> None:
    at.session_state["_page"] = page
    at.run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"


def _load_sample(at: AppTest, sample: str) -> None:
    at.selectbox(key="sample_sel").set_value(sample)
    btn = [b for b in at.button if "Cargar ejemplo" in b.label]
    assert btn, "no apareció el botón de ejemplo"
    btn[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["loaded"] is not None


@pytest.mark.parametrize("sample", list(G.DATASETS))
def test_full_walkthrough(sample):
    at = _app()
    assert not at.exception, [e.value for e in at.exception]
    _load_sample(at, sample)
    for page in ORDER:
        _go(at, page)
    ss = at.session_state
    assert ss["y"] is not None and ss["cmp"] is not None and ss["chosen"] is not None
    assert ss["y"] == G.CONTEXTS[sample]["y"], f"Y recomendada {ss['y']} ≠ esperada {G.CONTEXTS[sample]['y']}"
    cmp = ss["cmp"]
    assert cmp.models, "no se construyó ningún modelo"
    assert all(m.res.k >= 2 for m in cmp.models.values()), "todos los modelos deben tener 2 o más variables"
    assert cmp.models[cmp.recommendation.best_key].res.k >= 2


@pytest.mark.parametrize("sample,must_have,never", [
    ("Banca · monto de crédito aprobado (1000 registros)", "ingreso_mensual_millones", "monto_desembolsado_millones"),
    ("Salud · costo de hospitalización (1000 registros)", "dias_estancia", "costo_facturado_eps_millones"),
])
def test_ui_flow_uses_legit_predictors_and_drops_leaks(sample, must_have, never):
    """De punta a punta: el modelo recomendado incluye el predictor legítimo y nunca la variable con fuga de información."""
    at = _app()
    _load_sample(at, sample)
    for page in ORDER:
        _go(at, page)
    cmp = at.session_state["cmp"]
    best = cmp.models[cmp.recommendation.best_key]
    assert must_have in best.x_cols, f"se recomendó {best.key}"
    assert all(never not in m.x_cols for m in cmp.models.values())
    assert never not in at.session_state["x_cands"]          # el módulo 6 no la preselecciona (riesgo ALTO de fuga)


def test_pages_guard_without_data():
    at = _app()
    for page in ORDER[1:]:
        _go(at, page)


def test_second_split_mode_and_confidence_levels():
    """Cambios de configuración: partición temporal y 99 % no deben romper ninguna página."""
    at = _app()
    _load_sample(at, "Servicios · consumo de energía (serie con autocorrelación)")
    for page in ORDER[:6]:
        _go(at, page)
    at.session_state["split"] = {"method": "temporal", "test_size": 0.25, "seed": 1}
    at.session_state["conf"] = 99
    for page in ORDER[5:]:
        _go(at, page)
    assert at.session_state["cmp"].md.spec.method == "temporal"


def test_user_file_without_any_context_text():
    """Archivo propio sin escribir nada: la app propone Y, arma el modelo múltiple y todas las páginas funcionan."""
    df = pd.read_csv(Path(ROOT) / "sample_data" / "ventas_restaurante.csv", sep=";", decimal=",")
    at = _app()
    at.session_state["loaded"] = loader.from_dataframe(df, "ventas_restaurante.csv")
    at.session_state["data_id"] = 1
    at.session_state["source_name"] = "ventas_restaurante.csv"
    for page in ORDER:
        _go(at, page)
    ss = at.session_state
    assert ss["y"] == "Bowls"
    best = ss["cmp"].models[ss["cmp"].recommendation.best_key]
    assert best.res.k >= 2 and "Bowl Price" in best.x_cols
