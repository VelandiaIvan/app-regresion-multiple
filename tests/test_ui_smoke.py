"""Prueba de humo de la interfaz: recorre los 13 módulos con cada dataset de ejemplo sin excepciones.

AppTest no admite el radio de navegación de app.py (su valor difiere del texto mostrado), así que cada
módulo se ejecuta con un script mínimo que reutiliza `render()` de la página; la navegación real se verifica en el navegador.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from sample_data import generate as G

ROOT = str(Path(__file__).resolve().parents[1])
ORDER = ["m01_contexto", "m02_datos", "m03_preparacion", "m04_eda", "m05_y", "m06_x", "m07_modelos", "m08_evaluacion",
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
    at.session_state["_page"] = "m01_contexto"
    at.run()
    return at


def _go(at: AppTest, page: str) -> None:
    at.session_state["_page"] = page
    at.run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"


@pytest.mark.parametrize("sample", list(G.DATASETS))
def test_full_walkthrough(sample):
    at = _app()
    assert not at.exception, [e.value for e in at.exception]
    at.selectbox(key="sample_sel").set_value(sample)
    btn = [b for b in at.button if "Cargar ejemplo" in b.label]
    assert btn, "no apareció el botón de ejemplo"
    btn[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["loaded"] is not None
    for page in ORDER:
        _go(at, page)
    ss = at.session_state
    assert ss["y"] is not None and ss["cmp"] is not None and ss["chosen"] is not None
    assert ss["y"] == G.CONTEXTS[sample]["y"], f"Y recomendada {ss['y']} ≠ esperada {G.CONTEXTS[sample]['y']}"
    assert not [e for e in at.error if "No se pudo" in str(e.value)]


@pytest.mark.parametrize("sample,x_expected", [
    ("Banca · monto de crédito aprobado (1000 registros)", "ingreso_mensual_millones"),
    ("Salud · costo de hospitalización (1000 registros)", "dias_estancia"),
])
def test_ui_flow_picks_the_legitimate_predictor_on_1000_record_datasets(sample, x_expected):
    """De punta a punta por la interfaz (incluye los factores declarados en el módulo 1): gana el predictor legítimo."""
    at = _app()
    at.selectbox(key="sample_sel").set_value(sample)
    [b for b in at.button if "Cargar ejemplo" in b.label][0].click().run()
    for page in ORDER:
        _go(at, page)
    cmp = at.session_state["cmp"]
    best = cmp.models[cmp.recommendation.best_key]
    assert best.x_source == x_expected, f"se recomendó {best.key}"
    assert cmp.md.n_with_y == 1000 - 10 or cmp.md.n_with_y == 1000                      # los duplicados no se eliminan solos
    # ninguna variable con fuga alta entra por defecto a la comparación
    assert all(m.leak != "alta" for m in cmp.models.values())
    if "Banca" in sample:
        assert cmp.models["cuota_mensual_millones"].leak == "media"                     # se calcula después del monto
        assert cmp.models["ingreso_mensual_millones"].leak == "ninguna"                 # factor declarado por el usuario


def test_pages_guard_without_data():
    at = _app()
    for page in ORDER[1:]:
        _go(at, page)


def test_second_split_mode_and_confidence_levels():
    """Cambios de configuración: partición temporal y 99 % no deben romper ninguna página."""
    at = _app()
    key = "Servicios · consumo de energía (serie con autocorrelación)"
    at.selectbox(key="sample_sel").set_value(key)
    [b for b in at.button if "Cargar ejemplo" in b.label][0].click().run()
    for page in ORDER[1:6]:
        _go(at, page)
    at.session_state["split"] = {"method": "temporal", "test_size": 0.25, "seed": 1}
    at.session_state["conf"] = 99
    for page in ORDER[5:]:
        _go(at, page)
    assert at.session_state["cmp"].md.spec.method == "temporal"
