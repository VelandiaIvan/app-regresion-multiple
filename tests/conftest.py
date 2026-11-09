import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sample_data import generate  # noqa: E402


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("samples")
    generate.write_all(folder)
    return folder


@pytest.fixture(scope="session")
def retail():
    return generate.retail_ventas()


@pytest.fixture(scope="session")
def logistica():
    return generate.logistica_entregas()


@pytest.fixture(scope="session")
def manufactura():
    return generate.manufactura_costos()
