from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store_true",
        default=False,
        help="run tests that load the real DeepSeek checkpoint",
    )
    parser.addoption(
        "--apple-fm",
        action="store_true",
        default=False,
        help="run tests against Apple's on-device system model (apple-fm-sdk)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "model: needs the real checkpoint (slow)")
    config.addinivalue_line("markers", "applefm: needs Apple's on-device system model")


def pytest_collection_modifyitems(config, items):
    skip_model = pytest.mark.skip(reason="needs --model")
    skip_fm = pytest.mark.skip(reason="needs --apple-fm")
    for item in items:
        if "model" in item.keywords and not config.getoption("--model"):
            item.add_marker(skip_model)
        if "applefm" in item.keywords and not config.getoption("--apple-fm"):
            item.add_marker(skip_fm)
