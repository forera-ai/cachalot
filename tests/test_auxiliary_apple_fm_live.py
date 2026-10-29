"""
Integration tests against Apple's on-device system model. Skipped unless run with --apple-fm:

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src \
      ~/venvs/deepseek-v41/bin/python -m pytest -q --apple-fm tests/test_auxiliary_apple_fm_live.py
"""

from __future__ import annotations

import threading
import time

import pytest

from cachalot.auxiliary import errors as E
from cachalot.auxiliary.apple_fm import AppleFoundationModelsProvider
from cachalot.auxiliary.isolated import IsolatedProvider
from cachalot.auxiliary.provider import GenerationParams
from cachalot.auxiliary.tasks import ClassifyRequest, ExtractRequest, FieldSpec, classify, extract

pytestmark = pytest.mark.applefm


@pytest.fixture(scope="module")
def fm():
    p = AppleFoundationModelsProvider(timeout_s=20)
    if not p.is_available():
        pytest.skip(f"system model unavailable: {p.capabilities().unavailable_reason}")
    yield p
    p.close()


def test_capabilities_are_local_and_honest(fm):
    caps = fm.capabilities()
    assert caps.available and caps.local_only and caps.context_tokens and caps.context_tokens >= 4096
    assert not caps.logits and not caps.tokenizer_access and not caps.kv_access


def test_greedy_generate_is_repeatable(fm):
    a = fm.generate("Name the three primary colors, comma separated.", params=GenerationParams(max_tokens=24))
    b = fm.generate("Name the three primary colors, comma separated.", params=GenerationParams(max_tokens=24))
    assert a.text and a.text == b.text


def test_stream_concatenates_to_text(fm):
    text = "".join(fm.generate_stream("Count from 1 to 10 separated by spaces.", params=GenerationParams(max_tokens=40)))
    assert "1" in text and "10" in text


def test_overflow_is_typed(fm):
    with pytest.raises(E.AuxContextOverflow):
        fm.generate("The river flows past the mill. " * 2_000, params=GenerationParams(max_tokens=8))


def test_cancel_and_reuse(fm):
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t = time.perf_counter()
    with pytest.raises(E.AuxCancelled):
        fm.generate("Write a 1500 word essay about oceans.", params=GenerationParams(max_tokens=1500), cancel=cancel)
    assert time.perf_counter() - t < 3
    assert fm.generate("Say ok.", params=GenerationParams(max_tokens=4)).text


def test_tasks_on_the_real_model(fm):
    c = classify(ClassifyRequest("Refactor resident_store.py to use LFU with decay and keep the tests passing."), fm)
    assert c.source == "model" and c.route == "target"
    r = extract(ExtractRequest("Serve on host 127.0.0.1, port 8080.", (FieldSpec("host"), FieldSpec("port", "integer"),
                                                                       FieldSpec("password"))), fm)
    assert r.values.get("port") == 8080 and "password" in r.missing


def test_isolated_worker_round_trip():
    p = IsolatedProvider("apple_foundation_models", timeout_s=20)
    try:
        assert p.capabilities().available
        assert p.generate("Say ok.", params=GenerationParams(max_tokens=4)).text
    finally:
        p.close()
