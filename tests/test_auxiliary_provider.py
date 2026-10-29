"""AppleFoundationModelsProvider against a fake SDK: availability, deadlines, cancel, errors, admission."""

from __future__ import annotations

import threading
import time

import pytest

from cachalot.auxiliary import errors as E
from cachalot.auxiliary.apple_fm import AppleFoundationModelsProvider
from cachalot.auxiliary.provider import GenerationParams
from fake_apple_fm import (
    ConcurrentRequestsError,
    ExceededContextWindowSizeError,
    GenerationError,
    GuardrailViolationError,
    SystemLanguageModelUnavailableReason,
    make_sdk,
)


def provider(sdk, **kw):
    kw.setdefault("timeout_s", 2.0)
    return AppleFoundationModelsProvider(sdk=sdk, **kw)


def test_generate_and_capabilities():
    sdk = make_sdk(reply="hello")
    p = provider(sdk)
    caps = p.capabilities()
    assert caps.available and caps.local_only and caps.context_tokens == 8192
    assert caps.structured_output and caps.streaming and not caps.logits and not caps.kv_access
    r = p.generate("hi", instructions="be brief")
    assert r.text == "hello" and r.prompt_tokens is not None and r.latency_s >= 0
    p.close()


@pytest.mark.parametrize("reason,name", [
    (SystemLanguageModelUnavailableReason.APPLE_INTELLIGENCE_NOT_ENABLED, "apple_intelligence_not_enabled"),
    (SystemLanguageModelUnavailableReason.MODEL_NOT_READY, "model_not_ready"),
    (SystemLanguageModelUnavailableReason.DEVICE_NOT_ELIGIBLE, "device_not_eligible"),
])
def test_unavailable_reasons(reason, name):
    p = provider(make_sdk(available=False, reason=reason))
    assert not p.is_available()
    assert p.capabilities().unavailable_reason == name
    with pytest.raises(E.AuxUnavailable) as info:
        p.generate("x")
    assert info.value.reason == name


def test_context_precheck_overflows_before_generation():
    sdk = make_sdk(context_size=100, tokens_per_char=1.0)
    p = provider(sdk)
    with pytest.raises(E.AuxContextOverflow) as info:
        p.generate("x" * 50, params=GenerationParams(max_tokens=40))
    assert info.value.limit == 100
    assert sdk.calls.respond == 0


def test_timeout_cancels_native_task():
    sdk = make_sdk(delay_s=5.0)
    p = provider(sdk)
    t = time.perf_counter()
    with pytest.raises(E.AuxTimeout):
        p.generate("x", params=GenerationParams(timeout_s=0.2))
    assert time.perf_counter() - t < 1.5
    time.sleep(0.05)
    assert sdk.calls.cancelled == 1
    p.close()


def test_cancel_event():
    sdk = make_sdk(delay_s=5.0)
    p = provider(sdk)
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    t = time.perf_counter()
    with pytest.raises(E.AuxCancelled):
        p.generate("x", cancel=cancel)
    assert time.perf_counter() - t < 1.5
    p.close()


@pytest.mark.parametrize("exc,cls", [
    (ExceededContextWindowSizeError("x"), E.AuxContextOverflow),
    (GuardrailViolationError("x"), E.AuxRefused),
    (ConcurrentRequestsError("x"), E.AuxBusy),
    (GenerationError("status 255"), E.AuxProviderError),
    (GenerationError("Error FoundationModels.LanguageModelError:3 - The model refused to answer."), E.AuxRefused),
    (RuntimeError("boom"), E.AuxProviderError),
])
def test_error_mapping(exc, cls):
    p = provider(make_sdk(raise_exc=exc))
    with pytest.raises(cls):
        p.generate("x")
    p.close()


def test_structured_parses_and_rejects_non_object():
    p = provider(make_sdk(structured={"a": 1}))
    assert p.generate_structured("x", {"type": "object"}).structured == {"a": 1}
    p2 = provider(make_sdk(structured="[1, 2]"))
    with pytest.raises(E.AuxInvalidOutput):
        p2.generate_structured("x", {"type": "object"})
    p3 = provider(make_sdk(structured="not json"))
    with pytest.raises(E.AuxInvalidOutput):
        p3.generate_structured("x", {"type": "object"})


def test_stream_yields_deltas_from_snapshots():
    p = provider(make_sdk(snapshots=["He", "Hell", "Hello"]))
    assert list(p.generate_stream("x")) == ["He", "ll", "o"]


def test_stream_early_close_cancels():
    sdk = make_sdk(snapshots=["a", "ab", "abc", "abcd"], delay_s=0.3)
    p = provider(sdk)
    it = p.generate_stream("x")
    assert next(it) == "a"
    it.close()
    time.sleep(0.1)
    assert sdk.calls.cancelled == 1


def test_admission_is_bounded():
    sdk = make_sdk(delay_s=0.5)
    p = provider(sdk, max_concurrent_requests=1, queue_limit=1)
    errors: list[Exception] = []

    def run():
        try:
            p.generate("x")
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(3)]
    for t in threads:
        t.start()
        time.sleep(0.02)
    for t in threads:
        t.join()
    assert [type(e) for e in errors] == [E.AuxBusy]
    p.close()


def test_closed_provider_is_unavailable():
    p = provider(make_sdk())
    p.generate("x")
    p.close()
    with pytest.raises(E.AuxUnavailable) as info:
        p.generate("x")
    assert info.value.reason == "closed"


def test_missing_sdk_reports_reason(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "apple_fm_sdk":
            raise ImportError("no module")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    p = AppleFoundationModelsProvider()
    caps = p.capabilities()
    assert not caps.available
    assert caps.unavailable_reason in ("sdk_missing", "unsupported_os", "unsupported_hardware")
