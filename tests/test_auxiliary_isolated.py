"""IsolatedProvider: one child, restart limit, parent-side deadline, clean shutdown."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

from cachalot.auxiliary import errors as E
from cachalot.auxiliary import isolated
from cachalot.auxiliary.isolated import IsolatedProvider
from cachalot.auxiliary.provider import GenerationParams

READY = json.dumps({"id": 0, "ready": True, "pid": 1})


def child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_real_worker_with_deterministic_provider():
    env = dict(os.environ, CACHALOT_AUX_PROVIDER="deterministic")
    p = IsolatedProvider("apple_foundation_models", timeout_s=5, env=env)
    caps = p.capabilities()
    assert not caps.available and caps.unavailable_reason == "deterministic"
    with pytest.raises(E.AuxUnavailable):
        p.generate("x")
    pid = p.worker_pid
    p.close()
    assert pid is not None and p._proc is None


def test_child_death_is_an_error_then_a_crash_loop():
    p = IsolatedProvider("x", timeout_s=2, command=child(f"print({READY!r}, flush=True)"),
                         max_restarts=2, restart_window_s=60)
    for _ in range(3):
        with pytest.raises(E.AuxProviderError):
            p.generate("x")
        time.sleep(0.05)
    with pytest.raises(E.AuxUnavailable) as info:
        p.generate("x")
    assert info.value.reason == "crash_loop"
    p.close()


def test_parent_deadline_kills_a_silent_child(monkeypatch):
    monkeypatch.setattr(isolated, "DEADLINE_GRACE_S", 0.2)
    p = IsolatedProvider("x", timeout_s=0.3,
                         command=child(f"import time; print({READY!r}, flush=True); time.sleep(60)"))
    t = time.perf_counter()
    with pytest.raises(E.AuxTimeout):
        p.generate("x", params=GenerationParams(timeout_s=0.3))
    assert time.perf_counter() - t < 3
    assert p._proc.poll() is not None      # killed, not left running
    p.close()


def test_child_error_payload_round_trips():
    reply = json.dumps({"id": 1, "ok": False, "error": {"kind": "context_overflow", "message": "too long",
                                                        "needed": 9000, "limit": 8192}})
    code = (f"import sys; print({READY!r}, flush=True); sys.stdin.readline(); "
            f"print({reply!r}, flush=True); sys.stdin.read()")
    p = IsolatedProvider("x", timeout_s=2, command=child(code))
    with pytest.raises(E.AuxContextOverflow) as info:
        p.generate("x")
    assert info.value.limit == 8192 and info.value.needed == 9000
    p.close()
