"""DI-1 equality script helpers (Mac only; docs/DI1-BUILD-RECORD.md): the pure comparison functions on planted arrays; the model is never loaded."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import di1_tail_equality as eq  # noqa: E402


def test_kl_of_identical_logits_is_zero_and_grows_with_a_perturbation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=1000).astype(np.float32)
    assert eq.kl(a, a) == pytest.approx(0.0, abs=1e-12)
    assert eq.kl(a, a + rng.normal(scale=0.01, size=1000).astype(np.float32)) < eq.kl(a, a + rng.normal(scale=0.5, size=1000).astype(np.float32))


def test_compare_arrays_reports_equal_and_relative_error():
    a = np.ones((4, 4), dtype=np.float32) * 2
    assert eq.compare_arrays(a, a)["equal"] is True
    b = a.copy()
    b[0, 0] += 0.02
    r = eq.compare_arrays(a, b)
    assert r["equal"] is False and r["max_abs"] == pytest.approx(0.02, rel=1e-3) and r["rel_max"] == pytest.approx(0.01, rel=1e-3)


def test_compare_runs_on_two_planted_state_files(tmp_path, capsys):
    rng = np.random.default_rng(1)
    st = {"logits": rng.normal(size=50).astype(np.float32), "window_20": rng.normal(size=(8, 4)).astype(np.float32),
          "decode_tokens": np.arange(4, dtype=np.int64), "decode_logits": rng.normal(size=(4, 50)).astype(np.float32),
          "compressed_20": rng.normal(size=(6, 4)).astype(np.float32), "candidates": np.arange(5, dtype=np.int32)}
    for name, noise in (("a", 0.0), ("b", 1e-3)):
        d = tmp_path / name
        d.mkdir()
        np.savez(d / "state.npz", **{k: (v + noise if v.dtype.kind == "f" else v) for k, v in st.items()})
    assert eq.compare(tmp_path / "a", tmp_path / "b") == 0
    out = capsys.readouterr().out
    assert '"decode_tokens_equal": true' in out and '"windows_bit_equal": false' in out and '"candidates"' in out
