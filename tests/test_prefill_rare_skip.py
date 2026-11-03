"""CACHALOT_PREFILL_RARE_SKIP: the planning rule, its default-off state and its snapshot key (no model, no GPU)."""
from __future__ import annotations

import pytest

from cachalot.model import prefill_rare_skip, snapshot_store


def _assign(rows_by_expert):
    return {e: list(rows) for e, rows in rows_by_expert.items()}


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("CACHALOT_PREFILL_RARE_SKIP", raising=False)
    assert prefill_rare_skip.rare_skip() is None
    assert prefill_rare_skip.snapshot_tag() == ""
    for off in ("", "0", "off", "exact", "-1"):
        monkeypatch.setenv("CACHALOT_PREFILL_RARE_SKIP", off)
        assert prefill_rare_skip.rare_skip() is None


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("CACHALOT_PREFILL_RARE_SKIP", "2")
    monkeypatch.delenv("CACHALOT_PREFILL_RARE_MIN_ROWS", raising=False)
    assert prefill_rare_skip.rare_skip() == (2, 1024)
    monkeypatch.setenv("CACHALOT_PREFILL_RARE_MIN_ROWS", "512")
    assert prefill_rare_skip.rare_skip() == (2, 512)
    assert prefill_rare_skip.snapshot_tag() == "-rare2m512"


def test_below_min_rows_is_unchanged():
    a = _assign({1: [(0, 0, 0.5)], 2: [(0, 1, 0.5), (1, 0, 1.0)]})
    out, skipped = prefill_rare_skip.plan_rare_skip(a, n_tokens=2, k=1, min_rows=3)
    assert out is a and skipped == []


def test_skips_rare_experts_and_rescales_affected_rows():
    # token 0: experts 1 (0.2, rare), 2 (0.3), 3 (0.5); token 1: experts 2 (0.6), 3 (0.4); token 2: experts 2, 3
    a = _assign({
        2: [(0, 1, 0.3), (1, 0, 0.6), (2, 0, 0.5)],
        1: [(0, 0, 0.2)],
        3: [(0, 2, 0.5), (1, 1, 0.4), (2, 1, 0.5)],
    })
    out, skipped = prefill_rare_skip.plan_rare_skip(a, n_tokens=3, k=1, min_rows=1)
    assert skipped == [1]
    assert list(out) == [2, 3]  # order kept
    # token 0 lost 0.2 of 1.0: its kept weights scale by 1.0 / 0.8
    assert out[2][0] == (0, 1, pytest.approx(0.3 / 0.8))
    assert out[3][0] == (0, 2, pytest.approx(0.5 / 0.8))
    # tokens 1 and 2 are untouched, exactly
    assert out[2][1] == (1, 0, 0.6) and out[3][2] == (2, 1, 0.5)
    tok0 = sum(w for rows in out.values() for t, _, w in rows if t == 0)
    assert tok0 == pytest.approx(1.0)


def test_k_counts_rows_not_mass():
    a = _assign({7: [(0, 0, 0.9), (1, 0, 0.9)], 8: [(0, 1, 0.1), (1, 1, 0.1)]})
    out, skipped = prefill_rare_skip.plan_rare_skip(a, n_tokens=2, k=1, min_rows=1)
    assert skipped == [] and out is a
    out, skipped = prefill_rare_skip.plan_rare_skip(a, n_tokens=2, k=2, min_rows=1)
    assert sorted(skipped) == [7, 8] and out == {}


def test_snapshot_key_carries_the_setting(monkeypatch):
    monkeypatch.delenv("CACHALOT_PREFILL_DECODER_KEEP", raising=False)
    monkeypatch.delenv("CACHALOT_PREFILL_RARE_SKIP", raising=False)
    assert snapshot_store.numerics_version(False) == snapshot_store.NUMERICS_VERSION
    assert snapshot_store.numerics_version(True) == snapshot_store.NUMERICS_VERSION + "-di1.1"
    monkeypatch.setenv("CACHALOT_PREFILL_RARE_SKIP", "1")
    monkeypatch.delenv("CACHALOT_PREFILL_RARE_MIN_ROWS", raising=False)
    assert snapshot_store.numerics_version(False) == snapshot_store.NUMERICS_VERSION + "-rare1m1024"
    assert snapshot_store.numerics_version(True) == snapshot_store.NUMERICS_VERSION + "-di1.1-rare1m1024"


def test_log_field_is_empty_until_the_rule_applies(monkeypatch):
    monkeypatch.setattr(prefill_rare_skip, "STATS", prefill_rare_skip.Counter())
    assert prefill_rare_skip.log_field() == ""
    a = _assign({1: [(0, 0, 0.25)], 2: [(0, 1, 0.75), (1, 0, 1.0)]})
    prefill_rare_skip.plan_rare_skip(a, n_tokens=2, k=1, min_rows=1)
    assert prefill_rare_skip.log_field() == " rare_skip=1/12.50%"
