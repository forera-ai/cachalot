"""Offline predictor pricing (Mac only): the recorded predictor equals prefetch_coverage.py, and the other predictors behave as defined."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))
sys.path.insert(0, str(ROOT / "tests"))

import predictor_price as pp  # noqa: E402
import prefetch_coverage as pcov  # noqa: E402
from test_cuda_overlap_arena import synth  # noqa: E402


@pytest.fixture(scope="module")
def prepared():
    a = synth(tokens=12)
    return a, pp.prepare(a, train_tokens=6)


def test_the_recorded_predictor_equals_the_coverage_model_on_the_same_tokens(prepared):
    a, (items, real, rec, table, nl) = prepared
    for pct in (17, 50):
        mine = pp.replay(items, real, rec, table, nl, pct, "R", 0, 12)
        cov = pcov.coverage(a, pct, "pred", max_decode_tokens=12)
        assert mine["demand_misses"] == pytest.approx(cov["demand_misses_per_token"])
        assert mine["loads"] == pytest.approx(cov["prefetch_loads_per_token"])
        none = pp.replay(items, real, rec, table, nl, pct, "none", 0, 12)
        assert none["demand_misses"] == pytest.approx(pcov.coverage(a, pct, "none", max_decode_tokens=12)["demand_misses_per_token"])


def test_co_activation_is_trained_only_on_earlier_tokens_and_ranks_by_summed_counts(prepared):
    _a, (_items, real, _rec, table, _nl) = prepared
    assert table[:, :, :].sum() > 0
    t = 8                                                       # a held-out token (ordinal >= 6)
    ranked = pp.co_rank(table, real, t, 2)
    assert len(ranked) == len(set(ranked)) and all(0 <= e < pp.N_EXPERTS for e in ranked)
    score = table[2, real[(t, 1)]].sum(0)
    assert all(score[a] >= score[b] for a, b in zip(ranked, ranked[1:], strict=False))
    assert pp.co_rank(table, real, t, 0) == []                  # layer 0 has no previous layer


def test_predictor_sizes_and_the_temporal_and_mixed_definitions(prepared):
    _a, (_items, real, rec, table, _nl) = prepared
    t, la = 8, 2
    assert pp.predicted("TP6", table, real, rec, t, la) == real[(t - 1, la)]
    assert pp.predicted("TP6", table, real, rec, 0, la) == []
    r = pp.predicted("R", table, real, rec, t, la)
    assert len(r) == 6
    mix = pp.predicted("MIX6", table, real, rec, t, la)
    top4 = [e for _, e in sorted(zip(rec[(t, la)][1], rec[(t, la)][0], strict=True), reverse=True)][:4]
    assert mix[:4] == top4 and len(mix) <= 6 and len(set(mix)) == len(mix)
    rco = pp.predicted("RCO10", table, real, rec, t, la)
    assert rco[:6] == r and len(rco) <= 10 and len(set(rco)) == len(rco)
    assert pp.predicted("none", table, real, rec, t, la) == []
