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
    top4 = [e for _, e in sorted(zip(rec[1][(t, la)][1], rec[1][(t, la)][0], strict=True), reverse=True)][:4]
    assert mix[:4] == top4 and len(mix) <= 6 and len(set(mix)) == len(mix)
    rco = pp.predicted("RCO10", table, real, rec, t, la)
    assert rco[:6] == r and len(rco) <= 10 and len(set(rco)) == len(rco)
    assert pp.predicted("none", table, real, rec, t, la) == []


def _multi_lead(a):
    """The synthetic trace plus lead-2 and lead-3 records whose experts differ from lead 1's (so a lead mix-up changes results)."""
    out = dict(a)
    n = len(a["pred_target"])
    keep2 = np.nonzero(a["pred_target"] >= 2)[0]
    keep3 = np.nonzero(a["pred_target"] >= 3)[0]
    extra = {}
    for lead, keep in ((2, keep2), (3, keep3)):
        extra[lead] = {"tgt": a["pred_target"][keep], "pos": a["pred_position"][keep],
                       "exp": (a["pred_experts"][keep] + 7 * lead) % 24, "w": a["pred_weights"][keep]}
    out["pred_target"] = np.concatenate([a["pred_target"], extra[2]["tgt"], extra[3]["tgt"]])
    out["pred_source"] = np.concatenate([a["pred_source"], extra[2]["tgt"] - 2, extra[3]["tgt"] - 3])
    out["pred_position"] = np.concatenate([a["pred_position"], extra[2]["pos"], extra[3]["pos"]])
    out["pred_experts"] = np.concatenate([a["pred_experts"], extra[2]["exp"], extra[3]["exp"]]).astype(np.int16)
    out["pred_weights"] = np.concatenate([a["pred_weights"], extra[2]["w"], extra[3]["w"]])
    assert len(out["pred_target"]) > n
    return out


def test_a_multi_lead_trace_keeps_the_leads_apart_and_lead_1_is_unchanged():
    a = synth(tokens=12)
    m = _multi_lead(a)
    assert pcov.predicted_sets(m) == pcov.predicted_sets(a)                                  # lead 1 is exactly the old trace
    p2, p3 = pcov.predicted_sets(m, 2), pcov.predicted_sets(m, 3)
    assert p2 and p3 and set(p2) != set(pcov.predicted_sets(m)) and all(t >= 2 for _p, t in p2) and all(t >= 3 for _p, t in p3)
    assert all(p2[k] != pcov.predicted_sets(m)[k] for k in list(p2)[:5])
    assert pcov.coverage(m, 17, "pred") == pcov.coverage(a, 17, "pred")                       # default lead 1: unchanged
    import overlap_arena as oa

    assert oa.predicted_lookup(m) == oa.predicted_lookup(a)                                    # the arena harness ignores other leads


def test_lead_predictors_read_their_own_records():
    a = synth(tokens=12)
    m = _multi_lead(a)
    items, real, rec, table, _nl = pp.prepare(m, train_tokens=6)
    t, la = 8, 3
    r1, r2, r3 = (pp.predicted(x, table, real, rec, t, la) for x in ("R", "R2", "R3"))
    assert len(r1) == len(r2) == len(r3) == 6 and r1 != r2 and r2 != r3
    r12 = pp.predicted("R12", table, real, rec, t, la)
    assert r12[:6] == r1 and r12[6:] == [e for e in r2 if e not in r1] and len(r12) <= 12
    assert pp.predicted("R2", table, real, rec, t, 1) == []                                    # lead 2 into layer 1 has no source
    base = pp.replay(items, real, rec, table, _nl, 17, "none", 6, 12)
    for name in ("R", "R2", "R3", "R12"):
        r = pp.replay(items, real, rec, table, _nl, 17, name, 6, 12)
        assert r["demand_misses"] <= base["demand_misses"] + 1e-9 and r["loads"] >= 0
