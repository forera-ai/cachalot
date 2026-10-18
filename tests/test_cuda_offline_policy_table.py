"""CUDA lane N0a: the offline policy table agrees with cache_sim on a small synthetic trace (no files, no GPU)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import cuda_offline_policy_table as t  # noqa: E402
from cachalot.metrics.routing_trace import PHASE_DECODE, PHASE_PREFILL  # noqa: E402
from cache_sim import simulate  # noqa: E402


def synthetic(layers=4, prefill=12, decode=60, experts=24, topk=3, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(prefill):                                  # prefill: one row per (token, layer)
        for la in range(layers):
            rows.append((PHASE_PREFILL, la, 0, rng.choice(experts, topk, replace=False)))
    for pos in range(decode):
        for la in range(layers):
            hot = rng.choice(8, topk, replace=False) if rng.random() < 0.7 else rng.choice(experts, topk, replace=False)
            rows.append((PHASE_DECODE, la, pos, hot))
    return {"phase": np.array([r[0] for r in rows], dtype=np.uint8), "layer": np.array([r[1] for r in rows]),
            "position": np.array([r[2] for r in rows]), "experts": np.stack([r[3] for r in rows])}


@pytest.mark.parametrize("policy", ["lru", "slru", "lfu"])
@pytest.mark.parametrize("slots", [8, 20, 48])
def test_replay_equals_cache_sim_and_is_deterministic(policy, slots):
    arrays = synthetic()
    r = t.replay(arrays, slots, policy)
    ref = simulate(arrays, [], slots, policy, 0.0)
    assert (r["tokens"], r["requests"], r["misses"]) == (ref["tokens"], ref["requests"], ref["misses"])
    assert sum(r["per_token"]) == r["misses"] and len(r["per_token"]) == r["tokens"]
    assert t.replay(arrays, slots, policy)["per_token"] == r["per_token"]


def test_block_bootstrap_separates_a_real_gap_from_none_and_is_seeded():
    mean, lo, hi = t.block_bootstrap_mean([1.0, 2.0] * 60, 1)
    assert lo > 0 and mean == pytest.approx(1.5) and (lo, hi) == t.block_bootstrap_mean([1.0, 2.0] * 60, 1)[1:]
    import random
    rng = random.Random(5)
    _, lo0, hi0 = t.block_bootstrap_mean([rng.choice((-1.0, 1.0)) for _ in range(200)], 1)
    assert lo0 < 0 < hi0
