"""CUDA lane N2 stage 1 baseline: per-layer miss counts agree with the policy-table replay (no files, no GPU)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "cuda"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import layer_miss_profile as p  # noqa: E402
import routing_gen as rg  # noqa: E402

from cuda_offline_policy_table import replay  # noqa: E402


@pytest.mark.parametrize("slots", [10, 40, 120])
def test_layer_misses_sum_to_the_replay_and_group_by_layer(slots):
    arrays = rg.to_arrays(rg.RoutingSpec(layers=4, experts=24, topk=3, tokens=60, skew=1.0, locality=0.3), prefill_tokens=10)
    lm = p.layer_misses(arrays, slots)
    ref = replay(arrays, slots, "lru")
    assert sum(lm["events"]) == ref["misses"] and lm["tokens"] == ref["tokens"]
    assert len(lm["events"]) == ref["tokens"] * 4 and max(lm["events"]) <= 3
    s = p.summarize(lm["events"], lm["tokens"])
    assert s["event_histogram"]["0"] + sum(v for k, v in s["event_histogram"].items() if k != "0") == s["layer_events"]
