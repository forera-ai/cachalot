"""CUDA lane N2 stage 1: the seeded routing generator is deterministic, valid and feeds the existing simulator."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "cuda"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import routing_gen as rg  # noqa: E402

SMALL = dict(layers=4, experts=32, topk=3, tokens=120)


def test_same_spec_gives_identical_output_and_a_different_seed_does_not():
    a = list(rg.generate(rg.RoutingSpec(**SMALL, seed=1)))
    assert a == list(rg.generate(rg.RoutingSpec(**SMALL, seed=1)))
    assert a != list(rg.generate(rg.RoutingSpec(**SMALL, seed=2)))


def test_every_row_has_topk_distinct_valid_experts_in_trace_order():
    spec = rg.RoutingSpec(**SMALL, skew=1.2, locality=0.5)
    rows = list(rg.generate(spec))
    assert len(rows) == spec.tokens * spec.layers
    for i, (tok, layer, experts) in enumerate(rows):
        assert (tok, layer) == divmod(i, spec.layers)
        assert len(set(experts)) == spec.topk and all(0 <= e < spec.experts for e in experts)


@pytest.mark.parametrize("bad", [dict(topk=0), dict(topk=33), dict(skew=-1), dict(locality=1.5), dict(tokens=0)])
def test_invalid_specs_are_refused(bad):
    with pytest.raises(ValueError):
        list(rg.generate(rg.RoutingSpec(**{**SMALL, **bad})))


def test_skew_and_locality_raise_the_lru_hit_rate_as_designed():
    pytest.importorskip("numpy")
    from cache_sim import simulate

    def hit(**knobs):
        arrays = rg.to_arrays(rg.RoutingSpec(**{**SMALL, "tokens": 300}, **knobs), prefill_tokens=20)
        return simulate(arrays, [], 24, "lru", 0.0)["hit"]

    assert hit(skew=0.0, locality=0.0) < hit(skew=1.5, locality=0.0)
    assert hit(skew=0.0, locality=0.0) < hit(skew=0.0, locality=0.6)


def test_to_arrays_matches_the_trace_layout():
    pytest.importorskip("numpy")
    arrays = rg.to_arrays(rg.RoutingSpec(**SMALL), prefill_tokens=10)
    assert set(arrays) == {"phase", "layer", "position", "experts"}
    assert arrays["experts"].shape == (SMALL["tokens"] * SMALL["layers"], SMALL["topk"])
