"""
Seeded synthetic routing generator for the CUDA lane's N2 stage 1 (docs/CUDA-RESEARCH-PLAN.md section 6), written and
tested on the Mac with no GPU and no model. Standard library only; `to_arrays` needs numpy and is used by the Mac tests.

A token routes `topk` distinct experts in each layer. Three knobs control the pattern:

* `skew`: expert popularity follows a Zipf law with this exponent (0 = uniform). Each layer has its own seeded
  permutation, so the hot experts differ between layers, as in the recorded traces.
* `locality`: probability that a slot of this token's routing repeats an expert the same layer routed on the previous token.
* `seed`: everything is derived from it; the same arguments give byte-identical output.

This is a workload generator, not a model of any real router. A result obtained from it states the knobs it was run with
and is labelled stage 1 (plan N2 stage table); it says nothing about real model routing.
"""

from __future__ import annotations

import bisect
import itertools
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingSpec:
    layers: int = 40
    experts: int = 384
    topk: int = 6
    tokens: int = 200
    skew: float = 1.0
    locality: float = 0.3
    seed: int = 20261018

    def validated(self) -> RoutingSpec:
        if self.layers < 1 or self.experts < 1 or self.tokens < 1:
            raise ValueError("layers, experts and tokens must be positive")
        if not (1 <= self.topk <= self.experts):
            raise ValueError("topk must be in [1, experts]")
        if self.skew < 0 or not (0.0 <= self.locality <= 1.0):
            raise ValueError("skew must be >= 0 and locality in [0, 1]")
        return self


def _cdf(n: int, skew: float) -> list[float]:
    weights = [1.0 / ((rank + 1) ** skew) for rank in range(n)]
    total = sum(weights)
    return list(itertools.accumulate(w / total for w in weights))


def generate(spec: RoutingSpec):
    """Yield (token, layer, experts tuple of length topk) in trace order (token-major, then layer)."""
    spec = spec.validated()
    master = random.Random(spec.seed)
    perms = []
    for _ in range(spec.layers):
        perm = list(range(spec.experts))
        master.shuffle(perm)
        perms.append(perm)
    cdf = _cdf(spec.experts, spec.skew)
    rng = random.Random(master.getrandbits(64))
    previous: list[tuple[int, ...]] = [() for _ in range(spec.layers)]
    for token in range(spec.tokens):
        for layer in range(spec.layers):
            chosen: list[int] = []
            prev = previous[layer]
            while len(chosen) < spec.topk:
                if prev and rng.random() < spec.locality:
                    e = prev[rng.randrange(len(prev))]
                else:
                    e = perms[layer][min(bisect.bisect_left(cdf, rng.random()), spec.experts - 1)]
                if e not in chosen:
                    chosen.append(e)
            previous[layer] = tuple(chosen)
            yield token, layer, tuple(chosen)


def to_arrays(spec: RoutingSpec, prefill_tokens: int = 0):
    """Arrays shaped like a recorded trace (phase, layer, position, experts) for cache_sim / simulate_policies.

    `prefill_tokens` generated tokens come first as a prefill phase, the rest as decode.
    """
    import numpy as np

    from cachalot.metrics.routing_trace import PHASE_DECODE, PHASE_PREFILL

    rows = list(generate(spec))
    phase = np.array([PHASE_PREFILL if t < prefill_tokens else PHASE_DECODE for t, _, _ in rows], dtype=np.uint8)
    return {
        "phase": phase,
        "layer": np.array([la for _, la, _ in rows]),
        "position": np.array([t for t, _, _ in rows]),
        "experts": np.array([e for _, _, e in rows]),
    }
