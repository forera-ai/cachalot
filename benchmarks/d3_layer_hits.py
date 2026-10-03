"""
D3 pricing from a routing trace: the share of decode layers whose six experts are all resident, per budget, and the
net saving of GPU-side expert selection (host one layer behind) under a per-layer saving and rewind cost.

A layer with a missing expert must rewind (its read cannot be hidden), so only all-hit layers save time:
net = layers x (all_hit x saving - (1 - all_hit) x rewind). HANDOFF section 18.55.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/d3_layer_hits.py [trace.npz]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cachalot.metrics.routing_trace import PHASE_PREFILL, load_trace  # noqa: E402
from cache_sim import Q2_EXPERT_BYTES, _prefill_segment  # noqa: E402
from simulate_policies import GIB, N_LAYERS, Store  # noqa: E402

DEFAULT_TRACE = "benchmarks/results/trace_routing_v8_hermes.trace.npz"
PRICES = ((0.63, 0.0), (0.63, 0.35), (0.63, 0.7), (0.63, 1.4), (1.0, 0.7))  # ms saved, ms rewind, per layer


def layer_outcomes(arrays, slots: int) -> tuple[int, np.ndarray]:
    """(decode tokens, histogram of misses per decode layer-call, 0..6) under an LRU store of `slots`."""
    phase, layer, experts = arrays["phase"], arrays["layer"], arrays["experts"]
    change = np.nonzero(np.diff(phase.astype(np.int8)) != 0)[0] + 1
    bounds = [0, *change.tolist(), len(layer)]
    store = Store(slots, "lru")
    hist = np.zeros(experts.shape[1] + 1, dtype=np.int64)
    tokens = 0
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        if phase[lo] == PHASE_PREFILL:
            _prefill_segment(store, layer[lo:hi], experts[lo:hi])
            continue
        for row in range(lo, hi):
            la = int(layer[row])
            misses = 0
            for e in experts[row].tolist():
                key = (la, int(e))
                misses += not store._resident(key)
                store.decode_request(key)
            hist[misses] += 1
        tokens += (hi - lo) // N_LAYERS
    return tokens, hist


def main() -> None:
    arrays, _ = load_trace(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRACE)
    for gib in (36, 48, 52):
        tokens, hist = layer_outcomes(arrays, int(gib * GIB / Q2_EXPERT_BYTES))
        share = hist / hist.sum()
        print(f"{gib} GiB: {tokens} tokens, layers all-hit {share[0]:.3f}, misses per layer {share.round(3).tolist()}")
        for saving, rewind in PRICES:
            net = N_LAYERS * (share[0] * saving - (1 - share[0]) * rewind)
            print(f"   save {saving}/layer, rewind {rewind}: net {net:+.1f} ms a token")


if __name__ == "__main__":
    main()
