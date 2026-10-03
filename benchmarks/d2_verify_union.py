"""
D2 pricing from a routing trace: what a speculative verify of K consecutive tokens costs in expert reads.

A verify pass runs K positions through every layer at once, so a layer reads the union of the K tokens' experts.
The trace holds the tokens that were actually generated, so its K-blocks are the all-accepted case; the experts
of rejected draft positions are not in the trace and are approximated by the next real tokens (their misses are
probably higher, so the result is optimistic). Per accepted token:

    ms = (floor + marginal x (K - 1) + draft + miss_ms x block_misses) / accepted_per_step
    accepted_per_step = 1 + p + ... + p^(K-1)        (p = per-position acceptance)

Defaults are the research's (docs/SPEED-RESEARCH-2026-10-03.md: floor 70, marginal 8, draft 15, 2.0 ms a miss).
HANDOFF section 18.56.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/d2_verify_union.py [trace.npz] [--gib 48]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cachalot.metrics.routing_trace import PHASE_PREFILL, load_trace  # noqa: E402
from cache_sim import Q2_EXPERT_BYTES, _prefill_segment  # noqa: E402
from simulate_policies import GIB, N_LAYERS, Store  # noqa: E402

DEFAULT_TRACE = "benchmarks/results/trace_routing_v8_hermes.trace.npz"


def block_misses(arrays, slots: int, k: int) -> tuple[int, int]:
    """(decode tokens, expert misses) when decode runs in blocks of k consecutive tokens (a request's tail
    shorter than k is a smaller block); a layer reads the union of the block's experts."""
    phase, layer, position, experts = (arrays[x] for x in ("phase", "layer", "position", "experts"))
    change = np.nonzero(np.diff(phase.astype(np.int8)) != 0)[0] + 1
    bounds = [0, *change.tolist(), len(layer)]
    store = Store(slots, "lru")
    tokens = misses = 0
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        if phase[lo] == PHASE_PREFILL:
            _prefill_segment(store, layer[lo:hi], experts[lo:hi])
            continue
        seg_pos = position[lo:hi]
        order = np.unique(seg_pos)
        for start in range(0, len(order), k):
            wanted = order[start:start + k]
            in_block = np.isin(seg_pos, wanted)
            rows = np.nonzero(in_block)[0] + lo
            per_layer: dict[int, list[int]] = {}
            for row in rows:
                per_layer.setdefault(int(layer[row]), []).extend(experts[row].tolist())
            for la in range(N_LAYERS):
                for e in dict.fromkeys(per_layer.get(la, ())):
                    key = (la, int(e))
                    misses += not store._resident(key)
                    store.decode_request(key)
            tokens += len(wanted)
    return tokens, misses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default=DEFAULT_TRACE)
    ap.add_argument("--gib", type=float, default=48.0)
    ap.add_argument("--floor-ms", type=float, default=70.0)
    ap.add_argument("--marginal-ms", type=float, default=8.0)
    ap.add_argument("--draft-ms", type=float, default=15.0)
    ap.add_argument("--miss-ms", type=float, default=2.0)
    args = ap.parse_args()
    arrays, _ = load_trace(args.trace)
    slots = int(args.gib * GIB / Q2_EXPERT_BYTES)
    base_t, base_m = block_misses(arrays, slots, 1)
    base_ms = args.floor_ms + args.miss_ms * base_m / base_t
    print(f"{args.gib:g} GiB, {base_t} decode tokens; baseline {base_m / base_t:.1f} misses, {base_ms:.0f} ms a token")
    print("| K | misses a block | per token | p=0.6 | p=0.78 | p=0.9 |   (ms per accepted token, speed-up)")
    for k in (1, 2, 3, 4, 6):
        t, m = block_misses(arrays, slots, k)
        per_block = m / (t / k)
        cells = []
        for p in (0.6, 0.78, 0.9):
            accepted = sum(p**i for i in range(k))
            draft = args.draft_ms if k > 1 else 0.0
            step = args.floor_ms + args.marginal_ms * (k - 1) + draft + args.miss_ms * per_block
            ms = step / accepted
            cells.append(f"{ms:.0f} ms {base_ms / ms:.2f}x")
        print(f"| {k} | {per_block:.1f} | {m / t:.1f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
