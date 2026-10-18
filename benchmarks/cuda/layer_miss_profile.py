"""
CUDA lane N2 stage 1 baseline (docs/CUDA-N2S1-EXPERIMENT-RECORD.md): how many misses does a decode layer raise at once?

A discrete-GPU expert cache pays one host-to-device copy per missed expert, and S2 and V1 measured that the cost of a copy
depends on how many are queued together (RTX 5090) or not at all (V100). The replay's unit of work is therefore the misses
of one layer of one token: their count decides whether batching could ever matter. This script replays a routing trace through
the runtime's `Store` model (LRU, tau 0, no prefetch, as `cuda_offline_policy_table.replay` does) and records, per decode
(token, layer), the number of missed experts. Mac only, no GPU, no model. The aggregate miss count is checked against
`cuda_offline_policy_table.replay`; a difference voids the output.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/cuda/layer_miss_profile.py \
        benchmarks/results/routing-trace-deepseek-long/trace.npz --fractions 17,34 --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cachalot.metrics.routing_trace import (  # noqa: E402
    PHASE_DECODE,
    PHASE_PREFILL,
    _token_ordinals,
    load_trace,
)
from cache_sim import _prefill_segment, trace_layers  # noqa: E402
from cuda_offline_policy_table import replay  # noqa: E402
from simulate_policies import Store  # noqa: E402


def layer_misses(arrays, slots: int, policy: str = "lru") -> dict:
    """Misses of every decode (token, layer) in trace order, same walk as `cuda_offline_policy_table.replay`."""
    phase, layer, experts = arrays["phase"], arrays["layer"], arrays["experts"]
    token_of_row = np.full(len(layer), -1, dtype=np.int64)
    drows = np.nonzero(phase == PHASE_DECODE)[0]
    token_of_row[drows] = _token_ordinals(layer[drows])
    st = Store(slots, policy, n_layers=trace_layers(arrays))
    change = np.nonzero(np.diff(phase.astype(np.int8)) != 0)[0] + 1
    bounds = [0, *change.tolist(), len(layer)]
    per_event: list[int] = []
    tokens_seen: set[int] = set()
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b <= a:
            continue
        if phase[a] == PHASE_PREFILL:
            _prefill_segment(st, layer[a:b], experts[a:b])
            continue
        for i in range(a, b):
            tokens_seen.add(int(token_of_row[i]))
            la, m = int(layer[i]), 0
            for e in experts[i].tolist():
                key = (la, int(e))
                if not st._resident(key):
                    m += 1
                st.decode_request(key)
            per_event.append(m)
    return {"events": per_event, "tokens": len(tokens_seen)}


def summarize(per_event: list[int], tokens: int) -> dict:
    a = np.array(per_event)
    miss_events = a[a > 0]
    hist = {str(k): int((a == k).sum()) for k in range(0, 5)}
    hist["5+"] = int((a >= 5).sum())
    return {
        "layer_events": int(a.size), "tokens": tokens, "misses": int(a.sum()),
        "misses_per_token": float(a.sum() / tokens),
        "layer_events_with_a_miss_per_token": float(miss_events.size / tokens),
        "mean_misses_in_a_missing_event": float(miss_events.mean()) if miss_events.size else 0.0,
        "p50_p95_max_in_a_missing_event": ([float(np.percentile(miss_events, 50)), float(np.percentile(miss_events, 95)),
                                            int(miss_events.max())] if miss_events.size else [0, 0, 0]),
        "event_histogram": hist,
        "share_of_misses_in_events_of_3_or_more": float(miss_events[miss_events >= 3].sum() / a.sum()) if a.sum() else 0.0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("trace")
    ap.add_argument("--fractions", default="17,34", help="resident percent of all experts, comma separated")
    ap.add_argument("--policy", default="lru")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    arrays, _segments = load_trace(args.trace)
    total = trace_layers(arrays) * (int(arrays["experts"].max()) + 1)
    out = {"trace": args.trace, "policy": args.policy, "total_experts": total, "cells": {}}
    for f in (float(x) for x in args.fractions.split(",")):
        slots = int(round(total * f / 100))
        lm = layer_misses(arrays, slots, args.policy)
        ref = replay(arrays, slots, args.policy)
        ok = sum(lm["events"]) == ref["misses"] and lm["tokens"] == ref["tokens"]
        cell = {"slots": slots, "consistent_with_replay": ok, **summarize(lm["events"], lm["tokens"])}
        out["cells"][f"{f:g}"] = cell
        print(f"{f:g} % ({slots} slots): {cell['misses_per_token']:.2f} misses/token in "
              f"{cell['layer_events_with_a_miss_per_token']:.2f} layers/token, mean {cell['mean_misses_in_a_missing_event']:.2f} "
              f"per missing layer, p50/p95/max {cell['p50_p95_max_in_a_missing_event']}, consistent={ok}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0 if all(c["consistent_with_replay"] for c in out["cells"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
