"""
A row-level window model of the `P0L` arm (CUDA lane, Mac only, no GPU; record `docs/CUDA-BRIDGE-EXPERIMENT-RECORD.md`).

Replays the recorded trace with the stage A/B bookkeeping (lead 1) and counts, per decode row, the prefetch loads issued during the
previous layer (`n_pf`) and the demand misses at the row's boundary (`n_dem`). With a per-copy time `c` and a layer time `C`, a copy
stream that is idle at a layer's launch exposes, per row, `max(0, n_pf * c - C) + n_dem * c` (the prefetch that does not fit in the
previous layer's compute, plus every demand copy). The demand-only arm exposes `n_dem_no_prefetch * c` a row. The model ignores the
host, event overhead and interference: it asks only whether the two traces' row-level copy counts alone explain a different hidden
share at the same layer time.

Usage:
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/window_model.py --trace benchmarks/results/routing-trace-deepseek-long/trace.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prefetch_coverage as pc  # noqa: E402
import replay_arena as ra  # noqa: E402


def row_counts(arrays, percent: float, tokens: int, prefetch: bool):
    """Per decode row (in trace order, first `tokens` tokens): (token, layer, n_pf, n_dem)."""
    base = {k: arrays[k] for k in ("phase", "layer", "position", "experts")}
    pred = pc.predicted_sets(arrays, 1) if prefetch else {}
    slots = ra.slots_for(percent)
    lru: OrderedDict = OrderedDict()

    def touch(k):
        if k in lru:
            lru.move_to_end(k)
            return
        while len(lru) >= slots:
            lru.popitem(last=False)
        lru[k] = 1

    pos_iter = iter(int(base["position"][i]) for i in np.nonzero(base["phase"] == ra.PHASE_DECODE)[0])
    plan = [ev if ev[0] == "P" else (*ev, next(pos_iter)) for ev in ra.walk(base, tokens)]
    pre = ra.ArenaLRU(slots, int(base["layer"].max()) + 1)
    rows, started = [], set()
    pf_of: dict[int, int] = {}

    def issue(t):
        _, _tok, layer, _e, pos = plan[t]
        n = 0
        for e in pred.get((pos, layer), []):
            k = (layer, e)
            if k not in lru:
                touch(k)
                n += 1
        pf_of[t] = pf_of.get(t, 0) + n

    for j, ev in enumerate(plan):
        if ev[0] == "P":
            pre.lru, pre.free = lru, list(range(slots - len(lru)))
            pre.prefill_layer(ev[1], ev[2])
            continue
        _, tok, layer, exps, _p = ev
        if prefetch and j not in started:
            issue(j)
        n_dem = 0
        for e in exps:
            k = (layer, e)
            if k in lru:
                lru.move_to_end(k)
            else:
                n_dem += 1
                touch(k)
        rows.append((tok, layer, pf_of.pop(j, 0), n_dem))
        if prefetch and j + 1 < len(plan) and plan[j + 1][0] == "D":
            issue(j + 1)
            started.add(j + 1)
    return rows


def model(rows_pf, rows_none, c: float, layer_ms: float) -> dict:
    toks = {r[0] for r in rows_pf}
    n = max(1, len(toks))
    e_pf = sum(max(0.0, r[2] * c - layer_ms) + r[3] * c for r in rows_pf) / n
    e_d = sum(r[3] * c for r in rows_none) / n
    over = sum(1 for r in rows_pf if r[2] * c > layer_ms) / max(1, len(rows_pf))
    return {"layer_ms": layer_ms, "E_P0L_model": e_pf, "E_D_model": e_d, "H_P0L_model": 1.0 - e_pf / e_d, "overloaded_row_share": over}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--percent", type=float, default=17)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--c", type=float, default=0.194)
    ap.add_argument("--layers-ms", default="0.3,0.6,1.0")
    args = ap.parse_args(argv)
    with np.load(args.trace) as data:
        arrays = {k: data[k] for k in data.files if k != "segments"}
    pf = row_counts(arrays, args.percent, args.tokens, True)
    none = row_counts(arrays, args.percent, args.tokens, False)
    print(json.dumps([model(pf, none, args.c, float(x)) for x in args.layers_ms.split(",")], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
