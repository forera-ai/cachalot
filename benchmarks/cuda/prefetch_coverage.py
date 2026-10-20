"""
Offline layer-ahead prefetch coverage on a recorded routing trace (CUDA lane N3 design input, Mac only, no GPU).

Question: with an LRU expert arena of a given resident percentage, how many demand misses a decode token would remain if the
recorded next-layer predicted set (`pred_experts`, the runtime's own predictor output) were loaded one layer ahead, how many
loads that costs, and how many of them are wasted. Semantics (the same the N3 record fixes for the GPU harness):

* a decode row (token, layer l) first sees the predicted set recorded for (position, target = l), issued during layer l - 1:
  every predicted key not resident is loaded (inserted as most recently used, evicting from the LRU end) and marked pending;
* the row's real experts then hit (a pending hit is a *covered* miss), or miss and load on demand;
* pending keys of layer l that were not used are wasted loads.

Prefill is replayed as in `replay_arena.walk`. This is a bookkeeping model: it says nothing about time, nor about whether a
copy finishes before it is needed (that is what the GPU stage measures).

Usage:
    ~/venvs/deepseek-v41-mac/bin/python benchmarks/cuda/prefetch_coverage.py --trace benchmarks/results/routing-trace-deepseek-long/trace.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import replay_arena as ra  # noqa: E402


def lead_rows(arrays, lead: int = 1):
    """Indices of the predicted-set records made `lead` layers ahead (target minus source); lead 1 is every record of an older trace."""
    if "pred_source" not in arrays:
        return np.arange(len(arrays["pred_target"]))
    return np.nonzero(arrays["pred_target"].astype(np.int64) - arrays["pred_source"].astype(np.int64) == lead)[0]


def predicted_sets(arrays, lead: int = 1) -> dict[tuple[int, int], list[int]]:
    return {(int(arrays["pred_position"][i]), int(arrays["pred_target"][i])): arrays["pred_experts"][i].tolist() for i in lead_rows(arrays, lead)}


def predicted_weights(arrays, lead: int = 1) -> dict[tuple[int, int], list[float]]:
    return {(int(arrays["pred_position"][i]), int(arrays["pred_target"][i])): arrays["pred_weights"][i].tolist() for i in lead_rows(arrays, lead)}


def coverage(arrays, percent: float, mode: str = "pred", oracle: bool = False, max_decode_tokens: int | None = None,
             min_weight: float | None = None, lead: int = 1) -> dict:
    """`mode` 'none' disables prefetch; `oracle` replaces the recorded predicted set by the layer's real experts (an upper bound).
    `min_weight` gates the recorded prediction: only predicted experts whose recorded router weight is at least this are loaded."""
    base = {k: arrays[k] for k in ("phase", "layer", "position", "experts")}
    pred = predicted_sets(arrays, lead) if mode != "none" and not oracle else {}
    pred_w = predicted_weights(arrays, lead) if min_weight is not None and pred else {}
    slots = ra.slots_for(percent)
    lru: OrderedDict[tuple[int, int], int] = OrderedDict()

    def touch(k) -> bool:
        if k in lru:
            lru.move_to_end(k)
            return True
        while len(lru) >= slots:
            lru.popitem(last=False)
        lru[k] = 1
        return False

    positions = iter(int(base["position"][i]) for i in np.nonzero(base["phase"] == ra.PHASE_DECODE)[0])
    miss = covered = loads = 0
    toks: set[int] = set()
    pending: set[tuple[int, int]] = set()
    pre = ra.ArenaLRU(slots, int(base["layer"].max()) + 1)
    for ev in ra.walk(base):                       # in trace order: a prefill between decode rows changes the arena
        if ev[0] == "P":
            # the store's prefill admission rule (a per-layer quota) lives in `ArenaLRU.prefill_layer`; run it on this arena's state
            pre.lru, pre.free = lru, list(range(slots - len(lru)))
            pre.prefill_layer(ev[1], ev[2])
            continue
        _, tok, layer, exps = ev
        if max_decode_tokens is not None and tok >= max_decode_tokens:
            break
        pos = next(positions)
        toks.add(tok)
        if mode != "none":
            ahead = exps if oracle else pred.get((pos, layer), [])
            if min_weight is not None and not oracle:
                ahead = [e for e, w in zip(ahead, pred_w.get((pos, layer), []), strict=False) if w >= min_weight]
            for e in ahead:
                k = (layer, e)
                if k not in lru:
                    loads += 1
                    touch(k)
                    pending.add(k)
        for e in exps:
            k = (layer, e)
            if k in lru:
                if k in pending:
                    covered += 1
                    pending.discard(k)
                lru.move_to_end(k)
            else:
                miss += 1
                touch(k)
        pending = {k for k in pending if k[0] != layer}
    n = max(1, len(toks))
    return {"percent": percent, "mode": "oracle" if oracle else mode, "min_weight": min_weight, "tokens": len(toks), "demand_misses_per_token": miss / n,
            "covered_per_token": covered / n, "prefetch_loads_per_token": loads / n, "wasted_loads_per_token": (loads - covered) / n,
            "all_loads_per_token": (loads + miss) / n}


def coverage_timed(arrays, percent: float, lead: int = 1, max_decode_tokens: int | None = None) -> dict:
    """Prefetch with the issue point of a `lead`-layers-ahead prediction (N3 stage D): a row's recorded set of lead `lead` is loaded right
    after the demand of the row `lead` rows earlier of the same token (for lead 1: right after the previous row, as `coverage` does,
    including across a token boundary); a row whose set was not issued that way is issued just before its own demand. For lead 1 this
    equals `coverage`; for lead 2 the loads of layer l+2 enter the arena before layer l+1's demand and can evict what it needs."""
    base = {k: arrays[k] for k in ("phase", "layer", "position", "experts")}
    pred = predicted_sets(arrays, lead)
    slots = ra.slots_for(percent)
    lru: OrderedDict[tuple[int, int], int] = OrderedDict()

    def touch(k) -> None:
        if k in lru:
            lru.move_to_end(k)
            return
        while len(lru) >= slots:
            lru.popitem(last=False)
        lru[k] = 1

    pos_iter = iter(int(base["position"][i]) for i in np.nonzero(base["phase"] == ra.PHASE_DECODE)[0])
    plan = []
    for ev in ra.walk(base, max_decode_tokens):
        plan.append(ev if ev[0] == "P" else (*ev, next(pos_iter)))
    miss = covered = loads = 0
    toks: set[int] = set()
    pending: set[tuple[int, int]] = set()
    started: set[int] = set()
    pre = ra.ArenaLRU(slots, int(base["layer"].max()) + 1)

    def issue(t: int) -> None:
        nonlocal loads
        _, _tok, layer, _exps, pos = plan[t]
        for e in pred.get((pos, layer), []):
            k = (layer, e)
            if k not in lru:
                loads += 1
                touch(k)
                pending.add(k)

    for j, ev in enumerate(plan):
        if ev[0] == "P":
            pre.lru, pre.free = lru, list(range(slots - len(lru)))
            pre.prefill_layer(ev[1], ev[2])
            continue
        _, tok, layer, exps, _pos = ev
        toks.add(tok)
        if j not in started:
            issue(j)
        for e in exps:
            k = (layer, e)
            if k in lru:
                if k in pending:
                    covered += 1
                    pending.discard(k)
                lru.move_to_end(k)
            else:
                miss += 1
                touch(k)
        pending = {k for k in pending if k[0] != layer}
        t = j + lead
        if t < len(plan) and (plan[t][0] == "D" if lead == 1 else all(plan[x][0] == "D" and plan[x][1] == tok for x in range(j + 1, t + 1))):
            issue(t)
            started.add(t)
    n = max(1, len(toks))
    return {"percent": percent, "mode": f"timed-lead-{lead}", "tokens": len(toks), "demand_misses_per_token": miss / n,
            "covered_per_token": covered / n, "prefetch_loads_per_token": loads / n, "wasted_loads_per_token": (loads - covered) / n,
            "all_loads_per_token": (loads + miss) / n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--percents", default="8,12,17")
    ap.add_argument("--tokens", type=int, default=None, help="only the first N decode tokens (the N3 cells use 400)")
    ap.add_argument("--lead", type=int, default=1, help="use the predicted sets made this many layers ahead (needs a trace recorded with CACHALOT_PREDICT_AHEAD)")
    ap.add_argument("--timed", action="store_true", help="issue each prediction `lead` rows ahead (stage D semantics) instead of right before its row")
    ap.add_argument("--gates", default="", help="comma list of router-weight thresholds to add as gated prediction rows")
    args = ap.parse_args(argv)
    with np.load(args.trace) as data:
        arrays = {k: data[k] for k in data.files if k != "segments"}
    out = []
    if args.timed:
        for p in (float(x) for x in args.percents.split(",")):
            out += [coverage(arrays, p, "none", max_decode_tokens=args.tokens), coverage_timed(arrays, p, args.lead, args.tokens)]
        print(json.dumps(out, indent=1))
        return 0
    for p in (float(x) for x in args.percents.split(",")):
        out += [coverage(arrays, p, "none", max_decode_tokens=args.tokens), coverage(arrays, p, "pred", max_decode_tokens=args.tokens, lead=args.lead),
                coverage(arrays, p, "pred", oracle=True, max_decode_tokens=args.tokens)]
        out += [coverage(arrays, p, "pred", max_decode_tokens=args.tokens, min_weight=float(g), lead=args.lead) for g in args.gates.split(",") if g]
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
