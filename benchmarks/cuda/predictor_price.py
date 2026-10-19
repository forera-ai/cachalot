"""
Offline price of expert predictors against the runtime's recorded one (CUDA lane, Mac only, no GPU; record
`docs/CUDA-PREDICTOR-PRICE-RECORD.md`).

Replays the recorded routing trace through the stage A/B bookkeeping (LRU arena, prefetched keys inserted as most recently
used, covered and wasted counts) with a predictor under test and reports, on a held-out window of decode tokens, demand misses,
prefetch loads, copies and the serial bound `1 - misses / misses_without_prefetch`. Predictors: `R` (recorded), `CO6`/`CO10`
(co-activation table trained on earlier tokens, scored from the previous layer's real experts), `TP6` (the previous token's
experts at the layer), `MIX6`, `RCO10`. This is a bookkeeping model: it says nothing about time.

Usage:
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/predictor_price.py --trace benchmarks/results/routing-trace-deepseek-long/trace.npz
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

N_EXPERTS = 384
PREDICTORS = ("none", "R", "CO6", "CO10", "TP6", "MIX6", "RCO10")
LEAD_PREDICTORS = ("none", "R", "R2", "R3", "R12")                # need a trace recorded with CACHALOT_PREDICT_AHEAD >= 2
LEADS = (1, 2, 3)


def prepare(arrays, train_tokens: int):
    """Decode rows as dicts keyed (token, layer): real experts, recorded predicted set and weights; plus the co-activation table."""
    base = {k: arrays[k] for k in ("phase", "layer", "position", "experts")}
    items = list(ra.walk(base))
    positions = [int(base["position"][i]) for i in np.nonzero(base["phase"] == ra.PHASE_DECODE)[0]]
    lead_of = arrays["pred_target"].astype(np.int64) - arrays["pred_source"].astype(np.int64)
    rec_by_lead = {}
    for lead in LEADS:
        sel = np.nonzero(lead_of == lead)[0]
        rec_by_lead[lead] = {(int(arrays["pred_position"][i]), int(arrays["pred_target"][i])): (arrays["pred_experts"][i].tolist(), arrays["pred_weights"][i].tolist())
                             for i in sel}
    real, recorded, di = {}, {lead: {} for lead in LEADS}, 0
    for it in items:
        if it[0] != "D":
            continue
        _, tok, layer, exps = it
        real[(tok, layer)] = [int(e) for e in exps]
        for lead in LEADS:
            recorded[lead][(tok, layer)] = rec_by_lead[lead].get((positions[di], layer))
        di += 1
    n_layers = int(base["layer"].max()) + 1
    table = np.zeros((n_layers, N_EXPERTS, N_EXPERTS), np.float32)
    for (tok, layer), exps in real.items():
        if tok < train_tokens and layer >= 1 and (tok, layer - 1) in real:
            for p in real[(tok, layer - 1)]:
                table[layer, p, exps] += 1.0
    return items, real, recorded, table, n_layers


def co_rank(table, real, tok, layer):
    prev = real.get((tok, layer - 1))
    if not prev or layer < 1:
        return []
    score = table[layer, prev].sum(0)
    order = np.lexsort((np.arange(N_EXPERTS), -score))
    return [int(e) for e in order if score[e] > 0]


def predicted(pred: str, table, real, recorded, tok, layer) -> list[int]:
    rec = recorded[1].get((tok, layer))
    if pred in ("R2", "R3"):
        r = recorded[int(pred[1])].get((tok, layer))
        return list(r[0]) if r else []
    if pred == "R12":
        out = list(rec[0]) if rec else []
        r2 = recorded[2].get((tok, layer))
        out += [e for e in (r2[0] if r2 else []) if e not in out]
        return out
    if pred == "R":
        return list(rec[0]) if rec else []
    if pred in ("CO6", "CO10"):
        return co_rank(table, real, tok, layer)[: int(pred[2:])]
    if pred == "TP6":
        return list(real.get((tok - 1, layer), []))
    if pred in ("MIX6", "RCO10"):
        if not rec:
            return []
        if pred == "MIX6":
            ranked = [e for _, e in sorted(zip(rec[1], rec[0], strict=True), reverse=True)]
            out, size = ranked[:4], 6
        else:
            out, size = list(rec[0]), 10
        for e in co_rank(table, real, tok, layer):
            if len(out) >= size:
                break
            if e not in out:
                out.append(e)
        return out
    return []


def replay(items, real, recorded, table, n_layers: int, percent: float, pred: str, lo: int, hi: int) -> dict:
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

    pre = ra.ArenaLRU(slots, n_layers)
    miss = covered = loads = 0
    toks: set[int] = set()
    pending: set[tuple[int, int]] = set()
    for it in items:
        if it[0] == "P":
            pre.lru, pre.free = lru, list(range(slots - len(lru)))
            pre.prefill_layer(it[1], it[2])
            continue
        _, tok, layer, exps = it
        if tok >= hi:
            break
        counted = tok >= lo
        if counted:
            toks.add(tok)
        if pred != "none":
            for e in predicted(pred, table, real, recorded, tok, layer):
                k = (layer, e)
                if k not in lru:
                    touch(k)
                    pending.add(k)
                    loads += counted
        for e in exps:
            k = (layer, int(e))
            if k in lru:
                if k in pending:
                    covered += counted
                    pending.discard(k)
                lru.move_to_end(k)
            else:
                miss += counted
                touch(k)
        pending = {k for k in pending if k[0] != layer}
    n = max(1, len(toks))
    return {"percent": percent, "predictor": pred, "tokens": len(toks), "demand_misses": miss / n, "covered": covered / n, "loads": loads / n,
            "wasted": (loads - covered) / n, "copies": (miss + loads) / n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--percents", default="17,8")
    ap.add_argument("--train-tokens", type=int, default=1000)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--predictors", default=None, help="default: the expert-id predictors, or R,R2,R3,R12 when the trace has leads above 1")
    args = ap.parse_args(argv)
    with np.load(args.trace) as data:
        arrays = {k: data[k] for k in data.files if k != "segments"}
    has_leads = bool(np.any((arrays["pred_target"].astype(np.int64) - arrays["pred_source"].astype(np.int64)) > 1))
    names = args.predictors or ",".join(LEAD_PREDICTORS if has_leads else PREDICTORS)
    items, real, recorded, table, nl = prepare(arrays, args.train_tokens)
    lo, hi = args.train_tokens, args.train_tokens + args.window
    out = []
    for pct in (float(x) for x in args.percents.split(",")):
        rows = [replay(items, real, recorded, table, nl, pct, p, lo, hi) for p in names.split(",")]
        base = next(r for r in rows if r["predictor"] == "none")["demand_misses"] if any(r["predictor"] == "none" for r in rows) else None
        for r in rows:
            r["bound"] = None if base is None else 1.0 - r["demand_misses"] / base
        out += rows
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
