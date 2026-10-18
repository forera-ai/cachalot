"""
CUDA lane N0a (docs/CUDA-N0A-POLICY-RECORD.md): offline resident-fraction policy table, Mac only, no GPU.

Replays a routing trace through the runtime's `Store` model (simulate_policies.py) the way `cache_sim.simulate` does
(tau 0, no prefetch, no LFU decay) but keeps the misses of every decode token, so that a paired block bootstrap can say
whether a policy gap is larger than the trace's noise. Every cell's aggregate counts are checked against
`cache_sim.simulate`; a cell that differs voids the table.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/cuda_offline_policy_table.py \
        benchmarks/results/routing-trace-deepseek-long/trace.npz --fractions 8,12,17,25,34 --out OUT.json

A trace with fewer than 1,000 decode tokens is marked smoke-grade in the output and supports no conclusion.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cachalot.metrics.routing_trace import (  # noqa: E402
    PHASE_DECODE,
    PHASE_PREFILL,
    _token_ordinals,
    load_trace,
)
from cache_sim import (  # noqa: E402
    MIN_DECODE_TOKENS,
    Q2_EXPERT_BYTES,
    _prefill_segment,
    simulate,
    trace_layers,
)
from simulate_policies import GIB, Store  # noqa: E402
from simulate_policies import run as sp_run  # noqa: E402

TOTAL_EXPERTS = 40 * 384
POLICIES = ("lru", "slru", "lfu")
BLOCK = 20
BOOT = 2000


def replay(arrays, slots: int, policy: str, decay_every: int = 0, store=None, on_token=None) -> dict:
    """Same walk as cache_sim.simulate (tau 0, no prefetch); also returns misses per decode token, in order.

    `store` replaces the default `Store` (an instrumented subclass); `on_token(token, store)` is called after each decode token.
    `decay_every` > 0 halves LFU frequencies every that many decode tokens (simulate_policies.run does 16; cache_sim does not).
    """
    phase, layer, experts = arrays["phase"], arrays["layer"], arrays["experts"]
    token_of_row = np.full(len(layer), -1, dtype=np.int64)
    drows = np.nonzero(phase == PHASE_DECODE)[0]
    token_of_row[drows] = _token_ordinals(layer[drows])
    st = store if store is not None else Store(slots, policy, n_layers=trace_layers(arrays))
    change = np.nonzero(np.diff(phase.astype(np.int8)) != 0)[0] + 1
    bounds = [0, *change.tolist(), len(layer)]
    per_token: dict[int, int] = {}
    tokens = requests = misses = 0
    last_tok = None
    done = 0
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b <= a:
            continue
        if phase[a] == PHASE_PREFILL:
            _prefill_segment(st, layer[a:b], experts[a:b])
            continue
        tokens += int(np.unique(token_of_row[a:b]).size)
        for i in range(a, b):
            tok = int(token_of_row[i])
            if last_tok is not None and tok != last_tok:
                done += 1
                if decay_every and policy == "lfu" and done % decay_every == 0:
                    st.decay()
                if on_token is not None:
                    on_token(last_tok, st)
            last_tok = tok
            la = int(layer[i])
            for e in experts[i].tolist():
                key = (la, int(e))
                requests += 1
                if st._resident(key):
                    st.decode_request(key)
                else:
                    misses += 1
                    per_token[tok] = per_token.get(tok, 0) + 1
                    st.decode_request(key)
                per_token.setdefault(tok, 0)
    series = [per_token[k] for k in sorted(per_token)]
    return {"tokens": tokens, "requests": requests, "misses": misses, "per_token": series}


def block_bootstrap_mean(diff: list[float], seed: int) -> tuple[float, float, float]:
    """(mean, 2.5 %, 97.5 %) of the paired per-token difference, resampling blocks of BLOCK tokens."""
    n = len(diff)
    blocks = [diff[i:i + BLOCK] for i in range(0, n, BLOCK)]
    rng = random.Random(seed)
    means = []
    for _ in range(BOOT):
        pick = [x for blk in rng.choices(blocks, k=len(blocks)) for x in blk]
        means.append(sum(pick) / len(pick))
    means.sort()
    return sum(diff) / n, means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--fractions", default="8,12,17,25,34", help="percent of all experts resident")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--anchor-slots", type=int, default=-1,
                    help="a reference cache size in slots; default: the Mac's shipped 5,179 (48 GiB) for a DeepSeek trace, none otherwise; 0 = none")
    ap.add_argument("--expert-bytes", type=int, default=Q2_EXPERT_BYTES,
                    help="bytes of one expert record (DeepSeek 9,950,000; MiniMax slot image 21.1 MiB = 22,124,953; GLM 14,160,000)")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--out")
    ap.add_argument("--lfu-decay", type=int, default=0, help="also run LFU with a halving every N decode tokens (exploratory arm 'lfu-decay')")
    ap.add_argument("--determinism", action="store_true", help="run every lru/slru cell twice and require identical per-token series")
    ap.add_argument("--no-check", action="store_true", help="skip the cache_sim.simulate consistency check")
    args = ap.parse_args()

    arrays, segments = load_trace(args.trace)
    n_dec = int(np.unique(arrays["position"][arrays["phase"] == PHASE_DECODE]).size)
    smoke = n_dec < MIN_DECODE_TOKENS
    cells, t_all = [], time.time()
    total_experts = int(np.unique(arrays["layer"]).size) * (int(arrays["experts"].max()) + 1)
    anchor = args.anchor_slots
    if anchor < 0:
        anchor = 5179 if total_experts == TOTAL_EXPERTS else 0
    slot_list = [(f"{p}%", round(total_experts * float(p) / 100)) for p in args.fractions.split(",")]
    if anchor:
        slot_list.append(("anchor", anchor))
    policies = args.policies.split(",")
    series: dict[tuple[str, str], list[int]] = {}
    consistent = True
    deterministic = True
    cell_policies = [*policies, *(["lfu-decay"] if args.lfu_decay else [])]
    for label, slots in slot_list:
        for pol in cell_policies:
            t0 = time.time()
            real = "lfu" if pol == "lfu-decay" else pol
            r = replay(arrays, slots, real, args.lfu_decay if pol == "lfu-decay" else 0)
            check = None
            if args.determinism and pol in ("lru", "slru"):
                deterministic &= replay(arrays, slots, real)["per_token"] == r["per_token"]
            if pol == "lfu-decay" and not args.no_check and trace_layers(arrays) != 40:
                check = None        # simulate_policies.run assumes 40 layers (N_LAYERS), so it is no reference here
            elif pol == "lfu-decay" and not args.no_check:
                phase = arrays["phase"]
                chg = np.nonzero(np.diff(phase.astype(np.int8)) != 0)[0] + 1
                segs = [{"at": 0}, *({"at": int(i)} for i in chg)]
                ref = sp_run(arrays, segs, slots, "popularity", "lfu", args.lfu_decay)
                check = ref.decode_misses == r["misses"]
                consistent &= check
            elif not args.no_check:
                ref = simulate(arrays, segments, slots, real, 0.0)
                check = ref["misses"] == r["misses"] and ref["tokens"] == r["tokens"] and ref["requests"] == r["requests"]
                consistent &= check
            series[(label, pol)] = r["per_token"]
            cells.append({
                "fraction": label, "slots": slots, "resident_pct": round(100 * slots / total_experts, 2),
                "resident_gib": round(slots * args.expert_bytes / GIB, 2), "policy": pol, "tokens": r["tokens"],
                "requests": r["requests"], "misses": r["misses"], "misses_per_token": r["misses"] / r["tokens"],
                "hit": 1 - r["misses"] / r["requests"], "consistent_with_cache_sim": check,
                "seconds": round(time.time() - t0, 1),
            })
            print(f"{label:>6} {pol:5} slots {slots:5} misses/token {cells[-1]['misses_per_token']:7.2f} "
                  f"hit {cells[-1]['hit']:.4f} check {check} ({cells[-1]['seconds']} s)", flush=True)

    gaps = []
    if "lru" in policies:
        for label, _ in slot_list:
            lru = series[(label, "lru")]
            for pol in cell_policies:
                if pol == "lru":
                    continue
                diff = [a - b for a, b in zip(lru, series[(label, pol)], strict=True)]   # misses saved by pol
                mean, lo, hi = block_bootstrap_mean(diff, args.seed)
                base = sum(lru) / len(lru)
                gaps.append({"fraction": label, "policy": pol, "misses_saved_per_token": mean, "ci_low": lo,
                             "ci_high": hi, "relative_reduction": mean / base if base else 0.0,
                             "outside_noise": lo > 0 or hi < 0})
    out = {
        "trace": args.trace, "decode_tokens": n_dec, "smoke_grade": smoke, "expert_bytes": args.expert_bytes,
        "total_experts": total_experts, "block": BLOCK, "bootstrap": BOOT, "seed": args.seed,
        "all_cells_consistent_with_cache_sim": consistent if not args.no_check else None,
        "deterministic_repeat": deterministic if args.determinism else None,
        "cells": cells, "gaps_vs_lru": gaps, "seconds_total": round(time.time() - t_all, 1),
        "note": "trace-and-policy statement only: no copy cost, no VRAM overhead, no prefetch is modelled",
    }
    if smoke:
        print(f"\nSMOKE-GRADE: {n_dec} decode tokens < {MIN_DECODE_TOKENS}; supports no conclusion", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    print("\ngaps vs LRU (misses saved a token; positive = better than LRU):")
    for g in gaps:
        print(f"  {g['fraction']:>6} {g['policy']:5} {g['misses_saved_per_token']:+7.3f} "
              f"[{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] rel {g['relative_reduction']:+.2%} "
              f"{'outside noise' if g['outside_noise'] else 'inside noise'}")
    return 0 if (consistent or args.no_check) else 4


if __name__ == "__main__":
    sys.exit(main())
