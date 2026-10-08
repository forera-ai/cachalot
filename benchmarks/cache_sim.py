"""
Trace-driven decode cost model for DeepSeek: from a routing trace to misses a
token, milliseconds a token and tokens a second, per memory budget and per
miss-drop threshold, before any weights move.

It joins three things the repo already has but kept apart:

  * the runtime-shaped store of `simulate_policies.py` (per-layer prefill
    quotas, then a decode-time eviction policy), so hit rates match the
    runtime's own paths;
  * the cost model of docs/SPEED-RESEARCH-2026-10-03.md section 1.1,
    `token = floor + misses x cost per miss`, with the cost per miss of each
    drive as a parameter. `--context-tokens N` adds DS-FLOOR-CTX's context term. Defaults cite docs/LEDGER.md: floor 80 ms
    (DS-FLOOR-SRV, the server's all-resident token measured in 0.60.10; the
    70 ms of DS-FLOOR-48 was a fit intercept) and 2.0 ms a miss on the
    internal bank (DS-MISS-48, reproduced as 1.97 against that floor); the USB figure (10.2 ms) is still an estimate
    from the 1.0 GB/s wall (ST-X10);
  * the miss-drop rule of the MiniMax substitution (a missing expert whose
    router share of its layer is under tau is not read). It needs router
    weights, which traces recorded by 0.52.4 and later carry; older traces
    run with tau = 0 only.

What it can and cannot say. The hit rate of a trace is only as good as the
trace: the September traces in benchmarks/results hold 160 decode tokens from
four 512-token prompts, so their decode hit rates are noisy and say nothing
about a 25k-token Hermes session. The simulator prints the decode-token count
and warns below 1,000. The dropped router mass (mean share of a layer's
routing weight that the dropped experts held) is a quality proxy only: any
tau above 0 changes outputs and goes through the quality harness and Hamed
before it becomes a default. The substitute expert (best resident of the next
ranks) is not simulated, because a trace holds only the top six; the drop-only
arm is therefore a pessimistic bound on quality and an optimistic one on
nothing else.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/cache_sim.py \
        benchmarks/results/trace_routing_v7.trace.npz \
        --budgets-gib 44,52,60 --taus 0,0.06,0.10
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cachalot.metrics.routing_trace import PHASE_PREFILL, load_trace  # noqa: E402
from simulate_policies import GIB, Store  # noqa: E402

# The 2-bit affine g128 bank record (docs/SPEED-RESEARCH-2026-10-03.md section 1.1).
Q2_EXPERT_BYTES = 9_950_000
MIN_DECODE_TOKENS = 1_000
# docs/LEDGER.md DS-FLOOR-CTX (0.61.5, one clean arm): over the short-prompt floor the server token adds
# +4.0 ms by 1.8k tokens of context, then 0.09 ms per 1k (fit 82.7 + 0.09 x N/1k; 84.8 measured at 25k).
CTX_STEP_MS = 4.0
CTX_STEP_TOKENS = 1_800
CTX_SLOPE_MS_PER_1K = 0.09


def context_floor(floor_ms: float, context_tokens: int) -> float:
    """All-resident token at a context length. Below 1.8k tokens the +4 ms step is interpolated (not measured)."""
    if context_tokens <= 0:
        return floor_ms
    step = CTX_STEP_MS * min(context_tokens, CTX_STEP_TOKENS) / CTX_STEP_TOKENS
    return floor_ms + step + CTX_SLOPE_MS_PER_1K * max(context_tokens - CTX_STEP_TOKENS, 0) / 1000


def trace_layers(arrays) -> int:
    """MoE layers the trace routes through (DeepSeek 40, MiniMax 57, GLM 39): the prefill quota and the token count follow it."""
    return int(np.unique(arrays["layer"]).size)


def _prefill_segment(st: Store, layer: np.ndarray, experts: np.ndarray) -> None:
    for la in np.unique(layer).tolist():
        rows = experts[layer == la]
        if rows.size == 0:
            continue
        flat = rows.ravel().tolist()
        count = Counter(flat)
        first: dict[int, int] = {}
        for i, e in enumerate(flat):
            first.setdefault(e, i)
        order = sorted(count, key=lambda e: (-count[e], first[e]))
        st.prefill_layer(la, [(la, e) for e in order])


def simulate(arrays, segments, slots: int, policy: str, tau: float) -> dict:
    """Replay the trace; return decode counters. tau > 0 drops cheap misses."""
    phase, layer, position, experts = (arrays[k] for k in ("phase", "layer", "position", "experts"))
    weights = arrays.get("weights")
    if tau > 0 and weights is None:
        raise ValueError("tau > 0 needs a trace recorded with router weights")

    st = Store(slots, policy, n_layers=trace_layers(arrays))
    if not segments:
        # a server trace carries no marks (nobody calls tracer.mark): split at every prefill/decode change
        change = np.nonzero(np.diff(phase.astype(np.int8)) != 0)[0] + 1
        segments = [{"at": 0}] + [{"at": int(i)} for i in change]
    bounds = [s["at"] for s in segments] + [len(layer)]
    tokens = requests = misses = drops = 0
    dropped_mass = 0.0
    rows_seen = 0

    for si in range(len(segments)):
        a, b = bounds[si], bounds[si + 1]
        if b <= a:
            continue
        if phase[a] == PHASE_PREFILL:
            _prefill_segment(st, layer[a:b], experts[a:b])
            continue

        seg_pos = position[a:b]
        for pos in np.unique(seg_pos):
            m = seg_pos == pos
            for row_i in np.nonzero(m)[0] + a:
                la = int(layer[row_i])
                row = experts[row_i].tolist()
                share = None
                if weights is not None:
                    w = weights[row_i].astype(np.float64)
                    share = w / max(float(w.sum()), 1e-12)
                row_drop_mass = 0.0
                for k, e in enumerate(row):
                    key = (la, int(e))
                    requests += 1
                    if st._resident(key):
                        st.decode_request(key)
                    elif tau > 0 and share[k] < tau:
                        drops += 1
                        row_drop_mass += float(share[k])
                    else:
                        misses += 1
                        st.decode_request(key)
                dropped_mass += row_drop_mass
                rows_seen += 1
            tokens += 1

    if tokens == 0:
        raise ValueError("the trace holds no decode tokens")
    return {
        "tokens": tokens,
        "requests": requests,
        "misses": misses,
        "drops": drops,
        "hit": 1 - (misses + drops) / max(requests, 1),
        "misses_per_token": misses / max(tokens, 1),
        "mean_dropped_mass_per_layer": dropped_mass / max(rows_seen, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--budgets-gib", default="44,52,60")
    ap.add_argument("--taus", default="0", help="comma list; > 0 needs a weighted trace")
    ap.add_argument("--policy", default="lru", choices=("lru", "slru", "lfu"))
    ap.add_argument("--expert-bytes", type=int, default=Q2_EXPERT_BYTES)
    ap.add_argument("--floor-ms", type=float, default=80.0, help="all-resident token time (docs/LEDGER.md DS-FLOOR-SRV)")
    ap.add_argument("--context-tokens", type=int, default=0,
                    help="decode context length; adds LEDGER DS-FLOOR-CTX's context term to --floor-ms (0: short prompt)")
    ap.add_argument("--miss-ms", default="internal=2.0,usb=10.2",
                    help="name=ms per miss, comma list (LEDGER DS-MISS-48; usb is an estimate, ST-X10)")
    args = ap.parse_args()

    arrays, segments = load_trace(args.trace)
    floor_ms = context_floor(args.floor_ms, args.context_tokens)
    costs = {n: float(v) for n, v in (kv.split("=") for kv in args.miss_ms.split(","))}
    taus = [float(x) for x in args.taus.split(",")]
    if "weights" not in arrays:
        taus = [t for t in taus if t == 0] or [0.0]
        print("note: trace has no router weights; only tau = 0 is simulated\n")

    decode_tokens = int((arrays["phase"] == 1).sum() // trace_layers(arrays))
    print(f"trace {args.trace}: {decode_tokens} decode tokens, policy {args.policy}, "
          f"expert {args.expert_bytes / 1e6:.2f} MB, floor {floor_ms:.1f} ms"
          + (f" (short {args.floor_ms:g} + context {args.context_tokens:,} tokens)" if args.context_tokens else ""))
    if decode_tokens < MIN_DECODE_TOKENS:
        print(f"WARNING: under {MIN_DECODE_TOKENS} decode tokens; hit rates are noisy and "
              f"say nothing about a long session. Record a real trace (D0) before trusting them.\n")

    head = "| budget GiB | slots | tau | decode hit | misses/token | dropped mass/layer |"
    sep = "|---:|---:|---:|---:|---:|---:|"
    for n in costs:
        head += f" {n} ms | {n} tok/s |"
        sep += "---:|---:|"
    print(head)
    print(sep)
    for b in (float(x) for x in args.budgets_gib.split(",")):
        slots = int(b * GIB) // args.expert_bytes
        for tau in taus:
            r = simulate(arrays, segments, slots, args.policy, tau)
            line = (f"| {b:g} | {slots:,} | {tau:g} | {r['hit']:.1%} | {r['misses_per_token']:.1f} "
                    f"| {r['mean_dropped_mass_per_layer']:.3f} |")
            for c in costs.values():
                ms = floor_ms + r["misses_per_token"] * c
                line += f" {ms:.0f} | {1000 / ms:.1f} |"
            print(line, flush=True)


if __name__ == "__main__":
    main()
