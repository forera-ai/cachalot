"""
Price a router-weight gate on DeepSeek's decode prefetch, offline (0.61.4).

Replays a routing trace that carries predicted sets (recorded since 0.61.3) through the store's
residency rule and prices each gate threshold on a drive of a given bandwidth.

Residency rule (ResidentExpertStore.prefetch_decode, same file's _awaited path): a predicted expert
that is not resident is read into a transient slot, never evicting a resident; it becomes resident
only when the target layer routes it (then it counts as a miss whose read was started early). An
unused predicted load is wasted drive time and leaves no residency. A routed expert that is neither
resident nor predicted is a demand miss.

Cost rule (docs/LEDGER.md ST-BW-CURVE, 0.61.0), per decode token at B GB/s:
    token_ms = max(floor + M x (1.73 + 9.95 x (1/B - 1/6.8)), bytes / B)
with M = misses (demand misses plus predicted loads the layer used), bytes = every expert read,
demand and predicted. The model was fitted on the prediction-on arms; the replay is checked against
the measured arms before a gate is priced (printed first).

Modes: exact (the default path before 0.60.0) and budget0 (CACHALOT_DECODE_MISS_BUDGET=0, shipped
in serve.sh: a demand miss is dropped, not read; a predicted load a layer routes is still awaited).
budget0 changes outputs; the drop count a token is reported as a quality proxy only.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/pred_gate_price.py \
        benchmarks/results/pred-trace-0.61.3/smoke.trace.npz --bandwidths 1,2,6.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cachalot.metrics.routing_trace import PHASE_DECODE, PHASE_PREFILL, load_trace  # noqa: E402
from cache_sim import Q2_EXPERT_BYTES, _prefill_segment  # noqa: E402
from simulate_policies import GIB, Store  # noqa: E402

FLOOR_MS = 78.0  # ST-BW-CURVE's floor term (the curve was fitted with 78, not DS-FLOOR-SRV's 80)
INTERNAL_GBPS = 6.8


def replay(arrays, slots: int, th: float | None, mode: str) -> dict:
    """th None = no prediction at all; th 0.0 = every predicted load (shipped)."""
    phase, layer, experts = arrays["phase"], arrays["layer"], arrays["experts"]
    n = len(layer)
    st = Store(slots, "lru")

    decode_rows = np.nonzero(phase == PHASE_DECODE)[0]
    token_of_row = np.full(n, -1, dtype=np.int64)
    token_of_row[decode_rows] = np.cumsum(layer[decode_rows] == 0) - 1

    preds: dict[tuple[int, int], tuple[list[int], np.ndarray]] = {}
    if th is not None:
        pred_token = np.cumsum(arrays["pred_source"] == 0) - 1
        for i in range(len(pred_token)):
            preds[(int(pred_token[i]), int(arrays["pred_target"][i]))] = (
                arrays["pred_experts"][i].tolist(),
                arrays["pred_weights"][i],
            )

    tokens = demand = used_pred = pred_loads = drops = 0
    change = np.nonzero(np.diff(phase.astype(np.int8)) != 0)[0] + 1
    bounds = [0, *change.tolist(), n]

    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        if phase[a] == PHASE_PREFILL:
            _prefill_segment(st, layer[a:b], experts[a:b])
            continue
        for row in range(a, b):
            la = int(layer[row])
            tok = int(token_of_row[row])
            if la == 0:
                tokens += 1
            inflight: set[int] = set()
            rec = preds.get((tok, la))
            if rec is not None:
                for e, w in zip(*rec):
                    if w >= th and not st._resident((la, int(e))):
                        inflight.add(int(e))
                        pred_loads += 1
            for e in experts[row].tolist():
                key = (la, int(e))
                if st._resident(key):
                    st.decode_request(key)
                elif e in inflight:
                    used_pred += 1
                    st.decode_request(key)
                elif mode == "budget0":
                    drops += 1
                else:
                    demand += 1
                    st.decode_request(key)

    t = max(tokens, 1)
    return {
        "tokens": tokens,
        "misses": (demand + used_pred) / t,
        "demand": demand / t,
        "pred_loads": pred_loads / t,
        "pred_used": used_pred / t,
        "drops": drops / t,
        "reads": (demand + pred_loads) / t,
    }


def token_ms(r: dict, gbps: float, expert_bytes: int) -> float:
    miss_cost = 1.73 + 9.95 * (1 / gbps - 1 / INTERNAL_GBPS)
    mem = FLOOR_MS + r["misses"] * miss_cost
    return max(mem, r["reads"] * expert_bytes / 1e6 / gbps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--budget-gib", type=float, default=48.0)
    ap.add_argument("--bandwidths", default="1,2,6.8")
    ap.add_argument("--thresholds", default="0,0.1,0.15,0.2,0.25,0.3")
    ap.add_argument("--expert-bytes", type=int, default=Q2_EXPERT_BYTES)
    args = ap.parse_args()

    arrays, _ = load_trace(args.trace)
    if "pred_experts" not in arrays:
        raise SystemExit("trace has no predicted sets (recorded since 0.61.3)")
    slots = int(args.budget_gib * GIB) // args.expert_bytes
    bws = [float(x) for x in args.bandwidths.split(",")]
    ths = [float(x) for x in args.thresholds.split(",")]

    for mode in ("exact", "budget0"):
        print(f"\n## mode {mode}, {args.budget_gib:g} GiB = {slots:,} slots, "
              f"{int((arrays['layer'][arrays['phase'] == PHASE_DECODE] == 0).sum())} decode tokens")
        head = "| gate | misses/tok | predicted loads | used | reads/tok | drops/tok |" + "".join(
            f" {b:g} GB/s ms |" for b in bws)
        print(head)
        print("|---|---:|---:|---:|---:|---:|" + "---:|" * len(bws))
        rows = [("off", None)] + [(f">= {t:g}" if t else "all (shipped)", t) for t in ths]
        for name, th in rows:
            r = replay(arrays, slots, th, mode)
            used = f"{r['pred_used'] / r['pred_loads']:.0%}" if r["pred_loads"] else "-"
            line = (f"| {name} | {r['misses']:.1f} | {r['pred_loads']:.1f} | {used} "
                    f"| {r['reads']:.1f} | {r['drops']:.1f} |")
            for b in bws:
                line += f" {token_ms(r, b, args.expert_bytes):.0f} |"
            print(line, flush=True)


if __name__ == "__main__":
    main()
