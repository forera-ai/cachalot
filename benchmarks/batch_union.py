"""
Offline price of batched decode from a recorded routing trace (charter docs/RESEARCH-DIRECTION.md, L3 item 7; G11).

The server is single-flight, so `tokens/s = f(batch size)` cannot be measured live. This replays the decode streams of
one weighted server trace (`CACHALOT_ROUTING_TRACE`; HANDOFF 18.54's 22k-context Hermes run has 13 replies to 13
different user prompts after one shared system block) through the runtime-shaped store of `cache_sim.py`:

  * the first prefill segment (the shared system block) fills the store, as in the live run;
  * the decode streams are dealt round-robin into B lanes; a batched step takes the next token of every lane that
    still has one, and for each layer requests the *union* of the lanes' experts once (a batched MoE layer reads an
    expert once for every row that routes to it);
  * B = 1 is the same procedure with one lane (the single-flight baseline under this replay).

Reported per decoded token (a step of B lanes decodes B tokens): expert requests (unique per layer per step), store
misses, expert bytes read, and the union's share of the B x 6 routed uses (the overlap a batch gets for free).

A token-time column follows `step = floor(B) + miss_ms x misses_step` with two floor models, both hypotheses: "flat"
(the 70 ms floor is paid once per step, every lane free: the optimistic bound, since the trunk GEMV becomes a GEMM over
the same bytes) and "linear" (each extra lane adds `--lane-ms`, default 15 ms: attention, routed-expert compute and
sampling per row; a guess to be replaced by a measured batched step). Neither is a measurement.

Limits. The streams share one system block, so their routing overlaps more than unrelated users' would; per-stream
short prefills between turns are not replayed (in both B = 1 and B > 1); speculative prefetch is not modelled.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/batch_union.py \
        benchmarks/results/trace_routing_v8_hermes.trace.npz --budget-gib 48 --batches 1,2,4,8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache_sim import Q2_EXPERT_BYTES, _prefill_segment  # noqa: E402
from cachalot.metrics.routing_trace import PHASE_PREFILL, load_trace  # noqa: E402
from simulate_policies import GIB, N_LAYERS, Store  # noqa: E402


def split_streams(arrays) -> tuple[tuple[int, int] | None, list[list[list[np.ndarray]]]]:
    """(first prefill segment bounds, decode streams). A stream is a list of tokens; a token is a list of N_LAYERS
    expert rows (layer order)."""
    phase, layer, position, experts = (arrays[k] for k in ("phase", "layer", "position", "experts"))
    change = np.nonzero(np.diff(phase.astype(np.int8)) != 0)[0] + 1
    bounds = [0, *change.tolist(), len(layer)]
    first_prefill = None
    streams = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if phase[a] == PHASE_PREFILL:
            if first_prefill is None:
                first_prefill = (a, b)
            continue
        tokens = []
        seg_pos = position[a:b]
        for pos in np.unique(seg_pos):
            idx = np.nonzero(seg_pos == pos)[0] + a
            rows = [None] * N_LAYERS
            for i in idx:
                rows[int(layer[i])] = experts[i]
            if all(r is not None for r in rows):
                tokens.append(rows)
        if tokens:
            streams.append(tokens)
    return first_prefill, streams


def lanes_of(streams: list, batch: int) -> list[list]:
    """Deal streams round-robin into `batch` lanes; a lane runs its streams back to back."""
    lanes = [[] for _ in range(batch)]
    for i, s in enumerate(streams):
        lanes[i % batch].extend(s)
    return [lane for lane in lanes if lane]


def replay(arrays, slots: int, batch: int, policy: str = "lru") -> dict:
    first_prefill, streams = split_streams(arrays)
    st = Store(slots, policy)
    if first_prefill is not None:
        a, b = first_prefill
        _prefill_segment(st, arrays["layer"][a:b], arrays["experts"][a:b])
    lanes = lanes_of(streams, batch)
    steps = tokens = uses = requests = misses = 0
    width_sum = 0
    for t in range(max(len(lane) for lane in lanes)):
        live = [lane[t] for lane in lanes if t < len(lane)]
        steps += 1
        tokens += len(live)
        width_sum += len(live)
        for la in range(N_LAYERS):
            union = {int(e) for tok in live for e in tok[la]}
            uses += sum(len(tok[la]) for tok in live)
            for e in sorted(union):
                key = (la, e)
                requests += 1
                if not st._resident(key):
                    misses += 1
                st.decode_request(key)
    return {"batch": batch, "streams": len(streams), "steps": steps, "tokens": tokens,
            "mean_width": width_sum / steps, "union_share": requests / uses,
            "requests_per_token": requests / tokens, "misses_per_token": misses / tokens,
            "misses_per_step": misses / steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--budget-gib", type=float, default=48)
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--expert-bytes", type=int, default=Q2_EXPERT_BYTES)
    ap.add_argument("--floor-ms", type=float, default=70.0, help="docs/LEDGER.md DS-FLOOR-48")
    ap.add_argument("--miss-ms", type=float, default=2.0, help="docs/LEDGER.md DS-MISS-48")
    ap.add_argument("--lane-ms", type=float, default=15.0, help="hypothesis: added per extra lane in the 'linear' model")
    args = ap.parse_args()

    arrays, _ = load_trace(args.trace)
    slots = int(args.budget_gib * GIB) // args.expert_bytes
    print(f"trace {args.trace}, budget {args.budget_gib:g} GiB = {slots:,} slots, floor {args.floor_ms:g} ms, "
          f"{args.miss_ms:g} ms a miss, lane {args.lane_ms:g} ms (linear model)")
    print("| B | mean width | union share of uses | requests/token | misses/token | expert GB/token "
          "| ms/token flat | ms/token linear |")
    print("|---|---|---|---|---|---|---|---|")
    for b in (int(x) for x in args.batches.split(",")):
        r = replay(arrays, slots, b)
        w = r["mean_width"]
        step_reads = args.miss_ms * r["misses_per_step"]
        flat = (args.floor_ms + step_reads) / w
        linear = (args.floor_ms + args.lane_ms * (w - 1) + step_reads) / w
        print(f"| {b} | {w:.2f} | {r['union_share']:.1%} | {r['requests_per_token']:.1f} | "
              f"{r['misses_per_token']:.1f} | {r['misses_per_token'] * args.expert_bytes / 1e9:.3f} | "
              f"{flat:.1f} | {linear:.1f} |")


if __name__ == "__main__":
    main()
