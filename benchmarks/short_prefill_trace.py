"""
Per-chunk trace of a short prefill after a long context (LEDGER DS-PREFILL-SHORT, HANDOFF 18.79 and 18.88).

Measured so far: a 19-69 token follow-up after a 22-27k context costs 1.8-3.8 s (about 10-18 tokens a second), against
98-106 tok/s for a long prefill; the marginal cost falls from 39 to 7 ms a token as the chunk grows, so a fixed cost of
about 1.5 s a chunk dominates. This instrument splits that fixed cost.

Method. One runtime. For each context length (a small control, then the long one): prefill the context, snapshot, and for
each chunk size restore the snapshot and prefill that many new tokens, repeated REPS times with the SAME tokens. The
first repetition meets whatever experts are not resident (reads); later ones find them resident, so the difference
between repetition 1 and repetition 2 is the read cost, and what repetition 2+ keeps is the fixed cost with no reads.
The control context (512 tokens by default) separates attention over a long cache from per-layer overhead. Wrappers
from `profile_prefill_timeline.py` (imported for their side effect) record, per chunk: expert reads and their busy
time, time inside each layer's MoE phase, time between layers, `mx.eval` count and time, the Engram phase, the
route and prepare phases, and each attention-block function.

Instrumented runs are not speed baselines (charter rule 7): the wrappers add a few microseconds per call (a few
hundred marks a chunk), and every wall reported here is the instrumented one.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    benchmarks/guarded_run.sh --budget-gib 48 --max-seconds 2400 --tag shortprefill -- env \
      CACHALOT_EXPERT_BANK=/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128 CACHALOT_PAGE_CACHE=1 \
      PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/short_prefill_trace.py \
      --contexts 512,22000 --sizes 19,34,69,246 --reps 3 --out benchmarks/results/short-prefill-trace-0.61.12.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_prefill_timeline as ppt  # noqa: E402  (installs the wrappers; fills READS and MARKS)
from _common import MODEL_PATH  # noqa: E402
from cachalot.model.generation import load_official_encoding  # noqa: E402
from cachalot.model.text_decode_runtime import TextDecodeRuntime  # noqa: E402
from trace_routing import build_prompt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def union_busy(reads: list[tuple[float, float]]) -> float:
    if not reads:
        return 0.0
    order = sorted(reads)
    busy, (cs, ce) = 0.0, order[0]
    for s, e in order[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return busy + ce - cs


def summarize(t0: float, wall: float) -> dict:
    marks = list(ppt.MARKS)
    reads = list(ppt.READS)
    starts, ends = {}, {}
    pairs = defaultdict(float)  # label -> summed seconds, from *_start/*_end pairs
    open_at: dict[str, float] = {}
    for t, lab in marks:
        if lab.startswith("moe_start"):
            starts[lab.split()[1]] = t
        elif lab.startswith("moe_end"):
            ends[lab.split()[1]] = t
        for suffix_a, suffix_b in (("block_start", "block_end"), ("engram_start", "engram_end"),
                                   ("prepare_start", "prepare_end"), ("eval_start", "eval_end")):
            if lab.startswith(suffix_a):
                open_at[lab.replace(suffix_a, suffix_b)] = t
            elif lab.startswith(suffix_b) and lab in open_at:
                key = lab.split(" ", 1)[1] if " " in lab else lab.replace("_end", "")
                pairs[f"{suffix_b.split('_')[0]} {key}".strip()] += t - open_at.pop(lab)
    route = 0.0
    last_route = None
    for t, lab in marks:
        if lab == "route_start":
            last_route = t
        elif lab == "route_built" and last_route is not None:
            route += t - last_route
            last_route = None
    layers = sorted(starts, key=lambda k: int(k[1:]))
    moe_sum = sum(ends[k] - starts[k] for k in layers if k in ends)
    between = sum(starts[layers[i + 1]] - ends[layers[i]] for i in range(len(layers) - 1)
                  if layers[i] in ends) if len(layers) > 1 else 0.0
    n_eval = sum(1 for _, lab in marks if lab == "eval_start")
    return {
        "wall_s": round(wall, 3),
        "reads": len(reads),
        "read_busy_s": round(union_busy([(a, b) for a, b in reads]), 3),
        "read_sum_s": round(sum(b - a for a, b in reads), 3),
        "moe_layers": len(layers),
        "moe_sum_s": round(moe_sum, 3),
        "between_layers_s": round(between, 3),
        "route_s": round(route, 3),
        "evals": n_eval,
        "phase_s": {k: round(v, 3) for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])},
    }


def long_ids(rt, enc, n_tokens: int, offset_chars: int) -> list[int]:
    text = (REPO / "docs" / "HANDOFF.md").read_text()
    chunk = text[offset_chars: offset_chars + max(n_tokens * 6, 4000)]
    return build_prompt(rt, enc, chunk, n_tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="512,22000")
    ap.add_argument("--sizes", default="19,34,69,246")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    contexts = [int(x) for x in args.contexts.split(",")]
    sizes = [int(x) for x in args.sizes.split(",")]
    rows = []
    with TextDecodeRuntime(MODEL_PATH, max_seq_len=args.max_seq_len) as rt:
        print(f"runtime ready (expert budget {rt.expert_cache_budget_bytes / 2**30:.1f} GiB)", flush=True)
        enc = load_official_encoding(MODEL_PATH)
        # a suffix text disjoint from every context text (cold-text rule: no shared prefix, no shared content)
        suffix_all = list(rt.tokenizer.encode((REPO / "CHANGELOG.md").read_text()[:6000]))
        assert len(suffix_all) >= max(sizes), len(suffix_all)
        for ci, ctx in enumerate(contexts):
            ids = long_ids(rt, enc, ctx, offset_chars=200_000 + ci * 400_000)
            rt.reset()
            t0 = perf_counter()
            res = rt.prefill_tokens(ids)
            mx.eval(res.logits)
            print(f"context {len(ids)} tokens prefilled in {perf_counter() - t0:.1f} s", flush=True)
            snap = rt.snapshot(res.logits)
            for size in sizes:
                suffix = suffix_all[:size]
                for rep in range(args.reps):
                    rt.restore(snap)
                    ppt.READS.clear()
                    ppt.MARKS.clear()
                    st = rt.expert_store.stats()
                    miss0, hit0 = st.cache_misses, st.cache_hits
                    t0 = perf_counter()
                    out = rt.prefill_tokens(suffix)
                    mx.eval(out.logits)
                    wall = perf_counter() - t0
                    st = rt.expert_store.stats()
                    row = {"context": len(ids), "chunk": size, "rep": rep + 1, **summarize(t0, wall),
                           "misses": st.cache_misses - miss0, "hits": st.cache_hits - hit0}
                    rows.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != "phase_s"}), flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
        print("wrote", out)


if __name__ == "__main__":
    main()
