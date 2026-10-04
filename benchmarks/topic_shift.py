"""
The frozen-cache question of the decode miss budget (HANDOFF 18.61).

With `CACHALOT_DECODE_MISS_BUDGET=0` a decode never admits its own misses, so the resident set after a prefill stays
where the prefill and the prefetch left it. Does a long reply that changes topic pay for that? This teacher-forces one
fixed token stream that shifts topic twice (Python source, then changelog prose, then JSON) after a 256-token prefill of
the first topic, so every arm sees the same text, and reports per 100-token window the log-likelihood change against the
exact arm, the KL, the experts read and dropped, and the step time.

Each arm must run in its own fresh process: a shared one would let the exact arm warm the cache for the later arms.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/topic_shift.py run exact --out DIR
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/topic_shift.py run 0 --out DIR --ref DIR/exact.npz
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/topic_shift.py report DIR 0 1

Each arm takes about 7 minutes. One runtime at a time.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))

TOP_K = 64
PREFILL = 256
# (name, file, character offset, tokens used): the tokens a topic contributes to the stream, in order
SEGMENTS = [
    ("python", "src/cachalot/cache/wired_governor.py", 0, 656),
    ("prose", "CHANGELOG.md", 20000, 600),
    ("json", "benchmarks/pareto_tasks.json", 0, 601),
]
WINDOW = 100


def build_stream(tokenizer) -> tuple[list[int], list[int]]:
    """(token ids, the stream index at which each segment starts)"""
    ids: list[int] = []
    starts: list[int] = []
    for _, path, offset, n in SEGMENTS:
        text = (REPO / path).read_text()[offset:offset + 20000]
        piece = list(tokenizer.encode(text))[:n]
        if len(piece) < n:
            raise ValueError(f"{path} gives {len(piece)} tokens, need {n}")
        starts.append(len(ids))
        ids += piece
    return ids, starts


def run(arm: str, out: Path, ref: Path | None) -> None:
    import mlx.core as mx

    import pareto

    pareto.serve_defaults()
    sys.path.insert(0, str(REPO / "src"))
    from cachalot.model.api import V41Model

    out.mkdir(parents=True, exist_ok=True)
    budget = None if arm == "exact" else int(arm)
    ref_ids = np.load(ref)["ids"] if ref else None
    model = V41Model.from_pretrained(os.environ["CACHALOT_MODEL_PATH"], max_seq_len=4096,
                                     expert_cache_budget_bytes=int(48 * 2**30))
    model.set_decode_miss_budget(budget)
    rt = model.runtime
    ids, starts = build_stream(rt.tokenizer)
    rt.reset()
    res = rt.prefill_tokens(ids[:PREFILL])
    logits = res.logits
    cols: dict[str, list] = {k: [] for k in ("target_logp", "ids", "logp", "top1", "ms", "misses", "skipped")}
    for pos, i in enumerate(range(PREFILL, len(ids) - 1)):
        lp = logits.astype(mx.float32)
        lp = np.array(lp - mx.logsumexp(lp))
        top = ref_ids[pos] if ref_ids is not None else np.argsort(-lp)[:TOP_K]
        cols["target_logp"].append(lp[ids[i]])
        cols["ids"].append(top)
        cols["logp"].append(lp[top])
        cols["top1"].append(int(lp.argmax()))
        m0 = rt.expert_store.stats().cache_misses
        s0 = rt.expert_store.skipped_experts
        t0 = time.perf_counter()
        logits = rt.decode_token(ids[i]).logits
        mx.eval(logits)
        cols["ms"].append(1000 * (time.perf_counter() - t0))
        cols["misses"].append(rt.expert_store.stats().cache_misses - m0)
        cols["skipped"].append(rt.expert_store.skipped_experts - s0)
        if pos % 200 == 0:
            print(f"[{arm}] {pos} / {len(ids) - 1 - PREFILL}", file=sys.stderr, flush=True)
    np.savez_compressed(out / f"{arm}.npz", starts=np.array(starts), **{k: np.array(v) for k, v in cols.items()})
    model.close()


def report(out: Path, arms: list[str]) -> None:
    import pareto

    ref = np.load(out / "exact.npz")
    n = len(ref["ms"])
    starts = [s - PREFILL for s in ref["starts"]]
    names = [s[0] for s in SEGMENTS]

    def segment(pos: int) -> str:
        return names[max(i for i, s in enumerate(starts) if s <= pos)] if pos >= 0 else names[0]

    for arm in arms:
        a = np.load(out / f"{arm}.npz")
        d = ref["target_logp"] - a["target_logp"]
        kl = pareto.kl_top_tail(ref["logp"], a["logp"])
        print(f"\n## miss budget {arm} against exact ({n} teacher-forced decode steps after a {PREFILL}-token prefill)")
        print("| window | text | dNLL | KL mean | top-1 | exact misses | dropped (budget) | read (budget) | step ms exact / budget |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for lo in range(0, n, WINDOW):
            hi = min(lo + WINDOW, n)
            sl = slice(lo, hi)
            print(f"| {lo}-{hi} | {segment(lo)} | {d[sl].mean():+.4f} | {kl[sl].mean():.4f} | "
                  f"{(a['top1'][sl] == ref['top1'][sl]).mean():.0%} | {ref['misses'][sl].mean():.1f} | "
                  f"{a['skipped'][sl].mean():.1f} | {a['misses'][sl].mean():.1f} | "
                  f"{ref['ms'][sl].mean():.0f} / {a['ms'][sl].mean():.0f} |")
        print("\n| segment | positions | dNLL [95 % CI] | KL mean | KL max | step ms exact / budget |")
        print("|---|---|---|---|---|---|")
        for si, name in enumerate(names):
            lo = starts[si] if si else 0
            hi = starts[si + 1] if si + 1 < len(starts) else n
            sl = slice(lo, hi)
            m, c0, c1 = pareto.block_bootstrap_ci(d[sl], np.zeros(hi - lo, dtype=int))
            print(f"| {name} | {hi - lo} | {m:+.4f} [{c0:+.4f}, {c1:+.4f}] | {kl[sl].mean():.4f} | {kl[sl].max():.3f} | "
                  f"{ref['ms'][sl].mean():.0f} / {a['ms'][sl].mean():.0f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("arm", help="'exact' or the integer miss budget")
    r.add_argument("--out", required=True)
    r.add_argument("--ref")
    p = sub.add_parser("report")
    p.add_argument("out")
    p.add_argument("arms", nargs="+")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.arm, Path(args.out), Path(args.ref) if args.ref else None)
    else:
        report(Path(args.out), args.arms)


if __name__ == "__main__":
    main()
