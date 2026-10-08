"""
Quality of the GLM decode miss budget (HANDOFF 18.106): teacher-forced paired NLL and KL against exact decode.

For each text a context is prefilled once (exact, prefill is never budgeted) and snapshotted; each arm restores the
snapshot, sets `CACHALOT_GLM_DECODE_MISS_BUDGET`-style `set_decode_miss_budget(N)` and feeds the same forced tokens one at
a time through the decode path, saving every position's log-probs. Arms:

    exact       decode, no budget (the reference)
    exact2      the same again: a determinism / cache-state control, should equal exact exactly
    b<N>        decode with a budget of N non-resident experts a layer
    prefill     the same forced tokens in one prefill chunk: GLM's own prefill-against-decode spread (HANDOFF 18.43 saw
                mean KL 0.015), the scale a budget's cost is read against

Reported per arm and text: mean NLL, paired dNLL against exact with a bootstrap 95 % interval, KL(exact || arm)
mean / p99 / max, top-1 agreement, misses and dropped experts per token, and ms per token.

The machine is Hamed's working machine: the expert budget is chosen from what macOS says is available, the model's own
memory fit runs every few tokens, and arm order rotates by text so his load and the cache state do not always favour one
arm. Nothing is closed or moved.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/glm_miss_budget_quality.py --out benchmarks/results/glm-miss-budget-quality
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CACHALOT_PAGE_CACHE", "1")
os.environ.setdefault("CACHALOT_MLX_WIRED_LIMIT_GIB", "80")
os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
os.environ.setdefault("CACHALOT_GLM_PREDICT_TOPK", "0")  # serve-glm.sh's default since 0.62.14
BANK = "/Volumes/X10Pro/models/GLM-5.3-Flash-bank"
if "CACHALOT_GLM_BANK" not in os.environ and os.path.exists(f"{BANK}/bank.json"):
    os.environ["CACHALOT_GLM_BANK"] = BANK

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as rm  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("CACHALOT_MODEL_PATH", "/Volumes/X10Pro/models/GLM-5.3-Flash-MLX-4bit-MTP")
TEXTS = [  # (class, file, character offset)
    ("W1:prose", "README.md", 20000),
    ("W4:python", "src/cachalot/glm/experts.py", 0),
    ("W6:json", "benchmarks/pareto_tasks.json", 0),
]


def pick_budget_gib() -> float:
    """What the machine can give while Hamed works: his applications' memory is already out of 'available'."""
    from cachalot.glm.model import host_available

    avail = host_available() / 2**30
    for gib, need in ((46.0, 70.0), (44.0, 62.0), (40.0, 54.0)):
        if avail >= need:
            return gib
    return 36.0


def bootstrap_ci(d: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="16,8,4,2")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=100, help="forced tokens scored per text")
    ap.add_argument("--texts", default="0,1,2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from cachalot.glm.model import GlmModel

    budget_gib = pick_budget_gib()
    print(f"expert budget {budget_gib} GiB (chosen from host availability)", flush=True)
    man = rm.collect(workload="W4:glm-miss-budget-quality", budget_gib=budget_gib, instrument="glm_miss_budget_quality",
                     workload_text=json.dumps(TEXTS))
    mid = rm.write_manifest(out, man)
    t_start = time.perf_counter()
    m = GlmModel(MODEL, expert_budget_gib=budget_gib, heartbeat_seconds=0)
    store = m.store
    budgets = [int(b) for b in args.budgets.split(",")]
    results = []
    selected = [int(i) for i in args.texts.split(",")]
    for ti, idx in enumerate(selected):
        klass, rel, offset = TEXTS[idx]
        ids = list(m.tokenizer.encode((REPO / rel).read_text()[offset:], add_special_tokens=False))
        need = args.ctx + args.tokens
        if len(ids) < need:
            sys.exit(f"{rel} gives {len(ids)} tokens, need {need}")
        ctx, forced = ids[:args.ctx], ids[args.ctx:args.ctx + args.tokens]
        arms = ["exact"] + [f"b{b}" for b in budgets] + ["exact2", "prefill"]
        rot = ti % max(1, len(budgets))  # rotate the budget arms between texts
        bud_arms = arms[1:1 + len(budgets)]
        arms = ["exact"] + bud_arms[rot:] + bud_arms[:rot] + ["exact2", "prefill"]
        print(f"\n== text {idx} {klass} {rel}: ctx {args.ctx}, scoring {args.tokens - 1} tokens, arms {arms}", flush=True)

        m.set_decode_miss_budget(None)
        m._fit_prefill(len(ctx))
        cache = m.new_cache()
        logits = m.prefill(ctx, cache)
        mx.eval(logits)
        snap = m.snapshot(ctx, cache, None)
        first_lp = np.array((logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True))[0])
        del cache

        ref = None
        for arm in arms:
            t0 = time.perf_counter()
            cache = m.restore(snap)
            lps, miss, drop = [], [], []
            if arm == "prefill":
                m.set_decode_miss_budget(None)
                o = m.model(mx.array(forced[:-1], dtype=mx.int32)[None], cache=cache)
                full = getattr(o, "logits", o)[0].astype(mx.float32)
                lp = full - mx.logsumexp(full, axis=-1, keepdims=True)
                mx.eval(lp)
                lps = [np.array(lp[i]).astype(np.float16) for i in range(lp.shape[0])]
                store.release_prefill()
            else:
                m.set_decode_miss_budget(int(arm[1:]) if arm.startswith("b") else None)
                for i in range(args.tokens - 1):
                    s0, d0 = store.stats().cache_misses, store.skipped_experts
                    step = m._forward([forced[i]], cache).astype(mx.float32)
                    lp = step - mx.logsumexp(step, axis=-1, keepdims=True)
                    mx.eval(lp)
                    lps.append(np.array(lp[0]).astype(np.float16))
                    miss.append(store.stats().cache_misses - s0)
                    drop.append(store.skipped_experts - d0)
                    if i % 16 == 15:
                        m._fit_memory()  # the model's own memory governor, as in serving
                m.set_decode_miss_budget(None)
            dt = time.perf_counter() - t0
            lp_arr = np.stack(lps).astype(np.float32)
            tgt = np.array(forced[1:len(lps) + 1])
            nll = -lp_arr[np.arange(len(lps)), tgt]
            row = {"text": idx, "class": klass, "arm": arm, "nll_mean": float(nll.mean()), "seconds": dt,
                   "ms_token": 1000 * dt / len(lps), "misses_token": float(np.mean(miss)) if miss else None,
                   "dropped_token": float(np.mean(drop)) if drop else None}
            if ref is None:
                ref = (lp_arr, nll)
            else:
                d = nll - ref[1]
                lo, hi = bootstrap_ci(d)
                p = np.exp(ref[0].astype(np.float64))
                kl = (p * (ref[0].astype(np.float64) - lp_arr.astype(np.float64))).sum(axis=-1)
                row.update(dnll=float(d.mean()), dnll_lo=lo, dnll_hi=hi, kl_mean=float(kl.mean()),
                           kl_p99=float(np.quantile(kl, 0.99)), kl_max=float(kl.max()),
                           top1_agree=float((lp_arr.argmax(-1) == ref[0].argmax(-1)).mean()),
                           worse_tokens=float((d > 0.1).mean()))
            results.append(row)
            np.save(out / f"nll-t{idx}-{arm}.npy", nll)
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            fields = {"ms_token": (row["ms_token"], "measured"), "tokens": (float(len(lps)), "measured")}
            if miss:
                fields["misses_per_token"] = (row["misses_token"], "measured")
                fields["dropped_per_token"] = (row["dropped_token"], "measured")
            rm.append_rows(out / "scorecard.jsonl", [rm.row(mid, f"t{idx}:{arm}", fields,
                                                            note="NLL/KL are in results.jsonl (not scorecard fields)")])
            elapsed = (time.perf_counter() - t_start) / 60
            print(f"[{elapsed:5.1f} min] t{idx} {arm:8s} nll {row['nll_mean']:.4f}"
                  + (f" dNLL {row['dnll']:+.4f} [{row['dnll_lo']:+.4f}, {row['dnll_hi']:+.4f}] KL {row['kl_mean']:.4f} "
                     f"p99 {row['kl_p99']:.3f} top1 {row['top1_agree']:.3f}" if "dnll" in row else "")
                  + (f" miss/tok {row['misses_token']:.1f} drop/tok {row['dropped_token']:.1f}" if miss else "")
                  + f" {row['ms_token']:.0f} ms/tok", flush=True)
            del cache
        del snap
    m.close()
    print("\nSUMMARY (pooled over texts, tokens weighted equally)")
    for arm in dict.fromkeys(r["arm"] for r in results):
        rs = [r for r in results if r["arm"] == arm]
        print(f"  {arm:8s} nll {np.mean([r['nll_mean'] for r in rs]):.4f}"
              + (f" dNLL {np.mean([r['dnll'] for r in rs]):+.4f} KL {np.mean([r['kl_mean'] for r in rs]):.4f}"
                 f" top1 {np.mean([r['top1_agree'] for r in rs]):.3f}" if "dnll" in rs[0] else "")
              + (f" drop/tok {np.mean([r['dropped_token'] for r in rs]):.1f}" if rs[0].get("dropped_token") is not None else ""))


if __name__ == "__main__":
    main()
