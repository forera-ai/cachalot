"""
In-situ price of the sampled tracer default (CACHALOT_TOKEN_TRACE_EVERY=4): docs/L2-IN-SITU-PRICE-RECORD.md addendum, HANDOFF 18.143.

Through `Engine.chat` (the code `./serve.sh` runs). Each prompt is run twice back to back, once with a tracer recording one decode
token in four ('e4') and once with a tracer that only stamps token boundaries ('off'), the order alternating by prompt. A token's
wall time is the gap between consecutive boundary stamps (written in every state). Position-matched difference in differences: for
each prompt, d_k = wall_e4(k) - wall_off(k) at the positions both runs share (greedy text is identical, so token k is the same
token); T = mean(d at sampled positions, k % 4 == 0) - mean(d at the other positions). Pass-level drift and order effects are
common to all positions and cancel. T is the cost of a sampled token; the average over all tokens is T / 4. Control pairs run
'ctl' (a second stamp-only tracer) against 'off' with the same labels and estimate the noise floor of T.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    benchmarks/guarded_run.sh --budget-gib 34 --max-seconds 3600 --tag e4price -- \
        env PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/token_trace_price_sampled.py --budget-gib 34 --out benchmarks/results/l2-price/run-e4.json
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/token_trace_price_sampled.py --report benchmarks/results/l2-price/run-e4.json

One runtime at a time (it loads the model).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))
from token_trace_price import MIXED, SYSTEM  # noqa: E402

EVERY = 4


def run(out: Path, tokens: int, pairs: int, ctl_pairs: int, warm: int, budget_gib: float) -> None:
    import pareto

    pareto.serve_defaults()
    from cachalot.metrics import token_trace as tt
    from cachalot.model.api import V41Model
    from cachalot.model.generation import SamplingParams
    from cachalot.server.engine import ChatRequest, Engine

    model = V41Model.from_pretrained(os.environ["CACHALOT_MODEL_PATH"], max_seq_len=16384,
                                     expert_cache_budget_bytes=int(budget_gib * 2**30))
    engine = Engine(model)
    rt = model.runtime
    traces = {"e4": tt.TokenTrace(every=EVERY), "off": tt.TokenTrace(), "ctl": tt.TokenTrace()}
    traces["off"].alternate = -1
    traces["ctl"].alternate = -1
    rows: list[dict] = []

    def turn(cond: str, prompt: str, idx: int, role: str) -> None:
        trace = traces[cond]
        rt.set_token_trace(trace)
        trace.drain()
        req = ChatRequest(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                          params=SamplingParams(max_new_tokens=tokens, temperature=0.0))
        m0 = rt.expert_store.cache_misses
        res = engine.chat(req)
        n = max(res.completion_tokens, 1)
        d = trace.drain()
        row = {"cond": cond, "role": role, "pair": idx, "tokens": res.completion_tokens, "ms_tok": round(1000 * res.decode_seconds / n, 2),
               "misses_tok": round((rt.expert_store.cache_misses - m0) / n, 1), "text": res.raw_text,
               "stamps": d["token_stamps"].tolist(), "positions": d["token_positions"].tolist(), "dropped": int(d["dropped"])}
        rows.append(row)
        print({k: v for k, v in row.items() if k not in ("text", "stamps", "positions")}, flush=True)
        out.write_text(json.dumps({"every": EVERY, "rows": rows}))

    for i in range(warm):
        turn("off", MIXED[i % len(MIXED)], -1 - i, "warm")
    for i in range(pairs):
        order = ("e4", "off") if i % 2 == 0 else ("off", "e4")
        for c in order:
            turn(c, MIXED[i % len(MIXED)], i, "pair")
    for i in range(ctl_pairs):
        order = ("ctl", "off") if i % 2 == 0 else ("off", "ctl")
        for c in order:
            turn(c, MIXED[i % len(MIXED)], 1000 + i, "null")
    rt.set_token_trace(None)
    model.close()


def _walls(row: dict) -> dict[int, float]:
    st, pos = row["stamps"], row["positions"]
    return {pos[k]: (st[k + 1] - st[k]) / 1e6 for k in range(len(st) - 1) if pos[k + 1] == pos[k] + 1}


def matched_T(a: dict[int, float], b: dict[int, float], every: int = EVERY, min_positions: int = 20) -> float | None:
    """Cost of a sampled token: mean(a - b at positions k % every == 0) - mean(a - b at the other positions)."""
    ks = sorted(set(a) & set(b))
    if len(ks) < min_positions:
        return None
    d = np.array([a[k] - b[k] for k in ks])
    s = np.array([k % every == 0 for k in ks])
    if s.sum() < 3 or (~s).sum() < 3:
        return None
    return float(d[s].mean() - d[~s].mean())


def estimates(rows: list[dict], role: str, treated: str) -> np.ndarray:
    vals = []
    for p in sorted({r["pair"] for r in rows if r["role"] == role}):
        a = next((r for r in rows if r["role"] == role and r["pair"] == p and r["cond"] == treated), None)
        b = next((r for r in rows if r["role"] == role and r["pair"] == p and r["cond"] == "off"), None)
        if a is None or b is None:
            continue
        t = matched_T(_walls(a), _walls(b))
        if t is not None:
            vals.append(t)
    return np.array(vals)


def _boot_mean(x: np.ndarray, reps: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(1)
    means = [rng.choice(x, len(x)).mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(path: Path) -> None:
    rows = json.loads(path.read_text())["rows"]
    t = estimates(rows, "pair", "e4")
    n = estimates(rows, "null", "ctl")
    ident = [next(r["text"] for r in rows if r["pair"] == p and r["cond"] == "e4") == next(r["text"] for r in rows if r["pair"] == p and r["cond"] == "off")
             for p in sorted({r["pair"] for r in rows if r["role"] == "pair"})]
    print(f"bit-identity (e4 text equals off text, same prompt): {sum(ident)} of {len(ident)}")
    print(f"dropped events (e4 runs): {sum(r['dropped'] for r in rows if r['cond'] == 'e4')}")
    walls = np.array([w for r in rows if r["role"] == "pair" and r["cond"] == "off" for w in _walls(r).values()])
    print(f"token {walls.mean():.1f} ms (off runs), misses a token {np.mean([r['misses_tok'] for r in rows if r['role'] == 'pair']):.1f}")
    if len(t) > 1:
        lo, hi = _boot_mean(t)
        print(f"{len(t)} pairs: T (cost of a sampled token) = {t.mean():+.3f} ms (95 % pair bootstrap {lo:+.3f} to {hi:+.3f}), median {np.median(t):+.3f}")
        print(f"  average over all tokens (T / {EVERY}) = {t.mean() / EVERY:+.3f} ms ({lo / EVERY:+.3f} to {hi / EVERY:+.3f})")
    if len(n) > 1:
        nlo, nhi = _boot_mean(n)
        print(f"{len(n)} control pairs (second stamp-only tracer against off): T = {n.mean():+.3f} ms ({nlo:+.3f} to {nhi:+.3f}); per token {n.mean() / EVERY:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--budget-gib", type=float, default=34.0)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--ctl-pairs", type=int, default=10)
    ap.add_argument("--warm", type=int, default=4)
    args = ap.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    if args.report:
        report(Path(args.report))
    else:
        if not args.out:
            ap.error("--out is required to run")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        run(Path(args.out), args.tokens, args.pairs, args.ctl_pairs, args.warm, args.budget_gib)


if __name__ == "__main__":
    main()
