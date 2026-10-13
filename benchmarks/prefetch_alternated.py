"""
Alternated decode-prefetch pair on the 22k Hermes context (docs/L4B-RAW-LAYER-RECORD.md arm 8, HANDOFF 18.145).

One process, one model load: every Hermes prompt is run twice back to back, once with decode prefetch on (`moe_layer_metal.PREDICT_TOPK`
6, the shipped value) and once off (0), the order alternating by prompt, so a prompt's second run (warmer cache) falls on each side
equally and process drift cancels in the pair. `PREDICT_TOPK` is read at every MoE layer call, so flipping the module constant between
turns changes the next turn (checked in the data: prefetch-off turns must show no hidden read share). The text of a pair is
identical (greedy), so the same token sits at the same position in both runs. The tracer records every decode token (as the raw-row
arms did), which prices the same +0.55 ms into both sides.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    benchmarks/guarded_run.sh --budget-gib 42 --max-seconds 4200 --tag pfalt -- env PYTHONPATH=src ~/venvs/deepseek-v41/bin/python \
        benchmarks/prefetch_alternated.py --budget-gib 42 --dump benchmarks/results/hermes-live-0.61.1/requests.jsonl \
        --out benchmarks/results/prefetch-alt/run.json
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/prefetch_alternated.py --report benchmarks/results/prefetch-alt/run.json

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
from server_miss_budget_ab import USER as HERMES_USER  # noqa: E402

ON, OFF = 6, 0


def run(out: Path, dump: Path, tokens: int, pairs: int, warm: int, budget_gib: float, max_seq_len: int, design: str = "interleave",
        prompts: list[str] | None = None) -> None:
    import pareto

    pareto.serve_defaults()
    from cachalot.metrics import token_trace as tt
    from cachalot.model import moe_layer_metal as mlm
    from cachalot.model.api import V41Model
    from cachalot.model.generation import SamplingParams
    from cachalot.server.engine import ChatRequest, Engine

    users = prompts or HERMES_USER
    body = json.loads(dump.read_text().splitlines()[0])["body"]
    system, tools = body["messages"][0], body["tools"]
    model = V41Model.from_pretrained(os.environ["CACHALOT_MODEL_PATH"], max_seq_len=max_seq_len,
                                     expert_cache_budget_bytes=int(budget_gib * 2**30))
    engine = Engine(model)
    rt = model.runtime
    trace = tt.TokenTrace(every=1)
    rows: list[dict] = []
    print(f"prefetch top-k at start: {mlm.PREDICT_TOPK}", flush=True)

    def turn(cond: str, prompt: str, idx: int, role: str) -> None:
        mlm.PREDICT_TOPK = ON if cond == "on" else OFF
        rt.set_token_trace(trace)
        trace.drain()
        req = ChatRequest(messages=[system, {"role": "user", "content": prompt}], tools=tools,
                          params=SamplingParams(max_new_tokens=tokens, temperature=0.0))
        m0 = rt.expert_store.cache_misses
        res = engine.chat(req)
        n = max(res.completion_tokens, 1)
        d = trace.drain()
        summ = tt.summarize(d)
        row = {"cond": cond, "role": role, "pair": idx, "tokens": res.completion_tokens, "ms_tok": round(1000 * res.decode_seconds / n, 2),
               "misses_tok": round((rt.expert_store.cache_misses - m0) / n, 2), "reused": res.reused_prefix_tokens,
               "text": res.raw_text, "stamps": d["token_stamps"].tolist(), "positions": d["token_positions"].tolist(),
               "dropped": int(d["dropped"]), "summary": summ}
        rows.append(row)
        print({k: v for k, v in row.items() if k not in ("text", "stamps", "positions", "summary")}, flush=True)
        out.write_text(json.dumps({"rows": rows}))

    for i in range(warm):
        turn("on", users[i % len(users)], -1 - i, "warm")
    if design == "backtoback":  # arm 8, void: the second run of a prompt inherits the first run's resident experts
        for i in range(pairs):
            order = ("on", "off") if i % 2 == 0 else ("off", "on")
            for c in order:
                turn(c, users[i % len(users)], i, "pair")
    else:
        # arm 8b: passes over all prompts with the condition alternating by prompt parity and swapped between passes, so a prompt's two
        # runs of a pair are a whole pass apart (server_miss_budget_ab.py, HANDOFF 18.59). Passes 0,1 make pair id i, passes 2,3 make
        # pair id 100 + i; every prompt runs on, off, off, on or off, on, on, off, so each side has the same mix of positions.
        n = len(users)
        for k in range(pairs // n):
            for sub in (0, 1):
                for i in range(n):
                    on = (i % 2 == 1) if sub == 0 else (i % 2 == 0)
                    turn("on" if on else "off", users[i], 100 * k + i, "pair")
    mlm.PREDICT_TOPK = ON
    rt.set_token_trace(None)
    model.close()


def _walls(row: dict) -> dict[int, float]:
    st, pos = row["stamps"], row["positions"]
    return {pos[k]: (st[k + 1] - st[k]) / 1e6 for k in range(len(st) - 1) if pos[k + 1] == pos[k] + 1}


def pair_gap(on: dict[int, float], off: dict[int, float], min_positions: int = 20) -> float | None:
    """Mean over shared positions of (off - on) wall time, ms a token; positive means prefetch saves time."""
    ks = sorted(set(on) & set(off))
    if len(ks) < min_positions:
        return None
    return float(np.mean([off[k] - on[k] for k in ks]))


def pairs_of(rows: list[dict]) -> list[dict]:
    out = []
    for p in sorted({r["pair"] for r in rows if r["role"] == "pair"}):
        on = next((r for r in rows if r["role"] == "pair" and r["pair"] == p and r["cond"] == "on"), None)
        off = next((r for r in rows if r["role"] == "pair" and r["pair"] == p and r["cond"] == "off"), None)
        if on is None or off is None:
            continue
        g = pair_gap(_walls(on), _walls(off))
        if g is not None:
            first = "on" if rows.index(on) < rows.index(off) else "off"
            out.append({"pair": p, "gap": g, "first": first, "tokens": min(on["tokens"], off["tokens"]), "on": on, "off": off,
                        "srv_gap": off["ms_tok"] - on["ms_tok"]})
    return out


def _boot(x: np.ndarray, w: np.ndarray | None = None, reps: int = 5000, clusters: np.ndarray | None = None) -> tuple[float, float]:
    """Bootstrap of the weighted mean; with `clusters` (a prompt id per pair) whole prompts are resampled."""
    rng = np.random.default_rng(1)
    w = np.ones(len(x)) if w is None else w
    if clusters is None:
        m = [np.average(x[i], weights=w[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(reps))]
    else:
        ids = np.unique(clusters)
        members = {c: np.where(clusters == c)[0] for c in ids}
        m = []
        for _ in range(reps):
            idx = np.concatenate([members[c] for c in rng.choice(ids, len(ids))])
            m.append(np.average(x[idx], weights=w[idx]))
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(path: Path) -> None:
    rows = json.loads(path.read_text())["rows"]
    ps = pairs_of(rows)
    g = np.array([p["gap"] for p in ps])
    w = np.array([p["tokens"] for p in ps], dtype=float)
    sg = np.array([p["srv_gap"] for p in ps])
    ident = [p["on"]["text"] == p["off"]["text"] for p in ps]
    print(f"{len(ps)} pairs; text identical in {sum(ident)}; dropped events {sum(r['dropped'] for r in rows)}")
    cl = np.array([p["pair"] % 100 for p in ps])
    lo, hi = _boot(g, w, clusters=cl)
    print(f"position-matched gap (off - on): {np.average(g, weights=w):+.2f} ms a token (95 % bootstrap over prompts {lo:+.2f} to {hi:+.2f}), "
          f"unweighted {g.mean():+.2f}, median {np.median(g):+.2f}, sd {g.std(ddof=1):.2f}; on faster in {int((g > 0).sum())} of {len(g)}")
    slo, shi = _boot(sg, w, clusters=cl)
    print(f"server-timing gap (off - on): {np.average(sg, weights=w):+.2f} ms ({slo:+.2f} to {shi:+.2f})")
    for first in ("on", "off"):
        sel = np.array([p["first"] == first for p in ps])
        if sel.sum():
            print(f"  pairs with {first} first: n={int(sel.sum())}, gap {np.average(g[sel], weights=w[sel]):+.2f} ms")
    for cond in ("on", "off"):
        s = [x for p in ps for x in p[cond]["summary"]]
        if s:
            a = lambda k: np.mean([x[k] for x in s])
            busy = sum(x["read_busy_ms"] for x in s); hid = sum(x["hidden_read_ms"] for x in s)
            print(f"  {cond}: wall {a('wall_ms'):.1f}, sync {a('sync_ms'):.1f}, blocked {a('blocked_ms'):.1f}, busy {a('read_busy_ms'):.1f}, "
                  f"hidden share {100 * hid / busy:.1f} %, events {a('events'):.0f}, misses a token {np.mean([p[cond]['misses_tok'] for p in ps]):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--budget-gib", type=float, default=42.0)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--design", choices=("interleave", "backtoback"), default="interleave")
    ap.add_argument("--prompts", type=Path, help="JSON list of user prompts replacing the built-in 12 (a second workload)")
    ap.add_argument("--max-seq-len", type=int, default=32768)
    a = ap.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    if a.report:
        report(a.report)
        return
    if not a.out or not a.dump:
        ap.error("--out and --dump are required to run")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    run(a.out, a.dump, a.tokens, a.pairs, a.warm, a.budget_gib, a.max_seq_len, a.design,
        json.loads(a.prompts.read_text()) if a.prompts else None)


if __name__ == "__main__":
    main()
