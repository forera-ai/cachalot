"""
L2 in-situ price (docs/L2-IN-SITU-PRICE-RECORD.md, HANDOFF 18.132): what the per-token trace costs inside a real DeepSeek decode
token, through `Engine.chat` (the code `./serve.sh` runs), with recording switched per decode token by position parity.

Each condition is a pass: 'p0' records even positions, 'p1' records odd positions (the swapped parity), 'ctl' records nothing
(-1) while the parity labels stay, so on-against-off there is the noise floor of alternating tokens. A 'base' pass runs with no
tracer installed at all and gives the reference text for the bit-identity check. A token's wall time is the gap between
consecutive token-boundary stamps, written in every state.

Workloads: 'resident' repeats one short prompt (no misses); 'mixed' runs distinct prompts (about 20 misses a token).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    benchmarks/guarded_run.sh --budget-gib 48 --max-seconds 4500 --tag l2price -- \
        env PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/token_trace_price.py --budget-gib 48 --out benchmarks/results/l2-price/run.json
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/token_trace_price.py --report benchmarks/results/l2-price/run.json

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

SYSTEM = "You are a concise coding assistant."
RESIDENT_PROMPT = "Explain in two short paragraphs what a hash map is and why lookups are fast."
MIXED = [
    "List five common Linux commands for inspecting disk usage and say what each shows.",
    "Write a Python function that merges two sorted lists, and explain its complexity.",
    "Summarise the difference between TCP and UDP in a short table.",
    "Explain what a race condition is and give a tiny C# example.",
    "Describe how git rebase differs from merge and when each is safer.",
    "Write a bash one-liner that counts distinct IP addresses in an access log.",
    "Explain what a Bloom filter is and where it is used.",
    "Draft a commit message for a change that adds retry logic to an HTTP client.",
    "Explain the CAP theorem in four sentences.",
    "Write a SQL query that finds the second highest salary in a table.",
    "Describe what a memory-mapped file is and one pitfall of using it.",
    "Explain how a B-tree keeps its height small.",
]


def run(out: Path, tokens: int, resident_turns: int, mixed_turns: int, budget_gib: float, ctl_turns: int | None = None) -> None:
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
    trace = tt.TokenTrace()
    rows: list[dict] = []
    stats: list[dict] = []

    def turn(workload: str, cond: str, prompt: str, idx: int) -> str:
        if cond == "base":
            rt.set_token_trace(None)
        else:
            rt.set_token_trace(trace)
            trace.alternate = {"p0": 0, "p1": 1, "ctl": -1}[cond]
            trace.drain()  # clear anything left from a previous turn
        req = ChatRequest(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                          params=SamplingParams(max_new_tokens=tokens, temperature=0.0))
        misses0 = rt.expert_store.cache_misses
        res = engine.chat(req)
        n = max(res.completion_tokens, 1)
        row = {"workload": workload, "cond": cond, "turn": idx, "tokens": res.completion_tokens,
               "ms_tok": round(1000 * res.decode_seconds / n, 2), "misses_tok": round((rt.expert_store.cache_misses - misses0) / n, 1),
               "text": res.raw_text}
        if cond != "base":
            drained = trace.drain()
            row["stamps"] = drained["token_stamps"].tolist()
            row["positions"] = drained["token_positions"].tolist()
            summ = tt.summarize(drained)
            row["summary"] = summ
            row["dropped"] = drained["dropped"]
        rows.append(row)
        print({k: v for k, v in row.items() if k not in ("text", "stamps", "positions", "summary")}, flush=True)
        out.write_text(json.dumps({"rows": rows}))
        return res.raw_text

    # warm: the first turn loads the experts it needs and prefills; not analysed
    turn("resident", "base", RESIDENT_PROMPT, -1)
    if resident_turns:
        turn("resident", "base", RESIDENT_PROMPT, 0)
    for cond in ("p0", "p1", "ctl"):
        for i in range(resident_turns if cond != "ctl" or ctl_turns is None else min(ctl_turns, resident_turns)):
            turn("resident", cond, RESIDENT_PROMPT, i)
    for i in range(4):
        turn("mixed", "base", MIXED[i], i)
    for cond in ("p0", "p1", "ctl"):
        for i in range(mixed_turns if cond != "ctl" or ctl_turns is None else min(ctl_turns, mixed_turns)):
            turn("mixed", cond, MIXED[i % len(MIXED)], i)
    rt.set_token_trace(None)
    model.close()


def _pairs(row: dict) -> list[tuple[int, int, float]]:
    """(position, parity-label, wall_ms) for consecutive-position tokens of one turn."""
    st, pos = row["stamps"], row["positions"]
    return [(pos[k], pos[k] % 2, (st[k + 1] - st[k]) / 1e6) for k in range(len(st) - 1) if pos[k + 1] == pos[k] + 1]


def _block_boot(d: np.ndarray, block: int = 40, reps: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    nb = max(len(d) // block, 1)
    blocks = [d[i * block:(i + 1) * block] for i in range(nb)]
    means = [np.concatenate([blocks[j] for j in rng.integers(0, nb, nb)]).mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _walls(row: dict) -> dict[int, float]:
    st, pos = row["stamps"], row["positions"]
    return {pos[k]: (st[k + 1] - st[k]) / 1e6 for k in range(len(st) - 1) if pos[k + 1] == pos[k] + 1}


def matched_estimate(rows: list[dict]) -> dict[str, np.ndarray]:
    """
    Position-matched difference in differences, the estimator of record. A prompt gives the same tokens at the same positions
    in every pass, so token k of pass p0 pairs with token k of pass p1. p0 records even k, p1 records odd k:
    d_k = wall_p0(k) - wall_p1(k) is +T for even k and -T for odd k, T = (mean_even d - mean_odd d) / 2. Pass-level drift and
    the parity effect are common to even and odd positions and cancel. The null contrast is p0 against ctl at odd k
    (both unrecorded): it shows how large the pass-to-pass difference is that the estimator has to cancel. One value per turn.
    """
    out: dict[str, list] = {}
    for wl in ("resident", "mixed"):
        t_vals: list[float] = []
        null_vals: list[float] = []
        for t in sorted({r["turn"] for r in rows if r["workload"] == wl and r["cond"] == "p0"}):
            g = {c: next((r for r in rows if r["workload"] == wl and r["cond"] == c and r["turn"] == t), None) for c in ("p0", "p1", "ctl")}
            if g["p0"] is None or g["p1"] is None:
                continue
            w = {c: _walls(g[c]) for c in g if g[c] is not None}
            ks = sorted(set(w["p0"]) & set(w["p1"]))
            if len(ks) < 20:
                continue
            d = np.array([w["p0"][k] - w["p1"][k] for k in ks])
            ev = np.array([k % 2 == 0 for k in ks])
            t_vals.append((d[ev].mean() - d[~ev].mean()) / 2)
            if "ctl" in w:  # the control pass may be shorter than the others
                kc = [k for k, e in zip(ks, ev) if not e and k in w["ctl"]]
                if len(kc) >= 10:
                    null_vals.append(np.mean([w["p0"][k] - w["ctl"][k] for k in kc]))
        out[wl] = np.array(t_vals)
        out[wl + "_null"] = np.array(null_vals)
    return out


def _boot_mean(x: np.ndarray, reps: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(1)
    means = [rng.choice(x, len(x)).mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(path: Path) -> None:
    rows = json.loads(path.read_text())["rows"]
    for workload in ("resident", "mixed"):
        print(f"\n== {workload}")
        base = {r["turn"]: r["text"] for r in rows if r["workload"] == workload and r["cond"] == "base"}
        ident = [r["text"] == base[r["turn"]] for r in rows if r["workload"] == workload and r["cond"] != "base" and r["turn"] in base]
        print(f"bit-identity: {sum(ident)} of {len(ident)} traced turns equal the tracer-free text of the same prompt")
        for cond in ("p0", "p1", "ctl"):
            sel = [r for r in rows if r["workload"] == workload and r["cond"] == cond]
            d_all: list[float] = []
            for r in sel:
                pr = _pairs(r)
                phase = {"p0": 0, "p1": 1, "ctl": 0}[cond]
                # adjacent (on, off) pairs: "on" is the parity that records in this pass
                for (p_a, par_a, w_a), (p_b, par_b, w_b) in zip(pr, pr[1:]):
                    if p_b != p_a + 1:
                        continue
                    if par_a == phase and par_b != phase:
                        d_all.append(w_a - w_b)
                    elif par_b == phase and par_a != phase:
                        d_all.append(w_b - w_a)
            d = np.array(d_all)
            if len(d) < 20:
                print(f"{cond}: too few pairs ({len(d)})")
                continue
            lo, hi = _block_boot(d)
            walls = np.array([w for r in sel for (_, _, w) in _pairs(r)])
            print(f"{cond}: {len(d)} pairs, token {walls.mean():.1f} ms (median {np.median(walls):.1f}); on - off mean {d.mean():+.3f} ms "
                  f"(95 % block bootstrap {lo:+.3f} to {hi:+.3f}), median {np.median(d):+.3f}, sd {d.std(ddof=1):.2f}")
        recorded = [s for r in rows if r["workload"] == workload and r["cond"] in ("p0", "p1") for s in r["summary"]]
        if recorded:
            a = lambda k: np.array([s[k] for s in recorded])
            blocked, busy = a("blocked_ms"), a("read_busy_ms")
            print(f"recorded tokens {len(recorded)}: events {a('events').mean():.0f}, reads {a('reads').mean():.1f}, sync {a('sync_ms').mean():.1f} ms, "
                  f"blocked {blocked.mean():.1f} ms, read busy {busy.mean():.1f} ms, hidden read {a('hidden_read_ms').mean():.1f} ms, "
                  f"misses a token {np.mean([r['misses_tok'] for r in rows if r['workload'] == workload and r['cond'] in ('p0', 'p1')]):.1f}")
            if busy.sum() > 0:
                print(f"blocked / read busy {blocked.sum() / busy.sum():.2f}; dropped events {sum(r['dropped'] for r in rows if r['workload'] == workload and 'dropped' in r)}")

    est = matched_estimate(rows)
    print("\n== position-matched estimate of the tracer's cost on a decode token (estimator of record)")
    for wl in ("resident", "mixed"):
        t, nl = est[wl], est[wl + "_null"]
        if len(t) < 2:
            continue
        lo, hi = _boot_mean(t)
        nlo, nhi = _boot_mean(nl) if len(nl) > 1 else (float("nan"), float("nan"))
        print(f"{wl}: {len(t)} turns, T = {t.mean():+.3f} ms (95 % turn bootstrap {lo:+.3f} to {hi:+.3f}), median {np.median(t):+.3f}; "
              f"pass-to-pass difference the estimator cancels (p0 against ctl, both unrecorded) {nl.mean():+.2f} ms ({nlo:+.2f} to {nhi:+.2f})")
    both = np.concatenate([est["resident"], est["mixed"]])
    blo, bhi = _boot_mean(both)
    if len(est["resident"]) and len(est["mixed"]):
        print(f"pooled over both workloads (post hoc, not pre-registered): {len(both)} turns, T = {both.mean():+.3f} ms ({blo:+.3f} to {bhi:+.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--budget-gib", type=float, default=48.0)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--resident-turns", type=int, default=6)
    ap.add_argument("--mixed-turns", type=int, default=12)
    ap.add_argument("--ctl-turns", type=int, default=None, help="turns in the control pass (default: as many as the other passes)")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    if args.report:
        report(Path(args.report))
    else:
        if not args.out:
            ap.error("--out is required to run")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        run(Path(args.out), args.tokens, args.resident_turns, args.mixed_turns, args.budget_gib, args.ctl_turns)


if __name__ == "__main__":
    main()
