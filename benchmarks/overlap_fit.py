"""
L4(b): fit the overlap term of the DeepSeek token model from the L2 per-token rows and test it on held-out rows
(charter section 4.6, docs/L4B-OVERLAP-FIT-RECORD.md, HANDOFF 18.135). Offline: reads benchmarks/results/l2-price/*.json,
loads no model and touches nothing in src/.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/overlap_fit.py

Model, per decode token, with x = read busy time (ms):
    hidden  = min(x, H)              (A, constant)
              f x                    (B, proportional)
              clip(h0 + h1 x, 0, x)  (C, affine)
    exposed = x - hidden;  wall = w0 + exposed;  x = k reads
Baseline: the straight line wall = a + b reads (the form of benchmarks/whatif.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results" / "l2-price"
COLS = ("wall_ms", "sync_ms", "blocked_ms", "read_busy_ms", "hidden_read_ms", "reads")


def load(name: str) -> dict[str, np.ndarray]:
    """Per-token rows of one run file, with the turn index of each row. Rows without reads are kept."""
    rows = json.loads((RESULTS / name).read_text())["rows"]
    out: dict[str, list] = {c: [] for c in COLS}
    out["turn"] = []
    for i, row in enumerate(rows):
        for s in row.get("summary") or []:
            for c in COLS:
                out[c].append(s[c])
            out["turn"].append(i)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _lstsq(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(a, y, rcond=None)[0]


def fit_overlap(d: dict[str, np.ndarray]) -> dict[str, dict]:
    """Fit forms A, B, C and the wall model on rows `d`. Returns parameters per form."""
    x, hid, reads, wall = d["read_busy_ms"], d["hidden_read_ms"], d["reads"], d["wall_ms"]
    params: dict[str, dict] = {}
    # A: hidden = min(x, H); grid over H, least squares on hidden
    grid = np.linspace(0.0, float(x.max()), 400)
    errs = [float(np.mean((np.minimum(x, h) - hid) ** 2)) for h in grid]
    params["A"] = {"H": float(grid[int(np.argmin(errs))])}
    # B: hidden = f x through the origin
    params["B"] = {"f": float(np.dot(x, hid) / np.dot(x, x))}
    # C: hidden = h0 + h1 x
    h0, h1 = _lstsq(np.c_[np.ones_like(x), x], hid)
    params["C"] = {"h0": float(h0), "h1": float(h1)}
    # x = k reads
    k = float(np.dot(reads, x) / np.dot(reads, reads))
    params["k"] = {"k": k}
    # w0: mean of wall - exposed under each form, on the training rows
    for form in "ABC":
        exposed = x - hidden_of(form, params[form], x)
        params[form]["w0"] = float(np.mean(wall - exposed))
    # baseline straight line
    a, b = _lstsq(np.c_[np.ones_like(reads), reads], wall)
    params["line"] = {"a": float(a), "b": float(b)}
    return params


def hidden_of(form: str, p: dict, x: np.ndarray) -> np.ndarray:
    if form == "A":
        return np.minimum(x, p["H"])
    if form == "B":
        return p["f"] * x
    return np.clip(p["h0"] + p["h1"] * x, 0.0, x)


def score(params: dict[str, dict], d: dict[str, np.ndarray]) -> dict[str, dict]:
    """Root mean square error and bias (prediction minus measurement) on rows `d`."""
    reads, wall, blocked = d["reads"], d["wall_ms"], d["blocked_ms"]
    x_true = d["read_busy_ms"]
    x_hat = params["k"]["k"] * reads
    out: dict[str, dict] = {}
    for form in "ABC":
        p = params[form]
        # blocked given the measured busy time isolates the overlap shape; wall also carries the reads-to-busy step
        blk = x_true - hidden_of(form, p, x_true)
        wal = p["w0"] + (x_hat - hidden_of(form, p, x_hat))
        out[form] = {
            "blocked_rmse": float(np.sqrt(np.mean((blk - blocked) ** 2))),
            "blocked_bias": float(np.mean(blk - blocked)),
            "wall_rmse": float(np.sqrt(np.mean((wal - wall) ** 2))),
            "wall_bias": float(np.mean(wal - wall)),
        }
    ln = params["line"]["a"] + params["line"]["b"] * reads
    out["line"] = {"wall_rmse": float(np.sqrt(np.mean((ln - wall) ** 2))), "wall_bias": float(np.mean(ln - wall))}
    return out


def split_by_turn(d: dict[str, np.ndarray], parity: int) -> dict[str, np.ndarray]:
    m = (d["turn"] % 2) == parity
    return {k: v[m] for k, v in d.items()}


def bootstrap_wall_gain(params: dict, d: dict[str, np.ndarray], form: str = "C", n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Relative wall-error change of `form` against the line, resampling whole turns: (point, lo, hi); negative = better."""
    rng = np.random.default_rng(seed)
    turns = np.unique(d["turn"])
    pieces = {t: {k: v[d["turn"] == t] for k, v in d.items()} for t in turns}

    def gain(sample: list[int]) -> float:
        cat = {k: np.concatenate([pieces[t][k] for t in sample]) for k in d}
        s = score(params, cat)
        return s[form]["wall_rmse"] / s["line"]["wall_rmse"] - 1.0

    point = gain(list(turns))
    draws = [gain(list(rng.choice(turns, size=len(turns), replace=True))) for _ in range(n)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-reads", type=float, default=0.0, help="drop tokens with fewer reads (the resident arm has none)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    runs = {n: load(n) for n in ("run.json", "run30.json")}
    runs = {n: {k: v[runs[n]["reads"] >= args.min_reads] for k, v in d.items()} for n, d in runs.items()}
    report: dict[str, dict] = {}
    pairs = [("run.json", "run30.json"), ("run30.json", "run.json")]
    for tr, te in pairs:
        p = fit_overlap(runs[tr])
        report[f"train {tr} -> test {te}"] = {"params": p, "score": score(p, runs[te]), "gain_C_vs_line": bootstrap_wall_gain(p, runs[te])}
    for name, d in runs.items():
        p = fit_overlap(split_by_turn(d, 0))
        report[f"within {name}: even turns -> odd turns"] = {"params": p, "score": score(p, split_by_turn(d, 1)),
                                                              "gain_C_vs_line": bootstrap_wall_gain(p, split_by_turn(d, 1))}
    p_all = {n: fit_overlap(d) for n, d in runs.items()}
    report["h1 by run"] = {n: p["C"]["h1"] for n, p in p_all.items()}
    if args.json:
        print(json.dumps(report, indent=2))
        return
    for key, v in report.items():
        print(f"\n== {key}")
        if "score" not in v:
            print("  ", {k: round(x, 3) for k, x in v.items()})
            continue
        pr = v["params"]
        print("   fitted: H=%.1f  f=%.3f  h0=%.1f h1=%.3f  k=%.2f ms/read  w0(C)=%.1f  line=%.1f+%.2f x reads" % (
            pr["A"]["H"], pr["B"]["f"], pr["C"]["h0"], pr["C"]["h1"], pr["k"]["k"], pr["C"]["w0"], pr["line"]["a"], pr["line"]["b"]))
        for form in ("A", "B", "C"):
            s = v["score"][form]
            print("   %s  blocked rmse %5.1f bias %+5.1f | wall rmse %5.1f bias %+5.1f" % (
                form, s["blocked_rmse"], s["blocked_bias"], s["wall_rmse"], s["wall_bias"]))
        s = v["score"]["line"]
        print("   line                                | wall rmse %5.1f bias %+5.1f" % (s["wall_rmse"], s["wall_bias"]))
        g = v["gain_C_vs_line"]
        print("   C against the line, wall rmse change: %+.1f %% (95 %% turn bootstrap %+.1f to %+.1f)" % (100 * g[0], 100 * g[1], 100 * g[2]))


if __name__ == "__main__":
    main()
