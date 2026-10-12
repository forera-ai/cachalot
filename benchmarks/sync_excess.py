"""
The router-eval window against the read time that overlaps it, pooled over the raw-row arms (docs/L4B-SYNC-EXCESS-RECORD.md,
HANDOFF 18.146). Offline.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/sync_excess.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import layer_rows as lr

RESULTS = Path(__file__).resolve().parent / "results"
ARMS = ("l4b-raw", "l4b-raw-budget0", "l4b-raw-nopred", "l4b-raw-28gib", "l4b-raw-22gib", "l4b-raw-46gib-on", "l4b-raw-46gib-off")


def per_token(rows: list[dict]) -> dict[str, float]:
    keys = {(r["turn"], r["position"]) for r in rows}
    n = len(keys)
    return {"sync": sum(r["sync_ms"] for r in rows) / n, "sync_read": sum(r["sync_read_ms"] for r in rows) / n, "tokens": n}


def fit(rows_by_arm: dict[str, list[dict]]) -> dict:
    """Pooled least squares sync ~ layer constant + s x sync_read_ms, with layer fixed effects."""
    X, y, layers = [], [], sorted({r["layer"] for r in next(iter(rows_by_arm.values()))})
    idx = {l: i for i, l in enumerate(layers)}
    for rows in rows_by_arm.values():
        for r in rows:
            x = np.zeros(len(layers) + 1)
            x[idx[r["layer"]]] = 1.0
            x[-1] = r["sync_read_ms"]
            X.append(x)
            y.append(r["sync_ms"])
    X, y = np.array(X), np.array(y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return {"slope": float(beta[-1]), "layer_const": {l: float(beta[idx[l]]) for l in layers}, "resid_sd": float(resid.std())}


def predict_token(rows: list[dict], f: dict) -> float:
    keys = {(r["turn"], r["position"]) for r in rows}
    return sum(f["layer_const"][r["layer"]] + f["slope"] * r["sync_read_ms"] for r in rows) / len(keys)


def position_slope(rows: list[dict]) -> float:
    pos = np.array([r["position"] for r in rows], dtype=float)
    y = np.array([r["sync_ms"] for r in rows])
    layers = np.array([r["layer"] for r in rows])
    yy = y - np.array([y[layers == l].mean() for l in layers])  # per-layer centred
    pp = pos - pos.mean()
    return float(np.dot(pp, yy) / np.dot(pp, pp))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    a = ap.parse_args()
    data = {}
    for arm in a.arms:
        if (RESULTS / arm / "manifest.json").exists():
            data[arm] = lr.load_all(RESULTS / arm)[1]
    f = fit(data)
    print(f"pooled slope of sync on overlapping read time: {f['slope']:.3f} ms per ms (residual sd {f['resid_sd']:.2f} ms)")
    print(f"{'arm':20s} {'sync':>6s} {'read-in-sync':>12s} {'predicted':>9s} {'tail>3ms':>8s} {'pos slope':>9s}")
    for arm, rows in data.items():
        t = per_token(rows)
        tail = np.mean([r["sync_ms"] > 3.0 for r in rows])
        print(f"{arm:20s} {t['sync']:6.1f} {t['sync_read']:12.1f} {predict_token(rows, f):9.1f} {100 * tail:7.2f}% {position_slope(rows):9.4f}")
    if "l4b-raw" in data and "l4b-raw-budget0" in data:
        m = per_token(data["l4b-raw-budget0"])["sync"] - per_token(data["l4b-raw"])["sync"]
        p = predict_token(data["l4b-raw-budget0"], f) - predict_token(data["l4b-raw"], f)
        print(f"budget 0 minus exact (34 GiB, prefetch on): measured {m:+.2f} ms a token, predicted by overlapping reads {p:+.2f}, residual {m - p:+.2f}")
