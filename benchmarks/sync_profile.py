"""
Router-sync window by layer from the raw tracer rows (docs/L4B-SYNC-RECONCILE-RECORD.md, HANDOFF 18.142). Offline.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/sync_profile.py benchmarks/results/l4b-raw-46gib-on
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import layer_rows as lr


def profile(rows: list[dict]) -> dict:
    layers = sorted({r["layer"] for r in rows})
    by = {l: np.array([r["sync_ms"] for r in rows if r["layer"] == l]) for l in layers}
    return {
        "mean_by_layer": {l: float(v.mean()) for l, v in by.items()},
        "p10_by_layer": {l: float(np.percentile(v, 10)) for l, v in by.items()},
        "median_by_layer": {l: float(np.median(v)) for l, v in by.items()},
        "token_mean": float(sum(v.mean() for v in by.values())),
        "token_p10_sum": float(sum(np.percentile(v, 10) for v in by.values())),
        "token_median_sum": float(sum(np.median(v) for v in by.values())),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--layers", action="store_true")
    a = ap.parse_args()
    _, rows = lr.load_all(a.path)
    p = profile(rows)
    print(f"token mean {p['token_mean']:.1f} ms, sum of per-layer medians {p['token_median_sum']:.1f}, sum of per-layer p10 {p['token_p10_sum']:.1f}")
    if a.layers:
        for l, m in p["mean_by_layer"].items():
            print(l, round(m, 2), round(p["median_by_layer"][l], 2), round(p["p10_by_layer"][l], 2))
