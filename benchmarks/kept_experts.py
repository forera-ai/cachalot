"""
Does the number of experts a layer keeps change the next layer's router-eval window? (docs/L4B-SYNC-EXCESS-RECORD.md addendum,
HANDOFF 18.147). Offline: rows from a budget-0 recording made with 0.64.13 or later (GET_END carries the kept count) against the
exact arm. The window of layer L holds the GPU work queued since the previous eval: layer L-1's MoE compute, with the experts it kept,
and layer L's attention.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/kept_experts.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import layer_rows as lr

RESULTS = Path(__file__).resolve().parent / "results"


def with_previous(rows: list[dict]) -> list[dict]:
    """Attach the previous layer's kept and requested counts in the same token (None for the first layer)."""
    by = {(r["turn"], r["position"], r["layer"]): r for r in rows}
    out = []
    for r in rows:
        prev = by.get((r["turn"], r["position"], r["layer"] - 1))
        out.append({**r, "prev_kept": prev["kept"] if prev else None, "prev_requested": prev["requested"] if prev else None})
    return out


def layer_means(rows: list[dict]) -> dict[int, float]:
    return {l: float(np.mean([r["sync_ms"] for r in rows if r["layer"] == l])) for l in sorted({r["layer"] for r in rows})}


def split_excess(rows: list[dict], exact_means: dict[int, float], full: int = 6) -> dict:
    """Per-token tally of the excess over the exact arm for windows after full layers and after reduced layers."""
    n_tok = len({(r["turn"], r["position"]) for r in rows})
    tally = {"full": 0.0, "reduced": 0.0, "first": 0.0}
    counts = {"full": 0, "reduced": 0, "first": 0}
    for g_name in tally:
        pass
    groups: dict[tuple[int, str], list[float]] = {}
    for r in rows:
        g = "first" if r["prev_kept"] is None else ("full" if r["prev_kept"] >= full else "reduced")
        groups.setdefault((r["layer"], g), []).append(r["sync_ms"])
        counts[g] += 1
    for (layer, g), v in groups.items():
        tally[g] += (np.mean(v) - exact_means[layer]) * len(v) / n_tok
    return {"tally_ms_per_token": tally, "rows": counts, "per_layer_excess_full": _per_layer(groups, exact_means, "full"),
            "per_layer_excess_reduced": _per_layer(groups, exact_means, "reduced")}


def _per_layer(groups, exact_means, g):
    vals = [np.mean(v) - exact_means[l] for (l, name), v in groups.items() if name == g and len(v) >= 20]
    return float(np.mean(vals)) if vals else float("nan")


def fixed_effect_difference(rows: list[dict], full: int = 6) -> float:
    """Mean over layers of (window after a reduced layer - window after a full layer), layer by layer."""
    diffs, weights = [], []
    for l in sorted({r["layer"] for r in rows if r["prev_kept"] is not None}):
        red = [r["sync_ms"] for r in rows if r["layer"] == l and r["prev_kept"] is not None and r["prev_kept"] < full]
        ful = [r["sync_ms"] for r in rows if r["layer"] == l and r["prev_kept"] is not None and r["prev_kept"] >= full]
        if len(red) >= 10 and len(ful) >= 10:
            diffs.append(np.mean(red) - np.mean(ful))
            weights.append(min(len(red), len(ful)))
    return float(np.average(diffs, weights=weights))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget0", default="l4b-raw-budget0-kept")
    ap.add_argument("--exact", default="l4b-raw")
    a = ap.parse_args()
    b0 = with_previous(lr.load_all(RESULTS / a.budget0)[1])
    ex = lr.load_all(RESULTS / a.exact)[1]
    em = layer_means(ex)
    n = len({(r["turn"], r["position"]) for r in b0})
    kept = np.array([r["kept"] for r in b0]); req = np.array([r["requested"] for r in b0])
    print(f"{a.budget0}: {n} tokens, {len(b0)} layer rows; requested {req.mean():.2f}, kept {kept.mean():.2f}; share of layers with kept < requested {np.mean(kept < req):.3f}; "
          f"share with kept < 6 {np.mean(kept < 6):.3f}; dropped a token {(req - kept).sum() / n:.1f}")
    print("kept distribution:", {int(k): round(float(np.mean(kept == k)), 3) for k in sorted(set(kept.tolist()))})
    tot = sum(r["sync_ms"] for r in b0) / n - sum(r["sync_ms"] for r in ex) / len({(r["turn"], r["position"]) for r in ex})
    print(f"window excess over {a.exact}: {tot:+.2f} ms a token")
    s = split_excess(b0, em)
    print("tally of the excess (ms a token) by the previous layer's kept count:", {k: round(v, 2) for k, v in s["tally_ms_per_token"].items()},
          "rows", s["rows"])
    print(f"mean per-layer excess after full layers {s['per_layer_excess_full']:+.3f} ms, after reduced layers {s['per_layer_excess_reduced']:+.3f} ms")
    print(f"layer-fixed-effect difference, window after reduced minus after full: {fixed_effect_difference(b0):+.3f} ms")
    for k in sorted({r["prev_kept"] for r in b0 if r["prev_kept"] is not None}):
        v = [r["sync_ms"] - em[r["layer"]] for r in b0 if r["prev_kept"] == k]
        print(f"  previous layer kept {k}: {len(v)} rows, window minus the exact arm's layer mean {np.mean(v):+.3f} ms")
