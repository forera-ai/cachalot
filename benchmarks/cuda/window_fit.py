"""
Host-aware row-level window model: fit on the `D` and `P0L` cells, validate on arms it never saw (CUDA lane, Mac only, no GPU; record
`docs/CUDA-WINDOW-FIT-RECORD.md`).

Per decode row of an arm: `n_pf` prefetch loads arrive for it (issued during the previous layer; two layers for `L2`), `n_dem` demand
misses at its boundary. Exposed time a row: `max(0, L_pf + n_pf c - W)` (when `n_pf > 0`; `W = C` for lead 1, `2 C` for `L2`) plus
`L_dem + n_dem c` (when `n_dem > 0`). Three parameters: `c`, `L_dem`, `L_pf`. Step 1 fits `c` and `L_dem` to the measured `E_D`; step 2 fits
`L_pf` to the measured `E_P0L`; step 3 predicts `G10`, `G20`, `L2`, `O` with everything frozen and scores `|H_model - H_measured|`, where
`H_model = 1 - E_model / E_D_measured`.

Usage:
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/window_fit.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prefetch_coverage as pc  # noqa: E402
import replay_arena as ra  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TRACES = {"old": REPO / "benchmarks/results/routing-trace-deepseek-long/trace.npz", "new": REPO / "benchmarks/results/routing-trace-deepseek-leads/trace.npz"}
DATA = [("old", "docs/cuda-results/n3c-rtx5090-2026-10-19", "c-full-*-rows.jsonl", "stageC"),
        ("new", "docs/cuda-results/n3d-rtx5090-2026-10-20", "d-full-*-rows.jsonl", "stageD"),
        ("*", "docs/cuda-results/bridge-rtx5090-2026-10-20", "e-full-*-rows.jsonl", "bridge")]
ARMS = ("D", "P0L", "G10", "G20", "L2", "O")
GATES = {"G10": 0.10, "G20": 0.20}
GROUP_ARM = re.compile(r"n3\.p(?P<p>\d+)\.(?P<lv>L\d)\.C(?P<c>\d+\.\d+)")


def arm_rows(arrays, percent: float, tokens: int, arm: str):
    """Per decode row in trace order: (n_pf, n_dem) for the arm's prefetch policy, from the stage A/B bookkeeping semantics."""
    base = {k: arrays[k] for k in ("phase", "layer", "position", "experts")}
    lead = 2 if arm == "L2" else 1
    pred = {} if arm in ("D", "O") else pc.predicted_sets(arrays, lead)
    gate = GATES.get(arm)
    pw = pc.predicted_weights(arrays, lead) if gate is not None else {}
    slots = ra.slots_for(percent)
    lru: OrderedDict = OrderedDict()

    def touch(k):
        if k in lru:
            lru.move_to_end(k)
            return
        while len(lru) >= slots:
            lru.popitem(last=False)
        lru[k] = 1

    pos_iter = iter(int(base["position"][i]) for i in np.nonzero(base["phase"] == ra.PHASE_DECODE)[0])
    plan = [ev if ev[0] == "P" else (*ev, next(pos_iter)) for ev in ra.walk(base, tokens)]
    pre = ra.ArenaLRU(slots, int(base["layer"].max()) + 1)
    out, started, pf_of = [], set(), {}

    def issue(t):
        _, _tok, layer, exps, pos = plan[t]
        if arm == "O":
            ahead = [int(e) for e in exps]
        elif arm == "D":
            ahead = []
        else:
            ahead = pred.get((pos, layer), [])
            if gate is not None:
                ahead = [e for e, w in zip(ahead, pw.get((pos, layer), []), strict=False) if w >= gate]
        n = 0
        for e in ahead:
            k = (layer, e)
            if k not in lru:
                touch(k)
                n += 1
        pf_of[t] = pf_of.get(t, 0) + n

    for j, ev in enumerate(plan):
        if ev[0] == "P":
            pre.lru, pre.free = lru, list(range(slots - len(lru)))
            pre.prefill_layer(ev[1], ev[2])
            continue
        _, tok, layer, exps, _p = ev
        if arm != "D" and j not in started:
            issue(j)
        n_dem = 0
        for e in exps:
            k = (layer, int(e))
            if k in lru:
                lru.move_to_end(k)
            else:
                n_dem += 1
                touch(k)
        out.append((pf_of.pop(j, 0), n_dem))
        if arm != "D":
            t = j + lead
            ok = (t < len(plan) and plan[t][0] == "D") if lead == 1 else (t < len(plan) and all(plan[x][0] == "D" and plan[x][1] == tok for x in range(j + 1, t + 1)))
            if ok:
                issue(t)
                started.add(t)
    return np.array(out, dtype=np.float64)


def exposure(rows: np.ndarray, tokens: int, c: float, l_dem: float, l_pf: float, window: float) -> float:
    """Mean exposed ms a token under the model."""
    pf, dem = rows[:, 0], rows[:, 1]
    e_pf = np.where(pf > 0, np.maximum(0.0, l_pf + pf * c - window), 0.0)
    e_dem = np.where(dem > 0, l_dem + dem * c, 0.0)
    return float((e_pf + e_dem).sum() / tokens)


def load_groups() -> list[dict]:
    """One dict per measured group-run: trace, percent, C, level, and the measured E_<arm> values."""
    kinds = {json.loads((REPO / "benchmarks/cuda" / f).read_text())["trace_sha256"]: k for f, k in (("n3c-expected.json", "old"), ("n3d-expected.json", "new"))}
    groups = []
    for kind, directory, pattern, stage in DATA:
        for rpath in sorted((REPO / directory).glob(pattern)):
            mpath = rpath.with_name(rpath.name.replace("rows.jsonl", "run-manifest.json"))
            k = kinds.get(json.loads(mpath.read_text())["workload"]["trace_sha256"], kind) if kind == "*" else kind
            vals: dict = {}
            for line in rpath.read_text().splitlines():
                r = json.loads(line)
                if r["field"].startswith("E_") and GROUP_ARM.fullmatch(r["arm"]) and r.get("value") is not None:
                    vals.setdefault(r["arm"], {})[r["field"][2:]] = r["value"]
            for arm, e in vals.items():
                m = GROUP_ARM.fullmatch(arm)
                groups.append({"stage": stage, "file": rpath.name, "trace": k, "percent": float(m["p"]), "level": m["lv"], "C": float(m["c"]), "E": e})
    return groups


def fit(groups, counts, tokens: int):
    ds = [g for g in groups if "D" in g["E"]]
    best = None
    for c in np.arange(0.14, 0.3001, 0.002):
        for l_dem in np.arange(0.0, 0.6001, 0.01):
            err = sum((exposure(counts[(g["trace"], g["percent"], "D")], tokens, c, l_dem, 0.0, 0.0) - g["E"]["D"]) ** 2 for g in ds)
            if best is None or err < best[0]:
                best = (err, float(c), float(l_dem))
    _e, c, l_dem = best
    ps = [g for g in groups if "P0L" in g["E"]]
    bestp = None
    for l_pf in np.arange(-0.1, 0.5001, 0.005):
        err = sum((exposure(counts[(g["trace"], g["percent"], "P0L")], tokens, c, l_dem, l_pf, g["C"]) - g["E"]["P0L"]) ** 2 for g in ps)
        if bestp is None or err < bestp[0]:
            bestp = (err, float(l_pf))
    return c, l_dem, bestp[1]


def predict(g, counts, arm, tokens, params):
    c, l_dem, l_pf = params
    window = 2 * g["C"] if arm == "L2" else g["C"]
    return exposure(counts[(g["trace"], g["percent"], arm)], tokens, c, l_dem, l_pf, window)


def evaluate(groups, counts, tokens, params, arms) -> list[dict]:
    rows = []
    for g in groups:
        for arm in arms:
            if arm in g["E"] and "D" in g["E"]:
                e = predict(g, counts, arm, tokens, params)
                h_m, h_x = 1 - e / g["E"]["D"], 1 - g["E"][arm] / g["E"]["D"]
                rows.append({"stage": g["stage"], "trace": g["trace"], "percent": g["percent"], "C": g["C"], "arm": arm, "E_model": e, "E_meas": g["E"][arm],
                             "H_model": h_m, "H_meas": h_x, "error": abs(h_m - h_x)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    groups = load_groups()
    counts: dict = {}
    for kind, path in TRACES.items():
        with np.load(path) as d:
            arrays = {k: d[k] for k in d.files if k != "segments"}
        for pct in sorted({g["percent"] for g in groups if g["trace"] == kind}):
            for arm in ARMS:
                if any(arm in g["E"] for g in groups if g["trace"] == kind and g["percent"] == pct):
                    counts[(kind, pct, arm)] = arm_rows(arrays, pct, args.tokens, arm)
    params = fit(groups, counts, args.tokens)
    fitted = evaluate(groups, counts, args.tokens, params, ("D", "P0L"))
    held = evaluate(groups, counts, args.tokens, params, ("G10", "G20", "L2", "O"))
    ref_params = (0.194, 0.0, 0.0)
    ref = evaluate(groups, counts, args.tokens, ref_params, ("G10", "G20", "L2", "O"))
    res = {"params": {"c": params[0], "L_dem": params[1], "L_pf": params[2]}, "fit_cells": fitted, "held_out": held, "reference_held_out": ref,
           "mean_abs_error": {"fit": float(np.mean([r["error"] for r in fitted])), "held_out": float(np.mean([r["error"] for r in held])),
                              "reference_held_out": float(np.mean([r["error"] for r in ref]))}}
    text = json.dumps(res, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
