"""
Predicted-overlap version of the compute-slowdown term (Mac only; record `docs/CUDA-PREDICTED-OVERLAP-RECORD.md`, predictions PO-1 to PO-9).

The slowdown term (`compute_slowdown_fit.py`) takes each row's overlap with prefetch batches from the recorded event times. Here the
overlap is computed from the queue alone: the per-row demand and prefetch counts of the trace plus a per-copy cost, a base compute time
and a row gap, all fitted on run A. A first-in first-out copy stream gives each row's compute window and each prefetch batch's interval.
Nothing is read from run B except the counts (to predict) and the events (to score). It judges nothing for the writer: the verdict column
says only whether a number is inside the registered interval, past its falsifier, or between.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/predicted_overlap_fit.py RUN_A_DIR RUN_B_DIR --fix FIX_DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compute_slowdown_fit as cs  # noqa: E402
import row_timing as rt  # noqa: E402

RF = rt.RF
FIXED_POINT_PASSES = 8


def fit_params(p0l: dict, d: dict) -> dict:
    """Parameters of the queue model from one run's `P0L` and `D` cells at one level and C (run A only)."""
    rows = rt.rows_of(p0l)
    n_pf = np.nan_to_num(rows[:, RF["n_pf"]])
    pa, pb = rows[:, RF["pf_a"]], rows[:, RF["pf_b"]]
    m = (n_pf > 0) & np.isfinite(pa) & np.isfinite(pb)
    c_pf = float(np.median((pb[m] - pa[m]) / n_pf[m]))
    gaps, pre0 = [], []
    for tok in cs.tokens_of(p0l):
        if len(tok) < 2:
            continue
        pre0.append(tok[0, RF["pre"]])
        gaps.extend((tok[1:, RF["pre"]] - tok[:-1, RF["end"]]).tolist())
    gaps = np.array(gaps)
    return {"c_pf": c_pf, "c_dem": rt.per_miss(rt.rows_of(d))["d_per_miss"], "b0": cs.base_busy(d),
            "gap": float(np.median(gaps[np.isfinite(gaps)])), "pre0": float(np.nanmedian(pre0))}


def simulate(n_dem: np.ndarray, n_pf: np.ndarray, par: dict, s: float = 0.0) -> dict:
    """One token through a first-in first-out copy stream. `s` > 0 stretches a row's compute by `s` x its own overlap (closed loop)."""
    n = len(n_dem)
    start, end, pf_a, pf_b = np.zeros(n), np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    free, pre = 0.0, par["pre0"]
    for i in range(n):
        dem_a = max(pre, free)
        dem_b = dem_a + n_dem[i] * par["c_dem"]
        free = dem_b
        start[i] = dem_b
        if n_pf[i] > 0:
            pf_a[i] = max(dem_b, free)
            pf_b[i] = pf_a[i] + n_pf[i] * par["c_pf"]
            free = pf_b[i]
        stretch = 0.0
        for _ in range(FIXED_POINT_PASSES if s else 1):
            e = start[i] + par["b0"] + stretch
            lo = np.maximum(start[i], pf_a[: i + 1][np.isfinite(pf_a[: i + 1])])
            hi = np.minimum(e, pf_b[: i + 1][np.isfinite(pf_b[: i + 1])])
            ov = float(np.sum(np.maximum(0.0, hi - lo)))
            new = s * ov
            if abs(new - stretch) < 1e-9:
                stretch = new
                break
            stretch = new
        end[i] = start[i] + par["b0"] + stretch
        pre = end[i] + par["gap"]
    lo = np.maximum(start[:, None], pf_a[None, :])
    hi = np.minimum(end[:, None], pf_b[None, :])
    inter = np.where(np.isfinite(lo) & np.isfinite(hi), np.maximum(0.0, hi - lo), 0.0)
    return {"overlap": inter.sum(axis=1), "start": start, "end": end}


def predict_cell(cell: dict, par: dict, s: float = 0.0) -> list[np.ndarray]:
    """Predicted per-row overlap for every decode token of a cell, from its counts only."""
    out = []
    for tok in cs.tokens_of(cell):
        out.append(simulate(np.nan_to_num(tok[:, RF["n_dem"]]), np.nan_to_num(tok[:, RF["n_pf"]]), par, s)["overlap"] if len(tok) else np.zeros(0))
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def tok_mean_sum(parts: list[np.ndarray], k: float = 1.0) -> float:
    return float(np.mean([np.nansum(k * p) for p in parts]))


def score_level(raw_a: dict, raw_b: dict, c: float, level: int = 3) -> dict:
    ids = {role: rt.cell_id(level, c, role) for role in ("D", "P0L")}
    if not all(i in raw_a and i in raw_b and raw_a[i].get("rows") and raw_b[i].get("rows") for i in ids.values()):
        return {"available": False}
    par = fit_params(raw_a[ids["P0L"]], raw_a[ids["D"]])
    sa, sb = cs.stack(raw_a[ids["P0L"]]), cs.stack(raw_b[ids["P0L"]])
    base_a, base_b = cs.base_busy(raw_a[ids["D"]]), cs.base_busy(raw_b[ids["D"]])
    ya, yb = sa["busy"] - base_a, sb["busy"] - base_b
    pred_a, pred_b = predict_cell(raw_a[ids["P0L"]], par), predict_cell(raw_b[ids["P0L"]], par)
    pa_flat, pb_flat = np.concatenate(pred_a), np.concatenate(pred_b)
    s_pred = cs.fit_through_origin(pa_flat, ya)
    s_meas = cs.fit_through_origin(sa["overlap"], ya)
    closed_b = predict_cell(raw_b[ids["P0L"]], par, s_pred)
    d_parts = cs.stack(raw_b[ids["D"]])["parts"]
    p_tok = float(np.mean([np.nansum(x["busy"]) for x in sb["parts"]]))
    d_tok = float(np.mean([np.nansum(x["busy"]) for x in d_parts]))
    meas_diff = p_tok - d_tok
    meas_ov_tok = float(np.mean([np.nansum(x["overlap"]) for x in sb["parts"]]))
    return {"available": True, "params": par, "rows": int(len(yb)), "s_pred_run_a": s_pred, "s_measured_overlap_run_a": s_meas,
            "overlap_row_r_run_b": pearson(pb_flat, sb["overlap"]),
            "overlap_ratio_run_b": tok_mean_sum(pred_b) / meas_ov_tok if meas_ov_tok > 0 else float("nan"),
            "overlap_per_token_ms": {"v1_predicted": tok_mean_sum(pred_b), "v2_closed_loop": tok_mean_sum(closed_b), "measured": meas_ov_tok},
            "v2_over_v1": tok_mean_sum(closed_b) / tok_mean_sum(pred_b) if tok_mean_sum(pred_b) > 0 else float("nan"),
            "held_out_r2_v1": cs.r2(s_pred * pb_flat, yb), "held_out_r2_v3": cs.r2(s_meas * pb_flat, yb),
            "token_busy": {"measured_diff": meas_diff, "v1_predicted": tok_mean_sum(pred_b, s_pred), "v1_error": tok_mean_sum(pred_b, s_pred) - meas_diff,
                           "v3_predicted": tok_mean_sum(pred_b, s_meas), "v3_error": tok_mean_sum(pred_b, s_meas) - meas_diff}}


def hidden_share(raw_a: dict, raw_b: dict, c: float, s: float, par: dict) -> dict:
    """Row model of `compute_slowdown_fit.hidden_share` with the term's overlap replaced by V1's predicted overlap on run B."""
    pa, pb = raw_a.get(rt.cell_id(3, c, "P0L")), raw_b.get(rt.cell_id(3, c, "P0L"))
    db_cell = raw_b.get(rt.cell_id(3, c, "D"))
    l4 = [raw_b.get(rt.cell_id(4, c, role)) for role in ("N0", "N1", "D", "P0L")]
    if not (pa and pb and db_cell and all(l4)):
        return {"available": False}
    ca, _ = rt.fit_c(rt.rows_of(pa), c)
    rb, dbr = rt.rows_of(pb), rt.rows_of(db_cell)
    d = rt.derived(rb)
    qpred = np.maximum(0.0, d["ahead"] * ca - c)
    c_dem = rt.per_miss(dbr)["d_per_miss"]
    toks_b = max(1, len(pb["rows"]) - 1)
    ov_b = np.concatenate(predict_cell(pb, par))
    e_p0 = float(np.sum(qpred + d["n_dem"] * c_dem) / toks_b)
    e_p1 = e_p0 + s * float(np.nansum(ov_b)) / toks_b
    e_d = float(np.sum(rt.derived(dbr)["n_dem"] * c_dem) / max(1, len(db_cell["rows"]) - 1))
    floor = (rt.p50(l4[0]["wall_ms"][1:]) + rt.p50(l4[1]["wall_ms"][1:])) / 2
    ed, ep = rt.p50(l4[2]["wall_ms"][1:]) - floor, rt.p50(l4[3]["wall_ms"][1:]) - floor
    h_meas = 1.0 - ep / ed if ed > 0 else None
    h1 = 1.0 - e_p1 / e_d
    return {"available": True, "H_predicted_overlap": h1, "H_measured_L4": h_meas,
            "error_predicted": abs(h1 - h_meas) if h_meas is not None else None}


def level6_check(raw_a: dict, raw_b: dict, c: float, s_pred: float) -> dict:
    """PO-7: level-6 parameters from run A, the level-3 predicted-overlap slope, against run B's level-6 `P0L` minus `D` busy per token."""
    ids = {role: rt.cell_id(6, c, role) for role in ("D", "P0L")}
    if not all(i in raw_a and i in raw_b and raw_a[i].get("rows") and raw_b[i].get("rows") for i in ids.values()):
        return {"available": False}
    par = fit_params(raw_a[ids["P0L"]], raw_a[ids["D"]])
    p6, d6 = cs.stack(raw_b[ids["P0L"]]), cs.stack(raw_b[ids["D"]])
    meas = float(np.mean([np.nansum(x["busy"]) for x in p6["parts"]]) - np.mean([np.nansum(x["busy"]) for x in d6["parts"]]))
    pred = tok_mean_sum(predict_cell(raw_b[ids["P0L"]], par), s_pred)
    return {"available": True, "params": par, "measured_diff": meas, "predicted": pred, "error": pred - meas}


def verdicts(res: dict) -> dict:
    """Numbers against the registered intervals and falsifiers only (record section 4); no interpretation."""
    v = {}

    def put(key, ok, fal):
        v[key] = "in interval" if ok else ("past falsifier" if fal else "outside interval")

    meas_slope = {1.0: 0.0634, 0.3: 0.1053}
    conditional_error = {1.0: 0.045, 0.3: 0.008}
    for c in (1.0, 0.3):
        e = res["level3"].get(c, {})
        if not e.get("available"):
            continue
        r = e["overlap_row_r_run_b"]
        put(f"PO-1@{c}", r >= 0.85, not r >= 0.60)
        q = e["overlap_ratio_run_b"]
        put(f"PO-2@{c}", 0.85 <= q <= 1.05, not 0.65 <= q <= 1.25)
        dev = abs(e["s_pred_run_a"] - meas_slope[c]) / meas_slope[c]
        put(f"PO-3@{c}", dev <= 0.25, dev > 0.60)
        err = abs(e["token_busy"]["v1_error"])
        put(f"PO-4@{c}", err <= 0.10, err > 0.25)
        h = res["hidden"].get(c, {})
        if h.get("available") and h["error_predicted"] is not None:
            put(f"PO-5@{c}", h["error_predicted"] <= 0.05, h["error_predicted"] > 0.08)
            up = h["error_predicted"] - conditional_error[c]
            put(f"PO-6@{c}", up <= 0.03, up > 0.06)
        l6 = res["level6"].get(c, {})
        if l6.get("available"):
            put(f"PO-7@{c}", abs(l6["error"]) <= 0.15, abs(l6["error"]) > 0.40)
        dv = abs(e["v2_over_v1"] - 1.0)
        put(f"PO-8@{c}", dv <= 0.08, dv > 0.20)
        e3 = abs(e["token_busy"]["v3_error"])
        put(f"PO-9@{c}", e3 <= 0.15, e3 > 0.40)
    return v


def score(raw_a: dict, raw_b: dict) -> dict:
    res = {"level3": {}, "hidden": {}, "level6": {}}
    for c in (1.0, 0.3):
        e = score_level(raw_a, raw_b, c, 3)
        res["level3"][c] = e
        if e["available"]:
            res["hidden"][c] = hidden_share(raw_a, raw_b, c, e["s_pred_run_a"], e["params"])
            res["level6"][c] = level6_check(raw_a, raw_b, c, e["s_pred_run_a"])
        else:
            res["hidden"][c], res["level6"][c] = {"available": False}, {"available": False}
    res["verdict"] = verdicts(res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--fix", help="the f-fix run directory: its cells replace the voided C = 1.0 level-6 and level-4 cells of both runs")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    a, b = rt.load_samples(args.run_a), rt.load_samples(args.run_b)
    if args.fix:
        fx = rt.load_samples(args.fix)
        a, b = rt.merge_fix(a, fx), rt.merge_fix(b, fx)
    res = score(a, b)
    text = json.dumps(res, indent=2, sort_keys=True, default=float)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
