"""
Compute-slowdown term for the row model (Mac only; record `docs/CUDA-SLOWDOWN-RECORD.md`, predictions CS-1 to CS-7).

Reads the raw samples of the two full row-timing runs (A and B) and the fix run, as `row_timing.py` does, fits the extra busy time of
a `P0L` compute row against the time its compute window overlaps prefetch copy batches on run A, and scores the fit on run B and on
the level-6 cells. It judges nothing for the writer: the verdict column says only whether a number is inside the registered interval,
past its falsifier, or between.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/compute_slowdown_fit.py RUN_A_DIR RUN_B_DIR --fix FIX_DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import row_timing as rt  # noqa: E402

RF = rt.RF
BOOT = 1000
SEED = 20261020


def tokens_of(cell: dict) -> list[np.ndarray]:
    """One array (rows x ROW_FIELDS) per decode token; None becomes NaN; the warm-up token 0 is dropped."""
    out = []
    for toks in cell["rows"][1:]:
        rows = [[np.nan if v is None else v for v in r] for r in (toks or [])]
        out.append(np.array(rows, dtype=float).reshape(-1, len(RF)))
    return out


def row_terms(tok: np.ndarray) -> dict:
    """Per-row busy time, the time the row's compute window overlaps prefetch batches, and the copy-count counterpart."""
    start, end = tok[:, RF["start"]], tok[:, RF["end"]]
    pa, pb = tok[:, RF["pf_a"]], tok[:, RF["pf_b"]]
    n_pf = np.nan_to_num(tok[:, RF["n_pf"]])
    has = (n_pf > 0) & np.isfinite(pa) & np.isfinite(pb)
    overlap = np.zeros(len(tok))
    for i in range(len(tok)):
        if not (np.isfinite(start[i]) and np.isfinite(end[i])):
            continue
        lo = np.maximum(start[i], pa[has])
        hi = np.minimum(end[i], pb[has])
        overlap[i] = float(np.sum(np.maximum(0.0, hi - lo)))
    prev = np.concatenate([[0.0], n_pf[:-1]])
    return {"busy": end - start, "overlap": overlap, "count": n_pf + prev}


def stack(cell: dict) -> dict:
    parts = [row_terms(t) for t in tokens_of(cell) if len(t)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} | {"tok": [len(p["busy"]) for p in parts],
                                                                         "parts": parts}


def base_busy(d_cell: dict) -> float:
    s = stack(d_cell)
    b = s["busy"][np.isfinite(s["busy"])]
    return float(np.median(b))


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    den = float(np.sum(x[m] ** 2))
    return float(np.sum(x[m] * y[m]) / den) if den > 0 else 0.0


def sse(pred: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(y)
    return float(np.sum((pred[m] - y[m]) ** 2))


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(y)
    tot = float(np.sum((y[m] - np.mean(y[m])) ** 2))
    return 1.0 - sse(pred, y) / tot if tot > 0 else float("nan")


def boot_slope(parts: list[dict], base: float, key: str, n: int = BOOT) -> list[float]:
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(parts), len(parts))
        x = np.concatenate([parts[i][key] for i in idx])
        y = np.concatenate([parts[i]["busy"] - base for i in idx])
        out.append(fit_through_origin(x, y))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def score_level(raw_a: dict, raw_b: dict, c: float, level: int = 3) -> dict:
    ids = {role: rt.cell_id(level, c, role) for role in ("D", "P0L")}
    if not all(i in raw_a and i in raw_b and raw_a[i].get("rows") and raw_b[i].get("rows") for i in ids.values()):
        return {"available": False}
    pa, pb = stack(raw_a[ids["P0L"]]), stack(raw_b[ids["P0L"]])
    base_a, base_b = base_busy(raw_a[ids["D"]]), base_busy(raw_b[ids["D"]])
    ya, yb = pa["busy"] - base_a, pb["busy"] - base_b
    models = {}
    for name, key in (("M1_overlap", "overlap"), ("M2_count", "count")):
        s_a = fit_through_origin(pa[key], ya)
        s_b = fit_through_origin(pb[key], yb)
        models[name] = {"slope_run_a": s_a, "slope_run_b": s_b, "slope_diff_rel": abs(s_a - s_b) / max(abs(s_a), 1e-12),
                        "ci95_run_a": boot_slope(pa["parts"], base_a, key), "held_out_r2": r2(s_a * pb[key], yb),
                        "held_out_sse": sse(s_a * pb[key], yb)}
    f_a = float(np.nanmean(ya))
    models["M3_flat"] = {"constant_run_a": f_a, "held_out_r2": r2(np.full(len(yb), f_a), yb), "held_out_sse": sse(np.full(len(yb), f_a), yb)}
    models["M0_none"] = {"held_out_r2": r2(np.zeros(len(yb)), yb), "held_out_sse": sse(np.zeros(len(yb)), yb)}
    # the token total: the measured P0L - D busy time per token on run B, and the M1 prediction from run A's slope
    d_parts = stack(raw_b[ids["D"]])["parts"]
    p_tok = float(np.mean([np.nansum(x["busy"]) for x in pb["parts"]]))
    d_tok = float(np.mean([np.nansum(x["busy"]) for x in d_parts]))
    pred_tok = float(np.mean([np.nansum(models["M1_overlap"]["slope_run_a"] * x["overlap"]) for x in pb["parts"]]))
    return {"available": True, "rows": int(len(yb)), "base_busy_ms": {"run_a": base_a, "run_b": base_b},
            "models": models, "token_busy": {"p0l": p_tok, "d": d_tok, "measured_diff": p_tok - d_tok, "m1_predicted": pred_tok,
                                              "error": pred_tok - (p_tok - d_tok)}}


def hidden_share(raw_a: dict, raw_b: dict, c: float) -> dict:
    """Row model of `row_timing.score_rt7` with the M1 slowdown added to the P0L token cost, against the level-4 measured share."""
    pa, pb = raw_a.get(rt.cell_id(3, c, "P0L")), raw_b.get(rt.cell_id(3, c, "P0L"))
    db_cell, da_cell = raw_b.get(rt.cell_id(3, c, "D")), raw_a.get(rt.cell_id(3, c, "D"))
    l4 = [raw_b.get(rt.cell_id(4, c, role)) for role in ("N0", "N1", "D", "P0L")]
    if not (pa and pb and db_cell and da_cell and all(l4)):
        return {"available": False}
    ca, _ = rt.fit_c(rt.rows_of(pa), c)
    sa = stack(pa)
    s = fit_through_origin(sa["overlap"], sa["busy"] - base_busy(da_cell))
    rb, dbr = rt.rows_of(pb), rt.rows_of(db_cell)
    d = rt.derived(rb)
    qpred = np.maximum(0.0, d["ahead"] * ca - c)
    c_dem = rt.per_miss(dbr)["d_per_miss"]
    toks_b = max(1, len(pb["rows"]) - 1)
    ov_b = stack(pb)["overlap"]
    e_p0 = float(np.sum(qpred + d["n_dem"] * c_dem) / toks_b)
    e_p1 = e_p0 + s * float(np.nansum(ov_b)) / toks_b
    e_d = float(np.sum(rt.derived(dbr)["n_dem"] * c_dem) / max(1, len(db_cell["rows"]) - 1))
    floor = (rt.p50(l4[0]["wall_ms"][1:]) + rt.p50(l4[1]["wall_ms"][1:])) / 2
    ed, ep = rt.p50(l4[2]["wall_ms"][1:]) - floor, rt.p50(l4[3]["wall_ms"][1:]) - floor
    h_meas = 1.0 - ep / ed if ed > 0 else None
    h0, h1 = 1.0 - e_p0 / e_d, 1.0 - e_p1 / e_d
    return {"available": True, "slope_run_a": s, "H_without_term": h0, "H_with_term": h1, "H_measured_L4": h_meas,
            "error_without": abs(h0 - h_meas) if h_meas is not None else None, "error_with": abs(h1 - h_meas) if h_meas is not None else None}


def level6_check(raw_a: dict, raw_b: dict, c: float) -> dict:
    """CS-7: the level-3 slope from run A applied to the level-6 P0L rows against the level-6 measured P0L - D busy time per token."""
    l3 = score_level(raw_a, raw_b, c, 3)
    ids = {role: rt.cell_id(6, c, role) for role in ("D", "P0L")}
    if not l3["available"] or not all(i in raw_b and raw_b[i].get("rows") for i in ids.values()):
        return {"available": False}
    p6, d6 = stack(raw_b[ids["P0L"]]), stack(raw_b[ids["D"]])
    meas = float(np.mean([np.nansum(x["busy"]) for x in p6["parts"]]) - np.mean([np.nansum(x["busy"]) for x in d6["parts"]]))
    pred = float(np.mean([np.nansum(l3["models"]["M1_overlap"]["slope_run_a"] * x["overlap"]) for x in p6["parts"]]))
    return {"available": True, "measured_diff": meas, "predicted": pred, "error": pred - meas}


def verdicts(res: dict) -> dict:
    """Numbers against the registered intervals and falsifiers only (record section 4); no interpretation."""
    v = {}
    lo_hi = {1.0: (0.03, 0.06), 0.3: (0.05, 0.11)}
    for c in (1.0, 0.3):
        e = res["level3"].get(c, {})
        if not e.get("available"):
            continue
        m1, m2 = e["models"]["M1_overlap"], e["models"]["M2_count"]
        s = m1["slope_run_a"]
        lo, hi = lo_hi[c]
        v[f"CS-1@{c}"] = "in interval" if lo <= s <= hi else ("past falsifier" if not 0.01 <= s <= 0.15 else "outside interval")
        d = m1["slope_diff_rel"]
        v[f"CS-2@{c}"] = "in interval" if d <= 0.15 else ("past falsifier" if d > 0.40 else "outside interval")
        r = m1["held_out_r2"]
        v[f"CS-3@{c}"] = "in interval" if r >= 0.30 else ("past falsifier" if r < 0.10 else "outside interval")
        gap = 1.0 - m1["held_out_sse"] / m2["held_out_sse"] if m2["held_out_sse"] > 0 else float("nan")
        v[f"CS-4@{c}"] = "past falsifier" if m2["held_out_sse"] < m1["held_out_sse"] else ("in interval" if gap >= 0.10 else "outside interval")
        err = abs(e["token_busy"]["error"])
        v[f"CS-5@{c}"] = "in interval" if err <= 0.10 else ("past falsifier" if err > 0.25 else "outside interval")
        h = res["hidden"].get(c, {})
        if h.get("available") and h["error_with"] is not None:
            v[f"CS-6@{c}"] = "in interval" if h["error_with"] <= 0.05 else ("past falsifier" if h["error_with"] > 0.08 else "outside interval")
        l6 = res["level6"].get(c, {})
        if l6.get("available"):
            ae = abs(l6["error"])
            v[f"CS-7@{c}"] = "in interval" if ae <= 0.15 else ("past falsifier" if ae > 0.40 else "outside interval")
    return v


def score(raw_a: dict, raw_b: dict) -> dict:
    res = {"level3": {}, "hidden": {}, "level6": {}}
    for c in (1.0, 0.3):
        res["level3"][c] = score_level(raw_a, raw_b, c, 3)
        res["hidden"][c] = hidden_share(raw_a, raw_b, c)
        res["level6"][c] = level6_check(raw_a, raw_b, c)
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
