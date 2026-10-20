"""
Low layer time (Mac only; record `docs/CUDA-LOW-C-RECORD.md`, predictions L-1 to L-10): the compute-slowdown slope at C = 0.15 ms.

Reads the raw samples of two runs of the `h-full` profile (run A fits, run B scores) and prints each registered quantity beside its
interval and falsifier, plus the shapes the record derived through the slopes at C = 1.0, 0.6 and 0.3 ms. It judges nothing for the
writer: the verdict column says only whether a number is inside the registered interval, past its falsifier, or between.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/low_c_score.py RUN_A_DIR RUN_B_DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import predicted_overlap_fit as po  # noqa: E402
import row_timing as rt  # noqa: E402
import third_c_score as tc  # noqa: E402

C = 0.15
# Slopes of the three layer times already measured (record section 2): measured overlap, and V1's predicted overlap (C = 0.6 is run A).
KNOWN_MEASURED = {1.0: 0.0634, 0.6: 0.0765, 0.3: 0.1053}
KNOWN_V1 = {1.0: 0.0624, 0.6: 0.0767, 0.3: 0.1122}


def affine_fit(points: dict[float, float]) -> tuple[float, float]:
    """Least-squares `s = a + b / C`."""
    cs_ = np.array(sorted(points))
    a, b = np.linalg.lstsq(np.vstack([np.ones(len(cs_)), 1.0 / cs_]).T, np.array([points[c] for c in sorted(points)]), rcond=None)[0]
    return float(a), float(b)


def shapes(c: float = C, points: dict[float, float] | None = None) -> dict:
    """Affine in 1/C, power law in 1/C and linear in C, each least-squares through the given points, evaluated at `c`."""
    pts = points or KNOWN_MEASURED
    cs_ = np.array(sorted(pts))
    s = np.array([pts[x] for x in sorted(pts)])
    a, b = affine_fit(pts)
    pw = np.polyfit(np.log(1.0 / cs_), np.log(s), 1)
    ln = np.polyfit(cs_, s, 1)
    return {"affine_in_inverse_c": a + b / c, "power_law": float(np.exp(np.polyval(pw, np.log(1.0 / c)))), "power_exponent": float(pw[0]),
            "linear_in_c": float(np.polyval(ln, c)), "affine_a": a, "affine_b": b}


def extrapolated_v1_slope(c: float = C) -> float:
    a, b = affine_fit(KNOWN_V1)
    return a + b / c


def refit_with(slope: float, c: float = C) -> dict:
    """The four-point refit once the new slope is in: affine in 1/C and its largest residual."""
    pts = dict(KNOWN_MEASURED)
    pts[c] = slope
    a, b = affine_fit(pts)
    res = {x: pts[x] - (a + b / x) for x in sorted(pts)}
    return {"a": a, "b": b, "residuals": res, "max_abs_residual": max(abs(v) for v in res.values())}


def gate_summary(manifest: dict | None) -> dict:
    """The gates of a run manifest, when given (prediction L-9 reads `calibration` and `floor`)."""
    g = (manifest or {}).get("gates", {})
    out = {k: bool(v) for k, v in g.items() if k != "failed_cells"}
    if manifest:
        out["no_failed_cells"] = not g.get("failed_cells")
    return out


def demand_exposure(raw: dict, c: float = C) -> float | None:
    """Level 4: `D` minus the mean of the two `N` controls, median wall per token (ms)."""
    cells = [raw.get(rt.cell_id(4, c, role)) for role in ("N0", "N1", "D")]
    if not all(cells):
        return None
    floor = (rt.p50(cells[0]["wall_ms"][1:]) + rt.p50(cells[1]["wall_ms"][1:])) / 2
    return float(rt.p50(cells[2]["wall_ms"][1:]) - floor)


def score(raw_a: dict, raw_b: dict, c: float = C, manifest_a: dict | None = None, manifest_b: dict | None = None) -> dict:
    out: dict = {"c": c, "shapes": shapes(c), "shapes_v1_basis": shapes(c, KNOWN_V1), "extrapolated_v1_slope": extrapolated_v1_slope(c),
                 "gates_run_a": gate_summary(manifest_a), "gates_run_b": gate_summary(manifest_b)}
    e = po.score_level(raw_a, raw_b, c, 3)
    out["level3"] = e
    if not e.get("available"):
        return out
    ids = {role: rt.cell_id(3, c, role) for role in ("D", "P0L")}
    out["slope_measured_run_a"] = tc.slope_of(raw_a[ids["P0L"]], raw_a[ids["D"]])
    out["slope_measured_run_b"] = tc.slope_of(raw_b[ids["P0L"]], raw_b[ids["D"]])
    out["slope_v1_run_a"] = e["s_pred_run_a"]
    out["refit"] = refit_with(out["slope_measured_run_a"], c)
    out["hidden"] = po.hidden_share(raw_a, raw_b, c, e["s_pred_run_a"], e["params"])
    out["level6"] = po.level6_check(raw_a, raw_b, c, e["s_pred_run_a"])
    pred_b = po.predict_cell(raw_b[ids["P0L"]], e["params"])
    s_x = out["extrapolated_v1_slope"]
    out["extrapolated"] = {"slope": s_x, "token_busy_predicted": po.tok_mean_sum(pred_b, s_x),
                           "error": po.tok_mean_sum(pred_b, s_x) - e["token_busy"]["measured_diff"]}
    out["demand_exposure_ms"] = {"run_a": demand_exposure(raw_a, c), "run_b": demand_exposure(raw_b, c)}
    out["verdict"] = verdicts(out)
    return out


def verdicts(res: dict) -> dict:
    """Numbers against the registered intervals and falsifiers only (record section 4)."""
    v: dict = {}

    def put(key, ok, fal):
        v[key] = "in interval" if ok else ("past falsifier" if fal else "outside interval")

    gates = {**res.get("gates_run_a", {}), **{f"b.{k}": x for k, x in res.get("gates_run_b", {}).items()}}
    if gates:
        ok = all(gates.get(k, True) for k in gates if k.split(".")[-1] in ("calibration", "floor", "no_failed_cells"))
        put("L-9", ok, not ok)
    s = res["slope_measured_run_a"]
    put("L-1", 0.140 <= s <= 0.185, not 0.110 <= s <= 0.220)
    put("L-2", s >= 0.153, s < 0.125)
    sb = res["slope_measured_run_b"]
    dev = abs(sb - s) / s if s else float("inf")
    put("L-3", dev <= 0.15, dev > 0.30)
    h = res["hidden"]
    if h.get("available") and h.get("H_measured_L4") is not None:
        hm = h["H_measured_L4"]
        put("L-4", -0.25 <= hm <= 0.05, hm > 0.12 or hm < -0.40)
        put("L-7", h["error_predicted"] <= 0.07, h["error_predicted"] > 0.12)
    e = res["level3"]
    put("L-5", e["overlap_row_r_run_b"] >= 0.85, not e["overlap_row_r_run_b"] >= 0.60)
    q = e["overlap_ratio_run_b"]
    put("L-6", 0.85 <= q <= 1.05, not 0.65 <= q <= 1.25)
    err = abs(res["extrapolated"]["error"])
    put("L-8", err <= 0.30, err > 0.80)
    d = res["demand_exposure_ms"]["run_a"]
    if d is not None:
        put("L-10", 9.0 <= d <= 11.0, not 8.0 <= d <= 12.5)
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    def manifest(d):
        p = Path(d) / "run-manifest.json"
        return json.loads(p.read_text()) if p.exists() else None

    res = score(rt.load_samples(args.run_a), rt.load_samples(args.run_b), C, manifest(args.run_a), manifest(args.run_b))
    text = json.dumps(res, indent=2, sort_keys=True, default=float)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
