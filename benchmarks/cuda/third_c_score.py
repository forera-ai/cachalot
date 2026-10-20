"""
Third layer time (Mac only; record `docs/CUDA-THIRD-C-RECORD.md`, predictions G-1 to G-9): the compute-slowdown slope at C = 0.6 ms.

Reads the raw samples of two runs of the `g-full` profile (run A fits, run B scores) and prints each registered quantity beside its
interval and falsifier, plus the three shapes the record derived through the slopes at C = 1.0 and 0.3 ms. It judges nothing for the
writer: the verdict column says only whether a number is inside the registered interval, past its falsifier, or between.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/third_c_score.py RUN_A_DIR RUN_B_DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compute_slowdown_fit as cs  # noqa: E402
import predicted_overlap_fit as po  # noqa: E402
import row_timing as rt  # noqa: E402

C = 0.6
# Slopes of the two layer times already measured (record section 2): measured overlap, and V1's predicted overlap.
KNOWN_MEASURED = {1.0: 0.0634, 0.3: 0.1053}
KNOWN_V1 = {1.0: 0.0624, 0.3: 0.1122}


def affine_in_inverse_c(points: dict[float, float]) -> tuple[float, float]:
    """`s = a + b / C` through two points."""
    (c1, s1), (c2, s2) = sorted(points.items(), reverse=True)
    b = (s2 - s1) / (1.0 / c2 - 1.0 / c1)
    return s1 - b / c1, b


def shapes(c: float = C, points: dict[float, float] | None = None) -> dict:
    """The three smooth descriptions of the slope through the two known points, evaluated at `c` (record section 2)."""
    pts = points or KNOWN_MEASURED
    (c1, s1), (c2, s2) = sorted(pts.items(), reverse=True)
    a, b = affine_in_inverse_c(pts)
    k = (s2 - s1) / (c2 - c1)
    expo = np.log(s2 / s1) / np.log(c1 / c2)
    return {"linear_in_c": s2 + k * (c - c2), "affine_in_inverse_c": a + b / c, "power_law": s1 * (c1 / c) ** expo, "power_exponent": float(expo)}


def interpolated_v1_slope(c: float = C) -> float:
    a, b = affine_in_inverse_c(KNOWN_V1)
    return a + b / c


def slope_of(cell_p0l: dict, cell_d: dict) -> float:
    """Slope through the origin of the `P0L` row excess busy time on the measured overlap (level 3)."""
    sp = cs.stack(cell_p0l)
    return cs.fit_through_origin(sp["overlap"], sp["busy"] - cs.base_busy(cell_d))


def score(raw_a: dict, raw_b: dict, c: float = C) -> dict:
    out: dict = {"c": c, "shapes": shapes(c), "shapes_v1_basis": shapes(c, KNOWN_V1), "interpolated_v1_slope": interpolated_v1_slope(c)}
    e = po.score_level(raw_a, raw_b, c, 3)
    out["level3"] = e
    if not e.get("available"):
        return out
    ids = {role: rt.cell_id(3, c, role) for role in ("D", "P0L")}
    out["slope_measured_run_a"] = slope_of(raw_a[ids["P0L"]], raw_a[ids["D"]])
    out["slope_measured_run_b"] = slope_of(raw_b[ids["P0L"]], raw_b[ids["D"]])
    out["slope_v1_run_a"] = e["s_pred_run_a"]
    out["hidden"] = po.hidden_share(raw_a, raw_b, c, e["s_pred_run_a"], e["params"])
    out["level6"] = po.level6_check(raw_a, raw_b, c, e["s_pred_run_a"])
    pred_b = po.predict_cell(raw_b[ids["P0L"]], e["params"])
    s_i = out["interpolated_v1_slope"]
    out["interpolated"] = {"slope": s_i, "token_busy_predicted": po.tok_mean_sum(pred_b, s_i),
                           "error": po.tok_mean_sum(pred_b, s_i) - e["token_busy"]["measured_diff"]}
    out["verdict"] = verdicts(out)
    return out


def verdicts(res: dict) -> dict:
    """Numbers against the registered intervals and falsifiers only (record section 4)."""
    v: dict = {}

    def put(key, ok, fal):
        v[key] = "in interval" if ok else ("past falsifier" if fal else "outside interval")

    s = res["slope_measured_run_a"]
    put("G-1", 0.070 <= s <= 0.095, not 0.060 <= s <= 0.110)
    put("G-2", s < 0.085, s >= 0.091)
    sb = res["slope_measured_run_b"]
    dev = abs(sb - s) / s if s else float("inf")
    put("G-3", dev <= 0.15, dev > 0.30)
    h = res["hidden"]
    if h.get("available") and h.get("H_measured_L4") is not None:
        put("G-4", 0.37 <= h["H_measured_L4"] <= 0.45, not 0.33 <= h["H_measured_L4"] <= 0.48)
        put("G-7", h["error_predicted"] <= 0.05, h["error_predicted"] > 0.08)
    e = res["level3"]
    put("G-5", e["overlap_row_r_run_b"] >= 0.85, not e["overlap_row_r_run_b"] >= 0.60)
    q = e["overlap_ratio_run_b"]
    put("G-6", 0.85 <= q <= 1.05, not 0.65 <= q <= 1.25)
    err = abs(res["interpolated"]["error"])
    put("G-8", err <= 0.15, err > 0.40)
    l6 = res["level6"]
    if l6.get("available"):
        put("G-9", abs(l6["error"]) <= 0.15, abs(l6["error"]) > 0.40)
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    res = score(rt.load_samples(args.run_a), rt.load_samples(args.run_b))
    text = json.dumps(res, indent=2, sort_keys=True, default=float)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
