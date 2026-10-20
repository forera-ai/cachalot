"""
Scoring of the per-row and per-copy timing session (Mac only; record `docs/CUDA-ROW-TIMING-RECORD.md`, predictions RT-1 to RT-7).

Reads the `raw-samples.json` of two runs of `overlap_arena.py run --cells f-full` (run A and run B, a replicate on the same
instance) and prints the quantities of the record's section 4 with the interval each prediction registered. It judges nothing for
the writer: the verdict column says only whether a number is inside the registered interval, outside it, or past its falsifier;
the results file explains what that means.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/row_timing.py RUN_A_DIR RUN_B_DIR [--json OUT]

A run directory holds `raw-samples.json` as the harness wrote it. Cell ids are `p17.L<level>.C<c>.<role>` with the roles `N0`, `D`,
`P0L`, `N1`. Times are milliseconds since the token's first event (`overlap_arena.ROW_FIELDS`, `COPY_FIELDS`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import overlap_arena as oa  # noqa: E402

RF = {name: i for i, name in enumerate(oa.ROW_FIELDS)}
Q_THRESHOLD_MS = 0.02


def load_samples(run_dir) -> dict:
    return json.loads((Path(run_dir) / "raw-samples.json").read_text())["samples"]


def cell_id(level: int, c: float, role: str) -> str:
    return f"p17.L{level}.C{c}.{role}"


def rows_of(cell: dict) -> np.ndarray:
    """All decode rows of a cell as an array (rows x ROW_FIELDS); None becomes NaN; token 0 (the warm-up) is dropped."""
    out = []
    for toks in cell["rows"][1:]:
        for r in toks or []:
            out.append([np.nan if v is None else v for v in r])
    return np.array(out, dtype=float)


def copies_of(cell: dict) -> np.ndarray:
    out = []
    for ti, cs in enumerate(cell["copies"][1:], start=1):
        for r in cs or []:
            out.append([ti, *r])
    return np.array(out, dtype=float)                          # token, row, kind, k, t_done, dur


def derived(rows: np.ndarray) -> dict:
    """Per-row quantities: stall (compute waited), q (demand queue wait behind earlier copies), d (own demand batch), busy."""
    pre, start, end = rows[:, RF["pre"]], rows[:, RF["start"]], rows[:, RF["end"]]
    dem_a, dem_b = rows[:, RF["dem_a"]], rows[:, RF["dem_b"]]
    return {"stall": start - pre, "q": np.maximum(0.0, dem_a - pre), "d": dem_b - dem_a, "busy": end - start,
            "n_dem": rows[:, RF["n_dem"]], "n_pf": rows[:, RF["n_pf"]], "ahead": rows[:, RF["n_pf_ahead"]]}


def per_miss(rows: np.ndarray) -> dict:
    d = derived(rows)
    m = d["n_dem"] > 0
    if not m.any():
        return {"rows": 0}
    return {"rows": int(m.sum()), "stall_per_miss": float(np.mean(d["stall"][m] / d["n_dem"][m])),
            "q_per_miss": float(np.mean(d["q"][m] / d["n_dem"][m])), "d_per_miss": float(np.mean(d["d"][m] / d["n_dem"][m])),
            "q_share_of_rows": float(np.mean(d["q"][m] > Q_THRESHOLD_MS))}


def score_rt1_rt2(raw: dict, c: float = 1.0) -> dict:
    cell = raw.get(cell_id(6, c, "D"))
    if not cell or not cell.get("copies"):
        return {"available": False}
    cp = copies_of(cell)
    dem = cp[cp[:, 2] == 0]
    diffs = []
    for (_, _), grp in _groups(dem, (0, 1)):
        if len(grp) >= 2:
            diffs.append(grp[0, 5] - float(np.median(grp[1:, 5])))
    return {"available": True, "median_demand_copy_ms": float(np.median(dem[:, 5])), "p05": float(np.percentile(dem[:, 5], 5)),
            "p95": float(np.percentile(dem[:, 5], 95)), "first_minus_later_ms": float(np.median(diffs)) if diffs else None,
            "batches_ge2": len(diffs)}


def _groups(a: np.ndarray, cols: tuple[int, ...]):
    keys = {}
    for i, row in enumerate(a):
        keys.setdefault(tuple(row[list(cols)]), []).append(i)
    for k, idx in keys.items():
        grp = a[idx]
        yield k, grp[np.argsort(grp[:, 4])]


def score_rt3_rt4(raw: dict, c: float, level: int) -> dict:
    p, d = raw.get(cell_id(level, c, "P0L")), raw.get(cell_id(level, c, "D"))
    if not p or not d or not p.get("rows") or not d.get("rows"):
        return {"available": False}
    mp, md = per_miss(rows_of(p)), per_miss(rows_of(d))
    excess = mp["stall_per_miss"] - md["stall_per_miss"]
    return {"available": True, "p0l": mp, "d": md, "excess_ms": excess,
            "excess_ratio": excess / md["stall_per_miss"] if md["stall_per_miss"] else None,
            "q_share_of_excess": (mp["q_per_miss"] - md["q_per_miss"]) / excess if excess > 1e-9 else None}


def score_rt5(raw: dict, c: float) -> dict:
    cell = raw.get(cell_id(6, c, "P0L"))
    if not cell or not cell.get("copies"):
        return {"available": False}
    rows = [r for r in cell["rows"][1:]]
    cps = cell["copies"][1:]
    under, after = [], []
    for toks, cs in zip(rows, cps, strict=False):
        for row, kind, k, t, dur in cs or []:
            if kind != 1:
                continue
            r = toks[int(row)]
            (under if r[RF["start"]] < t <= r[RF["end"]] else after).append(dur)
    if not under or not after:
        return {"available": False}
    return {"available": True, "under_compute_ms": float(np.median(under)), "after_end_ms": float(np.median(after)),
            "slowdown": float(np.median(under) / np.median(after) - 1.0), "n_under": len(under), "n_after": len(after)}


def p50(xs) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2]


def score_rt6(raw: dict, c: float) -> dict:
    out = {}
    for arm in ("D", "P0L"):
        a, b = raw.get(cell_id(3, c, arm)), raw.get(cell_id(6, c, arm))
        if a and b:
            out[arm] = {"wall_L3": p50(a["wall_ms"][1:]), "wall_L6": p50(b["wall_ms"][1:]),
                        "excess_ms": p50(b["wall_ms"][1:]) - p50(a["wall_ms"][1:]), "closure_L6": b.get("closure", {}).get("share")}
    return out


def fit_c(rows: np.ndarray, window_ms: float) -> tuple[float, float]:
    """Grid fit of the per-copy cost `c` in q_pred = max(0, n_pf_ahead * c - window) against the measured q (least squares)."""
    d = derived(rows)
    best = (None, np.inf)
    for c in np.arange(0.05, 0.60, 0.005):
        err = float(np.mean((np.maximum(0.0, d["ahead"] * c - window_ms) - np.nan_to_num(d["q"])) ** 2))
        if err < best[1]:
            best = (float(c), err)
    return best


def score_rt7(raw_a: dict, raw_b: dict) -> dict:
    out = {}
    for c in (1.0, 0.3):
        pa, pb = raw_a.get(cell_id(3, c, "P0L")), raw_b.get(cell_id(3, c, "P0L"))
        db = raw_b.get(cell_id(3, c, "D"))
        if not (pa and pb and db and pa.get("rows") and pb.get("rows") and db.get("rows")):
            continue
        ca, _ = fit_c(rows_of(pa), c)
        rb, dbr = rows_of(pb), rows_of(db)
        d = derived(rb)
        qpred = np.maximum(0.0, d["ahead"] * ca - c)
        qmeas = np.nan_to_num(d["q"])
        r = float(np.corrcoef(qpred, qmeas)[0, 1]) if qpred.std() > 0 and qmeas.std() > 0 else None
        c_dem = per_miss(dbr)["d_per_miss"]
        e_p = float(np.sum(qpred + d["n_dem"] * c_dem) / max(1, len(pb["rows"]) - 1))
        e_d = float(np.sum(derived(dbr)["n_dem"] * c_dem) / max(1, len(db["rows"]) - 1))
        h_model = 1.0 - e_p / e_d if e_d > 0 else None
        h_meas = None
        l4 = [raw_b.get(cell_id(4, c, role)) for role in ("N0", "N1", "D", "P0L")]
        if all(l4):
            floor = (p50(l4[0]["wall_ms"][1:]) + p50(l4[1]["wall_ms"][1:])) / 2
            ed, ep = p50(l4[2]["wall_ms"][1:]) - floor, p50(l4[3]["wall_ms"][1:]) - floor
            h_meas = 1.0 - ep / ed if ed > 0 else None
        out[c] = {"c_fit_run_a": ca, "r_run_b": r, "H_model": h_model, "H_measured_L4": h_meas,
                  "token_error": abs(h_model - h_meas) if h_model is not None and h_meas is not None else None}
    return out


def verdicts(res: dict) -> dict:
    """Numbers against the registered intervals and falsifiers only (record section 4); no interpretation."""
    v = {}
    r12 = res["rt1_rt2"]
    if r12.get("available"):
        m = r12["median_demand_copy_ms"]
        v["RT-1"] = "in 0.17-0.22" if 0.17 <= m <= 0.22 else ("in slow 0.30-0.38" if 0.30 <= m <= 0.38 else ("outside, past falsifier" if not 0.15 <= m <= 0.40 else "outside interval"))
        f = r12["first_minus_later_ms"]
        if f is not None:
            v["RT-2"] = "in -0.03..+0.06" if -0.03 <= f <= 0.06 else ("past falsifier" if not -0.06 <= f <= 0.12 else "outside interval")
    q = res["rt3_rt4"]
    for c, key, hi, lo in ((1.0, "RT-3@1.0", 0.20, None), (0.3, "RT-3@0.3", None, (0.40, 0.75))):
        e = q.get(c, {}).get("L3", {})
        if e.get("available"):
            s = e["p0l"]["q_share_of_rows"]
            v[key] = (("in interval" if s <= hi else ("past falsifier" if s > 0.35 else "outside interval")) if hi is not None else
                      ("in interval" if lo[0] <= s <= lo[1] else ("past falsifier" if s < 0.25 else "outside interval")))
    e = q.get(1.0, {}).get("L3", {})
    if e.get("available") and e["excess_ratio"] is not None:
        ratio, share = e["excess_ratio"], e["q_share_of_excess"]
        v["RT-4"] = ("falsified" if (share is not None and share < 0.40) or abs(ratio) < 0.05 else
                     ("held" if 0.10 <= ratio <= 0.25 and share is not None and share >= 0.70 else "outside interval"))
    r5 = res["rt5"].get(0.3) or res["rt5"].get(1.0)
    if r5 and r5.get("available"):
        s = r5["slowdown"]
        v["RT-5"] = "in 0..+8 %" if 0.0 <= s <= 0.08 else ("past falsifier" if s > 0.15 or s < -0.03 else "outside interval")
    for c, d in res["rt6"].items():
        for arm, x in d.items():
            ok = x["excess_ms"] <= 0.5 and (x["closure_L6"] is None or x["closure_L6"] >= 0.90)
            v[f"RT-6@{c}.{arm}"] = "held" if ok else ("past falsifier" if x["excess_ms"] > 1.5 else "outside interval")
    for c, x in res["rt7"].items():
        r, te = x["r_run_b"], x["token_error"]
        v[f"RT-7@{c}"] = ("past falsifier" if (r is not None and r < 0.6) or (te is not None and te > 0.08)
                           else ("held" if r is not None and r >= 0.8 and te is not None and te <= 0.05 else "outside interval"))
    return v


def score(raw_a: dict, raw_b: dict) -> dict:
    res = {"rt1_rt2": score_rt1_rt2(raw_a), "rt3_rt4": {}, "rt5": {}, "rt6": {}, "rt7": score_rt7(raw_a, raw_b)}
    for c in (1.0, 0.3):
        res["rt3_rt4"][c] = {f"L{lv}": score_rt3_rt4(raw_b, c, lv) for lv in (3, 6)}
        res["rt5"][c] = score_rt5(raw_b, c)
        res["rt6"][c] = score_rt6(raw_b, c)
    res["verdict"] = verdicts(res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    res = score(load_samples(args.run_a), load_samples(args.run_b))
    print(json.dumps(res, indent=2, sort_keys=True, default=float))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, sort_keys=True, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
