"""
Copy-cost sweep (Mac only; record `docs/CUDA-COPY-COST-RECORD.md`, predictions SW-1 to SW-10): is the compute-slowdown slope a function
of copy time over layer time, `r = c / C`, on one L40S?

Reads the two run directories of the `i-full` profile (each holds `raw-samples.json`, `rows.jsonl` and `run-manifest.json`; run A first,
run B second) and prints each registered quantity beside its interval and falsifier. It judges nothing for the writer: the verdict
column says only whether a number is inside the registered interval, past its falsifier, or between. Rows that the record scores on
run B (SW-4 to SW-7) use run B's slopes and hidden shares and run B's own source qualification; SW-1, SW-2 and SW-9 use run A as
registered; SW-3 and SW-10 use both.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/copy_cost_score.py RUN_A_DIR RUN_B_DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import row_timing as rt  # noqa: E402
import third_c_score as tc  # noqa: E402

FRACTIONS = (1.0, 0.5, 0.25)
LAYER_TIMES = (0.3, 0.6)
DEMAND_PER_TOKEN = 52.15                    # demand misses a token with no prefetch, 17 % resident (record section 2)
G_A, G_B = 0.0498, 0.0811                   # the RTX 5090 fast form as a function of r = c / C (record section 2)
EQUAL_R_PAIRS = (((1.0, 0.6), (0.5, 0.3)), ((0.5, 0.6), (0.25, 0.3)))   # (f, C) pairs with about the same r


def gid(frac: float) -> str:
    return f"p17f{int(round(frac * 100))}"


def cid(frac: float, level: int, c: float, role: str) -> str:
    return f"{gid(frac)}.L{level}.C{c}.{role}"


def source_median(run_dir, frac: float) -> float | None:
    """The group's median source-record copy time from `rows.jsonl` (the first cell of every group carries it)."""
    p = Path(run_dir) / "rows.jsonl"
    if not p.exists():
        return None
    prefix = f"n3.{gid(frac)}."
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("field") == "source_copy_ms_median" and str(r.get("arm", "")).startswith(prefix) and r.get("value") is not None:
            return float(r["value"])
    return None


def slope(raw: dict, frac: float, c: float) -> float | None:
    p, d = raw.get(cid(frac, 3, c, "P0L")), raw.get(cid(frac, 3, c, "D"))
    if not (p and d and p.get("rows") and d.get("rows")):
        return None
    return tc.slope_of(p, d)


def hidden_measured(raw: dict, frac: float, c: float) -> tuple[float | None, float | None]:
    """Level 4: (hidden share of `P0L`, demand exposure `D` minus the `N` floor in ms)."""
    cells = [raw.get(cid(frac, 4, c, role)) for role in ("N0", "N1", "D", "P0L")]
    if not all(cells):
        return None, None
    floor = (rt.p50(cells[0]["wall_ms"][1:]) + rt.p50(cells[1]["wall_ms"][1:])) / 2
    ed, ep = rt.p50(cells[2]["wall_ms"][1:]) - floor, rt.p50(cells[3]["wall_ms"][1:]) - floor
    return (1.0 - ep / ed if ed > 0 else None), float(ed)


def hidden_model(raw: dict, frac: float, c: float, copy_ms: float) -> float | None:
    """The count-only window model with the measured copy time, on the row counts of the level 3 cells (`window_model.model`)."""
    p, d = raw.get(cid(frac, 3, c, "P0L")), raw.get(cid(frac, 3, c, "D"))
    if not (p and d and p.get("rows") and d.get("rows")):
        return None
    rp, rd = rt.rows_of(p), rt.rows_of(d)
    toks_p, toks_d = max(1, len(p["rows"]) - 1), max(1, len(d["rows"]) - 1)
    e_p = float(np.sum(np.maximum(0.0, rp[:, rt.RF["n_pf"]] * copy_ms - c) + rp[:, rt.RF["n_dem"]] * copy_ms)) / toks_p
    e_d = float(np.sum(rd[:, rt.RF["n_dem"]] * copy_ms)) / toks_d
    return 1.0 - e_p / e_d if e_d > 0 else None


def line_fit(xs: list[float], ys: list[float]) -> dict:
    x, y = np.array(xs), np.array(ys)
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    tot = float(np.sum((y - y.mean()) ** 2))
    return {"a": float(a), "b": float(b), "r2": float(1.0 - np.sum((y - pred) ** 2) / tot) if tot > 0 else float("nan")}


def points(raw: dict, run_dir) -> list[dict]:
    """One entry per (f, C) with everything the predictions read."""
    out = []
    for f in FRACTIONS:
        cm = source_median(run_dir, f)
        for c in LAYER_TIMES:
            h, exposure = hidden_measured(raw, f, c)
            out.append({"f": f, "c_layer": c, "copy_ms": cm, "r": (cm / c) if cm else None, "slope": slope(raw, f, c), "hidden_measured": h,
                        "hidden_model": hidden_model(raw, f, c, cm) if cm else None, "exposure_ms": exposure,
                        "exposure_predicted_ms": DEMAND_PER_TOKEN * cm if cm else None})
    return out


def gate_summary(manifest: dict | None) -> dict:
    g = (manifest or {}).get("gates", {})
    out = {k: bool(v) for k, v in g.items() if k != "failed_cells"}
    if manifest:
        out["no_failed_cells"] = not g.get("failed_cells")
    return out


def score(raw_a: dict, raw_b: dict, dir_a, dir_b, manifest_a: dict | None = None, manifest_b: dict | None = None) -> dict:
    pa, pb = points(raw_a, dir_a), points(raw_b, dir_b)
    out: dict = {"points_run_a": pa, "points_run_b": pb, "gates_run_a": gate_summary(manifest_a), "gates_run_b": gate_summary(manifest_b)}
    ok_b = [p for p in pb if p["slope"] is not None and p["r"] is not None]
    if len(ok_b) >= 3:
        out["line_run_b"] = line_fit([p["r"] for p in ok_b], [p["slope"] for p in ok_b])
        out["g_error_run_b"] = {f"f{p['f']}.C{p['c_layer']}": p["slope"] - (G_A + G_B * p["r"]) for p in ok_b}
    out["verdict"] = verdicts(out)
    return out


def _by(points_: list[dict]) -> dict:
    return {(p["f"], p["c_layer"]): p for p in points_}


def verdicts(res: dict) -> dict:
    """Numbers against the registered intervals and falsifiers only (record section 4)."""
    v: dict = {}

    def put(key, ok, fal):
        v[key] = "in interval" if ok else ("past falsifier" if fal else "outside interval")

    a, b = _by(res["points_run_a"]), _by(res["points_run_b"])
    gates = {**{f"a.{k}": x for k, x in res.get("gates_run_a", {}).items()}, **{f"b.{k}": x for k, x in res.get("gates_run_b", {}).items()}}
    if gates:
        ok = all(x for k, x in gates.items() if k.split(".")[-1] in ("calibration", "floor", "no_failed_cells"))
        put("SW-8", ok, not ok)
    c1 = a[(1.0, 0.3)]["copy_ms"]
    if c1:
        put("SW-1", 0.36 <= c1 <= 0.50, not 0.28 <= c1 <= 0.70)
        c5, c25 = a[(0.5, 0.3)]["copy_ms"], a[(0.25, 0.3)]["copy_ms"]
        if c5 and c25:
            r5, r25 = c5 / c1, c25 / c1
            put("SW-2", 0.50 <= r5 <= 0.62 and 0.26 <= r25 <= 0.40, not 0.45 <= r5 <= 0.70)
    # SW-3: both runs, both layer times
    mono, flat = [], []
    for pts in (a, b):
        for c in LAYER_TIMES:
            s1, s5, s25 = (pts[(f, c)]["slope"] for f in FRACTIONS)
            if None in (s1, s5, s25):
                continue
            mono.append(s1 > s5 > s25)
            flat.append(s1 <= s25)
    if mono:
        put("SW-3", all(mono), any(flat))
    line = res.get("line_run_b")
    if line:
        put("SW-4", line["r2"] >= 0.85, not line["r2"] >= 0.50)
        errs = list(res["g_error_run_b"].values())
        put("SW-6", sum(1 for e in errs if abs(e) <= 0.03) >= 4, float(np.mean(np.abs(errs))) > 0.06)
    devs = []
    for (f1, c1_), (f2, c2_) in EQUAL_R_PAIRS:
        s1, s2 = b[(f1, c1_)]["slope"], b[(f2, c2_)]["slope"]
        if s1 and s2:
            devs.append(abs(s1 - s2) / max(abs(s1), abs(s2)))
    if len(devs) == len(EQUAL_R_PAIRS):
        put("SW-5", all(d <= 0.25 for d in devs), any(d > 0.40 for d in devs))
    herr = [abs(p["hidden_model"] - p["hidden_measured"]) for p in res["points_run_b"] if p["hidden_model"] is not None and p["hidden_measured"] is not None]
    if herr:
        put("SW-7", sum(1 for e in herr if e <= 0.10) >= 4, float(np.mean(herr)) > 0.18)
    ex = [(p["exposure_ms"], p["exposure_predicted_ms"]) for p in res["points_run_a"] if p["exposure_ms"] is not None and p["exposure_predicted_ms"]]
    if ex:
        rel = [abs(e / q - 1.0) for e, q in ex]
        put("SW-9", all(x <= 0.15 for x in rel), any(x > 0.35 for x in rel))
    close = []
    for key, pa_ in a.items():
        sa_, sb_ = pa_["slope"], b[key]["slope"]
        if sa_ and sb_:
            close.append(abs(sb_ - sa_) / abs(sa_))
    if close:
        put("SW-10", sum(1 for x in close if x <= 0.15) >= 5, sum(1 for x in close if x <= 0.30) < 3)
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

    res = score(rt.load_samples(args.run_a), rt.load_samples(args.run_b), args.run_a, args.run_b, manifest(args.run_a), manifest(args.run_b))
    text = json.dumps(res, indent=2, sort_keys=True, default=float)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
