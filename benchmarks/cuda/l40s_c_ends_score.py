"""
L40S layer-time ends (Mac only; record `docs/CUDA-L40S-C-ENDS-RECORD.md`, predictions J-1 to J-10): does the RTX 5090's dependence of the
compute-slowdown slope on layer time exist on an L40S?

Reads the two run directories of the `j-full` profile (each holds `raw-samples.json`, `rows.jsonl` and `run-manifest.json`; run A first,
run B second) and prints each registered quantity beside its interval and falsifier. It judges nothing for the writer: the verdict
column says only whether a number is inside the registered interval, past its falsifier, or between. Rows the record scores on run B
(J-1, J-2, J-4, J-8) use run B, J-5 and J-9 run A, J-3, J-6, J-7 and J-10 both, as registered. It also prints the headline ratio
s(0.2) / s(1.0) per run, the affine-in-1/C fit through the four slopes and the 5090 fast form evaluated with this instance's copy time.

Usage:
    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/l40s_c_ends_score.py RUN_A_DIR RUN_B_DIR [--json OUT]
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

GROUP = "p17j"
LAYER_TIMES = (1.0, 0.6, 0.3, 0.2)     # 0.15 ms replaced by 0.2 ms after the smoke (record section 8)
FIRST_L40S_SLOPE = 0.093                    # record section 2: the first L40S instance, f = 1.0, C = 0.3 and 0.6
G_A, G_B = 0.0498, 0.0811                   # the RTX 5090 fast form as a function of r = c / C (record section 2)


def cid(level: int, c: float, role: str) -> str:
    return f"{GROUP}.L{level}.C{c}.{role}"


def source_median(run_dir) -> float | None:
    """The group's median source-record copy time from `rows.jsonl` (the first cell of the group carries it)."""
    p = Path(run_dir) / "rows.jsonl"
    if not p.exists():
        return None
    prefix = f"n3.{GROUP}."
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("field") == "source_copy_ms_median" and str(r.get("arm", "")).startswith(prefix) and r.get("value") is not None:
            return float(r["value"])
    return None


def slope(raw: dict, c: float) -> float | None:
    p, d = raw.get(cid(3, c, "P0L")), raw.get(cid(3, c, "D"))
    if not (p and d and p.get("rows") and d.get("rows")):
        return None
    return tc.slope_of(p, d)


def hidden_measured(raw: dict, c: float) -> tuple[float | None, float | None]:
    """Level 4: (hidden share of `P0L`, demand exposure `D` minus the `N` floor in ms)."""
    cells = [raw.get(cid(4, c, role)) for role in ("N0", "N1", "D", "P0L")]
    if not all(cells):
        return None, None
    floor = (rt.p50(cells[0]["wall_ms"][1:]) + rt.p50(cells[1]["wall_ms"][1:])) / 2
    ed, ep = rt.p50(cells[2]["wall_ms"][1:]) - floor, rt.p50(cells[3]["wall_ms"][1:]) - floor
    return (1.0 - ep / ed if ed > 0 else None), float(ed)


def points(raw: dict, run_dir) -> list[dict]:
    cm = source_median(run_dir)
    out = []
    for c in LAYER_TIMES:
        h, exposure = hidden_measured(raw, c)
        out.append({"c_layer": c, "copy_ms": cm, "r": (cm / c) if cm else None, "slope": slope(raw, c), "hidden_measured": h, "exposure_ms": exposure})
    return out


def gate_summary(manifest: dict | None) -> dict:
    g = (manifest or {}).get("gates", {})
    out = {k: bool(v) for k, v in g.items() if k != "failed_cells"}
    if manifest:
        out["no_failed_cells"] = not g.get("failed_cells")
    return out


def _by(points_: list[dict]) -> dict:
    return {p["c_layer"]: p for p in points_}


def ratio(pts: dict) -> float | None:
    s_lo, s_hi = pts[0.2]["slope"], pts[1.0]["slope"]
    return s_lo / s_hi if s_lo and s_hi else None


def score(raw_a: dict, raw_b: dict, dir_a, dir_b, manifest_a: dict | None = None, manifest_b: dict | None = None) -> dict:
    pa, pb = points(raw_a, dir_a), points(raw_b, dir_b)
    out: dict = {"points_run_a": pa, "points_run_b": pb, "gates_run_a": gate_summary(manifest_a), "gates_run_b": gate_summary(manifest_b)}
    out["ratio_run_a"], out["ratio_run_b"] = ratio(_by(pa)), ratio(_by(pb))
    for tag, pts in (("run_a", pa), ("run_b", pb)):
        ok = [p for p in pts if p["slope"] is not None]
        if len(ok) >= 3:
            b, a = np.polyfit([1.0 / p["c_layer"] for p in ok], [p["slope"] for p in ok], 1)
            out[f"affine_in_inverse_c_{tag}"] = {"a": float(a), "b": float(b), "residuals": {str(p["c_layer"]): float(p["slope"] - (a + b / p["c_layer"])) for p in ok}}
        if tag == "run_b":
            out["g_error_run_b"] = {str(p["c_layer"]): p["slope"] - (G_A + G_B * p["r"]) for p in ok if p["r"]}
    out["verdict"] = verdicts(out)
    return out


def verdicts(res: dict) -> dict:
    """Numbers against the registered intervals and falsifiers only (record section 4)."""
    v: dict = {}

    def put(key, ok, fal):
        v[key] = "in interval" if ok else ("past falsifier" if fal else "outside interval")

    a, b = _by(res["points_run_a"]), _by(res["points_run_b"])
    gates = {**{f"a.{k}": x for k, x in res.get("gates_run_a", {}).items()}, **{f"b.{k}": x for k, x in res.get("gates_run_b", {}).items()}}
    if gates:
        ok = all(x for k, x in gates.items() if k.split(".")[-1] in ("calibration", "floor", "no_failed_cells"))
        put("J-3", ok, not ok)
    s1, s2 = b[1.0]["slope"], b[0.2]["slope"]
    if s1 is not None:
        put("J-1", 0.075 <= s1 <= 0.110, s1 < 0.060 or s1 > 0.130)
    if s2 is not None:
        put("J-2", 0.085 <= s2 <= 0.140, s2 > 0.200)
    stab = [b[c]["slope"] for c in (0.3, 0.6)]
    if None not in stab:
        dev = [abs(x / FIRST_L40S_SLOPE - 1.0) for x in stab]
        put("J-4", all(d <= 0.15 for d in dev), any(d > 0.30 for d in dev))
    cm = a[1.0]["copy_ms"]
    if cm:
        put("J-5", 0.34 <= cm <= 0.42, not 0.30 <= cm <= 0.50)
    ra_, rb_ = res.get("ratio_run_a"), res.get("ratio_run_b")
    if ra_ is not None and rb_ is not None:
        put("J-6", all(0.80 <= x <= 1.30 for x in (ra_, rb_)), all(x > 1.60 for x in (ra_, rb_)))
    pairs = [(x[0.2]["slope"], x[1.0]["slope"]) for x in (a, b)]
    if all(None not in p for p in pairs):
        put("J-7", all(lo >= hi for lo, hi in pairs), all(lo < hi - 0.02 for lo, hi in pairs))
    h1, h02 = b[1.0]["hidden_measured"], b[0.2]["hidden_measured"]
    if h1 is not None and h02 is not None:
        put("J-8", 0.30 <= h1 <= 0.55 and h02 < 0.10, h02 > 0.25 or h1 < 0.15)
    ex = [p["exposure_ms"] for p in res["points_run_a"] if p["exposure_ms"] is not None]
    if len(ex) == len(LAYER_TIMES):
        put("J-9", all(14.0 <= x <= 20.0 for x in ex), max(ex) / min(ex) - 1.0 > 0.30)
    close = []
    for c in LAYER_TIMES:
        sa_, sb_ = a[c]["slope"], b[c]["slope"]
        if sa_ and sb_:
            close.append(abs(sb_ - sa_) / abs(sa_))
    if len(close) == len(LAYER_TIMES):
        put("J-10", sum(1 for x in close if x <= 0.15) >= 3, sum(1 for x in close if x <= 0.30) < 2)   # record section 7 erratum: four points, not eight
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
