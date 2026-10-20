"""
Score the bridge session (docs/CUDA-BRIDGE-EXPERIMENT-RECORD.md) from the four `e-full` runs copied home. Mac only.

Reads `*-run-manifest.json` / `*-rows.jsonl` pairs of the `e-full` cells in a results directory, tells the trace of each run by its sha256
(the two expected files), and prints per (trace, C): the hidden share of `P0L` in each run, their mean and spread, the new-minus-old
difference, `E_D`, and the row-level window model's hidden share (`window_model.py`, needs the two traces). Scoring against the
registered intervals is by hand from this table (the record's section 4).

Usage:
    ~/venvs/deepseek-v41/bin/python benchmarks/cuda/bridge_score.py DIR --old-trace OLD.npz --new-trace NEW.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import window_model as wm  # noqa: E402

HERE = Path(__file__).resolve().parent
GROUPS = {1.0: "n3.p17.L5.C1.0", 0.6: "n3.p17.L4.C0.6", 0.3: "n3.p17.L4.C0.3"}


def load_runs(directory: Path) -> list[dict]:
    kinds = {json.loads((HERE / f).read_text())["trace_sha256"]: kind for f, kind in (("n3c-expected.json", "old"), ("n3d-expected.json", "new"))}
    runs = []
    for mpath in sorted(directory.glob("*-run-manifest.json")):
        m = json.loads(mpath.read_text())
        if not str(m.get("workload", {}).get("id", "")).endswith("e-full"):
            continue
        rows = [json.loads(line) for line in mpath.with_name(mpath.name.replace("run-manifest.json", "rows.jsonl")).read_text().splitlines() if line.strip()]
        vals = {(r["arm"], r["field"]): r.get("value") for r in rows}
        runs.append({"created": m["created_utc"], "kind": kinds.get(m["workload"]["trace_sha256"], "unknown"), "status": m["status"],
                     "gates": m["gates"], "vals": vals, "file": mpath.name})
    return sorted(runs, key=lambda r: r["created"])


def table(runs: list[dict]) -> dict:
    out: dict = {"order": [r["kind"] for r in runs], "cells": {}}
    for c, arm in GROUPS.items():
        for kind in ("old", "new"):
            h = [r["vals"].get((arm, "H_P0L")) for r in runs if r["kind"] == kind]
            ed = [r["vals"].get((arm, "E_D")) for r in runs if r["kind"] == kind]
            h = [x for x in h if x is not None]
            ed = [x for x in ed if x is not None]
            out["cells"][f"{kind}.C{c}"] = {"H_runs": h, "H_mean": sum(h) / len(h) if h else None, "H_spread": (max(h) - min(h)) if h else None,
                                            "E_D_runs": ed}
        o, n = out["cells"][f"old.C{c}"]["H_mean"], out["cells"][f"new.C{c}"]["H_mean"]
        out["cells"][f"diff.C{c}"] = None if o is None or n is None else n - o
    return out


def model_table(old_trace: str, new_trace: str) -> dict:
    out = {}
    for kind, path in (("old", old_trace), ("new", new_trace)):
        with np.load(path) as d:
            arrays = {k: d[k] for k in d.files if k != "segments"}
        pf, none = wm.row_counts(arrays, 17, 400, True), wm.row_counts(arrays, 17, 400, False)
        for c in GROUPS:
            out[f"{kind}.C{c}"] = wm.model(pf, none, 0.194, c)["H_P0L_model"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--old-trace")
    ap.add_argument("--new-trace")
    args = ap.parse_args(argv)
    runs = load_runs(Path(args.directory))
    res = table(runs)
    res["runs"] = [{"kind": r["kind"], "status": r["status"], "file": r["file"], "gates_ok": all(v for k, v in r["gates"].items() if k != "failed_cells") and not r["gates"]["failed_cells"]}
                   for r in runs]
    if args.old_trace and args.new_trace:
        res["model"] = model_table(args.old_trace, args.new_trace)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
