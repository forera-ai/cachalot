"""
Command line for the CUDA research lane (docs/CUDA-RESEARCH-PLAN.md, phases N0b and N1).

    python benchmarks/cuda/cuda_bench.py doctor [--out FILE]
    python benchmarks/cuda/cuda_bench.py plan   --profile smoke|baseline [--max-seconds S --max-iters N --max-alloc-mib M]
    python benchmarks/cuda/cuda_bench.py run    --profile smoke|baseline [caps] [--execute] [--out-dir DIR] [--note TEXT]
    python benchmarks/cuda/cuda_bench.py validate PATH [PATH ...]

`run` without `--execute` is a dry run: it prints the plan and writes nothing. With `--execute` it runs on this machine's
GPU, under a hard wall-clock cap, and exits 3 (with a reason) if no CUDA device is visible. This tool never provisions,
starts or stops an instance and never reads provider credentials.

Exit codes: 0 ok, 2 bad arguments, 3 no usable CUDA device (informative skip), 4 validation or run failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import doctor  # noqa: E402
import manifest as mf  # noqa: E402
import suite  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO / "benchmarks" / "results" / "cuda"      # gitignored with benchmarks/results/


def _limits(args) -> suite.Limits:
    base = suite.PROFILE_DEFAULTS[args.profile]
    return suite.Limits(
        max_seconds=args.max_seconds if args.max_seconds is not None else base.max_seconds,
        max_iters=args.max_iters if args.max_iters is not None else base.max_iters,
        max_alloc_mib=args.max_alloc_mib if args.max_alloc_mib is not None else base.max_alloc_mib,
        warmup=base.warmup, seed=args.seed,
    ).validated()


def cmd_doctor(args) -> int:
    report = doctor.collect(note=args.note, region=args.region)
    ok, why = doctor.cuda_ready(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n# cuda usable by the framework: {ok} ({why})", file=sys.stderr)
    if args.out:
        mf.write_json(args.out, report)
        print(f"# wrote {args.out}", file=sys.stderr)
    return 0


def cmd_plan(args) -> int:
    try:
        print(json.dumps(suite.describe_plan(args.profile, _limits(args)), indent=2))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_run(args) -> int:
    try:
        limits = _limits(args)
        description = suite.describe_plan(args.profile, limits)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.execute:
        print(json.dumps(description, indent=2))
        print("\n# DRY RUN: nothing executed, nothing written. Add --execute on the provisioned GPU host.", file=sys.stderr)
        return 0
    report = doctor.collect(note=args.note, region=args.region, session_cap_s=int(limits.max_seconds))
    ok, why = doctor.cuda_ready(report)
    if not ok:
        print(f"SKIP: no CUDA run possible here: {why}", file=sys.stderr)
        return 3
    bench = suite.TorchBench(0)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{stamp}-{args.profile}-{report['manifest_id'][:6]}"
    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    mf.write_json(out / "machine-manifest.json", report)
    run_manifest, rows, raw = suite.run(args.profile, limits, bench, run_id=run_id, machine_id=report["manifest_id"],
                                        note=args.note, code=report["code"])
    problems = mf.validate_run(run_manifest)
    mf.write_json(out / "run-manifest.json", run_manifest)
    mf.write_json(out / "raw-samples.json", {"run": run_id, "samples": raw})
    (out / "rows.jsonl").write_text("")
    mf.append_rows(out / "rows.jsonl", rows)
    print(f"run {run_id}: status {run_manifest['status']}, {len(rows)} rows, "
          f"{run_manifest['limits']['elapsed_s']} s of {limits.max_seconds} s cap -> {out}")
    gates = run_manifest["gates"]
    bad = [k for k in ("copy_checksum", "kernel_reference", "sanity") if not gates[k]]
    if gates["failed_cells"]:
        bad.append(f"{len(gates['failed_cells'])} failed cell(s)")
    if problems or bad:
        print("FAIL: " + "; ".join(problems + bad), file=sys.stderr)
        return 4
    return 0


def cmd_validate(args) -> int:
    rc = 0
    for path in args.paths:
        p = Path(path)
        try:
            if p.suffix == ".jsonl":
                problems = []
                for i, line in enumerate(p.read_text().splitlines(), 1):
                    problems += [f"line {i}: {x}" for x in mf.validate_row(json.loads(line))]
            else:
                doc = mf.read_json(p)
                if doc.get("kind") == "machine":
                    problems = mf.validate_machine(doc)
                elif doc.get("kind") == "run":
                    problems = mf.validate_run(doc)
                elif "samples" in doc:                      # raw-samples.json: only the redaction rule applies
                    problems = [f"redaction: {x} ({k})" for x, k in mf.find_forbidden(doc)]
                else:
                    problems = [f"unknown kind {doc.get('kind')!r}"]
        except (OSError, ValueError) as exc:
            problems = [f"unreadable: {type(exc).__name__}"]
        print(f"{'OK  ' if not problems else 'FAIL'} {p}")
        for x in problems:
            print(f"     {x}")
        rc = rc or (4 if problems else 0)
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="print a machine manifest (non-secret inventory)")
    d.add_argument("--out")
    d.add_argument("--note", default="")
    d.add_argument("--region", help="coarse region only (continent or country)")
    for name in ("plan", "run"):
        s = sub.add_parser(name)
        s.add_argument("--profile", choices=sorted(suite.PROFILE_DEFAULTS), required=True)
        s.add_argument("--max-seconds", type=float, help=f"hard wall-clock cap (<= {suite.HARD_MAX_SECONDS})")
        s.add_argument("--max-iters", type=int, help=f"timed iterations per cell (<= {suite.HARD_MAX_ITERS})")
        s.add_argument("--max-alloc-mib", type=int, help=f"device/pinned allocation cap per cell (<= {suite.HARD_MAX_ALLOC_MIB})")
        s.add_argument("--seed", type=int, default=suite.Limits().seed)
    r = sub.choices["run"]
    r.add_argument("--execute", action="store_true", help="actually run on this machine's GPU (default: dry run)")
    r.add_argument("--out-dir", default=str(DEFAULT_RESULTS))
    r.add_argument("--note", default="")
    r.add_argument("--region")
    v = sub.add_parser("validate", help="check manifests/rows against the schema and the redaction rules")
    v.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    return {"doctor": cmd_doctor, "plan": cmd_plan, "run": cmd_run, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
