#!/usr/bin/env bash
# GPU host side of CUDA session S1 (docs/CUDA-S1-EXPERIMENT-RECORD.md, benchmarks/cuda/SESSION-RUNBOOK.md).
# Run from the directory the scaffold archive was unpacked into. No sudo, no system package changes, no downloads other
# than the pinned Python wheels named below, which `setup` shows before it installs them.
#
#   bash benchmarks/cuda/session.sh check|setup|doctor|smoke|baseline|s2|n2smoke|n2full|backing|hugepage|n3smoke|n3full|n3bsmoke|n3bfull|n3csmoke|n3cfull|n3dsmoke|n3dfull|n3esmoke|n3ebridge|n3fsmoke|n3ffull|n3ffix|n3gsmoke|n3gfull|n3hsmoke|n3hfull|n3ismoke|n3ifull|n3jsmoke|n3jfull|pinroute|n3ksmoke|n3kfull|results
#   (s2 = docs/CUDA-S2-EXPERIMENT-RECORD.md: overlap variants and batched expert copies; needs a clean smoke run first)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
BENCH="$ROOT/benchmarks/cuda/cuda_bench.py"
RESULTS="$ROOT/benchmarks/results/cuda"

# Pinned wheels (operator reviews these in the runbook before `setup`). Override only deliberately, and record it.
TORCH_SPEC="${TORCH_SPEC:-torch==2.7.1}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy==2.2.6}"
PIP_SPEC="${PIP_SPEC:-pip==25.1.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
MIN_DRIVER_MAJOR="${MIN_DRIVER_MAJOR:-570}"      # CUDA 12.8 wheels need a 570-series driver or newer
REGION="${CUDA_REGION:-unspecified}"             # coarse only: a country or continent, never a hostname or address

die() { echo "STOP: $*" >&2; exit 1; }

latest_run() {  # $1 = profile name; prints the newest run directory for it, or nothing
  ls -d "$RESULTS"/*Z-"$1"-* 2>/dev/null | sort | tail -1 || true
}

note() { echo "${SESSION_TAG:-S1} $1; freeze=$( [ -f "$ROOT/pip-freeze.txt" ] && shasum -a 256 "$ROOT/pip-freeze.txt" 2>/dev/null | cut -c1-16 || sha256sum "$ROOT/pip-freeze.txt" 2>/dev/null | cut -c1-16 || echo none )"; }

cmd="${1:-help}"
case "$cmd" in
  check)
    echo "== read-only inventory (no UUIDs, no addresses) =="
    uname -sr
    python3 --version
    python3 -c "import venv" 2>/dev/null && echo "python venv module: present" || echo "python venv module: MISSING (stop: needs an operator decision, not sudo from this script)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv || die "nvidia-smi failed: no usable NVIDIA driver"
    echo "locked-memory limit: $(ulimit -l)"
    df -h "$ROOT" | tail -1
    ;;

  setup)
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
    [ -n "$drv" ] || die "cannot read the driver version"
    [ "$drv" -ge "$MIN_DRIVER_MAJOR" ] || die "driver $drv is older than $MIN_DRIVER_MAJOR: the pinned wheel's CUDA runtime will not run (12.8 needs 570 or newer, 12.6 needs 525 or newer: MIN_DRIVER_MAJOR). Do not upgrade the driver from here; tell the agent."
    echo "will run, in a venv at $ROOT/.venv:"
    echo "  pip install --index-url $TORCH_INDEX $TORCH_SPEC"
    echo "  pip install $NUMPY_SPEC   (pip itself: $PIP_SPEC, hash-verified, only if the venv cannot bootstrap pip)"
    if ! python3 -m venv "$ROOT/.venv" 2>/dev/null; then
      # Ubuntu cloud images often lack the ensurepip package. Without sudo: a venv without pip, then a hash-verified,
      # pinned pip wheel from PyPI (no system change). Session S1, 2026-10-18, needed exactly this.
      echo "python3 -m venv could not bootstrap pip; using a venv without pip and a verified pip wheel ($PIP_SPEC)"
      rm -rf "$ROOT/.venv"
      python3 -m venv --without-pip "$ROOT/.venv" || die "python3 -m venv --without-pip failed. Operator decision needed; no sudo here."
      python3 - "$PIP_SPEC" <<'PYEOF' || die "could not fetch and verify the pip wheel"
import hashlib, json, sys, urllib.request
ver = sys.argv[1].split("==")[1]
meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/pip/{ver}/json", timeout=30))
f = next(x for x in meta["urls"] if x["filename"].endswith("py3-none-any.whl"))
data = urllib.request.urlopen(f["url"], timeout=60).read()
assert hashlib.sha256(data).hexdigest() == f["digests"]["sha256"], "pip wheel hash mismatch"
open("/tmp/" + f["filename"], "wb").write(data)
print("pip wheel verified:", f["filename"], f["digests"]["sha256"][:16])
PYEOF
      "$PY" "/tmp/pip-${PIP_SPEC#pip==}-py3-none-any.whl/pip" install --no-index "/tmp/pip-${PIP_SPEC#pip==}-py3-none-any.whl"
    fi
    "$PY" -m pip install --index-url "$TORCH_INDEX" "$TORCH_SPEC"
    "$PY" -m pip install "$NUMPY_SPEC"
    "$PY" -m pip freeze > "$ROOT/pip-freeze.txt"
    "$PY" -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'available', torch.cuda.is_available()); print('arch list', torch.cuda.get_arch_list())"
    # Arch gate (V1): the wheel must carry kernels for this device. A cu128 wheel has no sm_70, so it cannot run on a V100.
    "$PY" - "$ROOT/benchmarks/cuda" <<'PYEOF' || die "the installed torch has no kernels for this GPU's compute capability. Reinstall with the matching wheel (for a V100: TORCH_INDEX=https://download.pytorch.org/whl/cu126 MIN_DRIVER_MAJOR=525, see SESSION-RUNBOOK-V1.md); do not run any cell."
import sys
sys.path.insert(0, sys.argv[1])
import suite, torch
cc = torch.cuda.get_device_capability(0)
ok = suite.arch_supported(list(torch.cuda.get_arch_list()), cc)
print(f"arch gate: compute capability {cc[0]}.{cc[1]} supported by this torch build: {ok}")
sys.exit(0 if ok else 1)
PYEOF
    ;;

  doctor)
    [ -x "$PY" ] || die "run setup first"
    mkdir -p "$RESULTS"
    "$PY" "$BENCH" doctor --region "$REGION" --note "$(note doctor)" --out "$RESULTS/doctor-machine-manifest.json"
    ;;

  smoke)
    [ -x "$PY" ] || die "run setup first"
    "$PY" "$BENCH" run --profile smoke --execute --region "$REGION" --note "$(note smoke)"
    ;;

  baseline)
    [ -x "$PY" ] || die "run setup first"
    run="$(latest_run smoke)"
    [ -n "$run" ] || die "no smoke run found: run smoke first"
    "$PY" - "$run/run-manifest.json" <<'EOF' || die "the latest smoke run is not clean: do not start the baseline; bring the files home"
import json, sys
m = json.load(open(sys.argv[1]))
g = m["gates"]
ok = (m["status"] == "complete" and not g["failed_cells"] and g["copy_checksum"] and g["kernel_reference"]
      and g["sanity"] and not g["sanity_violations"])
print("smoke run clean:" , ok, "| status", m["status"], "| failed cells", len(g["failed_cells"]))
sys.exit(0 if ok else 1)
EOF
    "$PY" "$BENCH" run --profile baseline --execute --max-seconds 300 --max-alloc-mib 2048 --region "$REGION" --note "$(note baseline)"
    ;;

  s2)
    [ -x "$PY" ] || die "run setup first"
    run="$(latest_run smoke)"
    [ -n "$run" ] || die "no smoke run found: run smoke first"
    "$PY" - "$run/run-manifest.json" <<'EOF2' || die "the latest smoke run is not clean: do not start the S2 profile; bring the files home"
import json, sys
m = json.load(open(sys.argv[1]))
g = m["gates"]
ok = (m["status"] == "complete" and not g["failed_cells"] and g["copy_checksum"] and g["kernel_reference"]
      and g["sanity"] and not g["sanity_violations"])
print("smoke run clean:", ok, "| status", m["status"], "| failed cells", len(g["failed_cells"]))
sys.exit(0 if ok else 1)
EOF2
    SESSION_TAG="${SESSION_TAG:-S2}" "$PY" "$BENCH" run --profile s2 --execute --max-seconds 150 --max-alloc-mib 2048 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-S2}" note s2)"
    ;;

  n2smoke)
    # N2 stage 1 (docs/CUDA-N2S1-EXPERIMENT-RECORD.md): a short GPU smoke of the replay's stream code. Needs the trace file
    # copied to the host first (scp from the Mac); TRACE=path overrides the default location.
    [ -x "$PY" ] || die "run setup first"
    TRACE="${TRACE:-$ROOT/n2-trace.npz}"
    [ -f "$TRACE" ] || die "trace file not found at $TRACE (scp it from the Mac, see SESSION-RUNBOOK-N2S1.md)"
    SESSION_TAG="${SESSION_TAG:-N2}" "$PY" "$ROOT/benchmarks/cuda/replay_arena.py" run --trace "$TRACE" \
      --expected "$ROOT/benchmarks/cuda/n2s1-expected.json" --cells smoke --execute --max-seconds 180 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N2}" note n2smoke)"
    ;;

  n2full)
    [ -x "$PY" ] || die "run setup first"
    TRACE="${TRACE:-$ROOT/n2-trace.npz}"
    [ -f "$TRACE" ] || die "trace file not found at $TRACE"
    run="$(latest_run n2s1-smoke)"
    [ -n "$run" ] || die "no n2smoke run found: run n2smoke first"
    "$PY" - "$run/run-manifest.json" <<'EOF3' || die "the latest n2smoke run is not clean: do not start n2full; bring the files home"
import json, sys
m = json.load(open(sys.argv[1]))
g = m["gates"]
ok = m["status"] == "complete" and not g["failed_cells"] and all(g[k] for k in ("event_hash", "determinism", "readback", "sanity"))
print("n2smoke run clean:", ok, "| status", m["status"], "| failed cells", len(g["failed_cells"]))
sys.exit(0 if ok else 1)
EOF3
    SESSION_TAG="${SESSION_TAG:-N2}" "$PY" "$ROOT/benchmarks/cuda/replay_arena.py" run --trace "$TRACE" \
      --expected "$ROOT/benchmarks/cuda/n2s1-expected.json" --cells full --execute --max-seconds 2400 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N2}" note n2full)"
    ;;

  backing)
    # Session B1 (docs/CUDA-BACKING-EXPERIMENT-RECORD.md): four processes, in order, stopping at the first failure. The first
    # is a fresh-process allocation-order probe and doubles as the GPU smoke of the new instrument; the second repeats it
    # in a new process (B-P3); then the huge-page, reroll and slab cells; then the qualified copy curve.
    [ -x "$PY" ] || die "run setup first"
    export SESSION_TAG="${SESSION_TAG:-B1}"
    for step in "b1 P1" "b1 P2" "rest R" "curve C"; do
      set -- $step
      "$PY" "$ROOT/benchmarks/cuda/backing_probe.py" run --cells "$1" --tag "$2" --execute --max-seconds 300 --region "$REGION" \
        --note "$(note backing-$1-$2)" || die "backing step $1 ($2) failed: stop, bring the files home"
    done
    ;;

  n3smoke)
    # N3 stage A (docs/CUDA-N3-EXPERIMENT-RECORD.md): a short GPU smoke of the overlap harness's stream code. Needs the trace file copied
    # to the host first (scp from the Mac); TRACE=path overrides the default location.
    [ -x "$PY" ] || die "run setup first"
    TRACE="${TRACE:-$ROOT/n3-trace.npz}"
    [ -f "$TRACE" ] || die "trace file not found at $TRACE (scp it from the Mac, see SESSION-RUNBOOK-N3.md)"
    SESSION_TAG="${SESSION_TAG:-N3}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$TRACE" \
      --expected "$ROOT/benchmarks/cuda/n3-expected.json" --cells smoke --execute --max-seconds 300 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3}" note n3smoke)"
    ;;

  n3full)
    [ -x "$PY" ] || die "run setup first"
    TRACE="${TRACE:-$ROOT/n3-trace.npz}"
    [ -f "$TRACE" ] || die "trace file not found at $TRACE"
    run="$(latest_run n3a-smoke)"
    [ -n "$run" ] || die "no n3smoke run found: run n3smoke first"
    "$PY" - "$run/run-manifest.json" <<'EOF4' || die "the latest n3smoke run is not clean: do not start n3full; bring the files home"
import json, sys
m = json.load(open(sys.argv[1]))
g = m["gates"]
keys = ("event_hash", "determinism", "counts", "readback", "calibration", "drift", "closure", "sanity")
ok = m["status"] == "complete" and not g["failed_cells"] and all(g[k] for k in keys)
print("n3smoke run clean:", ok, "| status", m["status"], "| failed cells", len(g["failed_cells"]))
sys.exit(0 if ok else 1)
EOF4
    SESSION_TAG="${SESSION_TAG:-N3}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$TRACE" \
      --expected "$ROOT/benchmarks/cuda/n3-expected.json" --cells full --execute --max-seconds 2400 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3}" note n3full)"
    ;;

  hugepage)
    # Session B2 (docs/CUDA-HUGEPAGE-RETEST-RECORD.md): seven fresh processes, in order, stopping at the first failure. h7 first
    # (the fresh-process control and the check of this instance's buffer population), then h6, then the mapping cells h4 (the
    # shared control), h1, h2, h3 and h5 (populate; "unavailable" is a valid outcome, not a failure).
    [ -x "$PY" ] || die "run setup first"
    export SESSION_TAG="${SESSION_TAG:-B2}"
    for step in "h7 F" "h6 X" "h4 S" "h1 P" "h2 N" "h3 D" "h5 O"; do
      set -- $step
      "$PY" "$ROOT/benchmarks/cuda/backing_probe.py" run --cells "$1" --tag "$2" --execute --max-seconds 300 --region "$REGION" \
        --note "$(note hugepage-$1-$2)" || die "hugepage step $1 ($2) failed: stop, bring the files home"
    done
    ;;

  n3bsmoke)
    # N3 stage B (docs/CUDA-N3B-EXPERIMENT-RECORD.md): a short GPU smoke of the level code (digest off, lean book, views, bare events, the
    # CUDA graph). Needs the trace copied to the host as for n3smoke.
    [ -x "$PY" ] || die "run setup first"
    TRACE="${TRACE:-$ROOT/n3-trace.npz}"
    [ -f "$TRACE" ] || die "trace file not found at $TRACE (scp it from the Mac, see SESSION-RUNBOOK-N3B.md)"
    SESSION_TAG="${SESSION_TAG:-N3B}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$TRACE" \
      --expected "$ROOT/benchmarks/cuda/n3-expected.json" --cells b-smoke --execute --max-seconds 300 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3B}" note n3bsmoke)"
    ;;

  n3bfull)
    # The cap is explicit and below the 1 GPU-hour session cap: MAXS=seconds overrides (default 1500).
    [ -x "$PY" ] || die "run setup first"
    TRACE="${TRACE:-$ROOT/n3-trace.npz}"
    [ -f "$TRACE" ] || die "trace file not found at $TRACE"
    run="$(latest_run n3b-b-smoke)"
    [ -n "$run" ] || die "no n3bsmoke run found: run n3bsmoke first"
    "$PY" - "$run/run-manifest.json" <<'EOF5' || die "the latest n3bsmoke run is not clean: do not start n3bfull; bring the files home"
import json, sys
m = json.load(open(sys.argv[1]))
g = m["gates"]
keys = ("event_hash", "determinism", "counts", "chain2", "readback", "calibration", "drift", "closure", "sanity")
ok = m["status"] == "complete" and not g["failed_cells"] and all(g[k] for k in keys)
print("n3bsmoke run clean:", ok, "| status", m["status"], "| failed cells", len(g["failed_cells"]))
sys.exit(0 if ok else 1)
EOF5
    SESSION_TAG="${SESSION_TAG:-N3B}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$TRACE" \
      --expected "$ROOT/benchmarks/cuda/n3-expected.json" --cells b-full --execute --max-seconds "${MAXS:-1500}" --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3B}" note n3bfull)"
    ;;

  n3csmoke)
    # N3 stage C (docs/CUDA-N3C-EXPERIMENT-RECORD.md): a short GPU smoke of the gated arms (G10, G20) beside P0L at L5 and L4.
    [ -x "$PY" ] || die "run setup first"
    TRACE="${TRACE:-$ROOT/n3-trace.npz}"
    [ -f "$TRACE" ] || die "trace file not found at $TRACE (scp it from the Mac, see SESSION-RUNBOOK-N3C.md)"
    SESSION_TAG="${SESSION_TAG:-N3C}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$TRACE" \
      --expected "$ROOT/benchmarks/cuda/n3c-expected.json" --cells c-smoke --execute --max-seconds 300 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3C}" note n3csmoke)"
    ;;

  n3cfull)
    # The cap is explicit and below the 1 GPU-hour session cap: MAXS=seconds overrides (default 1200).
    [ -x "$PY" ] || die "run setup first"
    TRACE="${TRACE:-$ROOT/n3-trace.npz}"
    [ -f "$TRACE" ] || die "trace file not found at $TRACE"
    run="$(latest_run n3c-c-smoke)"
    [ -n "$run" ] || die "no n3csmoke run found: run n3csmoke first"
    "$PY" - "$run/run-manifest.json" <<'EOF6' || die "the latest n3csmoke run is not clean: do not start n3cfull; bring the files home"
import json, sys
m = json.load(open(sys.argv[1]))
g = m["gates"]
keys = ("event_hash", "determinism", "counts", "chain2", "coverage", "readback", "calibration", "drift", "closure", "sanity")
ok = m["status"] == "complete" and not g["failed_cells"] and all(g[k] for k in keys)
print("n3csmoke run clean:", ok, "| status", m["status"], "| failed cells", len(g["failed_cells"]))
sys.exit(0 if ok else 1)
EOF6
    SESSION_TAG="${SESSION_TAG:-N3C}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$TRACE" \
      --expected "$ROOT/benchmarks/cuda/n3c-expected.json" --cells c-full --execute --max-seconds "${MAXS:-1200}" --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3C}" note n3cfull)"
    ;;

  n3dsmoke)
    # N3 stage D (docs/CUDA-N3D-EXPERIMENT-RECORD.md): a short GPU smoke of the lead-2 arm `L2` beside `P0L`. Needs the NEW trace
    # (routing-trace-deepseek-leads, 39.8 MB) copied to the host as n3d-trace.npz, see SESSION-RUNBOOK-N3D.md.
    [ -x "$PY" ] || die "run setup first"
    TRACE="${TRACE:-$ROOT/n3d-trace.npz}"
    [ -f "$TRACE" ] || die "trace file not found at $TRACE (scp it from the Mac, see SESSION-RUNBOOK-N3D.md)"
    SESSION_TAG="${SESSION_TAG:-N3D}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$TRACE" \
      --expected "$ROOT/benchmarks/cuda/n3d-expected.json" --cells d-smoke --execute --max-seconds 300 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3D}" note n3dsmoke)"
    ;;

  n3dfull)
    # The cap is explicit and below the 1 GPU-hour session cap: MAXS=seconds overrides (default 1200).
    [ -x "$PY" ] || die "run setup first"
    TRACE="${TRACE:-$ROOT/n3d-trace.npz}"
    [ -f "$TRACE" ] || die "trace file not found at $TRACE"
    run="$(latest_run n3d-d-smoke)"
    [ -n "$run" ] || die "no n3dsmoke run found: run n3dsmoke first"
    "$PY" - "$run/run-manifest.json" <<'EOF7' || die "the latest n3dsmoke run is not clean: do not start n3dfull; bring the files home"
import json, sys
m = json.load(open(sys.argv[1]))
g = m["gates"]
keys = ("event_hash", "determinism", "counts", "chain2", "coverage", "readback", "calibration", "drift", "closure", "sanity")
ok = m["status"] == "complete" and not g["failed_cells"] and all(g[k] for k in keys)
print("n3dsmoke run clean:", ok, "| status", m["status"], "| failed cells", len(g["failed_cells"]))
sys.exit(0 if ok else 1)
EOF7
    SESSION_TAG="${SESSION_TAG:-N3D}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$TRACE" \
      --expected "$ROOT/benchmarks/cuda/n3d-expected.json" --cells d-full --execute --max-seconds "${MAXS:-1200}" --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3D}" note n3dfull)"
    ;;

  n3esmoke)
    # The bridge session (docs/CUDA-BRIDGE-EXPERIMENT-RECORD.md): a short smoke of the e-profile on BOTH traces. Needs the old trace
    # (n3-trace.npz, 11 MB) and the new multi-lead trace (n3d-trace.npz, 39.8 MB) on the host, see SESSION-RUNBOOK-BRIDGE.md.
    [ -x "$PY" ] || die "run setup first"
    OLD="${TRACE_OLD:-$ROOT/n3-trace.npz}"; NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$OLD" ] || die "old trace not found at $OLD"; [ -f "$NEW" ] || die "new trace not found at $NEW"
    for kind in old new; do
      if [ "$kind" = old ]; then T="$OLD"; X="$ROOT/benchmarks/cuda/n3c-expected.json"; else T="$NEW"; X="$ROOT/benchmarks/cuda/n3d-expected.json"; fi
      SESSION_TAG="${SESSION_TAG:-N3E}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$T" --expected "$X" --cells e-smoke --execute \
        --max-seconds 300 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3E}" note n3esmoke-$kind)" || die "n3esmoke $kind failed: stop, bring the files home"
    done
    ;;

  n3ebridge)
    # Four runs of e-full in the fixed order old, new, new, old. The cap is explicit and below the 1 GPU-hour session cap:
    # MAXS=seconds applies to each run (default 600). The trace of each run is identified by its sha256 in the run manifest.
    [ -x "$PY" ] || die "run setup first"
    OLD="${TRACE_OLD:-$ROOT/n3-trace.npz}"; NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$OLD" ] || die "old trace not found at $OLD"; [ -f "$NEW" ] || die "new trace not found at $NEW"
    for kind in old new new old; do
      if [ "$kind" = old ]; then T="$OLD"; X="$ROOT/benchmarks/cuda/n3c-expected.json"; else T="$NEW"; X="$ROOT/benchmarks/cuda/n3d-expected.json"; fi
      SESSION_TAG="${SESSION_TAG:-N3E}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$T" --expected "$X" --cells e-full --execute \
        --max-seconds "${MAXS:-600}" --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3E}" note n3ebridge-$kind)" || die "n3ebridge $kind failed: stop, bring the files home"
    done
    ;;

  n3fsmoke)
    # Row and per-copy timing (docs/CUDA-ROW-TIMING-RECORD.md): a short smoke of the f-profile on the NEW multi-lead trace only
    # (n3d-trace.npz, 39.8 MB, see SESSION-RUNBOOK-ROW-TIMING.md). Read every gate of the run manifest before n3ffull.
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    SESSION_TAG="${SESSION_TAG:-N3F}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
      --cells f-smoke --execute --max-seconds 300 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3F}" note n3fsmoke)" || die "n3fsmoke failed: stop, bring the files home"
    ;;

  n3ffull)
    # Two runs of f-full on the new trace (run A, then run B as the replicate). The cap is explicit and below the 1 GPU-hour session cap:
    # MAXS=seconds applies to each run (default 900).
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    for rep in A B; do
      SESSION_TAG="${SESSION_TAG:-N3F}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
        --cells f-full --execute --max-seconds "${MAXS:-900}" --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3F}" note n3ffull-$rep)" || die "n3ffull $rep failed: stop, bring the files home"
    done
    ;;

  n3ffix)
    # 0.66.46: the two (level, C = 1.0) groups of f-full that ran with the stand-in tuned to C = 0.3 (record section 8). One run on the new trace.
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    SESSION_TAG="${SESSION_TAG:-N3F}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
      --cells f-fix --execute --max-seconds "${MAXS:-420}" --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3F}" note n3ffix)" || die "n3ffix failed: stop, bring the files home"
    ;;

  n3gsmoke)
    # Third layer time (docs/CUDA-THIRD-C-RECORD.md): a short smoke of the g-profile (C = 0.6 ms) on the NEW multi-lead trace only.
    # Read every gate of the run manifest before n3gfull (SESSION-RUNBOOK-THIRD-C.md).
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    SESSION_TAG="${SESSION_TAG:-N3G}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
      --cells g-smoke --execute --max-seconds 300 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3G}" note n3gsmoke)" || die "n3gsmoke failed: stop, bring the files home"
    ;;

  n3gfull)
    # Two runs of g-full on the new trace (run A, then run B as the replicate), 12 cells each. MAXS=seconds applies to each run (default 600).
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    for rep in A B; do
      SESSION_TAG="${SESSION_TAG:-N3G}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
        --cells g-full --execute --max-seconds "${MAXS:-600}" --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3G}" note n3gfull-$rep)" || die "n3gfull $rep failed: stop, bring the files home"
    done
    ;;

  n3hsmoke)
    # Low layer time (docs/CUDA-LOW-C-RECORD.md): a short smoke of the h-profile (C = 0.15 ms) on the NEW multi-lead trace only.
    # Read every gate of the run manifest before n3hfull (SESSION-RUNBOOK-LOW-C.md); the calibration and floor gates are prediction L-9.
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    SESSION_TAG="${SESSION_TAG:-N3H}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
      --cells h-smoke --execute --max-seconds 300 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3H}" note n3hsmoke)" || die "n3hsmoke failed: stop, bring the files home"
    ;;

  n3hfull)
    # Two runs of h-full on the new trace (run A, then run B as the replicate), 12 cells each. MAXS=seconds applies to each run (default 600).
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    for rep in A B; do
      SESSION_TAG="${SESSION_TAG:-N3H}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
        --cells h-full --execute --max-seconds "${MAXS:-600}" --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3H}" note n3hfull-$rep)" || die "n3hfull $rep failed: stop, bring the files home"
    done
    ;;

  n3ismoke)
    # Copy-cost sweep (docs/CUDA-COPY-COST-RECORD.md): a short smoke of the i-profile (C = 0.3 ms, copy fractions 1.0 and 0.25) on the NEW trace.
    # Read every gate of the run manifest before n3ifull (SESSION-RUNBOOK-COPY-COST.md); note the source-copy medians of both groups.
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    SESSION_TAG="${SESSION_TAG:-N3I}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
      --cells i-smoke --execute --max-seconds 420 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3I}" note n3ismoke)" || die "n3ismoke failed: stop, bring the files home"
    ;;

  n3ifull)
    # Two runs of i-full on the new trace (run A, then run B as the replicate), 48 cells each. MAXS=seconds applies to each run (default 900).
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    for rep in A B; do
      SESSION_TAG="${SESSION_TAG:-N3I}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
        --cells i-full --execute --max-seconds "${MAXS:-900}" --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3I}" note n3ifull-$rep)" || die "n3ifull $rep failed: stop, bring the files home"
    done
    ;;

  n3jsmoke)
    # L40S layer-time ends (docs/CUDA-L40S-C-ENDS-RECORD.md): a short smoke of the j-profile (C = 1.0 and 0.2 ms, full copy; 0.15 ms did not calibrate, record section 8) on the NEW trace.
    # Read every gate of the run manifest before n3jfull (SESSION-RUNBOOK-L40S-C-ENDS.md); the calibration and floor gates at 0.2 ms are prediction J-3.
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    SESSION_TAG="${SESSION_TAG:-N3J}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
      --cells j-smoke --execute --max-seconds 300 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3J}" note n3jsmoke)" || die "n3jsmoke failed: stop, bring the files home"
    ;;

  n3jfull)
    # Two runs of j-full on the new trace (run A, then run B as the replicate), 32 cells each. MAXS=seconds applies to each run (default 720).
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    for rep in A B; do
      SESSION_TAG="${SESSION_TAG:-N3J}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
        --cells j-full --execute --max-seconds "${MAXS:-720}" --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3J}" note n3jfull-$rep)" || die "n3jfull $rep failed: stop, bring the files home"
    done
    ;;

  pinroute)
    # Session K (docs/CUDA-PIN-ROUTE-RECORD.md, SESSION-RUNBOOK-PIN-ROUTE.md): DC-3 and DC-6, six fresh processes in order (k0 classify, the
    # three registered routes, the cudaHostAlloc control, k5 classify again), stopping at the first failure. Each cell is its own process
    # because the allocation context is one of the variables. MAXS=seconds applies to each process (default 150).
    [ -x "$PY" ] || die "run setup first"
    for g in k0 k1 k2 k3 k4 k5; do
      SESSION_TAG="${SESSION_TAG:-K}" "$PY" "$ROOT/benchmarks/cuda/pin_route.py" run --cells "$g" --execute --max-seconds "${MAXS:-150}" \
        --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-K}" note pinroute-$g)" || die "pinroute $g failed: stop, bring the files home"
    done
    ;;

  n3ksmoke)
    # Session K, DC-5 (docs/CUDA-PIN-ROUTE-RECORD.md section 3.4): a short smoke of the k-profile (the guarded issue loop and the planted control) on the NEW trace.
    # Read every gate of the run manifest, including guard_sees_planted, before n3kfull (SESSION-RUNBOOK-PIN-ROUTE.md).
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    SESSION_TAG="${SESSION_TAG:-N3K}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
      --cells k-smoke --execute --max-seconds 300 --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3K}" note n3ksmoke)" || die "n3ksmoke failed: stop, bring the files home"
    ;;

  n3kfull)
    # One run of k-full on the new trace (27 cells; no replicate: the guard question is a count). MAXS=seconds (default 720).
    [ -x "$PY" ] || die "run setup first"
    NEW="${TRACE_NEW:-$ROOT/n3d-trace.npz}"
    [ -f "$NEW" ] || die "new trace not found at $NEW"
    SESSION_TAG="${SESSION_TAG:-N3K}" "$PY" "$ROOT/benchmarks/cuda/overlap_arena.py" run --trace "$NEW" --expected "$ROOT/benchmarks/cuda/n3d-expected.json" \
      --cells k-full --execute --max-seconds "${MAXS:-720}" --region "$REGION" --note "$(SESSION_TAG="${SESSION_TAG:-N3K}" note n3kfull)" || die "n3kfull failed: stop, bring the files home"
    ;;

  results)
    [ -d "$RESULTS" ] || die "no results directory"
    [ -f "$ROOT/pip-freeze.txt" ] && cp "$ROOT/pip-freeze.txt" "$RESULTS/pip-freeze.txt"
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    archive="$ROOT/cuda-s1-results-$stamp.tar.gz"
    tar -czf "$archive" -C "$ROOT/benchmarks" results/cuda
    (shasum -a 256 "$archive" 2>/dev/null || sha256sum "$archive")
    echo "archive: $archive"
    ;;

  *)
    sed -n '2,9p' "$0"
    ;;
esac
