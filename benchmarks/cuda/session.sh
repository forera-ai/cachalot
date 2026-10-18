#!/usr/bin/env bash
# GPU host side of CUDA session S1 (docs/CUDA-S1-EXPERIMENT-RECORD.md, benchmarks/cuda/SESSION-RUNBOOK.md).
# Run from the directory the scaffold archive was unpacked into. No sudo, no system package changes, no downloads other
# than the pinned Python wheels named below, which `setup` shows before it installs them.
#
#   bash benchmarks/cuda/session.sh check|setup|doctor|smoke|baseline|s2|results
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
  ls -d "$RESULTS"/*-"$1"-* 2>/dev/null | sort | tail -1 || true
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
    [ "$drv" -ge "$MIN_DRIVER_MAJOR" ] || die "driver $drv is older than $MIN_DRIVER_MAJOR: the pinned CUDA 12.8 wheel will not run. Do not upgrade the driver from here; tell the agent."
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
    SESSION_TAG=S2 "$PY" "$BENCH" run --profile s2 --execute --max-seconds 150 --max-alloc-mib 2048 --region "$REGION" --note "$(SESSION_TAG=S2 note s2)"
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
