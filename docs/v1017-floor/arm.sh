#!/bin/bash
# usage: docs/v1017-floor/arm.sh <tag> [ENV=VALUE ...]   one fresh ./serve.sh on DeepSeek-V4.1-Flash-1017, the driver, then the server stops.
# The 0.60.10 arm (benchmarks/results/floor-server-0.60.10/arm.sh, HANDOFF 18.76) with the 1017 checkpoint and its internal bank; the hotlist
# is the one serve.sh picks for the checkpoint (cachalot-hotlist-DeepSeek-V4.1-Flash-1017.json). Raw output goes to the gitignored results folder.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/floor-server-0.73.0-1017
mkdir -p "$D"
TAG=$1; shift
export CACHALOT_DECODE_MISS_BUDGET=off CACHALOT_SNAPSHOT_DIR=
for kv in "$@"; do export "$kv"; done
export CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-1017 CACHALOT_EXPERT_BANK=$HOME/DeepSeek-V4.1-Flash-1017-q2g128
export CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=80 MLX_METAL_FAST_SYNCH=1
echo "arm $TAG start $(date +%T) budget=$CACHALOT_DECODE_MISS_BUDGET $*"
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python - "$TAG" "$*" <<'PY'
import sys; sys.path.insert(0, "benchmarks"); import run_manifest as rm
from pathlib import Path
man = rm.collect(workload="W1:floor-server", workload_text=Path("docs/v1017-floor/driver.py").read_text(),
                 instrument="serve.sh + docs/v1017-floor/driver.py " + sys.argv[1] + " " + sys.argv[2], budget_gib=48,
                 cache_state="fresh server; floor prompt repeated (all-resident) + 12 mixed")
print("manifest", rm.write_manifest(Path("benchmarks/results/floor-server-0.73.0-1017"), man))
PY
benchmarks/guarded_run.sh --budget-gib 48 --max-seconds 2400 --force --tag "fsrv1017-$TAG" -- ./serve.sh > "$D/$TAG.server.log" 2>&1 &
GPID=$!
sleep 30; /usr/bin/python3 docs/v1017-floor/driver.py > "$D/$TAG.driver.jsonl" 2>&1
SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN)
[ -n "$SPID" ] && kill -INT $SPID
wait $GPID
echo "arm $TAG done $(date +%T)"
