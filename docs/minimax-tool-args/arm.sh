#!/bin/bash
# usage: docs/minimax-tool-args/arm.sh <tag> shipped|exact   one fresh ./serve-minimax.sh (shipped defaults: decode miss drop 0.20 with
# substitution 4, prefill miss drop 0.20, mirror on the X10Pro; 48 GiB, not 68: see RECORD section 3; no disk snapshots in either arm), 12 replayed
# Desktop tool requests (driver.py), then the server stops. exact sets both MiniMax drops to 0 (the exact path).
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/minimax-tool-args
mkdir -p "$D"
TAG=$1; MODE=$2
export CACHALOT_MINIMAX_SNAPSHOT_DIR=
if [ "$MODE" = exact ]; then export CACHALOT_MINIMAX_MISS_DROP=0 CACHALOT_MINIMAX_PREFILL_MISS_DROP=0; fi
echo "arm $TAG ($MODE) start $(date +%T)"
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python - "$TAG" "$MODE" <<'PY'
import sys; sys.path.insert(0, "benchmarks"); import run_manifest as rm
from pathlib import Path
man = rm.collect(workload="W1:hermes-tool-turn-replay-minimax", workload_text=Path("docs/minimax-tool-args/driver.py").read_text(),
                 instrument="serve-minimax.sh + docs/minimax-tool-args/driver.py " + sys.argv[1] + " " + sys.argv[2], budget_gib=48,
                 cache_state="fresh server, no snapshots")
print("manifest", rm.write_manifest(Path("benchmarks/results/minimax-tool-args"), man))
PY
benchmarks/guarded_run.sh --budget-gib 48 --max-seconds 4800 --force --tag "mmtool-$TAG" -- ./serve-minimax.sh --expert-budget-gib 48 > "$D/$TAG.server.log" 2>&1 &
GPID=$!
sleep 10; /usr/bin/python3 docs/minimax-tool-args/driver.py 12 > "$D/$TAG.driver.jsonl" 2>&1
SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN)
[ -n "$SPID" ] && kill -INT $SPID
wait $GPID
echo "arm $TAG done $(date +%T)"
