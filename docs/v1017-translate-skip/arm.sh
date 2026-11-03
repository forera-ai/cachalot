#!/bin/bash
# usage: docs/v1017-translate-skip/arm.sh <tag> budget0|exact   one fresh ./serve.sh (shipped 0.78.1 defaults: 1017, DI-1 + keep 140,
# decode miss budget 0, tool-exact, 48 GiB, hotlist, the shipped snapshot directory), 4 replies to each of the four recorded hummingbird
# turns (driver.py), then the server stops. exact sets CACHALOT_DECODE_MISS_BUDGET= (empty: no budget, every expert read).
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/translate-skip
mkdir -p "$D"
TAG=$1; MODE=$2
[ "$MODE" = exact ] && export CACHALOT_DECODE_MISS_BUDGET=
echo "arm $TAG ($MODE) start $(date +%T)"
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python - "$TAG" "$MODE" <<'PY'
import sys; sys.path.insert(0, "benchmarks"); import run_manifest as rm
from pathlib import Path
man = rm.collect(workload="W6:hermes-translate-turn-replay", workload_text=Path("docs/v1017-translate-skip/driver.py").read_text(),
                 instrument="serve.sh + docs/v1017-translate-skip/driver.py " + sys.argv[1] + " " + sys.argv[2], budget_gib=48,
                 cache_state="fresh server, shipped snapshot directory")
print("manifest", rm.write_manifest(Path("benchmarks/results/translate-skip"), man))
PY
benchmarks/guarded_run.sh --budget-gib 48 --max-seconds 3000 --force --tag "translate-$TAG" -- ./serve.sh > "$D/$TAG.server.log" 2>&1 &
GPID=$!
sleep 10; /usr/bin/python3 docs/v1017-translate-skip/driver.py 4 > "$D/$TAG.driver.jsonl" 2>&1
SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN)
[ -n "$SPID" ] && kill -INT $SPID
wait $GPID
echo "arm $TAG done $(date +%T)"
