#!/bin/bash
# usage: docs/glm-speed/arm.sh <tag> [budget GiB]   one fresh ./serve-glm.sh with no disk snapshots, the live session's five requests (driver.py), then the server stops.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
TAG=$1; GIB=${2:-46}
D=benchmarks/results/glm-speed
mkdir -p "$D"
export CACHALOT_GLM_VISION_MIN_PIXELS=810000 CACHALOT_GLM_SNAPSHOT_DIR=
echo "arm $TAG start $(date +%T)"
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python - "$TAG" "$GIB" <<'PY'
import sys; sys.path.insert(0, "benchmarks"); import run_manifest as rm
from pathlib import Path
man = rm.collect(workload="W5:glm-live-session-2026-11-06", workload_text=Path("docs/glm-speed/driver.py").read_text(),
                 instrument="serve-glm.sh + docs/glm-speed/driver.py " + sys.argv[1], budget_gib=int(sys.argv[2]), cache_state="fresh server, no disk snapshots")
print("manifest", rm.write_manifest(Path("benchmarks/results/glm-speed"), man))
PY
benchmarks/guarded_run.sh --budget-gib "$GIB" --max-seconds 2400 --force --tag "glm-speed-$TAG" -- ./serve-glm.sh --expert-budget-gib "$GIB" > "$D/$TAG.server.log" 2>&1 &
GPID=$!
sleep 10; ~/venvs/deepseek-v41/bin/python docs/glm-speed/driver.py > "$D/$TAG.driver.jsonl" 2>&1
SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN)
[ -n "$SPID" ] && kill -INT $SPID
wait $GPID
echo "arm $TAG done $(date +%T)"
