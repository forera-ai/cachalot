#!/bin/bash
# usage: docs/glm-speed/g6_arm.sh <tag> <speculate min tokens> [budget GiB]
# One fresh ./serve-glm.sh with no disk snapshots and CACHALOT_GLM_SPECULATE_MIN_TOKENS set, the short-prefill ladder
# (g6_driver.py), then the server stops. docs/glm-speed/G6-RECORD.md.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
TAG=$1; SPEC=$2; GIB=${3:-46}
D=benchmarks/results/glm-speed/g6
mkdir -p "$D"
export CACHALOT_GLM_VISION_MIN_PIXELS=810000 CACHALOT_GLM_SNAPSHOT_DIR= CACHALOT_GLM_SPECULATE_MIN_TOKENS=$SPEC
echo "arm $TAG spec=$SPEC start $(date +%T)"
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python - "$TAG" "$GIB" "$SPEC" <<'PY'
import sys; sys.path.insert(0, "benchmarks"); import run_manifest as rm
from pathlib import Path
man = rm.collect(workload="W5:glm-block-then-short-prefill-ladder", workload_text=Path("docs/glm-speed/g6_driver.py").read_text(),
                 instrument=f"serve-glm.sh + docs/glm-speed/g6_driver.py {sys.argv[1]} speculate_min={sys.argv[3]}", budget_gib=int(sys.argv[2]),
                 cache_state="fresh server, no disk snapshots")
print("manifest", rm.write_manifest(Path("benchmarks/results/glm-speed/g6"), man))
PY
benchmarks/guarded_run.sh --budget-gib "$GIB" --max-seconds 1500 --force --tag "glm-g6-$TAG" -- ./serve-glm.sh --expert-budget-gib "$GIB" > "$D/$TAG.server.log" 2>&1 &
GPID=$!
# at least 12 s, or the guard's preflight sees the driver as a second runtime (v319 rule)
sleep 15; ~/venvs/deepseek-v41/bin/python docs/glm-speed/g6_driver.py > "$D/$TAG.driver.jsonl" 2>&1
for _ in $(seq 1 120); do SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN); [ -n "$SPID" ] && break; kill -0 $GPID 2>/dev/null || break; sleep 5; done
[ -n "${SPID:-}" ] && kill -INT $SPID
wait $GPID
echo "arm $TAG done $(date +%T)"
