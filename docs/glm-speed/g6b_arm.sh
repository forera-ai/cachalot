#!/bin/bash
# usage: docs/glm-speed/g6b_arm.sh <tag> <speculate min tokens> <prefill predict top-k> [budget GiB]
# One fresh ./serve-glm.sh with no disk snapshots, CACHALOT_GLM_SPECULATE_MIN_TOKENS and CACHALOT_GLM_PREFILL_PREDICT_TOPK
# set, G6b's short-prefill ladder (g6_driver.py with G6_RUNGS), then the server stops. docs/glm-speed/G6B-RECORD.md.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
TAG=$1; SPEC=$2; TOPK=$3; GIB=${4:-46}
D=benchmarks/results/glm-speed/g6b
mkdir -p "$D"
export CACHALOT_GLM_VISION_MIN_PIXELS=810000 CACHALOT_GLM_SNAPSHOT_DIR= CACHALOT_GLM_SPECULATE_MIN_TOKENS=$SPEC \
  CACHALOT_GLM_PREFILL_PREDICT_TOPK=$TOPK G6_RUNGS=${G6_RUNGS:-25,60,120,160,250,400,528,640,800,950}
echo "arm $TAG spec=$SPEC topk=$TOPK start $(date +%T)"
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python - "$TAG" "$GIB" "$SPEC" "$TOPK" <<'PY'
import sys; sys.path.insert(0, "benchmarks"); import run_manifest as rm
from pathlib import Path
man = rm.collect(workload="W5:glm-block-then-short-prefill-ladder", workload_text=Path("docs/glm-speed/g6_driver.py").read_text(),
                 instrument=f"serve-glm.sh + docs/glm-speed/g6_driver.py {sys.argv[1]} speculate_min={sys.argv[3]} prefill_predict_topk={sys.argv[4]}",
                 budget_gib=int(sys.argv[2]), cache_state="fresh server, no disk snapshots")
print("manifest", rm.write_manifest(Path("benchmarks/results/glm-speed/g6b"), man))
PY
benchmarks/guarded_run.sh --budget-gib "$GIB" --max-seconds 1500 --force --tag "glm-g6b-$TAG" -- ./serve-glm.sh --expert-budget-gib "$GIB" > "$D/$TAG.server.log" 2>&1 &
GPID=$!
# at least 12 s, or the guard's preflight sees the driver as a second runtime (v319 rule)
sleep 15; ~/venvs/deepseek-v41/bin/python docs/glm-speed/g6_driver.py > "$D/$TAG.driver.jsonl" 2>&1
for _ in $(seq 1 120); do SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN); [ -n "$SPID" ] && break; kill -0 $GPID 2>/dev/null || break; sleep 5; done
[ -n "${SPID:-}" ] && kill -INT $SPID
wait $GPID
echo "arm $TAG done $(date +%T)"
