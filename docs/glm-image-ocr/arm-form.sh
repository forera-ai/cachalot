#!/bin/bash
# usage: docs/glm-image-ocr/arm-form.sh <tag> budget2|exact   one fresh ./serve-glm.sh at 40 GiB, five replies to each of the two form images.
# exact sets CACHALOT_GLM_DECODE_MISS_BUDGET=off. Budget-2 arm about 45 minutes, exact about 65.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/glm-image-ocr-form
mkdir -p "$D"
TAG=$1; MODE=$2
[ "$MODE" = exact ] && export CACHALOT_GLM_DECODE_MISS_BUDGET=off
echo "arm $TAG ($MODE) start $(date +%T)"
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python - "$TAG" "$MODE" <<'PY'
import sys; sys.path.insert(0, "benchmarks"); import run_manifest as rm
from pathlib import Path
man = rm.collect(workload="W5:screenshot-transcription-no-agent-context", workload_text=Path("docs/glm-image-ocr/driver-form.py").read_text(),
                 instrument="serve-glm.sh + docs/glm-image-ocr/driver-form.py " + sys.argv[1] + " " + sys.argv[2], budget_gib=40, cache_state="fresh server")
print("manifest", rm.write_manifest(Path("benchmarks/results/glm-image-ocr-form"), man))
PY
benchmarks/guarded_run.sh --budget-gib 40 --max-seconds 5400 --force --tag "glm-form-$TAG" -- ./serve-glm.sh --expert-budget-gib 40 > "$D/$TAG.server.log" 2>&1 &
GPID=$!
sleep 10; ~/venvs/deepseek-v41/bin/python docs/glm-image-ocr/driver-form.py 5 > "$D/$TAG.driver.jsonl" 2>&1
SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN)
[ -n "$SPID" ] && kill -INT $SPID
wait $GPID
echo "arm $TAG done $(date +%T)"
