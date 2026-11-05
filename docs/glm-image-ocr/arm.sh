#!/bin/bash
# usage: docs/glm-image-ocr/arm.sh <tag>   one fresh ./serve-glm.sh at 46 GiB on the shipped GLM defaults (decode miss budget 2, prefetch
# off, contiguous bank), five replies to each of four images (driver.py), then the server stops. About 35 minutes.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/glm-image-ocr
mkdir -p "$D"
TAG=$1
echo "arm $TAG start $(date +%T)"
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python - "$TAG" <<'PY'
import sys; sys.path.insert(0, "benchmarks"); import run_manifest as rm
from pathlib import Path
man = rm.collect(workload="W5:screenshot-transcription-no-agent-context", workload_text=Path("docs/glm-image-ocr/driver.py").read_text(),
                 instrument="serve-glm.sh + docs/glm-image-ocr/driver.py " + sys.argv[1], budget_gib=46, cache_state="fresh server")
print("manifest", rm.write_manifest(Path("benchmarks/results/glm-image-ocr"), man))
PY
benchmarks/guarded_run.sh --budget-gib 46 --max-seconds 3000 --force --tag "glm-ocr-$TAG" -- ./serve-glm.sh --expert-budget-gib 46 > "$D/$TAG.server.log" 2>&1 &
GPID=$!
sleep 10; ~/venvs/deepseek-v41/bin/python docs/glm-image-ocr/driver.py 5 > "$D/$TAG.driver.jsonl" 2>&1
SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN)
[ -n "$SPID" ] && kill -INT $SPID
wait $GPID
echo "arm $TAG done $(date +%T)"
