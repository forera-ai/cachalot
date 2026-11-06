#!/bin/bash
# usage: docs/glm-image-ocr/arm-knob.sh   one fresh ./serve-glm.sh at 40 GiB with CACHALOT_GLM_VISION_MIN_PIXELS=810000, driver-knob.py. About 30 minutes.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/glm-image-ocr-knob
mkdir -p "$D"
export CACHALOT_GLM_VISION_MIN_PIXELS=810000
echo "knob arm start $(date +%T)"
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python - <<'PY'
import sys; sys.path.insert(0, "benchmarks"); import run_manifest as rm
from pathlib import Path
man = rm.collect(workload="W5:screenshot-transcription-no-agent-context", workload_text=Path("docs/glm-image-ocr/driver-knob.py").read_text(),
                 instrument="serve-glm.sh + docs/glm-image-ocr/driver-knob.py", budget_gib=40, cache_state="fresh server")
print("manifest", rm.write_manifest(Path("benchmarks/results/glm-image-ocr-knob"), man))
PY
benchmarks/guarded_run.sh --budget-gib 40 --max-seconds 3600 --force --tag glm-knob -- ./serve-glm.sh --expert-budget-gib 40 > "$D/K.server.log" 2>&1 &
GPID=$!
sleep 10; ~/venvs/deepseek-v41/bin/python docs/glm-image-ocr/driver-knob.py > "$D/K.driver.jsonl" 2>&1
SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN)
[ -n "$SPID" ] && kill -INT $SPID
wait $GPID
echo "knob arm done $(date +%T)"
