#!/bin/bash
# usage: docs/v1017-image-upscale/arm.sh <tag> shipped|upscale   one fresh ./serve.sh (shipped 0.78.4 defaults: 1017, DI-1 + keep 140,
# decode miss budget 0, tool-exact, 48 GiB, hotlist, the shipped snapshot directory), 10 replies to each of the two 492-pixel screenshot
# turns (driver.py), then the server stops. upscale sets CACHALOT_VISION_MIN_PIXELS=810000 (900 x 900).
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/image-upscale
mkdir -p "$D"
TAG=$1; MODE=$2
[ "$MODE" = upscale ] && export CACHALOT_VISION_MIN_PIXELS=810000
echo "arm $TAG ($MODE) start $(date +%T)"
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python - "$TAG" "$MODE" <<'PY'
import sys; sys.path.insert(0, "benchmarks"); import run_manifest as rm
from pathlib import Path
man = rm.collect(workload="W5:hermes-image-turn-replay-small", workload_text=Path("docs/v1017-image-upscale/driver.py").read_text(),
                 instrument="serve.sh + docs/v1017-image-upscale/driver.py " + sys.argv[1] + " " + sys.argv[2], budget_gib=48,
                 cache_state="fresh server, shipped snapshot directory")
print("manifest", rm.write_manifest(Path("benchmarks/results/image-upscale"), man))
PY
benchmarks/guarded_run.sh --budget-gib 48 --max-seconds 3000 --force --tag "upscale-$TAG" -- ./serve.sh > "$D/$TAG.server.log" 2>&1 &
GPID=$!
sleep 10; /usr/bin/python3 docs/v1017-image-upscale/driver.py 10 > "$D/$TAG.driver.jsonl" 2>&1
SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN)
[ -n "$SPID" ] && kill -INT $SPID
wait $GPID
echo "arm $TAG done $(date +%T)"
