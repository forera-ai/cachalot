#!/bin/bash
# usage: docs/v1017-di2-server/arm.sh <tag> exact|keep140   one fresh ./serve.sh (shipped 1017 defaults: DI-1 on, budget 0, 48 GiB, hotlist),
# two cold prefills (driver.py), then the server stops. keep140 sets CACHALOT_PREFILL_DECODER_KEEP=140. No snapshots.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/di2-server
mkdir -p "$D"
TAG=$1; MODE=$2
export CACHALOT_SNAPSHOT_DIR=
[ "$MODE" = keep140 ] && export CACHALOT_PREFILL_DECODER_KEEP=140
echo "arm $TAG ($MODE) start $(date +%T)"
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python - "$TAG" "$MODE" <<'PY'
import sys; sys.path.insert(0, "benchmarks"); import run_manifest as rm
from pathlib import Path
man = rm.collect(workload="W3:hermes-block+W2:mid-prompt", workload_text=Path("docs/v1017-di2-server/driver.py").read_text()
                 + Path("docs/v1017-di2-server/mid_prompt.txt").read_text(),
                 instrument="serve.sh + docs/v1017-di2-server/driver.py " + sys.argv[1] + " " + sys.argv[2], budget_gib=48,
                 cache_state="fresh server, no snapshots; both prompts cold")
print("manifest", rm.write_manifest(Path("benchmarks/results/di2-server"), man))
PY
benchmarks/guarded_run.sh --budget-gib 48 --max-seconds 1500 --force --tag "di2srv-$TAG" -- ./serve.sh > "$D/$TAG.server.log" 2>&1 &
GPID=$!
sleep 10; /usr/bin/python3 docs/v1017-di2-server/driver.py "$HOME/cachalot-dump-1017.jsonl" docs/v1017-di2-server/mid_prompt.txt > "$D/$TAG.driver.jsonl" 2>&1
SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN)
[ -n "$SPID" ] && kill -INT $SPID
wait $GPID
echo "arm $TAG done $(date +%T)"
