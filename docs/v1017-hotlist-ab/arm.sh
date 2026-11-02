#!/bin/bash
# usage: docs/v1017-hotlist-ab/arm.sh <tag> on|off   one fresh ./serve.sh (shipped 1017 defaults: DI-1 on, budget 0, 48 GiB), one
# Hermes first turn, then the server stops. "off" sets CACHALOT_HOTLIST= (serve.sh keeps an explicit empty value). No snapshots, so the
# system block prefills cold in every arm. Raw output goes to the gitignored results folder.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/hotlist-ab-1017
mkdir -p "$D"
TAG=$1; MODE=$2
export CACHALOT_SNAPSHOT_DIR=
[ "$MODE" = off ] && export CACHALOT_HOTLIST=
echo "arm $TAG ($MODE) start $(date +%T)"
PYTHONPATH=src ~/venvs/deepseek-v41/bin/python - "$TAG" "$MODE" <<'PY'
import sys; sys.path.insert(0, "benchmarks"); import run_manifest as rm
from pathlib import Path
man = rm.collect(workload="W3:hermes-first-turn", workload_text=Path("docs/v1017-hotlist-ab/driver.py").read_text(),
                 instrument="serve.sh + docs/v1017-hotlist-ab/driver.py " + sys.argv[1] + " hotlist " + sys.argv[2], budget_gib=48,
                 cache_state="fresh server, no snapshots; request 0 of ~/cachalot-dump-1017.jsonl cold")
print("manifest", rm.write_manifest(Path("benchmarks/results/hotlist-ab-1017"), man))
PY
benchmarks/guarded_run.sh --budget-gib 48 --max-seconds 1500 --force --tag "hlab-$TAG" -- ./serve.sh > "$D/$TAG.server.log" 2>&1 &
GPID=$!
sleep 10; /usr/bin/python3 docs/v1017-hotlist-ab/driver.py "$HOME/cachalot-dump-1017.jsonl" > "$D/$TAG.driver.jsonl" 2>&1
SPID=$(lsof -ti tcp:8011 -sTCP:LISTEN)
[ -n "$SPID" ] && kill -INT $SPID
wait $GPID
echo "arm $TAG done $(date +%T)"
