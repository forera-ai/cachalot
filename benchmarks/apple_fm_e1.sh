#!/bin/bash
# E-1 (docs/APPLE-FM-E1-RECORD.md): one fresh ./serve.sh at 48 GiB, a warm-up, then A B B A x 6 replies with the
# auxiliary classification load off (A) and on (B), aux-alone baselines before and after. ~10-12 minutes.
#
#   cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/apple_fm_e1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VENV_PY=~/venvs/deepseek-v41/bin/python
OUT=benchmarks/results/apple-fm-e1/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
echo "out: $OUT"

benchmarks/settle.sh --budget-gib 48
PYTHONPATH=src "$VENV_PY" benchmarks/run_manifest.py > "$OUT/manifest-machine.json" 2>/dev/null || true

echo "aux-alone baseline (before), 30 s"
PYTHONPATH=src "$VENV_PY" benchmarks/apple_fm_e1.py load --seconds 30 --log "$OUT/aux-pre.jsonl"

echo "server: guarded ./serve.sh at 48 GiB"
benchmarks/guarded_run.sh --budget-gib 48 --force --max-seconds 1800 --tag apple-fm-e1 -- \
    env CACHALOT_SNAPSHOT_DIR= ./serve.sh > "$OUT/guard.txt" 2>&1 &
GPID=$!
# the driver and the load are Python processes the runtime guards would count: start them only after
# serve.sh's own one-runtime check has passed and the server process exists
for _ in $(seq 1 120); do
    pgrep -f 'cachalot\.cli serve' >/dev/null && break
    kill -0 "$GPID" 2>/dev/null || { echo "guarded run exited early:"; cat "$OUT/guard.txt"; exit 1; }
    sleep 1
done
pgrep -f 'cachalot\.cli serve' >/dev/null || { echo "server did not start"; cat "$OUT/guard.txt"; exit 1; }

set +e
/usr/bin/python3 benchmarks/apple_fm_e1.py drive --out "$OUT" --python "$VENV_PY" --cycles 6
DRIVE=$?
set -e
pkill -INT -f 'cachalot\.cli serve' || true
wait "$GPID" || true
SERVER_LOG=$(sed -n 's/^run: output -> //p' "$OUT/guard.txt")
tail -3 "$OUT/guard.txt"

echo "aux-alone baseline (after), 30 s"
PYTHONPATH=src "$VENV_PY" benchmarks/apple_fm_e1.py load --seconds 30 --log "$OUT/aux-post.jsonl"

/usr/bin/python3 benchmarks/apple_fm_e1.py score --out "$OUT" --server-log "$SERVER_LOG"
exit $DRIVE
