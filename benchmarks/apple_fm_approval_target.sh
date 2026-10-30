#!/bin/bash
# The 36-command approval gate through DeepSeek's server, as Hermes sends a review (docs/APPLE-FM-APPROVAL-RECORD.md
# addendum 1). One guarded ./serve.sh at 48 GiB, three passes, then the server is stopped. ~9-13 minutes.
#
#   cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/apple_fm_approval_target.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VENV_PY=~/venvs/deepseek-v41/bin/python
OUT=benchmarks/results/apple-fm-approval/target-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
echo "out: $OUT"
# settle at its default arithmetic: the machine offers ~70 of the 77 GiB a 48 GiB formula wants (18.230)
benchmarks/settle.sh --budget-gib 36
benchmarks/guarded_run.sh --budget-gib 48 --force --max-seconds 1800 --tag apple-fm-approval-target -- \
    env CACHALOT_SNAPSHOT_DIR= ./serve.sh > "$OUT/guard.txt" 2>&1 &
GPID=$!
for _ in $(seq 1 120); do
    pgrep -f 'cachalot\.cli serve' >/dev/null && break
    kill -0 "$GPID" 2>/dev/null || { echo "guarded run exited early:"; cat "$OUT/guard.txt"; exit 1; }
    sleep 1
done
for _ in $(seq 1 300); do
    curl -sf http://127.0.0.1:8011/health >/dev/null && break
    sleep 2
done
curl -sf http://127.0.0.1:8011/health >/dev/null || { echo "server not healthy"; cat "$OUT/guard.txt"; exit 1; }
echo "server healthy"
set +e
PYTHONPATH=src "$VENV_PY" benchmarks/apple_fm_approval_gate.py target --passes 3 | tee "$OUT/target.log"
RC=${PIPESTATUS[0]}
set -e
pkill -INT -f 'cachalot\.cli serve' || true
wait "$GPID" || true
grep -E 'footprint|result:' "$OUT/guard.txt" || true
exit $RC
