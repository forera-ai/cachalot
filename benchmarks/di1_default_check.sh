#!/bin/bash
# DI-1 default-on check (docs/DI1-DEFAULT-ON-PLAN.md section 4, D-1 to D-4): three ./serve.sh starts on one new snapshot directory,
# each replaying the live session's first request with max_tokens 1: (1) no knob in the environment, cold; (2) the same, after a
# restart, which should reuse the system block from disk; (3) CACHALOT_PREFILL_ENCODER_ONLY=0, which should find no snapshot of its mode.
#
#   benchmarks/di1_default_check.sh        # about 7 minutes
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${DI1_CHECK_OUT:-docs/di1-default-on}
DUMP=${DI1_AB_DUMP:-$HOME/cachalot-di1-live.jsonl}
SETTLE_GIB=${DI1_AB_SETTLE_GIB:-40}          # 48 needs more memory available than this machine has with Hamed's apps open (18.241)
mkdir -p "$OUT"
[ -f "$DUMP" ] || { echo "no dump at $DUMP" >&2; exit 1; }
snap=$(mktemp -d "${TMPDIR:-/tmp}/di1-default-snap.XXXXXX")
n=0
for mode in default default off; do
  n=$((n + 1))
  echo "run $n ($mode): settle $(date +%H:%M:%S)"
  benchmarks/settle.sh --budget-gib $SETTLE_GIB || { echo "settle failed before run $n" >&2; rm -rf "$snap"; exit 1; }
  if [ "$mode" = off ]; then
    env -u CACHALOT_SERVER_DUMP CACHALOT_PREFILL_ENCODER_ONLY=0 CACHALOT_SNAPSHOT_DIR="$snap" ./serve.sh > "$OUT/run-$n-$mode-server.log" 2>&1 &
    knob=0
  else
    env -u CACHALOT_SERVER_DUMP -u CACHALOT_PREFILL_ENCODER_ONLY CACHALOT_SNAPSHOT_DIR="$snap" ./serve.sh > "$OUT/run-$n-$mode-server.log" 2>&1 &
    knob=1
  fi
  SP=$!
  t0=$(date +%s)
  until curl -sf http://127.0.0.1:8011/v1/models >/dev/null 2>&1; do
    kill -0 $SP 2>/dev/null || { echo "server died in run $n" >&2; tail -5 "$OUT/run-$n-$mode-server.log" >&2; rm -rf "$snap"; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "server not up after 900 s in run $n" >&2; kill -INT $SP; wait $SP; rm -rf "$snap"; exit 1; }
    sleep 3
  done
  echo "run $n ($mode): server up $(date +%H:%M:%S)"
  /usr/bin/env python3 -I benchmarks/di1_server_ab_driver.py --dump "$DUMP" --out "$OUT/run-$n-$mode.json" --arm $n --knob $knob \
      --server-pid $SP --requests 1
  rc=$?
  sleep 5                                      # let the store finish writing before the stop
  kill -INT $SP 2>/dev/null; wait $SP 2>/dev/null
  echo "run $n ($mode): done rc=$rc $(date +%H:%M:%S); snapshot files: $(ls "$snap" | wc -l | tr -d ' ')"
  [ $rc -eq 0 ] || { rm -rf "$snap"; exit $rc; }
done
rm -rf "$snap"
grep -h "DI-1 encoder-only prefill on\|prefix snapshots:\|\[request\]" "$OUT"/run-*-server.log | cut -c1-160
