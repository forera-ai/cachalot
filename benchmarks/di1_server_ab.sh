#!/bin/bash
# DI-1 server A/B (docs/DI1-SERVER-AB-RECORD.md): four ./serve.sh arms in the order off, on, on, off, each with a new empty snapshot
# directory, replaying the live session's first four requests with max_tokens 1 (benchmarks/di1_server_ab_driver.py). The server starts
# first and the driver second, as in energy_arms.sh, because serve.sh's one-runtime guard refuses while a benchmarks/ python is alive.
#
#   benchmarks/di1_server_ab.sh            # about 26 minutes
#   DI1_AB_ORDER="0 1" benchmarks/di1_server_ab.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${DI1_AB_OUT:-docs/di1-server-ab}
DUMP=${DI1_AB_DUMP:-$HOME/cachalot-di1-live.jsonl}
ORDER=${DI1_AB_ORDER:-"0 1 1 0"}
# The settle gate checks that the previous arm gave its memory back. At 48 it needs 77 GiB available, which this machine does not reach with
# Hamed's apps open (73.7 at most on 2026-10-31); 40 needs 69. The server itself still holds serve.sh's 48 GiB.
SETTLE_GIB=${DI1_AB_SETTLE_GIB:-40}
mkdir -p "$OUT"
[ -f "$DUMP" ] || { echo "no dump at $DUMP" >&2; exit 1; }
n=0
for k in $ORDER; do
  n=$((n + 1))
  if [ "$k" = 1 ]; then tag=on; else tag=off; fi
  echo "arm $n ($tag): settle $(date +%H:%M:%S)"
  benchmarks/settle.sh --budget-gib $SETTLE_GIB || { echo "settle failed before arm $n" >&2; exit 1; }
  snap=$(mktemp -d "${TMPDIR:-/tmp}/di1-ab-snap.XXXXXX")
  env -u CACHALOT_SERVER_DUMP CACHALOT_PREFILL_ENCODER_ONLY=$k CACHALOT_SNAPSHOT_DIR="$snap" ./serve.sh > "$OUT/arm-$n-$tag-server.log" 2>&1 &
  SP=$!
  t0=$(date +%s)
  until curl -sf http://127.0.0.1:8011/v1/models >/dev/null 2>&1; do
    kill -0 $SP 2>/dev/null || { echo "server died in arm $n" >&2; tail -5 "$OUT/arm-$n-$tag-server.log" >&2; rm -rf "$snap"; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "server not up after 900 s in arm $n" >&2; kill -INT $SP; wait $SP; rm -rf "$snap"; exit 1; }
    sleep 3
  done
  echo "arm $n ($tag): server up $(date +%H:%M:%S)"
  /usr/bin/env python3 -I benchmarks/di1_server_ab_driver.py --dump "$DUMP" --out "$OUT/arm-$n-$tag.json" --arm $n --knob $k --server-pid $SP
  rc=$?
  kill -INT $SP 2>/dev/null; wait $SP 2>/dev/null
  rm -rf "$snap"
  echo "arm $n ($tag): done rc=$rc $(date +%H:%M:%S)"
  [ $rc -eq 0 ] || exit $rc
done
/usr/bin/env python3 -I benchmarks/di1_server_ab_driver.py --score "$OUT" > "$OUT/score.json"
cat "$OUT/score.json"
