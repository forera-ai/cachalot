#!/bin/bash
# Repeat one recording configuration N times and capture the machine state around each run (docs/L4B-DRIFT-RECORD.md, HANDOFF 18.148).
#
#   cd /Users/hamedprooshani/Projects/deepseek-v41-mac
#   benchmarks/drift_repeat.sh [N] [BUDGET_GIB]      # default 3 repeats at 34 GiB; results in benchmarks/results/drift-<budget>gib/
#
# Each run is `token_trace_raw.py` (exact decode, prefetch on, the 24 mixed turns of the raw-row arms) under guarded_run.sh, with
# settle.sh between runs. A state file next to each run records time, load, thermal notes, the busiest processes, VMware, swap.
set -uo pipefail
cd "$(dirname "$0")/.."
N=${1:-3}
BUDGET=${2:-34}
OUT=benchmarks/results/drift-${BUDGET}gib
mkdir -p "$OUT"
state() {  # $1 file, $2 label
  {
    echo "== $2 $(date '+%Y-%m-%d %H:%M:%S')"
    uptime
    pmset -g therm 2>&1 | head -4
    sysctl -n vm.swapusage
    echo "screensaver: $(pgrep -f Flurry.appex >/dev/null && echo running || echo not running)"
    echo "vmware: $(ps -Ao rss,comm | grep -i 'vmware' | grep -v grep | awk '{s+=$1} END {printf "%.0f MB over %d processes", s/1024, NR}')"
    top -l 2 -n 8 -o cpu -stats pid,command,cpu,mem 2>/dev/null | tail -9
  } >> "$1" 2>&1
}
for i in $(seq 1 "$N"); do
  state "$OUT/run$i.state.txt" "before run $i"
  benchmarks/guarded_run.sh --budget-gib "$BUDGET" --max-seconds 3000 --tag "drift$i" -- env PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/token_trace_raw.py --budget-gib "$BUDGET" --out "$OUT/run$i"
  echo "RUN${i}_EXIT=$?"
  state "$OUT/run$i.state.txt" "after run $i"
  [ "$i" -lt "$N" ] && benchmarks/settle.sh --budget-gib "$BUDGET" --timeout 900
done
echo "ALL_DONE"
