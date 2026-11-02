#!/bin/bash
# Two arms, quiet gate before each, settle between: A exact (floor, per-miss cost, hit rate), C shipped budget 0.
# The quiet gate reads the 5-minute load average (field 3 of `{ 1m 5m 15m }`; the 0.60.10 copy read field 2, the 1-minute one).
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
busy() {
  pgrep -f 'Python\.app/Contents/MacOS/Python .*(benchmarks/|cachalot)|deepseek-v41/bin/python|VBCSCompiler|dotnet (exec|build)' >/dev/null && return 0
  sysctl -n vm.loadavg | awk '{exit !($3 >= 2.5)}'
}
quiet() { while busy; do sleep 30; done; echo "quiet at $(date +%T), load $(sysctl -n vm.loadavg)"; }
quiet; docs/v1017-floor/arm.sh A
benchmarks/settle.sh --budget-gib 48 --timeout 600
quiet; docs/v1017-floor/arm.sh C CACHALOT_DECODE_MISS_BUDGET=0
echo "all arms done $(date +%T)"
