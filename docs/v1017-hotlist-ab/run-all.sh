#!/bin/bash
# ABBA: on, off, off, on. Quiet gate (no other runtime, 5-minute load under 3.5) before each arm; settle at 40 GiB between arms
# (at 48 it needs 77 GiB, which this machine does not reach with its applications open; the arms themselves run at 48 with --force).
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
busy() {
  pgrep -f 'Python\.app/Contents/MacOS/Python .*(benchmarks/|cachalot)|deepseek-v41/bin/python|VBCSCompiler|dotnet (exec|build)' >/dev/null && return 0
  sysctl -n vm.loadavg | awk '{exit !($3 >= 3.5)}'
}
quiet() { while busy; do sleep 30; done; echo "quiet at $(date +%T), load $(sysctl -n vm.loadavg)"; }
first=1
for spec in A1:on B1:off B2:off A2:on; do
  [ $first = 1 ] || benchmarks/settle.sh --budget-gib 40 --timeout 300
  first=0
  quiet; docs/v1017-hotlist-ab/arm.sh "${spec%%:*}" "${spec##*:}"
done
echo "all arms done $(date +%T)"
