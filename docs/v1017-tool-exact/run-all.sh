#!/bin/bash
# ABBA: budget0, toolexact, toolexact, budget0. Quiet gate (no other runtime, 5-minute load under 3.5) before each arm, settle at 40 GiB between.
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
busy() {
  pgrep -f 'Python\.app/Contents/MacOS/Python .*(benchmarks/|cachalot)|deepseek-v41/bin/python|VBCSCompiler|dotnet (exec|build)' >/dev/null && return 0
  sysctl -n vm.loadavg | awk '{exit !($3 >= 3.5)}'
}
quiet() { while busy; do sleep 30; done; echo "quiet at $(date +%T), load $(sysctl -n vm.loadavg)"; }
first=1
for spec in A1:budget0 B1:toolexact B2:toolexact A2:budget0; do
  [ $first = 1 ] || benchmarks/settle.sh --budget-gib 40 --timeout 300
  first=0
  quiet; docs/v1017-tool-exact/arm.sh "${spec%%:*}" "${spec##*:}"
done
echo "all arms done $(date +%T)"
