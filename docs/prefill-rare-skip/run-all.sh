#!/bin/bash
# ABBA: shipped, rare skip, rare skip, shipped (docs/prefill-rare-skip/RECORD.md; RARE_K picks k, default 2). Quiet gate (no other runtime, 5-minute load under 3.5) before each arm, settle at 40 GiB between.
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
busy() {
  pgrep -f 'Python\.app/Contents/MacOS/Python .*(benchmarks/|cachalot)|deepseek-v41/bin/python|VBCSCompiler|dotnet (exec|build)' >/dev/null && return 0
  sysctl -n vm.loadavg | awk '{exit !($3 >= 3.5)}'
}
quiet() { while busy; do sleep 30; done; echo "quiet at $(date +%T), load $(sysctl -n vm.loadavg)"; }
first=1
for spec in A1:exact B1:rare B2:rare A2:exact; do
  [ $first = 1 ] || benchmarks/settle.sh --budget-gib 40 --timeout 300
  first=0
  quiet; docs/prefill-rare-skip/arm.sh "${spec%%:*}" "${spec##*:}"
done
echo "all arms done $(date +%T)"
