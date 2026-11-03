#!/bin/bash
# Two arms, budget 0 (shipped) then exact (docs/v1017-translate-skip/RECORD.md). Quiet gate (no other runtime, 5-minute load under 3.5)
# before each arm, settle at 40 GiB between. About 70 minutes.
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
busy() {
  pgrep -f 'Python\.app/Contents/MacOS/Python .*(benchmarks/|cachalot)|deepseek-v41/bin/python|VBCSCompiler|dotnet (exec|build)|llama-server' >/dev/null && return 0
  sysctl -n vm.loadavg | awk '{exit !($3 >= 3.5)}'
}
quiet() { while busy; do sleep 30; done; echo "quiet at $(date +%T), load $(sysctl -n vm.loadavg)"; }
quiet; docs/v1017-translate-skip/arm.sh A budget0
benchmarks/settle.sh --budget-gib 40 --timeout 600
quiet; docs/v1017-translate-skip/arm.sh B exact
echo "all arms done $(date +%T)"
