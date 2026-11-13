#!/bin/bash
# usage: bash docs/glm-speed/g4_run_all.sh [budget GiB, default 56]   G4 (G4-RECORD.md): four arm.sh arms, 46, B, B, 46.
# Each B arm starts only when settle.sh counts the guard's full need for B (12 + B + 3 + 8 + 6 GiB) and runs without
# --force, so the guard may refuse it. The 46 arms wait for 73 GiB (the level 18.274-18.277's arms ran at) and keep
# arm.sh's --force, as those arms did. A failed gate stops the run before the next arm. Run it as a file, from Terminal.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
B=${1:-56}
[ "$(sysctl -n iogpu.wired_limit_mb)" = 88064 ] || { echo "g4: sysctl iogpu.wired_limit_mb is not 88064; run: sudo sysctl iogpu.wired_limit_mb=88064"; exit 2; }
echo "g4: budget $B, start $(date +%T)"
for arm in "G4A1 46" "G4A2 $B" "G4A3 $B" "G4A4 46"; do
  set -- $arm
  if [ "$2" = 46 ]; then
    benchmarks/settle.sh --budget-gib 44 --timeout 600 || { echo "g4: stop before $1, the machine did not settle"; exit 1; }
    docs/glm-speed/arm.sh "$1" 46
  else
    benchmarks/settle.sh --budget-gib "$2" --timeout 600 || { echo "g4: stop before $1, under the guard's need for $2 GiB"; exit 1; }
    ARM_FORCE= docs/glm-speed/arm.sh "$1" "$2"
  fi
  grep -h 'preflight\|footprint\|guardian' "benchmarks/results/glm-speed/$1.server.log" | cut -c1-300
done
echo "g4: all arms done $(date +%T)"
