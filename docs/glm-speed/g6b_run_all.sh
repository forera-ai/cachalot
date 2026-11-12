#!/bin/bash
# usage: docs/glm-speed/g6b_run_all.sh [budget GiB]   ABBA: S = 0.84.0 shipped (512, top-k 0), P = G6b (1200, top-k 5),
# settling between arms. docs/glm-speed/G6B-RECORD.md. Score: g6_analyse.py benchmarks/results/glm-speed/g6b S1,P1,P2,S2
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
GIB=${1:-46}
for arm in "S1 512 0" "P1 1200 5" "P2 1200 5" "S2 512 0"; do
  set -- $arm
  benchmarks/settle.sh
  docs/glm-speed/g6b_arm.sh "$1" "$2" "$3" "$GIB"
done
echo "all arms done $(date +%T)"
