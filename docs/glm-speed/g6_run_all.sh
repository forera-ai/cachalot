#!/bin/bash
# usage: docs/glm-speed/g6_run_all.sh [budget GiB]   ABBA: shipped 128, then 1200, 1200, 128, settling between arms. docs/glm-speed/G6-RECORD.md.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
GIB=${1:-46}
for arm in "S1 128" "D1 1200" "D2 1200" "S2 128"; do
  set -- $arm
  benchmarks/settle.sh
  docs/glm-speed/g6_arm.sh "$1" "$2" "$GIB"
done
echo "all arms done $(date +%T)"
