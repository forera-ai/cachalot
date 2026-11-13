#!/bin/bash
# usage: docs/glm-speed/g6b_block_run.sh [budget GiB]   G6b's block check (G6B-RECORD.md section 7): the live block cold plus
# one 35-token rung, floor 1,200 in every arm, CACHALOT_GLM_PREFILL_PREDICT_TOPK 5, 0, 0, 5, the per-layer prefill log on.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
GIB=${1:-46}
export G6_RUNGS=25 CACHALOT_GLM_PREFILL_LOG=1
for arm in "B1k5 1200 5" "B2k0 1200 0" "B3k0 1200 0" "B4k5 1200 5"; do
  set -- $arm
  benchmarks/settle.sh --timeout 120
  docs/glm-speed/g6b_arm.sh "$1" "$2" "$3" "$GIB"
done
echo "all arms done $(date +%T)"
