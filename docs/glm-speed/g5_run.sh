#!/bin/bash
# usage: bash docs/glm-speed/g5_run.sh [n, default 4]   G5 (HANDOFF 18.279, G5-RECORD.md): blind replay, exact against
# GLM decode miss budget 1, in one guarded process (benchmarks/quality_blind_ab.py, CACHALOT_QB_FAMILY=glm).
# serve-glm.sh's environment except the miss budget, which each arm sets; its own snapshot directory. Run it as a file.
set -uo pipefail
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
N=${1:-4}
D=benchmarks/results/g5-glm-b1
DUMP=$HOME/.hermes/sessions/request_dump_local-923d0cac4e08f458181be56d5f9ff218d57b611ac4e323eb7fb06dc9f4ef9a75_20261102_183608_266378.json
[ "$(sysctl -n iogpu.wired_limit_mb)" = 88064 ] || { echo "g5: sysctl iogpu.wired_limit_mb is not 88064; run: sudo sysctl iogpu.wired_limit_mb=88064"; exit 2; }
[ -f "$HOME/GLM-5.3-Flash-bank/.cachalot-verified" ] || { echo "g5: no verified internal GLM bank"; exit 2; }
mkdir -p "$D"
[ -f "$D/bodies.jsonl" ] || ~/venvs/deepseek-v41/bin/python docs/glm-speed/g5_bodies.py "$DUMP" "$D/bodies.jsonl" || exit 1

export CACHALOT_MODEL_PATH=/Volumes/X10Pro/models/GLM-5.3-Flash-MLX-4bit-MTP
[ -d "$HOME/GLM-5.3-Flash-MLX-4bit-MTP" ] && export CACHALOT_MODEL_PATH=$HOME/GLM-5.3-Flash-MLX-4bit-MTP
export CACHALOT_MODEL_FAMILY=glm CACHALOT_PAGE_CACHE=1 CACHALOT_MLX_WIRED_LIMIT_GIB=80 PYTHONPATH=src MLX_METAL_FAST_SYNCH=1
export CACHALOT_GLM_BANK=$HOME/GLM-5.3-Flash-bank CACHALOT_GLM_PREDICT_TOPK=0 CACHALOT_GLM_SPECULATE_MIN_TOKENS=512
export CACHALOT_SNAPSHOT_DIR=$PWD/$D/snapshots
unset CACHALOT_GLM_DECODE_MISS_BUDGET
export CACHALOT_QB_FAMILY=glm CACHALOT_QB_BUDGET_GIB=46 CACHALOT_QB_DUMP=$D/bodies.jsonl CACHALOT_QB_ROWS=0,1,2,3,4,5
export CACHALOT_QB_TOOL_BODIES=0 CACHALOT_QB_MAX_TOKENS=700

echo "g5: n=$N start $(date +%T)"
benchmarks/settle.sh --budget-gib 44 --timeout 600 || { echo "g5: the machine did not settle"; exit 1; }
benchmarks/guarded_run.sh --budget-gib 46 --max-seconds 14400 --force --tag g5-glm-b1 -- \
    ~/venvs/deepseek-v41/bin/python benchmarks/quality_blind_ab.py run --budget 1 --n "$N" --out "$D" 2>&1 | tee -a "$D/run.log"
echo "g5: done $(date +%T)"
