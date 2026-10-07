#!/bin/bash
# GLM-5.3-Flash (MLX 4-bit, Vontra/GLM-5.3-Flash-MLX-4bit-MTP) as an OpenAI-compatible HTTP server.
#
# Same shape and port as serve.sh, so an agent harness switches models by restarting the server:
# ./serve.sh serves DeepSeek V4.1 Flash, ./serve-glm.sh serves GLM-5.3-Flash, both on
# http://127.0.0.1:8011/v1. Model id here: glm-5.3-flash. Only one runtime runs at a time (the
# guard below), because both use the same expert-cache memory.
#
# The routed experts (12,096 x 13.5 MiB) stream from SSD into a 52 GiB wired cache;
# the rest of the model (5.5 GiB) stays resident. Text and images since 0.51.0 (the vision tower loads on the first image); no MTP (priced and closed, HANDOFF 18.40). HANDOFF sections 17, 18.39.
#
# Any argument is passed through to `cachalot.cli serve`, e.g. ./serve-glm.sh --expert-budget-gib 44
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -fl "deepseek-v41/bin/python|cachalot\.cli|Python\.app/Contents/MacOS/Python .*(benchmarks/|cachalot)" >/dev/null 2>&1; then
    echo "a runtime is already running; not starting a second one:" >&2
    pgrep -fl "deepseek-v41/bin/python|cachalot\.cli|Python\.app/Contents/MacOS/Python .*(benchmarks/|cachalot)" >&2
    exit 1
fi

# The internal copy was deleted on 2026-09-25 to make room for MiniMax-M3 (HANDOFF 18); GLM reads over USB
# until it is copied back. CACHALOT_GLM_PATH overrides.
# GLM (priority 3) lives on the X10Pro since 2026-10-03, freeing the internal SSD for DeepSeek's bank (HANDOFF 18.51); an internal copy, if one
# exists again, wins.
if [ -d "$HOME/GLM-5.3-Flash-MLX-4bit-MTP" ]; then
    GLM_DEFAULT_PATH="$HOME/GLM-5.3-Flash-MLX-4bit-MTP"
else
    GLM_DEFAULT_PATH="/Volumes/X10Pro/models/GLM-5.3-Flash-MLX-4bit-MTP"
fi
export CACHALOT_MODEL_PATH=${CACHALOT_GLM_PATH:-$GLM_DEFAULT_PATH}
export CACHALOT_MODEL_FAMILY=glm
export CACHALOT_PAGE_CACHE=1
export CACHALOT_MLX_WIRED_LIMIT_GIB=${CACHALOT_MLX_WIRED_LIMIT_GIB:-80}
export PYTHONPATH=src
export MLX_METAL_FAST_SYNCH=${MLX_METAL_FAST_SYNCH:-1}
# The snapshot where an agent's system prompt ends survives a restart (HANDOFF 17.1), in a directory of its
# own next to DeepSeek's. Empty disables it.
export CACHALOT_SNAPSHOT_DIR=${CACHALOT_GLM_SNAPSHOT_DIR-$HOME/.cache/cachalot/prefix-snapshots-glm}
# The contiguous expert bank (one pread an expert instead of 5-9; bit-identical, HANDOFF 18.34, 18.86): on the X10Pro it reads
# 6.6 % faster and a token is 6.1 % shorter. Used when its directory exists; CACHALOT_GLM_BANK= (empty) or
# CACHALOT_GLM_BANK_ENABLED=0 turns it off. Build: benchmarks/glm_bank.py --write MODEL OUT (~7 min, 163 GB).
GLM_BANK_DEFAULT=/Volumes/X10Pro/models/GLM-5.3-Flash-bank
if [ -z "${CACHALOT_GLM_BANK+x}" ] && [ -f "$GLM_BANK_DEFAULT/bank.json" ]; then
    export CACHALOT_GLM_BANK=$GLM_BANK_DEFAULT
fi

exec ~/venvs/deepseek-v41/bin/python -m cachalot.cli serve \
    --model "$CACHALOT_MODEL_PATH" \
    --expert-budget-gib 52 \
    --max-seq-len 131072 \
    --port 8011 \
    --model-id glm-5.3-flash \
    --default-max-tokens 8192 \
    --default-temperature 0.6 \
    "$@"
