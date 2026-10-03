#!/bin/bash
# GLM-5.3-Flash in the terminal: the GLM counterpart of chat.sh (DeepSeek V4.1 Flash).
#   ./chat-glm.sh                 interactive
#   ./chat-glm.sh --thinking      with GLM's reasoning shown
#   ./chat-glm.sh "a question"    one turn
# Only one runtime runs at a time. HANDOFF section 17.
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >/dev/null 2>&1; then
    echo "a runtime is already running; not starting a second one:" >&2
    pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >&2
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

# The serve script's directory (0.30.0, HANDOFF 18.12): the resident expert set is read back at startup and saved
# after every turn, and the prompt head's snapshot survives a restart. Empty disables it.
export CACHALOT_SNAPSHOT_DIR=${CACHALOT_GLM_SNAPSHOT_DIR-$HOME/.cache/cachalot/prefix-snapshots-glm}

exec ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat \
    --model "$CACHALOT_MODEL_PATH" \
    --expert-budget-gib 52 \
    --max-seq-len 131072 \
    --max-new-tokens 2000 \
    --temperature 0.6 \
    "$@"
