#!/bin/bash
# The shipped interactive configuration, as one command that cannot be split.
#
# The equivalent one-liner in docs/HANDOFF.md section 4 carries eight
# environment variables in front of the interpreter. Pasted into a terminal
# that wraps it, zsh runs each wrapped line as its own command: the assignments
# on the leading lines become shell parameters that are never exported, the
# runtime starts without CACHALOT_EXPERT_BANK and silently serves FP4 off the
# USB drive, and the arguments on the trailing line come back as
# "command not found: --max-seq-len". That happened on 2026-09-21 and the
# session read 0.6 tok/s against the configuration's 7.6.
#
# Any argument given here is passed through to `cachalot.cli chat`, so
#   ./chat.sh --temperature 0.2
# overrides the default below (the last value of a repeated flag wins).
#
# --max-new-tokens is 2000 because 1024 cut a coding turn mid-method in three
# separate sessions, and a stop=length reply is not a quality signal. At 2000
# the two Objective-C turns of 2026-09-21 finished on stop=stop at 1,483 and
# 1,788 tokens (HANDOFF section 7.2.4).
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -fl "deepseek-v41/bin/python|cachalot\.cli|Python\.app/Contents/MacOS/Python .*(benchmarks/|cachalot)" >/dev/null 2>&1; then
    echo "a runtime is already running; not starting a second one:" >&2
    pgrep -fl "deepseek-v41/bin/python|cachalot\.cli|Python\.app/Contents/MacOS/Python .*(benchmarks/|cachalot)" >&2
    exit 1
fi

export CACHALOT_MODEL_PATH=${CACHALOT_MODEL_PATH:-/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash}
# Each checkpoint has its own bank and hotlist, named after the checkpoint directory, so pointing CACHALOT_MODEL_PATH at another
# revision (DeepSeek-V4.1-Flash-1017, docs/V41-1017-RUNBOOK.md) never pairs its trunk with another revision's experts.
DS_NAME=$(basename "$CACHALOT_MODEL_PATH")
# The 2-bit bank is read from the internal SSD (6.8 GB/s) when a verified copy exists there, else from beside the checkpoint on the
# X10Pro (1 GB/s, six times dearer a miss; HANDOFF 18.51). The internal copy is trusted only when it carries the marker, written after
# a byte-for-byte compare with the X10Pro copy or after a build that finished and verified.
if [ -f "$HOME/$DS_NAME-q2g128/.cachalot-verified" ]; then
    DS_BANK_DEFAULT="$HOME/$DS_NAME-q2g128"
else
    DS_BANK_DEFAULT="$(dirname "$CACHALOT_MODEL_PATH")/$DS_NAME-q2g128"
fi
export CACHALOT_EXPERT_BANK=${CACHALOT_EXPERT_BANK:-$DS_BANK_DEFAULT}
export CACHALOT_PAGE_CACHE=1
export CACHALOT_MLX_WIRED_LIMIT_GIB=${CACHALOT_MLX_WIRED_LIMIT_GIB:-72}
# The hotlist ranks experts by one checkpoint's routing; another revision starts without one until its own is built. Empty disables it.
if [ "$DS_NAME" = "DeepSeek-V4.1-Flash" ]; then
    DS_HOTLIST_DEFAULT="$HOME/cachalot-hotlist.json"
else
    DS_HOTLIST_DEFAULT="$HOME/cachalot-hotlist-$DS_NAME.json"
fi
[ -f "$DS_HOTLIST_DEFAULT" ] || DS_HOTLIST_DEFAULT=""
export CACHALOT_HOTLIST=${CACHALOT_HOTLIST-$DS_HOTLIST_DEFAULT}
export CACHALOT_HOTLIST_GIB=8
export PYTHONPATH=src
# Shared-memory Metal fences: bit-identical, faster decode in a display-on slow
# window (HANDOFF section 15.9). MLX reads it once, so it has to be in the env.
export MLX_METAL_FAST_SYNCH=${MLX_METAL_FAST_SYNCH:-1}

exec ~/venvs/deepseek-v41/bin/python -m cachalot.cli chat \
    --expert-budget-gib 44 \
    --max-seq-len 32768 \
    --max-new-tokens 2000 \
    --temperature 0.6 \
    "$@"
