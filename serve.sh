#!/bin/bash
# The shipped interactive configuration, as an OpenAI-compatible HTTP server.
#
# Same env vars as chat.sh (see that file's header for why the pgrep guard
# and the exported-not-piped form matter), pointed at `cachalot serve`
# instead of `cachalot chat`. Point an agent harness (Hermes, OpenCode,
# Continue, aider, ...) at http://127.0.0.1:8011/v1 with model id
# `deepseek-v4.1-flash` and any placeholder API key.
#
# Any argument given here is passed through to `cachalot.cli serve`, so
#   ./serve.sh --port 8080 --api-key secret
# overrides the defaults below.
#
# --max-seq-len is 65536, not chat.sh's 32768: Hermes Agent refuses any
# endpoint whose /v1/models max_context_length is below 64,000 and never
# sends a request (HANDOFF section 15.1). The extra compressed-KV cache
# costs about 84 MB.
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >/dev/null 2>&1; then
    echo "a runtime is already running; not starting a second one:" >&2
    pgrep -fl "deepseek-v41/bin/python|cachalot\.cli" >&2
    exit 1
fi

export CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash
# The 2-bit bank is read from the internal SSD (6.8 GB/s) when a verified copy exists there, else from the X10Pro (1 GB/s, six times dearer a miss;
# HANDOFF 18.51). The internal copy is trusted only when it carries the marker written after a byte-for-byte compare with the X10Pro one.
if [ -f "$HOME/DeepSeek-V4.1-Flash-q2g128/.cachalot-verified" ]; then
    DS_BANK_DEFAULT="$HOME/DeepSeek-V4.1-Flash-q2g128"
else
    DS_BANK_DEFAULT="/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q2g128"
fi
export CACHALOT_EXPERT_BANK=${CACHALOT_EXPERT_BANK:-$DS_BANK_DEFAULT}
export CACHALOT_PAGE_CACHE=1
export CACHALOT_MLX_WIRED_LIMIT_GIB=${CACHALOT_MLX_WIRED_LIMIT_GIB:-80}
export CACHALOT_HOTLIST=/Users/hamedprooshani/cachalot-hotlist.json
export CACHALOT_HOTLIST_GIB=8
export PYTHONPATH=src
# Shared-memory Metal fences: bit-identical, faster decode in a display-on slow
# window (HANDOFF section 15.9). MLX reads it once, so it has to be in the env.
export MLX_METAL_FAST_SYNCH=${MLX_METAL_FAST_SYNCH:-1}
# The snapshot where an agent's system prompt ends survives a restart, so the
# first request after one reuses it instead of re-prefilling ~13.5k tokens
# (HANDOFF section 15.4). Empty disables it.
export CACHALOT_SNAPSHOT_DIR=${CACHALOT_SNAPSHOT_DIR-$HOME/.cache/cachalot/prefix-snapshots}

# Decode drops every non-resident expert a layer would read (Hamed's default since 0.60.0, HANDOFF 18.66): -22 to -27 %
# a token, no sign of harm in the blind checks (18.60, 18.61). It changes outputs; CACHALOT_DECODE_MISS_BUDGET=off is the exact path.
export CACHALOT_DECODE_MISS_BUDGET=${CACHALOT_DECODE_MISS_BUDGET-0}

# A request without max_tokens gets 8192 (cut to what fits in max_seq_len). At 2000, Hermes's context
# summaries and one long delegate_task call were cut mid-output, and the truncated tool call reached
# the client as raw markup (HANDOFF section 15.10).
# 48 GiB, not 52 (HANDOFF 18.52, 18.53): once the whole system's wired memory passes ~74.5 GiB the GPU pages and an all-resident token
# costs 160+ ms instead of 70. 52 GiB crossed it on this machine (4.8 tok/s against 8.3). The runtime's wired governor (0.54.0,
# CACHALOT_WIRED_CEILING_GIB, default 73) now gives slots back when anything else wires memory; 48 stays the measured default.
exec ~/venvs/deepseek-v41/bin/python -m cachalot.cli serve \
    --expert-budget-gib 48 \
    --max-seq-len 65536 \
    --port 8011 \
    --default-max-tokens 8192 \
    --default-temperature 0.6 \
    "$@"
