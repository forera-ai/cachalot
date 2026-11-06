# G2: GLM's system block across restarts and across days

Written 2026-11-06 against runtime 0.82.0, before the pin build. Hamed: "Yes, do G2". HANDOFF section 18.273 when scored.

## 1. Baseline (measured)

Same block after a restart, `restart-check.sh RC2`: the 15,564-token disk snapshot written during the live session loaded, `reused=15564`, prefill 2.09 s, turn 5.73 s, against 208 s cold (`G3-RECORD.md`). So G2's first half needs nothing.

A changed block is the open case. GLM has no in-block pins: any difference re-prefills all of it, 205 s. Rendered with GLM's template, today's first request shares 15,434 of 15,568 tokens with 2026-11-03's block, 15,217 with a 2026-11-02 block and 11,419 with another: the tools come first and the date sits near the end, unlike DeepSeek's (18.65).

## 2. Hypothesis

Keeping the last chunk boundaries inside the block on disk lets a new day resume at the last boundary before the change. Prefill stays in the same 2,048-token chunks from zero, so the result is bit-identical to a cold prefill.

## 3. Design

`GlmModel.BLOCK_PINS` (`CACHALOT_GLM_BLOCK_PINS`, default 3, 0 off): the last three chunk boundaries below a block's end go to disk through `SnapshotStore.persist_pin`, not into memory. `PIN_MAX_TOKENS` rises for GLM; MiniMax stays at 0 (its snapshots are about ten times larger). Pins are pruned apart from blocks, `keep_pins` 8, about 0.3 GiB each.

`pin-check.sh`: three fresh servers with temporary snapshot directories, one short request each. A: today's request, cold. B: restart on A's directory, the 2026-11-03 request. C: empty directory, the 2026-11-03 request, cold. `compare_blocks.py` compares B's and C's block snapshots tensor by tensor.

## 4. Predictions (registered before the run)

- **P-1.** Arm A writes pins at 10,240, 12,288 and 14,336 and the block, and its prefill stays within 15 s of 205 s.
- **P-2.** Arm B reuses 14,336 tokens and prefills the rest in 15-45 s.
- **P-3.** Arm B's and C's block snapshots are equal, every tensor, byte for byte.
- **Falsifier.** P-3 fails: the pins stay off.

Runtime estimate: about 10 minutes.

## 5. Results

Not yet run.

## 6. Kind

Not yet scored.
