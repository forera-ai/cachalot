# The v270 Mac jobs: results (DI-1 saves 39 % of a 22k cold prefill inside the noise band; the reference's `act_quant(o)` is quality-neutral, built off; the new `encoding.py` renders every recorded Hermes request identically)

Run 2026-10-30 (local evening, about 23:34 to 00:29 CET) against runtime 0.70.1 plus this work. Record: `docs/V270-MAC-JOBS-RECORD.md` (T-1, T-2, L-1 to L-3, A-1 to A-3, C-1, committed first in c2f2e70 before any build or run). Raw outputs in `docs/v270-results/`. Every number is **measured** unless marked derived. Machine: `iogpu.wired_limit_mb` 88064 (Hamed set it), the internal 2-bit bank, the server's environment, every arm one process under `guarded_run.sh` at a 34 GiB budget with `settle.sh` between, swap 3.2 GB in use before the first arm and not growing. The first launch was refused by the guard's preflight (42.6 GiB available, 63 needed) while an unrelated `llama-server` held memory; Hamed stopped it.

## 1. DI-1 at 22,000 tokens (T-1, T-2)

| prompt | calls | deferred (encoder-only, then `finish_decoder`) | full | ratio | logits KL (full to deferred) | top-1, 16 greedy tokens | decode KL max |
|---|---|---|---|---|---|---|---|
| 14,796 tokens (the default source module, which tops out there) | 4 | 95.8 s | 153.9 s | 0.622 | 0.0359 | equal, equal | 0.0055 |
| **22,000 tokens** (`docs/HANDOFF.md` at c2f2e70 from character 1,057,561, via the new `--text-file`) | 6 | **136.5 s** | **224.6 s** | **0.608** | 0.0127 | equal, equal | 0.0183 |

In both pairs every window of layers 0 to 20, layer 20's compressed KV and index keys, the candidates and the top-k are bit-identical; the decoder windows (layers 21 to 39) differ in rounding (relative max up to 0.68 at 14.8k and 0.75 at 22k). The deferred arm ran first in each pair (cold), so the saving is if anything understated. Peak wired 61.0 against 61.8 GiB at 22k.

**T-1 held** (0.608 in 0.55 to 0.66; the derived 39 to 42 % of layer time, measured as 39.2 % of the whole prefill). **T-2 held.** Deviation from the record, stated: the record named `di1_tail_equality.py --tokens 22000` without a source; its default source produced 14,796 tokens, so the script gained `--text-file`/`--text-start` and the 22k pair was run from a pinned copy of `docs/HANDOFF.md`.

## 2. DI-1's long-context quality gate (L-1 to L-3)

`pareto.py` in its new long mode (`benchmarks/pareto_arms.long.json`): three texts (prose, code, numeric markdown), 8,192 tokens each prefilled in 4,096-token calls, then 192 teacher-forced positions each (576), no battery.

| arm | verdict | dNLL nats [95 % CI] | KL mean / max | top-1 | ms a step |
|---|---|---|---|---|---|
| ref (full, 4,096-token calls) | reference | | | | 155 |
| noise-chunk3000 (full, 3,000-token calls) | noise arm | +0.0049 [-0.0114, +0.0237] | 0.0101 / 0.295 | 95.0 % | 149 |
| **di1** (encoder-only calls, decoder once) | **inside noise** | **+0.0003 [-0.0036, +0.0043]** | **0.0014 / 0.047** | **97.9 %** | 151 |

**L-1 held** (the band exists: KL 0.0101). **L-2 held** (interval contains 0). **L-3 held**, and by a wide margin: DI-1 is seven times closer to the full prefill (KL 0.0014) than a plain re-chunking is (0.0101), with a nine times narrower dNLL interval.

## 3. The reference's `act_quant(o)` (A-1 to A-3)

**Reference change read (measured from the files):** `deepseek-ai/DeepSeek-V4.1-Flash` at revision `314113f` (2026-10-16) adds `act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)` before the grouped `wo_a` einsum in the main attention and in the DSpark block; `act_quant` is the same blockwise FP8 round trip (32 values, power-of-two scales, E4M3) the runtime already implements exactly for its KV path (`fp8_fused_metal.fp8_roundtrip_fused`). The reference's other change (publishing `index_k` on incomplete steps) is already in this runtime.

**Build:** `fp8_fused_metal.act_quant_o()` behind `CACHALOT_ATTN_ACT_QUANT_O` (default off), called before `wo_a` at all seven sites (`attention_layer0.py`, `attention_sliding_window.py`, three in `attention_compressed.py`, `attention_prefill_batched.py`, `dspark_draft.py`; every `block_*` caller reaches `wo_a` through these). `tests/test_attn_act_quant_o.py`: off returns the same array; on equals the repo's round trip bit for bit on three model shapes and inside `mx.compile`. **A-1 held.**

Standard gate (`benchmarks/pareto_arms.actq.json`, 384-token prefill, 576 positions, the 21-task battery, MLX 0.32.4):

| arm | harness verdict | dNLL nats [95 % CI] | KL mean / max | top-1 | tasks | ms a step |
|---|---|---|---|---|---|---|
| ref | reference | | | | 20/21 | 140 |
| noise-chunk96 | noise arm | +0.0028 [-0.0134, +0.0177] | 0.0140 / 0.226 | 94.3 % | 20/21 | 142 |
| **actq** | outside noise | **+0.0047 [-0.0156, +0.0261]** | **0.0171 / 0.379** | 92.5 % | 20/21 | 142 |

**A-2 held** as registered (dNLL in ±0.02 with the interval containing 0; KL mean 1.2 times the noise arm's, under 2). The harness calls it "outside noise" because its rule needs the KL mean at or below the band's (0.0171 against 0.0140); it is a little more divergent from the runtime's own reference than a re-chunking, and not measurably better or worse in NLL or tasks. **A-3 held** (142 against 140 ms, +1.4 %; one process each, not alternated, indicative). The reference is not available as a ground truth on this machine, so "closer to the reference model" is not measured; the gate only says the port is not a loss.

## 4. `encoding.py` 2026-10-16 (C-1)

`diff` against the checkpoint's: tool messages may carry content blocks; a user message of tool results only no longer counts as a new turn; tool-message merging and result sorting move before image collection, so tool-result images follow prompt order; image-token validation runs before merging. All 142 request bodies of the recorded Hermes session (`benchmarks/results/hermes-live-di4/requests.jsonl`), rendered with the server's own message preparation and its `reasoning_effort` handling, in chat and thinking modes: **142 of 142 identical prompt text and identical image lists in both modes**, including the 37 requests with images in tool results (up to 6 images). **C-1 held.** Nothing in the checkpoint was changed.

## 5. What it means (derived; decisions are Hamed's)

* **DI-1 is ready for Hamed's default decision on quality grounds:** -39 % of a 22k cold prefill, and closer to the full prefill than the runtime's own chunk-size noise on the long gate. What is not measured: the server path with the knob on (a live Hermes session), and the snapshot policy (snapshots taken with the knob on differ in rounding and are not marked; §18.221).
* **`act_quant(o)` is optional parity with the reference:** neutral on quality and speed here; turning it on would change outputs (and snapshot numerics would then need a tag bump), with no measured benefit.
* **The `encoding.py` update is safe for this user's traffic:** no recorded prompt changes. Replacing the checkpoint's file is Hamed's call; the runtime would then match the reference's ordering for tool-result images in sessions where results arrive out of call order.

## 6. Limits

One machine state, one 22k prompt, one long-gate configuration (8,192 tokens, three texts); speed numbers are single processes; the DI-1 timings are process-internal `prefill_s`, not server latencies. The encoding check covers one recorded session's traffic.
