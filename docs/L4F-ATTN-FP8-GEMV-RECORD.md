# L4(f) step 3: pricing a faster FP8 GEMV for `wo_b` and `wq_b` (experiment record)

Written 2026-10-15 against runtime 0.64.17, before any measurement of the arms below. Status: **run 2026-10-15 (Hamed's go), two processes A and B; stop rule met; no change to `src/` yet, waiting for Hamed to say whether to build it.** Follows `docs/L4F-FLOOR-ROOFLINE-RECORD.md` (step 2).

## Goal

Step 2 found the FP8 attention projections below a plain `mx.sum` over the same bytes: `wo_b` 0.115 ms against 0.087, `wq_b` 0.104 against 0.087 (42 MB a layer each). The arithmetic ceiling of closing both gaps is 0.045 ms a layer, about 1.8 ms of the 80 ms floor. Price whether a faster `fp8_linear` kernel gets a meaningful part of that bit-identically; build it in `src/` only if the stop rule below is met.

## Baseline (measured, from the repo)

- `micro_attn_projection_roofline.py`, two processes (A / B): `wo_b` 0.1154 / 0.1144 ms, `wq_b` 0.1036 / 0.1022, `wq_a` 0.0392 / 0.0345, `wkv` 0.0276 / 0.0279, `wo_a` bf16 0.1066 / 0.1056; five chained 0.3798 / 0.3776. `mx.sum` ceiling at 42 MB: 485 GB/s (0.087 ms).
- The measured `fp8_linear` arm is not the GEMV alone. It issues the activation-quantize kernel, then the v2 GEMV (`fp8_gemv_decoded`, `_gemv_v2_kernel`), then `.astype(bfloat16)`, then the arm's `mx.sum`. The 0.115 ms therefore contains launches that a kernel change to the GEMV cannot touch. This split was never measured.
- The v2 GEMV decodes each weight byte by a lookup in a 256-entry `constant float` table (`e4m3_table`), four lookups per 32-bit word, 32 per block. Summation order: per block a sequential sum over 32 products, then `acc += block_acc * act_scale * weight_scale` over a lane's blocks, then an xor-shuffle reduction over `lanes_per_row` lanes. Bit-identity means this order is kept.

## Method (all in a new benchmark, no change to `src/`)

New `benchmarks/micro_fp8_gemv_variants.py`, random buffers, no model, 40 distinct layers chained per arm, best and median of 9 reps (same harness as step 2), shapes `wo_b` [5120, 8192] and `wq_b` [32768, 1280]:

1. Decomposition of the shipped path: `full` (`fp8_linear`), `quant` (quantize kernel only, chained), `gemv` (shipped GEMV on a pre-quantized activation, bf16 cast and sum included), `sum` (ceiling).
2. Prototype GEMV variants defined inside the benchmark file, each checked against the shipped GEMV bit for bit on the same inputs before it is timed (a variant that is not bit-identical is reported and not timed): `half` (arithmetic E4M3 decode through a half-precision bit trick instead of the table, NaN code kept equal to the table's 0.0), `half-nonan` (the same without the NaN select, valid for checkpoints that never hold 0x7F/0xFF; reported separately, bit-identical on valid codes only), and threadgroup sizes 64, 128, 512 for the best decode.
3. Equality is checked with the arrays compared as bit patterns over every one of the 256 weight byte values (random bytes in 0..255 cover them).

Under `guarded_run.sh --budget-gib 4`, two processes A then B with `settle.sh` between only if the first result clears the stop rule. No model load, no server, no `sudo`. Arms run sequentially.

## Estimate

About 6 minutes for one process (`wo_b` and `wq_b` weights 42 MB x 40 layers x 2 shapes, 4 to 8 GEMV arms each); 12 with the B repeat. Needs Hamed's go before the run.

## Prediction (written first)

- P1: `gemv` alone (shipped kernel) takes **0.100-0.110 ms** for `wo_b` and **0.090-0.100** for `wq_b`; the quantize kernel plus cast is 0.005-0.015 ms of the `full` time. So at most 70-90 % of the step 2 gap to `sum` is in the GEMV.
- P2: the table lookup is a real limiter (constant-memory gathers serialize across a simdgroup when bytes differ): the `half` decode saves **0.008-0.020 ms** on `wo_b` and **0.005-0.015** on `wq_b`, combined **0.013-0.035 ms a layer**, 0.5-1.4 ms of the 80 ms token.
- P3: no threadgroup size changes the best decode variant by more than 3 %.
- P4: `half` with the NaN select is within 2 % of `half-nonan`.

## Falsifier and stop rule

- P1 fails if `gemv` alone is below 0.095 or above 0.115 ms for `wo_b`.
- P2 fails if the combined saving is below 0.008 or above 0.045 ms a layer (above the arithmetic ceiling of 0.045 means the baseline arithmetic is wrong).
- **Stop rule: no kernel in `src/` unless a bit-identical variant saves at least 0.020 ms a layer combined over `wo_b` and `wq_b` (0.8 ms of the 80 ms floor, 1 %), reproduced in both processes A and B within 5 % of each other.** Below that, the record closes the lever as priced and the numbers are written into LEDGER. A variant that is only bit-identical on valid checkpoint codes (`half-nonan`) counts only if the shipped path is also shown never to see 0x7F/0xFF, and is proposed to Hamed, not shipped by default.
- If a build is warranted: behind `CACHALOT_FP8_GEMV_DECODE` (default off until Hamed decides), a trace-shaped test (the GEMV inside an `mx.compile`d function with no `mx.eval`), a bit-identity test over all 256 codes and both shapes, and a version bump with CHANGELOG and README.

## Limits

Random buffers measure the kernel at the step 2 conditions (40 distinct layers, no reuse of cache contents), not the model in the server, where the 80 ms floor also holds CPU graph building and GPU idle. A saving in the microbenchmark is an upper bound on the saving in the token. 1.8 ms of 80 ms is below the drift of one exact configuration (sd 1.5 %, LEDGER 0.64.14), so the token-level effect would not be provable with an end-to-end A/B; the claim, if any, rests on the microbenchmark plus bit-identity.

## Result (measured on the machine 2026-10-15, processes A then B, bit-identity checked first)

Raw: `benchmarks/results/l4f/fp8-gemv-variants-A.json`, `-B.json`; logs `benchmarks/results/guarded/fp8-gemv-variants_*`. Per layer, best of 9 reps, 40 distinct layers chained (A / B, ms). B started after `settle.sh` timed out at 603 s ("quiet 0/20s" never reached, pressure 1, swap 0, no runtime alive); the chain continued, so B did not run behind a passed gate. A and B agree within 3 % on every arm below.

| arm | wo_b A / B | wq_b A / B |
|---|---|---|
| `mx.sum` ceiling | 0.0876 / 0.0875 | 0.0910 / 0.0865 |
| `fp8_linear` (full) | 0.1128 / 0.1169 | 0.1005 / 0.0968 |
| quantize kernel only | 0.0230 / 0.0233 | 0.0199 / 0.0184 |
| shipped GEMV (+cast, sum) | 0.1107 / 0.1117 | 0.0937 / 0.0931 |
| table variant (control, bit-identical) | 0.1122 / 0.1138 | 0.0928 / 0.0924 |
| `half` decode, tg256 | 0.0811 / 0.0821 | 0.0822 / 0.0808 |
| `half` decode, tg64 | 0.0775 / 0.0784 | 0.0801 / 0.0792 |
| `half` decode, tg128 | 0.0786 / 0.0788 | 0.0806 / 0.0793 |
| `half` decode, tg512 | 0.0798 / 0.0812 | 0.0847 / 0.0840 |

Every `half` variant is bit-identical to the shipped GEMV (all outputs equal as uint32 patterns, random bytes covering all 256 codes, including the NaN code, which the guard maps to 0.0 as the table does). `half-nonan` is not bit-identical on random bytes (every output differs) because 0x7F and 0xFF decode to 480 instead of 0.0; it is dropped, there is no reason to take the risk since the guarded version is faster than the shipped one anyway and its cost was not separated.

**Saving a layer against the shipped GEMV, `wo_b` + `wq_b`:** at the shipped threadgroup 256: A 0.0296 + 0.0115 = **0.041**, B 0.0296 + 0.0123 = **0.042 ms**; at threadgroup 64: A 0.0332 + 0.0136 = **0.047**, B 0.0333 + 0.0139 = **0.047 ms**. About 40 attention layers: **1.6-1.9 ms of the 80 ms floor (2.0-2.4 %)** in the microbenchmark, an upper bound for the token. The half-decode GEMV reaches 62-66 % of 819 GB/s and runs **6-10 % faster than `mx.sum`** over the same bytes: the step 2 "ceiling" was not a ceiling.

**Stop rule met** (at least 0.020 ms a layer, bit-identical, A and B within 5 %: 0.041 / 0.042 and 0.047 / 0.047).

### Against the predictions

- P1: shipped `gemv` alone 0.1107 / 0.1117 ms for `wo_b` (predicted 0.100-0.110, missed by 1 %; falsifier band 0.095-0.115 held) and 0.0937 / 0.0931 for `wq_b` (0.090-0.100 held). The quantize kernel plus cast costs only 0.002-0.005 ms of `full` in the chain (predicted 0.005-0.015, missed low): nearly all of the step 2 gap is in the GEMV.
- P2: **missed high**. Combined saving 0.041-0.047 ms a layer (predicted 0.013-0.035); `wo_b` alone 0.030-0.033 (predicted 0.008-0.020). The falsifier ceiling (0.045) was crossed only at threadgroup 64 (0.047), meaning the step 2 arithmetic ceiling was too low, not that the baseline is wrong. The table lookup is the limiter, by far more than predicted.
- P3: **missed**: threadgroup 64 beats 256 by 4.4-4.6 % on `wo_b` and 2.5-2.0 % on `wq_b`; 512 is worse on `wq_b` by 3-5 %.
- P4: not testable as written (`half-nonan` is not bit-identical on random bytes).

### Limits and next step

Random buffers, one activation per shape, 40 layers; not the model in the server. The saving is a microbenchmark upper bound; 1.6-1.9 ms is below the end-to-end drift (sd 1.5 % = 1.2 ms), so the token effect cannot be proven by an A/B and rests on bit-identity plus this measurement. The same table decode is used by every other FP8 GEMV of the runtime (`wq_a`, `wkv`, shared experts, indexer): the same change is likely to help them too, unpriced. The expert FP8 path that goes through other kernels is not covered.

If Hamed approves: build `_gemv_v2_kernel` with the guarded half decode behind `CACHALOT_FP8_GEMV_DECODE` (default off until he decides), threadgroup 64 or 128 priced per shape, a trace-shaped test (GEMV in an `mx.compile`d function, no `mx.eval`), a bit-identity test over all 256 codes and both lane counts, version bump with CHANGELOG and README. Nothing in `src/` has been touched.
