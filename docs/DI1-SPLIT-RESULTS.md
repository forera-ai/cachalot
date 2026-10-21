# DI-1 first measurement: results (the decoder half is 46 % of a cold 4,096-token prefill chunk)

Run 2026-10-21 under `docs/DI1-SPLIT-RECORD.md` (six predictions D-1 to D-6, committed before the run in ad72a6e), runtime 0.66.60, `benchmarks/prefill_half_split.py 4096 2` under `guarded_run.sh` (exit 0, 99 s, peak wired 61.4 GiB, peak swap 0, pressure 1). Raw rows: `docs/di1-split-2026-10-21.json`. Every number is **measured** unless marked **derived**.

## 1. Setup and what went wrong first

One process, internal 2-bit bank (`~/DeepSeek-V4.1-Flash-q2g128`, 9.49 MiB per expert), the server's environment (page cache, wired limit 80, fast synch), expert budget **34 GiB** (the guardian refused 44: 64.2 GiB available against 73 needed), `iogpu.wired_limit_mb` 88064, prompt `A_code_review` of 4,096 tokens (the first prompt of `trace_routing.prompt_sources()`), call 0 cold, call 1 warm. Two earlier launches were stopped before producing a number: one was killed when the previous session ended (not by the guardian) and two loaded the **X10Pro FP4 bank** because I did not pass the server's `CACHALOT_EXPERT_BANK`; the numbers below are from the launch that printed `expert bank: ... q2g128 (affine, 2-bit, 9.49 MiB/expert)`.

## 2. Result

| | cold | warm |
|---|---|---|
| wall | 42.86 s (95.6 tok/s) | 38.35 s (106.8 tok/s) |
| encoder half, layers 0 to 19 | 22.85 s | 20.62 s |
| decoder half, layers 20 to 39 | 19.87 s | 17.59 s |
| **decoder share `d`** | **0.465** | **0.460** |
| MoE part, encoder / decoder | 18.53 / 18.35 s | 16.01 / 16.10 s |
| non-MoE part, encoder / decoder | 4.32 / 1.52 s | 4.61 / 1.50 s |
| SSD reads, encoder / decoder | 7,088 / 7,070 | 5,248 / 5,243 |
| after the last layer | 0.14 s | 0.13 s |

Layer 1 costs 3.86 s cold and 4.02 s warm against 0.8 to 1.2 s for every other layer (unexplained here; the first compressed-source layer and the Engram phase are candidates, not tested). Without layer 1 the cold decoder share is 19.87 / (42.72 - 3.86 + 1.0) = **0.50** (derived, replacing layer 1 by a typical 1.0 s). A chunk reads about 354 of a layer's 384 experts per layer (7,088 / 20 layers), so a chunk is a near-full pass over each layer's experts.

## 3. Scores of D-1 to D-6

| id | result | score |
|---|---|---|
| D-1 | cold `d` 0.465 | in interval (0.40 to 0.60) |
| D-2 | warm `d` 0.460 | in interval |
| D-3 | the decoder half starts 49.9 % of the reads (7,070 of 14,158) | in interval (40 to 60 %) |
| D-4 | layer intervals plus the time after the last layer equal the wall (42.86 s) | holds, but **vacuously**: the intervals tile the call by construction, so this is not a test of the marks; the record's check could not fail |
| D-5 | 4,096 tokens in 42.86 s | in interval (40 to 56 s) |
| D-6 | warm / cold = 0.895 | in interval, at its edge (40 to 90 %) |

Five of six held; D-4 is reported as not informative, not as a pass.

## 4. What it means for the lever (derived, then hypothesis)

With `d` = 0.465 and a Hermes system block of about 22.4k tokens in six 4,096-token calls, an encoder-only prefill with an exact 2,541-row decoder tail runs the encoder over all six calls (6 x 22.85 = 137 s of layer time) and the decoder once, over the tail. Because a chunk reads nearly every expert of each layer, one decoder pass over 2,541 rows costs between its row-scaled time (2,541 / 4,096 x 19.87 = 12.3 s, compute-bound) and a full pass (19.9 s): the block's layer time falls from 6 x 42.9 = **257 s** to **about 149 to 157 s, a saving of about 39 to 42 %** (derived; the recorded block is 205 to 227 s, which scales to about 120 to 135 s if the same fraction holds, a **hypothesis**). That is inside the lane's -33 to -46 % bracket and above its 25 % falsifier; the lane's D-1 falsifier (`d` outside 0.35 to 0.65) did not fire.

**Limits.** One prompt of code review text, one machine state, a 34 GiB expert budget (the warm call had 5,248 reads per half, so residency was partial), host-time marks that attribute work to the layer where the host waited, GPU overlap into the next layer is not separated, and layer 1's spike is unexplained. The saving is a derivation from one chunk's split; the exact tail itself is unbuilt and its output equality (not bit-identity: the rounding band is KL 0.0144) and the snapshot policy (DI-3, Hamed's call) are the real work.

## 5. Open

(1) What layer 1's 3.9 s is (a one-layer profile would say, and it is an encoder-side cost that the shortcut does not remove); (2) the L1 metadata matrix and the exact-tail build (DM-3, DI-1 build) and the `pareto.py` gate; (3) DI-4's hit counts for the snapshot policy (needs a longer Hermes dump); (4) Hamed's go on any build.
