# Can the Neural Engine make Cachalot faster? Every lever reassessed against measured numbers

Written 2026-10-30 against runtime 0.69.5, after E-1 to E-3 (`docs/APPLE-FM-E1-RESULTS.md`, `docs/APPLE-FM-E2-E3-RESULTS.md`).
Hamed asked whether the Neural Engine can accelerate throughput in any way — speculative decoding, next-layer guesses,
anything else. Labels: measured (with its source), derived, hypothesis. Nothing here was built or run unless it says so.

## 1. Two different things share the name

* **Apple's system model through the Foundation Models SDK** (what 0.69.x integrated): a closed ~3B model behind a text
  API. It runs on the ANE at 4.3-4.7 W (measured, 0.69.5), but it computes only *its own* model: no logits, no
  tokenizer, no access to our tensors (`Capabilities.logits/tokenizer_access/kv_access` false).
* **The ANE as hardware for our own computation**: reachable only through Core ML / Core AI (`coremlc` is in Xcode 27;
  `coremltools`, the usual model builder, is not installed). The ANE runs *compiled, static* programs: weights are
  baked in at compile time, shapes are fixed, the formats are fp16, int8 and palettized (lookup-table) weights.

Every "the ANE speeds up the target" idea is the second kind. The first kind can only take whole jobs off the target.

## 2. Where a token's time is, today (measured)

* **DeepSeek, shipped (budget 0, internal bank):** ~76 ms all-resident exact (0.69.3), ~78-80 ms with budget 0;
  ~53 ms GPU work, ~18 ms host work between 44 syncs (0.60.9). Under budget 0 a miss is *dropped*, not read: decode
  time is not waiting on the drive (the drops cost quality, measured as small, 18.60-18.61).
* **DeepSeek exact path:** token ≈ 80 + 2.0 ms x misses (0.60.10); 57 % of read time exposed (0.64.0).
* **GLM from the X10Pro:** 0.91 s a token at 57 misses (E-2, measured); 57 x 13.5 MiB ≈ 770 MiB a token against a
  ~0.97 GB/s drive ≈ 0.8 s (derived): **the drive is the bottleneck**, and the mean read took 24 ms for a 13.5 MiB
  record (queueing).
* **MiniMax:** decode drive-bound, prefill at the GPU's matmul peak (0.38.x).
* **Prefill, DeepSeek cold:** ~99 tok/s (0.58.2).
* **Agent sessions:** on Hermes's 142-request session, 7 summary calls hold ~28 % of prefill and ~37 % of decode tokens,
  32 title/short calls ~12 % of prefill (derived from the dump, 0.69.5 chat).

## 3. The levers

| # | lever | what the ANE would do | ceiling (derived from section 2) | blocker | verdict |
|---|---|---|---|---|---|
| 1 | **Offload Hermes side requests** to the system model | answer titles and short calls; later, summaries | titles ~6 min a session; summaries ~35-40 min a session (estimated target time 46 min, system-model time ~8 min) | changes outputs; summaries need a fact-retention gate (compression dropped 9 of 11 critical numbers once) | **best measured opportunity**; titles low-risk, summaries gated |
| 2 | **ANE as a second matmul engine** (our own Core ML programs for the dense trunk in prefill) | run attention / shared-expert / dense GEMMs while the GPU runs routed experts | if the ANE sustained ~half the GPU's 18 TFLOP/s fp16 on these shapes, prefill could gain up to ~30-50 % where compute-bound (hypothesis; unmeasured) | ANE weights are compiled in: the streamed 2-bit experts cannot run there, so only the 12 GiB trunk, *duplicated* in another format (12 GiB less expert cache, more misses); every layer would hand activations between MLX and Core ML (two syncs a layer); ANE throughput on these shapes unknown | **the only route to more raw throughput; unpriced**; one microbenchmark decides it (section 4) |
| 3 | **ANE-run expert predictor** (a learned "which experts next" model, several layers ahead) | predict routing earlier and more precisely, for prefetch | DeepSeek shipped: none in time (misses are dropped, not waited for; better prediction = fewer drops = quality). GLM X10Pro: none (the drive is saturated; earlier reads cannot add bandwidth). DeepSeek exact: part of 57 % exposed read time, but lead and width were already swept and closed (§9.19) | the current predictor is the next layer's router on this layer's hidden state, inside the same sync: its compute is negligible, so moving it to the ANE saves nothing; its ~5 ms a token (DS-PRED-FLOOR) is host-side load issue, not math | **no speed lever**; at most a quality lever under budget 0 |
| 4 | **Speculative decoding, ANE drafter** (system model, or our own draft on Core ML) | draft tokens for free beside the GPU | none: even a free draft needs the target to verify K positions through the prefill path (243 ms + 27 ms a position, §9.20) and verify widens expert reads (515 to 600 MiB a token at width 2); the target's own trained drafter (DSpark) closed at 1.03x | the target's verify cost, not the draft | **no-go**, unchanged (`docs/APPLE-FM-SPECULATIVE-FEASIBILITY.md`) |
| 5 | **Speculative routing** (an ANE draft runs a few tokens ahead only to prefetch experts; outputs exact) | predict *future tokens'* experts | same ceiling as lever 3: drops (DeepSeek budget 0) or a saturated drive (GLM, MiniMax) | needs a draft whose hidden states map to the target's routing | **no speed lever** on the shipped paths |
| 6 | **Vision tower on the ANE** | encode images while the GPU decodes | images are rare in agent sessions; the tower is already cached per image (0.58.1) | conversion of the ViT | minor; not worth it now |
| 7 | **Prompt-side helpers** (classification, extraction) | label requests | 0.82 s a call, under ~1 ms of a target token beside decode (E-1); no target time saved unless a request is skipped | — | built, useful only as part of lever 1's routing |
| 8 | **Semantic cache policy** | label prefix snapshots | one recoverable eviction on the measured session, which 3 GiB fixes | — | closed (0.69.4) |
| 9 | **Context compression before target prefill** | shrink cold context | loses 4.9-16 s per 1,000 tokens against 99 tok/s cold prefill, omits facts | the system model writes at ~52 tok/s | closed (0.69.0) |

## 4. What would settle lever 2 (proposed, needs Hamed's go for the install)

A microbenchmark, no runtime: build three fp16 Core ML programs with `coremltools` in a scratch venv (Apple's package,
BSD-3) — a batch-1 GEMV of a shared-expert size (bandwidth-bound, decode-shaped), a 512-token GEMM of an attention
projection (prefill-shaped), and the same with int8 or 4-bit palettized weights — compile them for the ANE
(`computeUnits = .cpuAndNeuralEngine`), and time sustained throughput alone and beside the MLX GEMM stand-in, with the
IOReport ANE/GPU watts. **Go:** the ANE sustains at least ~5 TFLOP/s on the prefill GEMM while the GPU loses under 10 %
beside it, and a GPU-to-ANE activation hand-off costs under ~0.5 ms. **No-go:** any of those fails; lever 2 then closes on
measurement, and the Neural Engine's value to Cachalot is lever 1 only. Estimated 30-45 minutes including the build.

## 5. Recommendation

1. **Lever 1, titles first** (low risk, ~6 min a session, frees the single-flight server for the real turns): a
   server-side switch, off by default, that answers structurally detected title/short side requests from the system
   model; replay the Hermes dump, then one live session. Output-changing, Hamed's call.
2. **Lever 2's microbenchmark** (section 4): the only experiment that can show the ANE adding raw throughput to the target.
3. **Lever 1, summaries**, only after an offline fact-retention comparison against DeepSeek's own summaries on the dump's
   7 calls.

Everything else in section 3 is closed or has no speed ceiling on the shipped paths.
