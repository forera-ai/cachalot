# DeepInfer lane: levers, methods and CUDA-lane techniques found in `deepseek-ai/DeepInfer` and the V4.1 technical report

Written 2026-10-21 against runtime 0.66.56 (HEAD 0959397). Status: **planned, nothing built, nothing run**. No `src/` file, default, version or launcher was touched by the session that wrote this file. Hamed asked for a full-depth read of DeepInfer, a professional checklist of the levers and possibilities it contains, a lane for them in the unresolved-lever lists, and the techniques and methods that help the CUDA side. This file is that checklist.

It adds a research lane. It does not change Hamed's lane order (Hermes, vision, speed), the model priority, or any output-changing default (those stay Hamed's call).

## 0. How to read this file

Evidence labels follow the charter (`docs/RESEARCH-DIRECTION.md` section 7.2) and `docs/CUDA-RESEARCH-PLAN.md` section 0:

| label | meaning |
|---|---|
| **read** | stated in a file this session opened; the file and line are given |
| **recorded** | a Cachalot number from the ledger or HANDOFF, not reproduced here |
| **derived** | arithmetic from read or recorded values; the arithmetic is shown |
| **hypothesis** | an expectation no measurement here tests |
| **unverified** | something the reading could not settle; named so nobody builds on it |

Lever ids: **DI-n** for runtime levers, **DM-n** for methods worth importing into the Mac tooling, **DC-n** for CUDA-lane techniques. Status words: **UNRESOLVED** (open, nobody has priced it), **PRICE FIRST** (a cheap offline or short measurement decides whether it is worth a session), **REFERENCE** (a design note for an existing open lever), **CLOSED** (settled on paper, with the reason, so nobody re-reads).

## 1. Provenance and coverage

**What was read.**

- The repository `https://github.com/deepseek-ai/DeepInfer`, Apache-2.0, shallow clone at `774ce24` ("update Q&A EN doc"; the page shows 3 commits, 11.7k stars). Cloned into a scratchpad, never installed, built or run: it needs Linux, GCC 13, Python 3.12, CUDA 13 and an SM100 (Blackwell datacenter) GPU or an Ascend 950, none of which this Mac or the CUDA lane's cards (RTX 5090 is sm_120, L40S sm_89, V100 sm_70) can provide.
- The official V4.1 technical report and reference code that already sit in Cachalot's checkpoint directory: `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash/DeepSeek_V41_Tech_Report.pdf` (text-extracted with `pdftotext`; sections 1, 2.1 to 2.4, 3.1.2, 3.1.3, 3.2 and 6 read, figures and tables not inspected) and `inference/model.py` (the attention, block and transformer classes read, about a third of the 1,309 lines).
- Both public HF configs (`deepseek-ai/DeepSeek-V4.1-Flash` and `...-1017`, the checkpoint DeepInfer defaults to).

**Coverage of the DeepInfer tree** (count made by script on the clone, 336 tracked non-vendor files, 63,632 lines, doc images excluded):

| class | lines | share |
|---|---:|---:|
| read in full | 21,241 | 33 % |
| read in part (headers, signatures, tests, reference implementations) | 7,108 | 11 % |
| Ascend (`asc/`, `npu_ops/`) mirrors of the CUDA kernels, not read | 7,629 | 12 % |
| not read | 27,654 | 43 % |

Read in full, by group: the engine and scheduler (`csrc/engine/scheduler.hpp`, `sequence.hpp`, `mtp_varlen.hpp`, `flops_estimator.hpp`, `zos.cpp`, `environ.py`, `decode_perf_cache.py`), the worker (`model_runner.py`, `cuda_graph_runner.py`, `worker.py`, `cache_engine.py`, `llm_engine.py`, `utils.py`, `timer.py`), the model (`attn.py`, `mlp.py`, `hc.py`, `index_attn.py`, `cross_layer.py`, `yoco.py`, `engram.py`, `compress.py`, `deepseek.py`, `decoder_layer.py`, `mtp/dflash.py`, `config.py`), the sampler (`sampler.py`, `sampler_utils.py`, `sampling_params.py`), the metadata path (`prepare_tensor_input_v4.cpp`, `input_metadata.py`, `yoco.py`, `prepare_tensor_input_zos.py`, `model_metadata_utils.py`), EPLB (`policy.py`, `__init__.py`, `shuffle.py`), structured output (all four files), the loader (`weights.py`, `config.py`), logging and profiling (`stats_logger.py`, `profile_server.py`), the benchmark and sweep (`examples/benchmark.py`, `run_sweep.sh`, `_trace.py`, `_config.py`, the examples README), the equivalence harness (`examples/yoco/yoco_equiv.py`), the tests that define the methodology (`test_kl.py`, `test_batch_invariant.py`, `test_offline_inference.py`, `test_cuda_sync_guards.py`), the eval README and `passk.py`, the Q&A document and the README.

**Not read, and what that limits.** The bodies of most TileLang kernels (`store_kv_cache`, `load_paged_kv_cache`, `get_kv_index`, `norm`, `gather*`, `prepare_cp_mirror`, `image_embed_ops`, the rest of `fused_transform`, `sampler` and `fused_sample`), the vendored kernel libraries (FlashMLA, DeepGEMM, DeepEP, TileKernels, DeepSelect are git submodules that were not fetched), `csrc/engine/outputs.hpp`, `prepare_sample.cpp`, `block_manager.hpp` (partly), the pybind files, the vision encoder and its processor, `tests/worker/test_state_cache_ops.py`, the eval scoring code, and the Ascend mirrors. So this file is complete for the engine, scheduling, cache layout, model structure, sampling, test and benchmark design. It is **not** a review of any kernel's inner loop. Nothing in the lever list depends on a kernel body, and every lever that would is marked.

**Never executed.** Every DeepInfer statement below is read from source or from the report. No number from DeepInfer's own benchmarks was reproduced (the README gives charts without numbers; the Q&A gives a few figures, quoted with their labels).

## 2. What the reading changes

1. **Cachalot spends the full 40 layers on every prompt token; the model was designed for 20.** The report's Causal Encoder-Decoder (CED, section 2.2) projects the decoder's global KV from the encoder's final hidden state, so "most prompt tokens bypass full decoder computation" and the model "activates 8B parameters per token during prefill and 16B during decode" (**read**, report lines 135 to 137, 327 to 330, 1969 to 1973 of the extracted text). Cachalot's layer-major prefill loops `for layer_id in range(2, N_LAYERS)` over every row (`src/cachalot/model/text_decode_runtime.py:2239`), exactly as the official reference `inference/model.py` does. This is the largest unpriced lever the reading found (DI-1).
2. **Cachalot's checkpoint has the same CED structure as the checkpoint DeepInfer targets.** Both public configs carry `compress_ratios` `[0, 0, 2 x 18, 1 x 20, 0, 0, 0]`, `kv_source_layer_ids [2, 8, 14, 20]`, `index_source_layer_ids [2, 8, 14, 20, 24, 28, 32, 36]`, candidate source layer 20 (2,048 blocks of 8) and DSpark target layers `[37, 38, 39]` (**read**, both `config.json` files). DeepInfer derives its encoder/decoder split as the first layer whose ratio is 1, which is layer 20 (`deepinfer/hugging_face/config.py:158-161`). The `-1017` card calls itself "a minor update" of V4.1-Flash (**read**, model page). The technical report that ships with the original V4.1-Flash weights (Cachalot's checkpoint) already says that post-training "additionally simulate[s] the same replay ... for train-aware adaptation" (**read**, report 3.2.2), so the original weights were adapted to the bounded replay too. What stays **unverified** is how much quality it costs on Cachalot's gate, which is what DI-2 measures.
3. **DeepInfer is not a macOS candidate and has nothing for SSD expert streaming.** Platform enum GPU or NPU only (`deepinfer/platform.py`), every heavy kernel is CUDA or CANN, weights are assumed HBM-resident across 4 to 128 devices, and the only SSD in the design is the closed-source KV store (3FS). Section 6 of this file and the answer given to Hamed on 2026-10-20 explain why a Mac port of it is the wrong target. What transfers is **design knowledge**, not code.
4. **Parity is better than expected.** Image routing bias (`bias_vl`), the Engram image mask, candidate-block pre-filtering at layer 20, shared-KV segments and a DSpark draft already exist in Cachalot (`moe_prefill_batched.py:137`, `vision_ablation.py`, `block_compressed_source_prefill.py:292`, `dspark_draft.py`). The model-level reading found no missing semantics. The levers are about *what the engine does with the model's structure*, not about model correctness.

## 3. The lever checklist

Ranked by expected gain x confidence / cost (`SESSION-PROTOCOL.md` section 2). Numbers are labelled. "Hook" names the Cachalot code a build would touch. No lever here is authorised to build; each says what to measure first.

### DI-1. Encoder-only prefill with an exact decoder tail (the CED shortcut) — PRICE FIRST, then build (first measurement run 0.66.61: `docs/DI1-SPLIT-RESULTS.md`: decoder half 46.5 % of a cold chunk, derived saving 39 to 42 % of a 22.4k block's layer time; build waits for Hamed)

**Source.** Report 2.2 and 3.2.2; `doc/inference_deployment_qa.md:183-187`; `deepinfer/model_executor/models/deepseek.py:188-208` (the boundary layer switches metadata to the consumer view), `decoder_layer.py:134-142` (keeps the full-length shared hidden), `attn.py:672-721` (`_store_yoco_full_kv`: the decoder half still writes main KV and index-K for every token from the shared hidden), `v4_modules/cross_layer.py` (shared-KV segments).

**Mechanism.** Layers 0 to 19 (the encoder) run over every token. A decoder layer's global KV is a projection of layer 20's input, which is the encoder output (**read**: report 2.2 equation 1; official `Attention` docstring "only kv_source_layers compress, the rest read that same cache"). So for a token that is not near the end of the prompt, nothing in layers 20 to 39 is ever read again except its sliding-window (SWA) K and V, and those are read only by the next 128 positions at each layer. The decoder half therefore only needs to run over the last tokens of the prompt.

**Exact tail length (derived, mine; consistent with the report).** SWA dependencies cascade one window per layer, so the first decode token depends on decoder-layer hidden states back to `19 x 127 + 128 = 2,541` positions, and on the encoder output back to `2,541 + 127 = 2,668` positions (the report says 2,560 for `n_win x L/2`). Running the decoder layers over exactly the last 2,541 positions, with the retained encoder outputs as their layer-20 input, reproduces the full forward's state for decode **in exact arithmetic**. This is the exact variant; DeepInfer ships the approximate one (DI-2).

**Cachalot today (read).** `prepare_prompt` cuts a prompt into 4,096-token calls (`generation.py:302-322`) and `TextDecodeRuntime._prefill_tokens_impl` runs layers 0 to 39 over each call. A snapshot is added at every call boundary (`generation.py:411-423`), pinned inside a system block.

**Size (derived).** Layer passes = chunks x layers; token-layers = tokens x layers (compute). 4,096-token chunks, exact tail of 2,541 rows; no saving when the prompt is no longer than the tail:

| prompt (source of the length) | chunks | layer passes: now to new | token-layers: now to new |
|---|---:|---|---|
| 4,501 (DS-PREFILL-COLD) | 2 | 80 to 60 (-25 %) | 180k to 141k (-22 %) |
| 12,288 | 3 | 120 to 80 (-33 %) | 492k to 297k (-40 %) |
| 22,411 (DS-PREFILL-BLOCK, a Hermes system block) | 6 | 240 to 140 (-42 %) | 896k to 499k (-44 %) |
| 65,536 | 16 | 640 to 340 (-47 %) | 2.62M to 1.36M (-48 %) |

The asymptote is -50 %. Prefill is partly read-bound and partly compute-bound (DS-PREFILL-BLOCK 205 to 227 s, about 100 tok/s, **recorded**; a chunk reads most of a layer's experts, DS-PREFILL-SHORT-SPLIT, **recorded**); both terms fall with the passes saved, so the table's two columns bracket the expectation: **-33 % to -46 % of a cold prompt of 12k tokens or more (derived)**. A cold Hermes system block of 205 to 227 s would fall to roughly 125 to 150 s (**hypothesis** on the split between reads and compute).

**Cost and risks.**

- The decoder half needs the encoder outputs of the last 2,668 rows kept until the tail pass: 2,668 x 4 hyper-connection copies x 5,120 x 2 bytes = **104 MiB** (derived, assuming bf16 activations; the whole 22.4k prompt would be 0.85 GiB, which is why only the tail is kept).
- Layer 20 must still produce its shared KV and its index-K for **all** tokens (a projection and a compressor, no MoE, no query path), and the candidate pool it builds (2,048 blocks of 8) is needed only for the tail queries.
- **Snapshot conflict (the decisive design question).** A snapshot at an intermediate 4,096-token boundary needs the decoder-layer SWA rings of the last 128 positions. Skipping the decoder half in non-final calls leaves them missing. Each retained intermediate snapshot would cost one decoder tail pass (20 layer passes), which cancels the saving. Three policies, in order of simplicity: (a) drop unpinned intermediate snapshots in a cold prefill and keep only the pinned end-of-system-block and end-of-prompt ones (DS-PIN-4096 saved 26 s of a 212 s cold prefill once; **recorded**); (b) DI-3 below (deferred completion); (c) keep today's behaviour for any request with boundaries, and use the shortcut only for prompts without them. Which one is right depends on how often an in-block snapshot is hit; that count is **unknown** and is what DI-4's offline pass would also produce.
- Prefill with a continuation (a hit from a full-state snapshot plus a short suffix) cannot use the exact variant: the encoder outputs of the cached tokens are not stored. It needs DI-2.
- The tail pass is a different grouping of tokens into expert-major GEMMs than the chunked forward. Results are equal in exact arithmetic, not bit-identical; the rounding band of a chunking change is **recorded** at KL mean 0.0144, dNLL +0.0081 [-0.0071, +0.0243] (Q-NOISE, `pareto.py` noise arm). The gate is `pareto.py` against that band, not bit-identity.

**First measurement (cheap, no build).** The per-layer cost of a 4,096-token cold chunk split into the encoder half and the decoder half: `benchmarks/profile_prefill_layer_gpu.py` or `profile_prefill_timeline.py` under `guarded_run.sh` on one chunk. If the decoder half is much cheaper than half (for example because the encoder has the ratio-2 compressors and indexers at layers 2, 8, 14), the table above shrinks accordingly. A long test: estimate and get Hamed's confirmation first (`feedback-long-test-estimate-confirm`).

**Draft prediction and falsifier (register formally when building, charter 7.2).** Decoder-half share of a cold chunk 40 to 60 %; end-to-end cold 22.4k prefill -33 % to -46 %. Falsified if the share is outside 35 to 65 % or the end-to-end saving is under 25 %.

**Status.** UNRESOLVED. Hamed decides whether a snapshot policy change is acceptable (it touches the in-block pins of 0.13.0 and 0.59.0).

### DI-2. Approximate bounded replay (keep 140, DeepInfer's setting) — PRICE FIRST with a mask-only quality curve

**Source.** Report 3.2.2; `doc/inference_deployment_qa.md:183-191`; `deepinfer/hugging_face/config.py` and `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:129-168` (`yoco_swa_visibility`: a query at `pos` sees keys from `max(pos - W + 1, data_len - keep_tokens)`), `environ.py:188-194` (mode `off`, `mask`, `compact`; `keep_tokens >= window + max_backtracking`).

**Mechanism.** Replay only the last `keep_tokens` (140 = 128 + 6 backtracking slots, DeepInfer's V4.1 default) through the decoder, truncating the window at the replay start. DeepInfer's own `mask` mode keeps all rows and changes only the visibility, with the same numerics as `compact` (`yoco_equiv.py` proves the two bitwise equal; it does **not** compare against a full forward).

**Why it is separate from DI-1.** (a) It works for continuations and for any suffix length, because it needs no encoder outputs of cached tokens. (b) It is approximate: the report says so and claims a negligible effect, and says post-training simulated the same replay, which makes the truncated state *on-distribution* for the released weights. (c) For Cachalot, whose passes are partly read-bound, the extra saving over the exact tail is the compute of 2,401 fewer decoder rows: -5 points at 22k, and -26 points at 4.5k (derived: token-layers 180k to 93k, -48 %, against -22 % for the exact tail). For a single-chunk suffix there is no pass saving, only compute (579 tokens: 23k to 14k token-layers, -38 %, **derived**), and a read-bound short follow-up gains nothing (DS-PREFILL-SHORT-SPLIT: 246 tokens, 6.7 s, 3,250 reads; **recorded**).

**First measurement (offline, no compute change).** A mask-mode arm: a visibility mask on the decoder layers' SWA in `_prefill_tokens_impl`, swept over `keep` in {140, 256, 512, 1,024, 2,560}, scored by `benchmarks/pareto.py` (paired NLL with block bootstrap, KL, top-1, the 21-task battery) against the full forward and against the noise arm. It costs the same as a normal arm and builds no new compute path. It also decides whether the exact tail of DI-1 can be shortened (the report cites work showing the effective receptive field is far smaller than 2,560).

**Draft prediction and falsifier.** dNLL inside the noise band for `keep >= 256`; top-1 agreement at least 90 % at `keep = 140`. Falsified by dNLL above +0.05 at `keep = 256` or top-1 below 88 %. An output-changing default is Hamed's call regardless (charter and `SKILL.md` section 5).

**Status.** UNRESOLVED.

### DI-3. Deferred decoder completion for intermediate snapshots — UNRESOLVED (variant of the snapshot policy in DI-1)

Store, with an intermediate snapshot, the encoder outputs of its last 2,668 rows (104 MiB, derived) instead of the decoder state, and finish the decoder tail only when a request hits that snapshot. A cold prefill then pays no decoder passes for snapshots nobody hits; a hit pays one 20-layer tail pass over 2,541 rows (the decoder half of one chunk, about the cost the cold prefill saved). Disk cost 6 x 104 MiB for a 22k block (derived) against the 1.5 GiB snapshot budget (README `prefix_cache_bytes`). Price with DI-4's hit counts first. Needs a new "partial snapshot" type in `snapshot_store.py` and `prefix_cache.py`; bump `NUMERICS_VERSION` only if KV bits can move (they should not, in exact arithmetic).

### DI-4. Token-granular (128) global-KV prefix reuse with bounded replay — PRICE FIRST, offline from request dumps, no GPU

**Source.** Report 3.2.1; `doc/inference_deployment_qa.md:163-191` (store only at multiples of 128; hit length from the block KV; back off one block when the SWA state is missing); `worker/cache_engine.py:32,113-117,221-306` (`KVSTORE_GRAIN = 128`, the dropped region for decoder SWA and compression state: "compression windows end at the grain, so only SWA is persisted").

**Mechanism.** The persistent state at a 128-aligned position is only the global KV (main KV and index-K); the compressor ring is empty at a boundary divisible by the ratio, and the SWA windows are rebuilt by replaying the last 128 tokens. The reuse point can therefore be any multiple of 128 inside a cached prefix, not only the chunk boundaries and pins Cachalot snapshots.

**Cachalot today (recorded).** Snapshots at call boundaries and pins; "a mid-block change such as Hermes compression reuses up to the last chunk before it" (README 0.13.0). The average waste at a mid-block divergence is half a chunk, up to 2,048 tokens, about 20 s at 100 tok/s (derived).

**Price.** `benchmarks/prefix_pin_replay.py` already replays a `CACHALOT_SERVER_DUMP` request dump through the real `prepare_prompt` and `PrefixCache` with **no model loaded** and prints the tokens each arm prefilled per request (HANDOFF 15.7 and 15.10: 479,899 tokens under 0.14.0's eviction against 256,597 under 0.15.0's on Hamed's subagent session). Add one arm that credits a hit at the largest multiple of 128 inside the longest shared prefix with an earlier request, minus the 128-token replay, and compare the total tokens prefilled against the existing arms. The per-request distribution of `(grain hit) - (snapshot hit)` is the whole price; it also yields how often an intermediate in-block snapshot was the hit (DI-1's snapshot policy). No GPU, no model, no sysctl. The recorded snapshot size used by that tool is about 5.3 MB plus 3,050 bytes per token.

**Draft prediction and falsifier.** At least a quarter of mid-session requests diverge inside a chunk (not at a snapshot point), recovering a median of 1,000+ tokens. Falsified if 80 % or more diverge exactly at a snapshot point or at the end of the prompt.

**Risks.** Output-changing by the replay approximation (DI-2's gate). Smaller snapshots (global KV only) are a side effect. Cachalot stores about 3 KB per token against the model's native 890 B (derived below), so the format is a second, smaller question.

**Status.** UNRESOLVED; pairs with DI-2 and DI-3.

### DI-5. DSpark: a cost-curve verification scheduler, a rollback-free state ring, and a low-variance acceptance estimate — REFERENCE for the closed lever D2

**State of the lever (recorded).** Closed twice, the last time at 1.03x against a 1.15x bar, because the only multi-position forward was prefill: 242.6 ms plus 26.9 ms per extra position (DS-D2-VERIFY, `docs/SPEED-RESEARCH-2026-10-03.md` D2). It reopens under about 8 ms per verified position and needs a decode-shaped packed verify first. **DeepInfer does not change that arithmetic.** What it adds is design material for the three parts a decode-shaped verify needs:

1. **The scheduler (read, `csrc/engine/mtp_varlen.hpp:150-267`).** Choose the total verified positions by maximising `expected tokens / (baseline_ms(batch) + slope x batch + sampler_ms)`. Expected tokens add the cumulative product of the confidence head's per-position probabilities, ranked greedily across requests; the budget uses the confidences of two rounds ago, the per-request split uses the previous round's; `baseline_ms` is a measured curve and `slope` is refitted online (median of per-size medians over 5,000 steps). For one stream the same rule picks `k` from `E[tokens](k) / cost(k)`. In Cachalot `cost(k)` is `floor + misses(k) x cost per miss` (token = 80 ms + 2.0 ms x misses, recorded), so the curve must be bytes and misses, not graph size. `benchmarks/speculation_policy.py` already holds the family; the import is the *calibrated online cost curve* and the two-round-lag rule.
2. **Rollback-free state (read, `cache_engine.py:88-110,119-139`; `tilelang_ops/compress.py` reference decode).** The SWA ring is `window + max_backtracking` slots and the compression ring `ratio + max_backtracking - 1`, both addressed by absolute position modulo the ring, so rejected draft positions are simply overwritten by the next write and no copy-back exists. This is the missing "KV rollback of the 128-window and compressed KV" in D2's design.
3. **Acceptance accounting (read, `tilelang_ops/fused_sample.py:121-248`, `sampler.py` verification).** Lossless rejection sampling at temperature above 0: accept a draft token with probability `min(1, p_target / p_draft)` using a position-keyed hash uniform, resample the first rejection from the residual `max(p_target - p_draft, 0)`. The kernel also returns `accept_prob = 1 - sum(residual)`, which is `sum_i min(p_target_i, p_draft_i)`, the *expected* acceptance of that row given both distributions. DeepInfer logs `|accept_prob - confidence|` summed per batch as the head's calibration error (`stats_logger.py`, `batch_confidence_error`). For pricing, `sum min(p, q)` is a far lower-variance estimate of acceptance than counting sampled accepts, and it is the right measure at the 0.7 temperature the servers use. Cachalot's measured acceptance (2.85 tokens per forward at width 5, SPEED-RESEARCH line 121) can be re-expressed this way from one pass of `benchmarks/dspark_acceptance.py` with the two distributions saved. It will not reopen the lever alone (the per-position verify cost does), but it removes sampling noise from the one input that is still a measurement.

**Status.** REFERENCE for D2. Re-price only; do not build unless the decode-shaped verify exists.

### DI-6. Top-k-first sampling (`DEEPINFER_SAMPLER_FORCE_TOPK`) — PRICE FIRST (probably closed by arithmetic)

DeepInfer takes the top-k of the raw logits first (benchmark default 1,024; offline example 4,096), then temperature, top-p and softmax in k-space, remapping draft top-k indices to the target's (`sampler_utils.py`, `tilelang_ops/sampler.py:165-224`). Cachalot's sampler runs over the 129,280-wide vocabulary. The price is the sampler's share of the 80 ms floor, which the floor profile does not list separately (head, Engram and router together 3.2 ms; routed experts 6.9; **recorded**, `SPEED-RESEARCH` line 36). If sampling is below 0.3 ms the lever is closed. One micro-benchmark decides it. Note that top-k-first changes the sampled distribution for tokens outside the k, so it is output-changing at the tail; DeepInfer asserts `max_kl == 0 or force_topk == 0` in its own KL test for that reason.

### DI-7. Trigger-gated grammar decoding for tool calls — PRICE FIRST (measure the failure rate before building)

**Source.** `deepinfer/structure_output/*`, `sampling_params.py:39-44`. It wraps `xgrammar`: a grammar (JSON schema, regex or EBNF) compiled off the decode path on idle CPUs, a per-step token bitmask applied to the logits on a side stream, and **trigger tokens**: generation is free until a given token sequence appears, then the grammar binds (optionally forcing the trigger sequence itself with `bound_token_id`). It supports rollback for speculative steps (`MAX_ROLLBACK_TOKENS = nextn + 2`).

**Why it matters.** The trigger design is exactly the shape of an agent's tool call: free reasoning, then a fixed delimiter, then arguments that must be valid. Cachalot's recorded failures in this family are a MiniMax tool call without its namespace token (0.39.2) and invented APIs in generated code (DeepSeek C# case, GLM garbling). A grammar fixes only syntactic validity, not invented APIs.

**Price.** Count malformed tool calls in the recorded Hermes dumps by model. If the DeepSeek rate is near zero the lever is closed. Otherwise the build is a dependency (`xgrammar`, a C++ library with Python bindings; its availability for arm64 macOS is **unverified**), a CPU mask fill per token (its cost is **unverified**: no timing of it was read) and an MLX mask apply. Output-changing in the sense that it constrains.

### DI-8. Single-pass mHC fusion — likely CLOSED, check lever 9.2 first

The report's Single-Pass mHC shifts the mixing coefficients by one block so one traversal of the residual does the update, mixing and coefficient prediction (activation traffic `(4n+4)d` to `(2n+2)d`), implemented as `deep_gemm.mega_mhc` (`hc.py:229-315`). Cachalot already implements the shifted form (the checkpoint has no `hc_head` tensors) and measures hyper-connection glue at 4.2 to 4.9 ms of the 77 to 80 ms floor (**recorded**, `SPEED-RESEARCH` line 36, `L4F-FLOOR-ROOFLINE-RECORD.md`). For a single decode token the residual is `4 x 5,120 x 2 = 40 KiB`: the glue is dispatch-bound, not traffic-bound, which is what lever 9.2 (dispatch count) and 9.16/9.17 (trace and scalar parameters) already attacked and closed. Do not reopen unless a Metal kernel that fuses post, pre, norm and the FP8 cast into one launch is priced against the 4.2 ms.

### DI-9. FP4-packed global KV storage — CLOSED for decode; minor for snapshots

The model's native global KV is E2M1 with one E4M3 scale per 16 channels (288 B per compressed token) plus an FP4 index-K (68 B): 890 B per token over the whole network (**derived** from `cache_common.py:109-137` and the 4 + 4 shared sources; matches the report's 890). Cachalot's snapshot is about 5 MB + 3 KB per token (README line 260, **recorded**), 3.4x larger. At the contexts Cachalot serves (25k tokens, about 75 MB of KV, **derived**) this is 0.15 % of a 48 GiB expert budget, so no decode lever; the snapshot budget (1.5 GiB) would hold 3.4x more snapshots and disk loads would shrink. Only worth doing together with DI-4.

### DI-10. Quality-evaluation protocol — METHOD (see DM-9)

### Items found and closed (so nobody re-reads them)

| item | where read | why closed for Cachalot |
|---|---|---|
| EP/DP/CP, EPLB placement, redundant experts, shuffle | `eplb/*`, `deepseek.py:324-365` | one device, no ranks. The policy objective (minimise the per-step peak rank load over a history window, greedy replica choice) has no single-device analogue: residency maximises hits, not balance. |
| Prefill/decode disaggregation, load balancers, 3FS KV store, 72-hour retention, autoscaling | Q&A | serving fleet. |
| Mega-MoE, DeepEP, DeepGEMM, FlashMLA fused kernels | `mlp.py`, `attention_backend.py` | CUDA-only; the *fusion boundaries* (norm + rope + quantise + cache store in one launch; hc post + pre + norm + cast) are the only transferable idea and are covered by DI-8. |
| Async scheduling (ZOS) | `worker.py`, `zos.cpp` | Cachalot already runs the host check one step behind (0.31.0 design, D3). |
| Batch-invariant inference, deterministic sampling | `environ.py:197` | single stream. The *position-keyed RNG* behind deterministic sampling is imported as DM-6. |
| Engram tables in host memory with GPU gather | `engram.py`, `weights.py` | Cachalot's tables live on SSD with prefetch (0.9.0). The report's one prefetch point (the first module overlaps the first block) is already how `_start_engram_prefetch` is used (`text_decode_runtime.py:2064`). |
| Hierarchical sparse indexer | report 2.3.2, `index_attn.py`, `candidate_blocks.py` | present (candidate source layer 20). |
| Vision pre-processing | `ve/processor/image_processor.py` | official `inference/image_processor.py` is local and is Cachalot's reference; not re-diffed. |

## 4. Methods worth importing into the Mac tooling

Each says what DeepInfer does (**read**) and how it maps onto an existing Cachalot instrument.

- **DM-1. A host-sync budget guard.** `utils.py:72-140` wraps model launch and sampling in torch's `set_sync_debug_mode("error")` through a per-thread intent stack, with `allow` scopes for lifecycle code and seven tests (`tests/test_cuda_sync_guards.py`: nesting, baseline restore, an interleaved-exit leak, a stress run). Cachalot's equivalent bug class is documented in `SKILL.md` section 4: `mx.eval` inside a compile trace, a host read-back that is a second GPU round trip (about 21 ms a token on MiniMax). **Import:** a test fixture that counts `mx.eval`, `.item()` and `np.array(mx_array)` calls in a decode token and asserts the expected 44 for DeepSeek (the floor's 44 host syncs), failing on any extra. Cheap, protects the floor.
- **DM-2. Steady-state windows that drop interference.** `examples/benchmark.py:380-456` and the examples README 47 to 58: warm-up, then a window of intervals, discarding any interval that collides with a periodic background event (the MoE load-balance flush), logging off during the window, and **the profiler run last** because it leaves a 7 to 11 % residual slowdown and inflates step time by up to 50 %. `gc.freeze()` before measuring. **Import:** the same mask for Cachalot's periodic events (governor fits, memwatch samples, snapshot writes), and a rule that no timing is taken after a profiler or sampler has been attached. Consistent with `MEASUREMENT.md` and the slow-window sampler; the new part is discarding by event rather than by threshold.
- **DM-3. A two-level equivalence ladder with weightless L1.** `examples/yoco/README.md` and `yoco_equiv.py`: L1 compares metadata (visible keys, store slots, compressed ranges) for a matrix of boundary shapes on one device with no weights, against an analytic reference (`_reference_visible`); L2 dumps teacher-forced step logprobs per rank to JSON with a format version and an environment snapshot, and a separate compare script (no framework import) refuses to compare unless every environment variable except the one under test is identical. **Import:** for any output-changing lever (DI-1, DI-2, DI-4) write the L1 matrix first (prompt shorter than, equal to, longer than the tail; chunk wholly before, crossing, wholly inside the replay window; the boundary on a chunk start; mixed batches) so the metadata bugs are found without loading 142 GiB. The compare-offline and env-comparability rules fit `pareto.py`.
- **DM-4. A kernel contract suite.** Every TileLang kernel ships a reference function, a pytest against it and a `benchmark` test that records `time_us` and achieved bandwidth (`io_bytes / time`); the harness gates regressions at 15 % **and** 3 microseconds (`tilelang_ops/conftest.py`: `TK_BENCHMARK_REGRESSION_THRESHOLD=0.15`, `TK_BENCHMARK_MIN_DELTA_US=3.0`) against committed baselines, plus a per-kernel GPU-memory profile. **Import:** for Cachalot's Metal kernels (`fp8_fused_metal.py`, the attention and gather kernels), a reference, a test and a bandwidth-fraction record against the M3 Ultra's 819 GB/s, with the two-threshold gate so a 2 % wobble on a 5 us kernel does not fail.
- **DM-5. Raw-quantity JSONL and a consistency assertion at plot time.** `benchmark.py` appends window tokens, window seconds and the configuration, not tokens per second; `plot_pareto.py` refuses to draw points whose model, parallel size, context, `nextn` or redundancy differ; each sweep point is a fresh process on a fresh port (TIME_WAIT); `PROFILE=0` for sweeps. The accounting ("full-acceptance" tokens) is stated explicitly and a note warns not to multiply a varlen throughput by acceptance twice. **Import:** the same discipline for Cachalot Lab's comparisons (raw quantities, assertion on mismatched manifests).
- **DM-6. Position-keyed random numbers for paired sampled A/Bs.** `tilelang_ops/random_sampling.py`: the uniform for a sample is a SplitMix64 hash of `(seq_seed, position)`, independent of batch composition and order; draft and target use distinct streams. Two arms of a lever test that share the seed then share their noise (common random numbers). **Import:** the MiniMax derail rate was measured at 5/12 (temperature 1.0), 1/12 (0.7), 0/12 (0.5); a position-keyed sampler lets the 12-run comparisons be paired and shrinks the interval for the same runs. Output-neutral for the default sampler if it stays opt-in.
- **DM-7. Analytic per-step cost with live roofline fractions.** `csrc/engine/flops_estimator.hpp` computes each step's FP8-equivalent FLOPs from the model (attention pairs by compression ratio, top-k and candidate caps, SWA pairs, the producer/consumer row split, hyper-connection mixes, Engram projection, logits) and the stats logger reports MFU against a table of peaks. Cachalot has `whatif.py`, `cache_sim.py` and the floor roofline (record L4F); it does not report a per-token bytes-and-FLOPs fraction in the server. **Import:** an analytic bytes per token (11.94 GB, recorded) and a live fraction of the chip's bandwidth in `/v1/stats`.
- **DM-8. Calibration checks as first-class metrics.** `batch_confidence_error` for the draft's confidence head, `moe_lb` for load, `prefill_caching_rate`, per-position accept rates and average verified length, all in one step line (`stats_logger.py`). **Import:** when DSpark is priced again, log accept rate per position and the calibration error from DI-5(3), not a single tokens-per-forward number.
- **DM-9. A quality protocol with an unbiased Pass@k.** `eval/common/passk.py` uses `1 - C(n-c, k)/C(n, k)` (the HumanEval estimator), approximates a task with fewer than k samples by `min(k, n)` instead of dropping it (dropping biases the score up), refuses a Pass@1 unless `coverage_complete`, and reports the agreement between the rule judge and the LLM judge. The offline test compares per-sample decode perplexity **exactly** against a per-platform golden (`tests/models/offline_decode_ppl.json`, refreshed by a script, with a lock for multi-node). **Import:** the estimator and the agreement metric for the 21-task battery; a golden file of exact teacher-forced NLL per (platform, workload) for bit-identical levers, refreshed deliberately.
- **DM-10. Decode-versus-prefill self-consistency as a test.** `tests/models/test_kl.py`: run teacher-forced generation to get decode-path logprobs, then feed the same tokens back as a prompt and compare the prompt logprobs (KL and rmse), with the prompt portion required to match exactly. **Import:** if `benchmarks/nll_sanity.py` does not already pair the decode path with the batched prefill path on the real checkpoint, this is the cheapest invariant test of the two paths DI-1 and DI-2 would diverge.
- **DM-11. Test isolation hygiene.** An autouse fixture snapshots and restores `os.environ` and re-arms the lazily-resolved environment descriptors (`tests/conftest.py`); multi-rank tests run in their own process group with a bounded lifetime and `killpg` (`tests/models/_runner.py`). Cachalot reads many knobs at import; the same fixture removes order dependence in `pytest -q`.
- **DM-12. Persisting measured curves with a key and a time limit.** `decode_perf_cache.py` writes the profiled batch-size curve to a file named by a SHA-256 of (model, batch sizes, parallel layout), with a one-hour TTL read window and a longer delete window, atomic temp-and-rename with `fsync`. Matches the charter rule that constants carry date and regime; useful if the governor's fitted constants are ever persisted.

## 5. Techniques for the CUDA lane

What DeepInfer's CUDA code teaches, mapped to the open observations of `docs/CUDA-RESEARCH-PLAN.md`. None needs DeepInfer to run; each is an arm or a protocol for the existing harness (`benchmarks/cuda/`), and each needs Hamed's go for any GPU time, as every lane item does.

- **DC-1. Reserve streaming multiprocessors for communication instead of letting copies contend.** `userbuffer.py:48-60` (`limit_num_sms`) lowers the compute libraries' SM count by `DEEPINFER_USERBUFFER_COMM_SMS` while a communication kernel runs, and `environ.py:108` sets `NCCL_NUM_SMS=32`. The lane's open question is the slowdown slope (extra compute per millisecond of overlap with prefetch copies: 0.0634 at a 1.0 ms layer to 0.1534 at 0.15 ms on a fast 5090; flat near 0.09 to 0.10 on two L40S cards; 0.2234 on a slow-population 5090). **Arm:** run the stand-in compute with a reduced SM count (or a stream priority) while the copies run, and measure whether the slope falls. Falsified if it does not move more than the replicate spread (about 3 to 4 %).
- **DC-2. Copy with SMs, not the copy engine, for many mid-size transfers.** The Q&A (line 157) says the CUDA copy engine "performs poorly on the fine-grained copies of KV cache blocks" and DeepInfer swaps in with GPUDirect RDMA from a NIC, which saturates PCIe "without consuming SMs". Our expert copies are 9.95 MB (not fine-grained), but the lane found copy time depends on the destination/source *buffer* (0.18 against 0.35 ms for pinned 9.95 MB buffers, B1/B2) and on the instance population. **Arm:** a copy implemented as a kernel reading pinned host memory through a mapped (zero-copy) pointer and writing the slot, against `cudaMemcpyAsync`, per buffer, on both a fast-population and a slow-population instance, with the L2 flush of the plan's measurement rules. Falsified if the slow class is unchanged.
- **DC-3 (run 0.66.59, `docs/CUDA-PIN-ROUTE-RESULTS.md`: registering your own memory 14 to 26 % faster than `cudaHostAlloc` in the interleaved pattern, allocation order the larger effect). Pin existing memory with `cudaHostRegister` as well as allocating pinned buffers.** `platform.py:65-68` pins with `cudaHostRegister(ptr, nbytes, 0)`, and `shared_memory.py:115-119` registers its POSIX shared-memory buffer that way (memory that already exists), while the engine's per-step tensors use `pin_memory=True` allocations (`prepare_tensor_input_*`). B1/B2 compared allocation routes and huge-page backing and falsified huge-page backing as the cause. **Arm:** register an ordinary aligned mapping versus allocate pinned, same 9.95 MB, fresh process, per instance class. Cheap (minutes), and it either explains or removes the buffer-dependent rate.
- **DC-4. Per-segment device time inside a CUDA graph with no synchronisation.** `timer.py:78-217` bakes event-record nodes into a captured graph (captured with `keep_graph=True`, instantiated, nodes matched by event identity) and swaps two event buffers with `set_event` before each replay; `flush_out` reads a finished step's pair after the step's end event. The lane's stage B found that half of the host term in stage A was the harness's own instrumentation. **Arm:** if the stand-in or the layer loop is captured as a graph, time segments this way and compare the quoted host term. Falsified if the quote does not move.
- **DC-5 (run 0.66.59: no synchronising call in the issue loop, planted control seen). Prove no hidden synchronisation in the issue loop.** DM-1's guard (`set_sync_debug_mode("error")` around the launch of N3's issue loop) turns "the issue loop ran 1.35 ms" into "and it contained no blocking call". Cost: zero, one wrapper.
- **DC-6 (run 0.66.59: the rule built on the interleaved pattern reads any instance slow; needs the batch-first pattern). Population classification by a baseline kernel with a committed threshold.** DM-4's regression gate (15 % and 3 us against committed baselines) applied to one canonical copy and one canonical stream kernel at the start of every session classifies the instance as fast or slow population *before* the registered run, replacing the post hoc median-source-copy reading (0.194 versus 0.349 ms). The lane already reads the median in the smoke; the change is to commit the threshold in the record.
- **DC-7. Synthetic routing with a controlled skew.** `mlp.py:329-357` generates routing by drawing scores that are `U(0, f)` for the local experts and `U(0, 1)` for the rest, with `f` solved by a mean-field formula so that one rank receives `r` times the average share; `model_runner.py:444-512` profiles with it (`FORCE_IMBALANCE_RATIO = 1.8`) and `gate_force_random` makes the decode cost curve routing-independent. **Arm:** a routing generator with a target hit rate (a hot set drawn with a chosen mass) for the CUDA replay arena, so the predictor and copy-count findings can be swept over locality instead of measured only on the recorded DeepSeek and MiniMax traces. It separates "workload locality" from "model" in the transfer question the plan asks (plan section 1).
- **DC-8. A measured cost curve before a scheduling decision.** `model_runner.py:561-631`: one warm-up replay and five timed replays per batch size with CUDA events, trimmed mean, averaged across the group, run with forced random routing; cached by DM-12's key. The lane's models (`whatif`, the window model, the slowdown fit) are fitted offline; a short on-instance profile of the stand-in at each C in the plan, stored with its manifest hash, would make the fit inputs reproducible across instances.
- **DC-9. Do not try to run DeepInfer on the lane's cards.** SM100 only; the 5090 (sm_120), L40S (sm_89) and V100 (sm_70) cannot, and a B200 class machine with 4 to 8 devices is needed even for the `mask-compact` L2 check. The L1 metadata check (`yoco_equiv.py mask-compact`) needs the compiled extension but no weights; whether it can be built without CUDA is **unverified**. License: Apache-2.0; porting an idea is free, copying code requires keeping their notices in an MIT repository.

## 6. Why not ask DeepSeek for a macOS version (summary of the 2026-10-20 answer)

The repository states its scope as NVIDIA SM100 and Ascend 950, has no Metal path, assumes HBM-resident weights, and ships 3 commits and an offline-inference release. The right contact, if Hamed wants one, is a specific technical question after Apple-side numbers exist: which keep length and which boundary cases the report's experiments covered (the report names "SWA state reconstruction at cache-resumption boundaries" as an area it will keep stress-testing), and whether the exact 2,560-token replay was ever compared with the 128-token one. Nothing is sent without his decision.

## 7. Order of work and what needs Hamed

Ranked by expected gain x confidence / cost:

1. **DI-4 offline pass** (hours, no GPU, no model): the hit-granularity distribution from the recorded dumps. Also yields the in-block snapshot hit counts that decide DI-1's snapshot policy.
2. **DI-1 first measurement:** the encoder/decoder time split of one cold chunk (`profile_prefill_layer_gpu.py`). A long test: estimate and confirmation first.
3. **DI-2 mask-mode quality curve** with `pareto.py`: decides the approximate variants and shortens or confirms the exact tail.
4. **DM-1 sync-budget test** and **DM-11 fixture**: small, protective, no behaviour change.
5. DI-1 build (L1 matrix first, DM-3), then the snapshot policy that fits the measured hit counts.
6. DI-5 / DI-6 / DI-7 prices as time allows.
7. CUDA arms DC-1 to DC-8 only with Hamed's go for GPU time. The cheapest are DC-3, DC-5 and DC-6 (minutes, no new harness).

**Needs Hamed:** any change to snapshot behaviour (DI-1/3/4); any output-changing default (DI-2/4/6/7); the confirmation before each long run; GPU time for DC items; whether to contact DeepSeek; the version bump and release routine for this documentation change (not done: nothing is committed or pushed).

## 8. Evidence index (for verification by the next session)

DeepInfer paths are relative to the clone at `774ce24`; report lines are of the `pdftotext -layout` extraction.

| claim | where |
|---|---|
| CED computes only the first half of the layers in prefill; decoder KV from `H_{L/2}` | report 2.2 (lines 395-411); `doc/inference_deployment_qa.md:183-187` |
| exact decoder SWA reconstruction needs `n_win x L/2` tokens; bounded replay is approximate and simulated in post-training | report 2.2 and 3.2.2 (lines 405-411, 990-1030) |
| encoder/decoder split = first ratio-1 layer | `deepinfer/hugging_face/config.py:158-161` |
| visibility rule of the truncation | `csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:129-168,203-346` |
| `mask` and `compact` are bitwise equal; no comparison with a full forward | `examples/yoco/yoco_equiv.py:864-937,979-997` |
| decoder half still writes KV for all tokens | `deepinfer/model_executor/models/v4_modules/attn.py:672-721` |
| state rings sized for rollback | `deepinfer/worker/cache_engine.py:88-110,119-139,221-306` |
| verification scheduler | `csrc/engine/mtp_varlen.hpp:150-267`; cost curve `worker/model_runner.py:561-631`, `decode_perf_cache.py` |
| rejection sampling, residual, expected acceptance | `layers/tilelang_ops/fused_sample.py:121-248`; `layers/sampler.py:410-595`; `tilelang_ops/sampler.py:584-589` |
| trigger-gated grammar | `structure_output/matcher.py:58-206`; `sampling_params.py:39-44` |
| sync guard | `deepinfer/utils.py:72-140`; `tests/test_cuda_sync_guards.py` |
| in-graph event timers | `deepinfer/timer.py:78-217` |
| benchmark windows and profiler-last rule | `examples/benchmark.py:380-456,512-519`; `examples/README.md:47-58` |
| kernel regression gate | `deepinfer/model_executor/layers/tilelang_ops/conftest.py` |
| SM reservation | `deepinfer/model_executor/parallel_utils/userbuffer.py:48-60`; `environ.py:108-109` |
| synthetic skewed routing | `deepinfer/model_executor/models/v4_modules/mlp.py:329-357`; `worker/model_runner.py:444-512` |
| native KV bytes (288, 528, 68 per token; 890 total) | `layers/tilelang_ops/cache_common.py:109-137`; report conclusion (line 1969) |
| Cachalot prefill loop and chunking | `src/cachalot/model/text_decode_runtime.py:1977,2239`; `src/cachalot/model/generation.py:302-454` |
| recorded Cachalot prefill numbers | `docs/LEDGER.md` DS-PREFILL-COLD, -BLOCK, -SHORT, -SHORT-SPLIT, DS-PIN-4096, Q-NOISE, DS-D2-VERIFY |
