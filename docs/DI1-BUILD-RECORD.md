# DI-1 build: encoder-only prefill with an exact decoder tail, behind `CACHALOT_PREFILL_ENCODER_ONLY` (default off): design, predictions before the equality runs

Written 2026-10-21 against runtime 0.66.63 plus the uncommitted build of this session, **before any equality run on the model**. Charter section 7.2 fields 1 to 3 are fixed here; results go in `docs/DI1-BUILD-RESULTS.md`. A prediction edited after its run is void. It builds the lever priced in `docs/DI1-SPLIT-RESULTS.md` (the decoder half is 46.5 % of a cold chunk; derived saving of a 22.4k block 39 to 42 %) after Hamed's "continue the DI-1 build" (2026-10-21). **Nothing here changes a default**: the knob is off, and an output-changing default is his call.

## 1. Design (derived from the code, checked against the official reference)

**What a decoder layer leaves behind.** In the official `inference/model.py` only `kv_source_layers` (2, 8, 14, 20) compress KV, and the indexer's keys are written by the layers that own them (the same four); layers 24, 28, 32, 36 run their own indexer for **queries** only, against the keys of the source layers (`Indexer.owns_k = layer_id in kv_source_layers`). So the global state after layer 20 is the compressed KV and index keys of layer 20, computed from layer 20's input on every row. Every other decoder-layer state is the 128-row sliding-window ring of that layer's K, read only by the next 128 positions.

**The plan** (`src/cachalot/model/prefill_tail_plan.py`): run layers 0 to 19 over every call as today; for layer 20 run only what later rows read, `compressed_source_kv_only` (the compressor, index keys and compressed-KV writes, no query path, attention or MoE); keep the encoder outputs (the layer-20 input `x` and its `pre_mix`) of the last `TAIL_ROWS` rows; at a checkpoint (a requested boundary, the prompt's end) run layers 20 to 39 over those rows with **empty window rings**. A row's output at a decoder layer is exact iff its 127 predecessors exist and are exact, so exactness enters at `start + 127 j` after layer j; the last layer's window needs layer 38's output from row `n - 128`: `TAIL_ROWS = 127 x 19 + 128 = 2,541`. A brute-force row simulation written without the closed form (`tests/test_prefill_tail_plan.py`) agrees and shows one row fewer fails.

**Correction to the lane file:** DI-1 said the encoder outputs of the last **2,668** rows must be kept (2,541 + 127, for the layer-20 window context). With empty rings that context is the first 127 rows of the same pass, so **2,541** rows suffice (retained memory 2,541 x 4 x 5,120 x 2 B = 104 MB, derived). The lane file keeps its figure with this note.

**Integration.** `_prefill_tokens_impl` has its layer loop moved, verbatim (checked by diff), into `_prefill_layer`; `prefill_tokens(..., encoder_only=True)` stops after layer 19, runs the layer-20 source part and keeps the rows; `finish_decoder()` runs layers 20 to 39 over them and returns the last token's result; `snapshot` and `decode_token` refuse while a decoder half is pending, `reset` and `restore` clear it. `generation.prepare_prompt` with the knob on and a text prompt of more than one call runs every call encoder-only and finishes at the requested boundaries (a snapshot there) and at the end; **the unpinned and in-block intermediate chunk-boundary snapshots are not taken** (DI-4's replay of a real session found none of them served a request). A truncated tail (more rows since the last exact point than 2,541) starts the decoder windows from zeros; a shorter one runs every row with the real windows and is the old computation.

**Not covered:** image prompts and MTP hidden capture (the call refuses or falls back), the MiniMax and GLM runtimes (DeepSeek only).

## 2. Baselines and arms (recorded where they exist)

Prompt: the 6,100-token `A_code_review` prompt of the split record, two calls (4,096 + 2,004), then 16 greedy decode tokens. Arms (one process each, the server's environment and the internal 2-bit bank, 34 GiB, `guarded_run.sh`): **B** the pre-change code (a `git worktree` of HEAD 7dba490), mode `full`; **F** the changed code, mode `full`; **D** the changed code, mode `deferred`. Compared with `benchmarks/di1_tail_equality.py compare`.

## 3. Predictions (registered before the runs)

| id | quantity | prediction | falsifier |
|---|---|---|---|
| E-1 | the refactor is a no-op | F against B: logits, every window ring, the layer-20 caches and the 16 decoded tokens **bit-identical** (probability 0.90) | any difference |
| E-2 | the encoder half is untouched | D against F: windows of layers 0 to 19 and the compressed caches of layers 2, 8, 14 bit-identical (probability 0.85) | any difference |
| E-3 | the final logits | D against F: same top-1 and KL(F to D) below 0.001 (probability 0.70) | top-1 differs or KL above 0.05 |
| E-4 | the decoder windows | D against F: relative max difference of every decoder-layer window ring at most 0.02 (probability 0.60) | above 0.2 at any layer |
| E-5 | layer 20's global state | D against F: compressed KV and index keys of layer 20 relative max at most 0.01 (probability 0.70; the tail rows are recomputed in a different grouping) | above 0.1 |
| E-6 | decode continues the same | the 16 greedy tokens equal (probability 0.75) | the first 4 differ |
| E-7 | the saving | D's prefill wall is 0.65 to 0.85 of F's (probability 0.50; derived: the decoder runs once over 2,541 rows instead of 4,096 + 2,004) | above 0.95 |
| E-8 | memory | D's peak wired exceeds F's by at most 1 GiB (probability 0.80; 104 MB retained) | more than 3 GiB |

## 4. What the outcomes mean

E-1 false means the move of the loop body changed numerics (a revert, not a result). E-3 or E-5 failing means the tail derivation or the layer-20 source-only part is wrong for the real model (the likeliest place: the sparse index scores of the tail rows against keys written in another grouping). E-3 to E-6 holding says the shortcut is equivalent to rounding on this prompt, which is a precondition for the quality arms (`pareto.py` against the Q-NOISE band) and not a verdict on them. E-7 is the measured size of the saving at 6.1k tokens (smaller than at 22k).

## 5. Stop rules

The three runs are about 100 s each; stop on a guardian kill, a swap increase, an exception, or E-1 false. Not a release if E-1 is false.

## 6. A control arm added after the first three results (written before it is run)

The first three runs gave: F equals B bit for bit (E-1 holds); D equals F bit for bit on the encoder windows, the compressed caches of layers 2, 8, 14 and 20, layer 20's index keys, the candidates and the top-k (E-2 holds, E-5 beaten); but the decoder windows of layers 21 to 39 differ by a relative max of 0.10 to 0.45 (E-4's falsifier, 0.2, fired), the final logits by KL 0.0359 with the same top-1 (E-3's 0.001 missed, under its 0.05 falsifier), and the 16 greedy tokens are equal (E-6 holds; per-token decode KL max 0.0079). Whether that is the known rounding of regrouping the MoE (the lane's Q-NOISE: KL mean 0.0144 for a chunking change) or a semantic error in the tail is not decided by those numbers alone. The control is **N**: the changed code, mode `full`, `CACHALOT_PREFILL_CHUNK=3050` (calls of 3,050 + 3,050 instead of 4,096 + 2,004), compared with F by the same script.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| N-1 | N against F is a noise reference of the same size as D against F | N's decoder-window relative max (layers 21 to 39) is at least 0.1 and its logits KL lies between 0.003 and 0.1 (probability 0.60) | N's window relative max below 0.02 (regrouping alone is nearly exact: then D's difference is a defect) |
| N-2 | the decoder windows | D's relative max per layer is within a factor of 3 of N's at 15 of the 19 layers (probability 0.55) | D above 3 x N at more than 10 layers: D is noisier than a regrouping |
| N-3 | the greedy tokens | N's 16 greedy tokens equal F's (probability 0.70) | the first 4 differ |

Reading: N-1 and N-2 holding make D's differences a regrouping effect (Q-NOISE-class), and the quality gate (`pareto.py` against that band) the next step; N-1's falsifier firing, or N-2's, makes the tail suspect and the build is not released until the cause is found.
