# DI-1 first measurement: how a cold prefill chunk's time splits between the encoder half (layers 0 to 19) and the decoder half (layers 20 to 39) (experiment record; predictions before the run)

Written 2026-10-21 against runtime 0.66.59, **before the measurement is run** (the instrument `benchmarks/prefill_half_split.py` is built and its analysis tested on planted marks; the runtime has not been loaded). It is the first job of `docs/DEEPINFER-LEVER-LANE.md` DI-1 ("price first"); Hamed asked for the lane's jobs on 2026-10-21 ("go for all 4 in order"). Charter section 7.2 fields 1 to 3 are fixed here. A long test: the estimate and Hamed's confirmation come first (`feedback-long-test-estimate-confirm`), and the machine gate (the wired limit) is his.

## 1. Baseline (recorded and derived)

* Cachalot's cold prefill runs all 40 layers over every 4,096-token call (`text_decode_runtime.py:2239`). Recorded: 4,501 tokens in 52 s (86.5 tok/s, DS-PREFILL-COLD, int bank, 48 GiB, n = 1); a Hermes system block of about 22.4k tokens costs 205 to 227 s (DS-PREFILL-BLOCK, about 100 tok/s).
* The model is a causal encoder-decoder; an encoder-only prefill with an exact decoder tail of 2,541 rows would skip the decoder half's passes for all but the last 2,541 tokens (derived in the lane file). With `d` the decoder half's share of a cold chunk's time and a prompt of `n` tokens, the saving of the whole prefill, **if the decoder half scales with the rows it processes** (compute-bound), is `d x (1 - 2541 / n)` (0.887 d at 22.4k tokens), and **if it is read- and pass-bound** (each pass reads most of a layer's experts) it is `d x (1 - 1 / passes)` (0.833 d for 6 passes). The two bracket the expectation (derived).
* Not known: `d`. Layers 2, 8, 14 carry the ratio-2 compressors and indexers (encoder side); the decoder half has ratio-1 layers with the sparse indexer at 24, 28, 32 and 36 (config), so `d` could be below or above 0.5.

## 2. What is measured

`benchmarks/prefill_half_split.py 4096 2`: one runtime, one prompt of 4,096 tokens, **cold** (first call) and **warm** (second call, experts resident as the budget allows), with the per-layer marks of `profile_prefill_timeline.py` (host-time marks only, no added synchronisation, `CACHALOT_PROFILE_SYNC` unset). Per layer: the interval between consecutive `moe_end` marks (the call's start for layer 0) and the MoE part; the encoder sum (layers 0 to 19), the decoder sum (20 to 39), the decoder share `d`, the time after the last layer, and the SSD reads that started in each half. The marks attribute host-observed time to the layer in which the host waited; GPU work that overlaps the next layer is attributed there (stated limit).

## 3. Predictions

All **predicted**; probabilities mine.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| D-1 | decoder share `d` of the cold chunk's layer time (sum of the 40 layer intervals) | 0.40 to 0.60 (probability 0.65) | outside 0.35 to 0.65 (the lane file's registered falsifier) |
| D-2 | `d` of the warm chunk | 0.40 to 0.60 (probability 0.60) | outside 0.35 to 0.65 |
| D-3 | the cold chunk's reads | the decoder half starts 40 to 60 % of the reads (probability 0.55) | below 30 % or above 70 % |
| D-4 | the cold chunk's layer sum against its wall | the 40 layer intervals plus the time after the last layer equal the wall within 1 % (a consistency check on the marks; probability 0.95) | more than 3 % apart (the marks miss work) |
| D-5 | the cold chunk's speed | 4,096 tokens in 40 to 56 s (probability 0.60; DS-PREFILL-COLD 86.5 tok/s gives 47 s) | outside 30 to 70 s |
| D-6 | warm against cold | the warm chunk takes 40 to 90 % of the cold time (probability 0.60) | more than the cold time (no effect of resident experts) |

## 4. What each outcome means

* `d` in 0.40 to 0.60: the saving bracket of section 1 puts a cold 22.4k-token block at about -33 % to -44 % of its layer passes' time (derived), and the snapshot-policy question (DI-3, Hamed's call) is worth designing. `d` below 0.35: the shortcut saves less than a quarter and the lever is dropped from the list. `d` above 0.65: it saves more, and the exact tail becomes the first lever of the lane.
* D-4 falsified means the marks miss work (a different phase) and `d` is only a lower bound on what a skipped half would remove.
* The measurement is of one prompt on one machine state; the end-to-end claim needs the build and `pareto.py` (DI-1's later steps). Nothing here changes a default.

## 5. Machine gate, estimate and stop rule (Hamed's confirmation needed)

`sysctl iogpu.wired_limit_mb` reads **0** (the machine rebooted about an hour ago: `sudo sysctl iogpu.wired_limit_mb=88064` is Hamed's to run). Load average 5.3 at 12:46 (other applications); the guardian refuses when pressure is not normal or free plus reclaimable memory cannot hold the budget. **Estimate:** runtime start and model load 1 to 3 minutes (not measured here), the cold chunk about 47 s (D-5), the warm chunk 20 to 45 s: **about 5 to 8 minutes in all, under the guardian's 1,200 s cap**, one process, nothing else running, no repeat unless the guardian or a gate fails. Stop on a guardian kill, a swap increase or a crash.
