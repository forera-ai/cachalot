# Router sync: 59-66 ms in the tracer against 9-12 ms in the floor profile (record opened 2026-10-12, before the analysis)

Hamed's go: "go, reconcile the router sync" (2026-10-12). Offline: reads the saved raw rows of `benchmarks/results/l4b-raw*/` and the code, loads no model, nothing in `src/` changes.

## Baseline (the apparent contradiction)
- The tracer's sync window (`moe_layer_metal.py`, SYNC_START to SYNC_END around the router `mx.eval`) sums to 56-66 ms a token over the 40 MoE layers in every arm of `docs/L4B-RAW-LAYER-RECORD.md` (1.4-1.65 ms a layer). HANDOFF 18.138 wrote this as a "router sync" of 66 ms and suggested a per-layer floor.
- HANDOFF 18.75 (all-resident, in-process, 512-token context) says: 44 evals a token, **55.5 ms inside `mx.eval`** (40 router syncs: 53.8 ms in eval, 16.5 ms of CPU graph building in the gap before them) and an isolated eval round trip of 0.272 ms (x 44 = 11.9 ms, "9-12 ms"), which mostly overlaps inside the token.
- The two numbers are not about the same thing if the sync window contains the GPU compute that is queued before the router eval (attention, hyper-connection glue, shared expert, the router itself). The hypothesis to test is that the tracer's "sync" is GPU compute plus a small round trip, and that 18.138's wording ("router sync", "per-layer floor") was wrong.

## Prediction (written before the analysis; the aggregate per-layer means were already seen, the percentiles and the layer-type pattern were not)
1. The per-layer mean sync is higher by 0.3-0.7 ms in the layers that carry the heavier attention (compressed attention with the indexer: the layers the aggregate table shows near 1.7-2.2 ms, namely 2, 8, 14, 20, 24, 28, 32, 36) than in their neighbours: a pure round trip would not depend on the layer type.
2. The 10th-percentile sync of a layer row, times 40 layers, is 42-54 ms (the wait-free GPU time), close to 18.75's 53.8 ms in eval.
3. The sync difference between prefetch on and off at the same budget is 3-5 ms a token (the predicted-set arrays are inside the same `mx.eval`; 18.75: prediction costs ~5 ms of an all-resident token).
4. Sync rises with misses (reads slow the GPU part of a token; 18.4): 46 GiB < 34 GiB < 28 GiB < 22 GiB, by 1-3 ms a step.

## Falsifier
Prediction 2 below 35 or above 60 ms (the window holds something other than GPU work, or the percentile is not wait-free); prediction 1 absent (a layer-type-independent window); prediction 3 outside 1-8 ms.

## Limits
Different instruments, contexts (512 tokens against short chat prompts of 20-130 tokens plus 128 generated) and budgets; the tracer's cost (+0.55 ms a token) is inside its rows; `mx.eval` blocks until its inputs are computed, so the window's length is the GPU work queued since the previous eval, not the cost of the eval call.

## Result (measured offline from the raw rows, 2026-10-12; `benchmarks/sync_profile.py`)

Per-layer sync window, 40 MoE layers, all seven recorded arms (2,428 tokens each):

| arm | mean sum (ms a token) | sum of per-layer medians | sum of per-layer 10th percentiles |
|---|---|---|---|
| 46 GiB prefetch on | 59.5 | 57.3 | 47.9 |
| 46 GiB prefetch off | 56.1 | 50.8 | 46.5 |
| 34 GiB on (arm 1) | 63.1 | 64.3 | 48.9 |
| 34 GiB off | 59.2 | 54.2 | 46.6 |
| 28 GiB on | 65.1 | 66.4 | 49.6 |
| 22 GiB on | 66.0 | 66.8 | 50.2 |
| 34 GiB budget 0 | 66.1 | 60.7 | 49.2 |

By layer type (46 GiB on, mean ms): ordinary layers 1.40-1.44 (10th percentile 1.13); layer 0 1.04; heavier layers 1 (1.70), 2 (1.83), 8 (1.79), 14 (2.10), 20 (1.88), 24 / 28 / 32 / 36 (1.62-1.64).

## Against the prediction

1. **Held, a little under at the low end.** Layers 2, 8, 14, 20 are 0.37-0.68 ms above their neighbours; 24, 28, 32 and 36 are about 0.2 ms above (predicted 0.3-0.7). The window depends on the layer type, which a pure round trip would not.
2. **Held.** The ten-percent-quantile sum is 46.5-50.2 ms (predicted 42-54), against 18.75's 53.8 ms inside `mx.eval` for the 40 router syncs.
3. **Held.** Prefetch on minus off: +3.4 ms at 46 GiB and +3.9 ms at 34 GiB (predicted 3-5): the predicted-set arrays share the router's `mx.eval`.
4. **Partly.** Sync rises as the budget falls (59.5, 63.1, 65.1, 66.0) but by +3.6, +2.0, +0.9 per step, not a steady 1-3. And the budget-0 arm has the fewest misses (24.2) and the highest sync (66.1), which a pure misses-slow-the-GPU story does not explain; unexplained.
No falsifier fired.

## Reconciliation

The tracer's "sync" is not a synchronization cost. `mx.eval` blocks until the GPU has finished everything queued since the previous eval (the layer's attention, hyper-connection glue, shared expert, the router itself and, with prefetch, the next layers' predicted-set arrays), so the window is mostly GPU compute. Its wait-free part is about 47-50 ms a token (about 1.13 ms for an ordinary layer), matching 18.75's 53.8 ms in eval at a 512-token context; the isolated round trip of 0.27 ms x 40 = 11 ms (the "9-12 ms") is a part of that, which mostly overlaps with the GPU work inside it. The remaining 9-16 ms between the 10th percentile and the mean (59.5-66.0) is contention and variability: prefetch's extra arrays (3.4-3.9 ms), reads in flight while the GPU computes, and a heavy tail in the distribution.

18.138's wording ("router sync", "1.5 ms a layer, a per-layer floor", "router-sync-bound") was wrong and is corrected in 18.142: the budget-0 token is GPU-compute-bound (about 66 ms of GPU work between router evals, plus 16 ms of waiting and about 30 ms of CPU and other work), not bound by a synchronization overhead. There is no 40-60 ms synchronization saving to find; the levers 18.75 named stand (the CPU graph building outside the evals, 17.9 ms; kernel efficiency of the attention rows).

## Limits
Different instruments and contexts; tracer cost inside; no run done this session for this question, so the reconciliation is read from existing data. The cause of the budget-0 sync being highest is not isolated.
