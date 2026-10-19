# CUDA lane: offline price of better expert predictors (experiment record; Mac only, no GPU)

Written 2026-10-20 against runtime 0.66.35, **before any predictor-pricing code**. This is v240 candidate job 1. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void (plan section 12). Labels follow plan section 0.

## 1. What this asks, and what it cannot claim

N3 stage B and C showed that on the RTX 5090 the recorded predictor hides 0.339 of the demand exposure at 1 ms layers (bound 0.365) and 0.096 at 0.3 ms, against an oracle's 0.85 and 0.71, and that fewer copies (a weight gate) does not help. What is left is **coverage**: the share of demand misses a predictor turns into prefetched hits. The question: **can a predictor built only from data the trace holds beat the runtime's recorded next-layer predictor on the serial bound `1 - misses_with / misses_without`, at the same availability (known when layer l-1's router has run)?** Priced offline with the stage A/B bookkeeping (`prefetch_coverage.py` semantics: LRU arena, prefetched keys inserted as most recently used, covered and wasted counts).

It cannot claim: any time on any hardware (the bound is a serial model; stage B showed the recorded predictor reaches 93 % of it on a 5090 at 1 ms and far less at 0.3 ms); that a predictor is implementable in a runtime (the co-activation table below is trained offline from the same trace and its table size is not priced); generalization beyond this one trace (`routing-trace-deepseek-long`, the Mac's workload mix) and one split of it; anything about another model.

## 2. Baseline (recorded)

At 17 % resident on the first 400 decode tokens: demand misses a token 50.45 without prefetch, 32.03 with the recorded predictor (covered 23.00, loads 57.43, wasted 34.42, bound 0.365); at 8 %: 91.72 and 54.44 (bound 0.406). The recorded predictor emits six experts a target layer with a recorded weight each (`pred_experts`, `pred_weights`); a gate on that weight only lost (stage C). The trace also holds every row's real experts, so layer l-1's real experts at the same token are known, and every earlier token's.

## 3. Design (script `benchmarks/cuda/predictor_price.py`, built after this record is committed)

**Window and training.** Decode tokens are numbered by ordinal in trace order. Tables are trained on decode tokens with ordinal below **1000** and every predictor is scored on the **held-out window of 400 decode tokens, ordinals 1000 to 1399**; the arena is replayed from the start of the trace with the predictor under test on for every token (the recorded predictor also for the baseline row), counts taken in the window only. The baseline demand-only misses are measured on the same window.

**Predictors** (all emit a list of experts per target layer l at token t, available once layer l-1's router has run; the arena loads each not resident):

| id | definition |
|---|---|
| `R` | the recorded predicted set (six experts) |
| `CO6`, `CO10` | co-activation: for each layer l >= 1 a count table `T[l][p][e]` = number of training tokens whose layer l-1 real set contained `p` and whose layer l real set contained `e`; the score of `e` is the sum over the six real experts `p` of layer l-1 at token t; the top 6 (or 10) by score, ties to the lower expert id |
| `TP6` | temporal: the six real experts of layer l at the previous decode token (none for the first token) |
| `MIX6` | the four recorded experts with the highest recorded weight, then the top `CO` experts not already in the set until six |
| `RCO10` | the recorded six, then the top `CO` experts not in the set until ten |

Metrics per predictor and resident share (17 % and 8 %): demand misses, prefetch loads, covered, wasted, copies a token (demand + loads), bound `1 - misses / misses_without`, and, as a derived host price, copies x 6 microseconds (stage C's measured host cost of a removed copy; an assumption, not a measurement for these arms).

**Decision rule (fixed here).** A predictor justifies a GPU stage only if its 17 % bound exceeds `R`'s by at least **0.05** on this window; copies a token are reported but are not a criterion (stage C: fewer copies bought almost nothing).

## 4. Predictions (registered before any code)

All **predicted**. Probabilities are my own. "Bound" is at 17 % resident on the held-out window unless said.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| PP-1 | `R` on the held-out window | bound between 0.30 and 0.42 (centre 0.36; the first 400 tokens read 0.365) (probability 0.7) | outside 0.25 to 0.48 |
| PP-2 | `CO6` against `R` | bound lower than `R`'s, between 0.12 and 0.32 (a co-activation table sees only experts, not the hidden state the router uses) (probability 0.75 that it is lower) | `CO6` at or above `R` |
| PP-3 | more loads help | `CO10` bound above `CO6` by 0.05 to 0.15, with loads a token 1.4 to 1.8 times `CO6`'s (probability 0.6) | `CO10` above `CO6` by under 0.02 |
| PP-4 | temporal | `TP6` bound under 0.10 (the arena already holds what the last token used) (probability 0.8) | `TP6` at or above 0.15 |
| PP-5 | `MIX6` | bound within -0.04 and +0.03 of `R` (probability 0.5) | above `R` by 0.05 or more (the decision rule fires) |
| PP-6 | `RCO10` | bound above `R` by 0.03 to 0.12, copies a token at least 15 % above `R`'s (probability 0.5) | above `R` by under 0.01 |
| PP-7 | ceiling | no predictor reaches a bound of 0.50 (probability 0.9) | any predictor at or above 0.50 |
| PP-8 | ranking at 8 % | the order of `R`, `CO6`, `CO10`, `RCO10` by bound is the same at 8 % as at 17 % (probability 0.8) | any pair swaps |

## 5. What each outcome means

* **PP-2 and PP-5 hold, PP-6 below the decision rule:** the recorded predictor already uses what is cheaply knowable; a better predictor needs new information (the router's hidden state two layers ahead), which only the live runtime can supply; the lane closes the predictor lever offline for this trace.
* **The decision rule fires for any predictor:** a stage D on the GPU is worth designing, with that predictor as an arm and predictions first.
* **PP-1 failing:** the held-out window is unlike the first 400 tokens; every comparison is read against `R` on this window only.
* Nothing here is a statement about a GPU, a runtime, or another trace.

## 6. Needs Hamed

Nothing: Mac only, no instance, no `src/` change. Housekeeping from earlier stages stands.
