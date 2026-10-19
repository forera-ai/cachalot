# CUDA lane: offline price of better expert predictors (results; Mac only, no GPU)

Run 2026-10-20 under `docs/CUDA-PREDICTOR-PRICE-RECORD.md` (eight predictions committed in 6c072de before any code). Script `benchmarks/cuda/predictor_price.py` (3 tests), raw output `docs/cuda-results/predictor-price-2026-10-20.json`. Trace `routing-trace-deepseek-long`; tables trained on decode tokens below 1000, scored on the held-out window of tokens 1000 to 1399. Labels follow plan section 0: everything here is **derived** from a bookkeeping model on one trace; it says nothing about time or any hardware.

## 1. Sanity checks (before reading the table)

On the first 400 tokens (in-sample for the tables) the recorded predictor `R` reproduces `prefetch_coverage.py` exactly (32.03 demand misses, 89.46 copies, bound 0.365); a test asserts the equality on a synthetic trace at two resident shares. Recall of the six real experts regardless of residency, held-out, layers 1 to 39: `R` 0.667, `CO6` 0.272, `CO10` 0.359 (chance would be 0.016).

## 2. Results (held-out window, 400 decode tokens; per token)

| resident | predictor | demand misses | loads | covered | wasted | copies | bound |
|---|---|---:|---:|---:|---:|---:|---:|
| 17 % | none | 52.99 | 0 | 0 | 0 | 52.99 | 0 |
| 17 % | `R` recorded | 28.33 | 55.80 | 27.39 | 28.41 | 84.12 | **0.465** |
| 17 % | `CO6` | 51.74 | 20.20 | 2.58 | 17.62 | 71.94 | 0.024 |
| 17 % | `CO10` | 52.40 | 41.32 | 4.20 | 37.13 | 93.72 | 0.011 |
| 17 % | `TP6` | 53.03 | 0.05 | 0 | 0.05 | 53.08 | -0.001 |
| 17 % | `MIX6` | 34.76 | 38.04 | 19.51 | 18.52 | 72.80 | 0.344 |
| 17 % | `RCO10` | 28.34 | 77.04 | 29.36 | 47.68 | 105.39 | 0.465 |
| 8 % | none | 91.07 | | | | 91.07 | 0 |
| 8 % | `R` | 45.20 | 95.97 | 51.67 | 44.31 | 141.18 | **0.504** |
| 8 % | `CO6` | 87.64 | 52.34 | 7.87 | 44.47 | 139.98 | 0.038 |
| 8 % | `CO10` | 87.71 | 109.80 | 14.36 | 95.44 | 197.51 | 0.037 |
| 8 % | `TP6` | 91.07 | 0.07 | 0 | 0.07 | 91.14 | 0.000 |
| 8 % | `MIX6` | 56.65 | 74.19 | 38.59 | 35.60 | 130.84 | 0.378 |
| 8 % | `RCO10` | 44.40 | 147.25 | 57.46 | 89.78 | 191.65 | 0.512 |

## 3. Scoring the registered predictions

| id | outcome |
|---|---|
| PP-1 (`R` bound 0.30 to 0.42 on the window) | **missed high**: 0.465 (inside the falsifier range 0.25 to 0.48); the window is easier than the first 400 tokens (0.365) |
| PP-2 (`CO6` lower than `R`, 0.12 to 0.32) | **missed low**: lower, but 0.024; the falsifier (at or above `R`) did not fire |
| PP-3 (`CO10` above `CO6` by 0.05 to 0.15, loads 1.4 to 1.8 times) | **falsified**: `CO10` is 0.013 *below* `CO6`; loads 2.05 times |
| PP-4 (`TP6` under 0.10) | **held**: -0.001 |
| PP-5 (`MIX6` within -0.04 and +0.03 of `R`) | **missed low**: -0.121; the falsifier (`R` + 0.05) did not fire |
| PP-6 (`RCO10` above `R` by 0.03 to 0.12, copies +15 %) | **falsified**: -0.0003 (copies +25 %, as predicted) |
| PP-7 (no predictor reaches 0.50 at 17 %) | **held**: best 0.465 |
| PP-8 (the order of `R`, `CO6`, `CO10`, `RCO10` is the same at 8 % as at 17 %) | **falsified, narrowly**: `RCO10` ties `R` at 17 % (0.4651 against 0.4654) and is ahead at 8 % (0.5125 against 0.5036) |

Of the eight: **2 held, 3 missed (PP-1, PP-2, PP-5), 3 falsified (PP-3, PP-6, PP-8), PP-8 by a margin under 0.001 at 17 %.** The decision rule (a predictor's 17 % bound at least 0.05 above `R`'s) is **not met by any predictor**.

## 4. What the result is, and is not

* **Derived: the recorded predictor already holds the usable information.** A co-activation table from the previous layer's real experts predicts 27 % of the real experts (recall) but turns only 0.024 of the demand misses into prefetched hits: what it predicts is mostly the popular experts the arena already holds. The recorded predictor reads the layer's hidden state through the router, which an expert-id table cannot reproduce. Adding the table's picks to the recorded six (`RCO10`) buys nothing at 17 % and 0.009 at 8 %, at 25 to 36 % more copies; replacing two of the six (`MIX6`) loses 0.12.
* **Derived: more loads of the same kind do not help** (`CO10` below `CO6`): the extra picks are resident or wrong.
* **Derived: temporal reuse is already in the arena** (`TP6` -0.001).
* **Consequence for the lane.** A better predictor cannot be built from the trace's expert ids, nor from a gate on the recorded weight (stage C); the large term left (oracle 0.85 against 0.34 on the card at 1 ms) needs new information, such as the router's own scores two layers ahead, which only the live runtime can supply and which this trace does not contain. No GPU stage is justified for these predictors. A trace that records the router's logits two layers early is the next input, not a GPU hour (hypothesis).
* **Not shown.** Other train/test splits or tables (a table conditioned on the previous token's layer-l experts, or on the hidden state), the hidden-share on a GPU (the serial bound overstates it: stage B measured 0.339 at a 0.365 bound at 1 ms and 0.10 at 0.3 ms), another trace, another model.
