# CUDA lane: predictions at leads 1, 2 and 3 on one recorded trace (results; Mac only, no GPU analysis)

Recorded 2026-10-20 by Hamed under `docs/CUDA-LEAD-TRACE-RECORD.md` (seven predictions committed in 849b8f2 before any recording) with the runbook `docs/CUDA-LEAD-TRACE-RUNBOOK.md`: `CACHALOT_ROUTING_TRACE`, `CACHALOT_DECODE_MISS_BUDGET=off`, `CACHALOT_PREDICT_AHEAD=3`, `CACHALOT_PREDICT_SUBMIT=0`, 44 GiB, the eight requests of `routing-trace-deepseek-long` (cap 600, temperature 0.7). The trace is git-ignored (`benchmarks/results/routing-trace-deepseek-leads/trace.npz`, 39.8 MB); raw output of the analysis: `docs/cuda-results/lead-trace-price-2026-10-20.json`. Analysis by `benchmarks/cuda/predictor_price.py` (tables not used; only the recorded predictions), tokens 1000 to 1399 of 1,897 as the window. Labels follow plan section 0: **derived** from a bookkeeping model on one trace; no time, no hardware. The agent started no server.

## 1. The recording

Eight requests, 646 s of replay (the first request prefilled its 22,411 tokens cold, 259 s; the earlier recording had them snapshot-warm), trace file written when Hamed stopped the server. **1,897 decode tokens** (75,880 decode rows); predicted rows per lead: 73,983 / 72,086 / 70,189, exactly 1,897 x (40 - lead) for leads 1, 2 and 3. The replies differ from the earlier trace's (temperature 0.7), so absolute numbers are read against this trace's own lead 1.

## 2. Results (held-out window, 400 decode tokens; per token)

| resident | predictor | demand misses | loads | wasted | copies | bound | recall of the real set |
|---|---|---:|---:|---:|---:|---:|---:|
| 17 % | none | 56.27 | 0 | 0 | 56.27 | 0 | |
| 17 % | `R` lead 1 | 28.59 | 58.81 | 28.36 | 87.40 | **0.492** | 0.684 |
| 17 % | `R2` lead 2 | 35.68 | 59.33 | 34.44 | 95.01 | **0.366** | 0.596 |
| 17 % | `R3` lead 3 | 41.09 | 59.74 | 38.92 | 100.83 | **0.270** | 0.521 |
| 17 % | `R12` lead 1 then lead-2 extras | 27.05 | 90.38 | 54.40 | 117.43 | **0.519** | |
| 8 % | none | 93.87 | 0 | 0 | 93.87 | 0 | |
| 8 % | `R` | 44.15 | 97.53 | 43.46 | 141.68 | 0.530 | |
| 8 % | `R2` | 55.42 | 97.34 | 52.69 | 152.76 | 0.410 | |
| 8 % | `R3` | 64.07 | 96.97 | 59.65 | 161.04 | 0.318 | |
| 8 % | `R12` | 40.25 | 145.30 | 82.09 | 185.54 | 0.571 | |

Recall counts the six real experts of layers 3 to 39 found in the prediction, whatever the arena holds.

## 3. Scoring the registered predictions

| id | outcome |
|---|---|
| LT-1 (`R1` bound 0.30 to 0.50) | **held**: 0.492, at the top of the band |
| LT-2 (`R2` below `R1` by 0.02 to 0.12) | **missed, narrowly**: 0.126 below; the falsifier (at or above `R1`, or more than 0.20 below) did not fire |
| LT-3 (`R3` below `R2` by 0.02 to 0.10) | **held**: 0.096 |
| LT-4 (recall lead 1 0.60 to 0.72, lead 2 0.54 to 0.66, lead 3 0.50 to 0.62, each 0.02 to 0.10 below the last) | **held**: 0.684, 0.596, 0.521; steps 0.088 and 0.075 |
| LT-5 (`R12` above `R1` by 0.00 to 0.06 with loads at least 1.6 times) | **bound held, loads clause missed**: +0.027; loads 1.54 times |
| LT-6 (rows as for the old trace; predicted rows per lead = layers minus lead) | **held**: exact |
| LT-7 (inside 25 minutes, under 60 MB, at least 1,400 decode tokens) | **held**: 11 minutes of replay, 39.8 MB, 1,897 tokens |

**5 held, 2 missed narrowly (LT-2, LT-5's loads clause), 0 falsified.**

## 4. Decision rules (fixed in the record before the recording)

* **Lead 2 needed `R2` at least `R1` minus 0.05 (0.442): it is 0.366. Not met: the lead lever is closed offline by the rule.**
* **The union needed `R12` at least `R1` plus 0.05 (0.542): it is 0.519. Not met.** Its 31.0 extra loads a token (1.54 times) buy 0.027.

## 5. What the result is, and is not

* **Derived: each layer of lead costs about 0.12 to 0.13 of the serial bound** at 17 % resident (0.492, 0.366, 0.270) and about 0.07 of recall (0.684, 0.596, 0.521); the fall is smooth, with no cliff at lead 2. The earlier offline recall table (65.0 % against 71.5 %, a 6.5-point step) understated the step on this trace (8.8 points); the runtime's recorded predictor is the lead-1 router on the layer's own input, and a router applied two layers early sees a residual stream that has drifted.
* **What the rule does not price.** The bound is a serial model with no window. Stage B measured that the window decides the hidden share at a 0.3 ms layer (P40, 0.4 of a layer later, was 0.17 below `P0L`; `P0L` reached 0.096 of a 0.365 bound), so a lead-2 prediction, a full layer earlier, could help where the lead-1 predictor is window-limited even though its bound is 0.126 lower. That is a hypothesis this offline price cannot test; it would need a GPU stage with a lead-2 arm, a harness change (predictions keyed by lead) and Hamed's go. The pre-registered rule says no; an exception needs a new record with predictions first.
* **Derived: the union adds copies, not coverage.** Lead 2's extra picks are 54 % wasted at the margin (31.6 loads for 1.6 fewer demand misses at 17 %).
* **Not shown.** Any time or hardware effect; another trace; the window effect above; a router-score predictor of another form (for example the router's second-layer scores from the live state, or a learned predictor), none of which this trace holds.

## 6. Housekeeping

The recording stopped by Hamed; no server of the agent's remains. A stale background polling shell from the first CUDA session (an `ssh` loop to a long-gone `cuda-s1` host) still showed in the process list; it is harmless and was not touched.
