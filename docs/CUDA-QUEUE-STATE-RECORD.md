# CUDA lane: a queue-state per-miss cost in the row-level model (experiment record; Mac only, no GPU)

Written 2026-10-20 against runtime 0.66.43, **before any queue-state code is run**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file. A prediction edited after its run is void. Labels follow plan section 0. Nothing here starts an instance; this is the last offline CUDA item Hamed asked for ("then stop CUDA").

## 1. What this asks, and what it cannot claim

`docs/CUDA-WINDOW-FIT-RESULTS.md` found that the fixed per-row host terms of the row-level model fit to zero and that the remaining error is structural: the misses that `P0L` leaves behind cost about **0.22 ms each against 0.187 ms for `D`'s misses on the new trace** (0.197 against 0.189 on the old trace). **H-queue:** a miss costs more when its row also has prefetch copies in the queue, or when misses come in bursts, so a per-miss cost that depends on queue state removes the new trace's overstated hidden share and part of the lead-2 and oracle errors. It is tested on the cells already recorded (stages C and D, the bridge); only whole-token wall times are measured, so a row-level cost can be tested only through its aggregate effect on a token's exposure, and the three terms below are hypotheses about that effect, not measurements of queue state. It cannot claim anything about a GPU's copy engine, another card or host, or a mechanism: a term that fits is a description.

## 2. Baseline (recorded)

The fitted window model of the previous record: `c` = 0.190 ms, `L_dem` = 0, `L_pf` = -0.025 ms; hidden-share error on `P0L` cells 0.026 (six of 19 group-runs above 0.05, all on the new trace at 1.0 and 0.6 ms, 0.073 to 0.094), gated arms 0.026, oracle 0.100, lead 2 0.187, all held-out cells **0.0986** against the copy-count-only screen's **0.0944**. The data and the fitting grids are those of `benchmarks/cuda/window_fit.py` (19 `D` and 19 `P0L` group-runs; 5 gated, 5 oracle, 4 lead-2 cells held out).

## 3. The terms and the procedure (fixed here)

Each term adds a cost to a token's exposure from the row-level counts (`n_pf` loads that arrive for a row, `n_dem` misses at its boundary), with one new parameter beta (ms):

* **T1, contended misses:** `beta * sum over rows of n_dem * [n_pf > 0]` (a miss in a row that also has prefetch copies);
* **T2, contended rows:** `beta * sum over rows of [n_pf > 0 and n_dem > 0]` (a row with both kinds of copies);
* **T3, burstiness:** `beta * sum over rows of n_dem * (n_dem - 1) / 2` (pairs of misses in one row; applies to every arm).

**Procedure.** `c` and `L_dem` are fitted on the `D` cells as before (for T3, `c` and beta are fitted on `D`, with `L_dem` = 0); then `L_pf` and beta are fitted jointly on the `P0L` cells (T1, T2; `D` has `n_pf` = 0 so beta does not enter) or `L_pf` alone (T3), by grid least squares; everything is frozen and the held-out arms (`G10`, `G20`, `L2`, `O`) are predicted; error = `|H_model - H_measured|` with the measured `E_D` in the denominator, as before. The comparison baselines are the previous fit (0.0986) and the screen (0.0944).

## 4. Predictions (registered before any run)

All **predicted**. Probabilities are my own.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| QS-1 | T1's parameter and fit | fitted beta between 0.00 and 0.06 ms; the `P0L` fit-cell mean error falls from 0.026 to at most 0.020 (probability 0.55) | beta outside -0.02 to 0.12, or no fall below 0.024 |
| QS-2 | the new trace's overstatement | under T1 or T2, the new-trace `P0L` error at C = 1.0 and 0.6 ms (six group-runs, now 0.073 to 0.094) is at most 0.05 in at least 4 of 6 (probability 0.45) | at most 2 of 6 under both |
| QS-3 | burstiness against contention | T3's all-held-out mean error is not below the better of T1 and T2 (probability 0.6) | T3 below both by 0.01 or more |
| QS-4 | the main claim | the best term's mean error over all 14 held-out cells is at most 0.075 (probability 0.35) | the best term at or above 0.09 |
| QS-5 | lead 2 | the best term lowers the `L2` mean error from 0.187 to at most 0.12 (probability 0.3) | not below 0.17 |
| QS-6 | the oracle | the oracle's mean error stays above 0.10 under every term (a synchronization floor is not a queue-state effect) (probability 0.8) | some term brings it below 0.06 |

## 5. Decision rules (fixed here)

* **A queue-state model is adopted for the `P0L` family** (replacing the copy-count-only screen there) only if **QS-2 and QS-4 both hold**; it is then used with its measured held-out error and never for multi-layer prefetch or the oracle unless QS-5 and QS-6 also fail in its favour.
* **Otherwise no model is adopted, the screen stays at about 0.1, and the CUDA lane stops** (Hamed's instruction) with its remaining items (fewer and larger copies, a native issue path, ROCm, V100 or L40S, instance population) recorded as open and needing his go and a GPU.
* Nothing here predicts a new GPU measurement.

## 6. Needs Hamed

Nothing.
