# CUDA lane: a queue-state per-miss cost in the row-level model (results; Mac only, no GPU)

Run 2026-10-20 under `docs/CUDA-QUEUE-STATE-RECORD.md` (six predictions committed in 372e836 before any queue-state code was run). Script `benchmarks/cuda/window_fit.py --queue-state` (5 tests in `tests/test_cuda_window_fit.py`), raw output `docs/cuda-results/queue-state-2026-10-20.json`. Data and fitting grids as in `docs/CUDA-WINDOW-FIT-RESULTS.md`. Labels follow plan section 0: **derived** from models fitted to measured cells; no GPU was used.

## 1. Fits

| term | fitted parameters | fit cells (`D` + `P0L`) mean error | `P0L` cells only | new-trace `P0L` at 1.0 and 0.6 ms (six group-runs) | held out, all 14 | `G10` / `G20` / `L2` / `O` |
|---|---|---:|---:|---|---:|---|
| none (previous fit) | `c` 0.190, `L_pf` -0.025 | 0.0259 | 0.0416 | 0.073 to 0.094 (0 of 6 within 0.05) | 0.0986 | 0.024 / 0.044 avg 0.026 / 0.187 / 0.100 |
| T1 contended misses | beta 0.012, `L_pf` -0.04 | 0.0233 | 0.0364 | 0.055 to 0.072 (0 of 6) | 0.0920 | 0.024 / 0.013 / 0.163 / 0.110 |
| T2 contended rows | beta 0.024, `L_pf` -0.05 | 0.0227 | 0.0354 | 0.045 to 0.062 (2 of 6) | **0.0919** | 0.024 / 0.012 / **0.155** / 0.116 |
| T3 burstiness | beta 0.009, `c` 0.182, `L_pf` 0.0 | 0.0283 | 0.0491 | 0.086 to 0.105 (0 of 6) | 0.1034 | 0.033 / 0.031 / 0.199 / 0.098 |
| copy-count-only screen | `c` 0.194, no latency | | | | 0.0944 | |

## 2. Scoring the registered predictions

| id | outcome |
|---|---|
| QS-1 (T1 beta 0.00 to 0.06; `P0L` fit error falls from 0.026 to at most 0.020) | **missed**: beta 0.012 is in the band, but the fit error is 0.0233 (all fit cells; 0.0364 for `P0L` alone against 0.0416 before); the falsifier (no fall below 0.024) did not fire |
| QS-2 (new-trace `P0L` error at most 0.05 in at least 4 of 6 under T1 or T2) | **falsified**: 0 of 6 under T1, 2 of 6 under T2 |
| QS-3 (T3 not below the better of T1 and T2) | **held**: 0.1034 against 0.0919 |
| QS-4 (best term's held-out mean at most 0.075) | **falsified**: best 0.0919 (T2), at the falsifier's 0.09 |
| QS-5 (best term lowers `L2` to at most 0.12) | **missed**: 0.155 (T2); the falsifier (not below 0.17) did not fire |
| QS-6 (oracle error above 0.10 under every term) | **missed narrowly**: 0.110, 0.116 and **0.098** (T3); the falsifier (below 0.06) did not fire |

**1 held, 3 missed, 2 falsified.** The registered baseline for QS-1 (0.026) is the mean over `D` and `P0L` fit cells; I give the `P0L`-only numbers beside it because the record's wording ("the `P0L` fit-cell mean error") was ambiguous: the outcome is a miss either way.

## 3. Decision rule

A queue-state model is adopted only if **QS-2 and QS-4 both hold**. Neither holds. **No model is adopted; the copy-count-only screen stays at an error of about 0.1, and the CUDA lane stops here, as Hamed asked.**

## 4. What the result is, and is not

* **Derived: a contention term helps a little and is not the explanation.** T2 (a cost per row that has both prefetch and demand copies) lowers the new trace's `P0L` error from 0.073 to 0.094 down to 0.045 to 0.062 and the all-held-out mean from 0.0986 to 0.0919, which is 3 % below the screen's 0.0944: not enough to prefer it. Burstiness (T3) does not help. The per-miss cost difference (0.22 ms for the misses prefetch leaves behind against 0.187 ms for `D`'s) is only partly absorbed by any of the three row-level counts.
* **Not shown.** Which queue state (if any) sets the per-miss cost; the cost of a miss may depend on its position in the layer, on the copy stream's phase relative to the router synchronization, or on interactions the whole-token wall times cannot reveal: the harness measures a token's wall, not a row's. A per-row timing instrument would be needed, on a GPU.
* **Consequence for the lane.** The row-level screen's error of about 0.1 is a property of its inputs (whole-token walls, row counts), not of a missing latency term or of the three queue-state terms tried. The CUDA lane's open items all need a GPU and Hamed's go: per-row timing, fewer and larger copies, a native issue path, ROCm, V100 or L40S, instance population.
