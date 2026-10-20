# CUDA lane: a host-aware row-level window model, fitted and validated on the measured cells (experiment record; Mac only, no GPU)

Written 2026-10-20 against runtime 0.66.42, **before any fitting code is run**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file. A prediction edited after its run is void. Labels follow plan section 0. Nothing here starts an instance and nothing is measured on a GPU: all data are the cells already recorded.

## 1. What this asks, and what it cannot claim

The copy-count-only row-level model (`window_model.py`) was a useful screen in the bridge (right sign, errors up to 0.10 in hidden share). The lane now has many measured cells from one harness whose numbers reproduce across sessions to about 0.005 (bridge): stage C (old trace; arms `D`, `P0L`, `G10`, `G20`, `O`), stage D (new trace; `D`, `P0L`, `L2`, `O`) and the bridge (both traces; `D`, `P0L`). Candidate job 6 of v247: **fit a small host-aware model on the `D` and `P0L` cells, then test it on arms it never saw (`G10`, `G20`, `L2`, `O`)**, so the lane has a validated cheap predictor of an arm's hidden share from the trace's row-level copy counts, and a stated error. It cannot claim anything about another card, another host CPU, a native issue path, another stand-in compute, or arms whose mechanism the model does not contain (copy-stream contention between layers).

## 2. Data (recorded, all 17 % unless said; hidden share `H = 1 - E_X / E_D`, `E` = wall minus the group's floor, p50 per token)

Per group (trace, resident share, C, level): the measured `E_D`, `E_P0L` and, where run, `E_G10`, `E_G20`, `E_L2`, `E_O` (rows `E_<arm>` of each run). Groups: **stage C** (old trace): 17 % C = 1.0 ms L5 and C = 0.3 ms L4 (`D`, `P0L`, `G10`, `G20`, `O`), 8 % C = 1.0 ms L5 (`D`, `P0L`, `G20`); **stage D** (new trace): 17 % C = 1.0 ms L5, 0.6 and 0.3 ms L4 (`D`, `P0L`, `L2`, `O`), 8 % C = 0.3 ms L4 (`D`, `P0L`, `L2`); **bridge** (both traces, two runs each, so 12 group-runs): 17 % C = 1.0 ms L5, 0.6 and 0.3 ms L4 (`D`, `P0L`). Row-level counts of every arm come from the stage A/B bookkeeping on the matching trace (gated sets by weight, `L2` by the timed lead-2 semantics, `O` by the next row's real set), exactly the semantics the harness's expected files verify.

## 3. The model and the fitting procedure (fixed here)

Per decode row r of an arm: `n_pf` = prefetch loads that arrive for the row (issued during the previous layer; for `L2` during the two previous layers), `n_dem` = demand misses at its boundary.

* exposed prefetch tail: `max(0, L_pf + n_pf * c - W)` when `n_pf > 0` (else 0), with `W = C` for lead-1 arms and `W = 2 C` for `L2`; no contention term between layers (a deliberate omission: it is what `L2` tests);
* exposed demand: `L_dem + n_dem * c` when `n_dem > 0` (else 0);
* `E_X` = the mean over tokens of the sum of the row terms.

Three parameters: `c` (effective copy time a copy of 9.95 MB costs on the critical path, start 0.194 ms), `L_dem` (a fixed latency a row with demand copies pays), `L_pf` (a fixed lag before a row's prefetch queue starts). **Step 1:** fit `c` and `L_dem` to the measured `E_D` of every group (the `D` arm has no prefetch) by least squares on a grid. **Step 2:** with those frozen, fit `L_pf` to the measured `E_P0L` of every group (grid, least squares). **Step 3:** with all three frozen, predict `E_G10`, `E_G20`, `E_L2`, `E_O` and score `H_model = 1 - E_X_model / E_D_measured` against the measured `H` (the measured `E_D` in the denominator isolates the arm's own error). The reference model is the copy-count-only `window_model` (`c` = 0.194, `L_pf` = `L_dem` = 0).

## 4. Predictions (registered before any fitting code is run)

All **predicted**. Probabilities are my own. "Error" = `|H_model - H_measured|`.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| WF-1 | step 1 (the `D` arm) | fitted `c` between 0.17 and 0.23 ms, `L_dem` between 0 and 0.30 ms, and every group's `E_D` residual at most 0.35 ms (probability 0.7) | `c` outside 0.15 to 0.26, or any residual above 0.6 ms |
| WF-2 | step 2 (the `P0L` arm) | `L_pf` between 0 and 0.25 ms; the `P0L` error at most 0.05 in at least 80 % of the group-runs (probability 0.6) | `L_pf` outside -0.1 to 0.5 ms, or at most 50 % of group-runs within 0.05 |
| WF-3 | held-out gated arms | `G10` and `G20` error at most 0.05 in at least 75 % of their cells (probability 0.7) | more than half above 0.08 |
| WF-4 | held-out oracle | `O` error at most 0.08 in at least 3 of 4 cells (probability 0.5) | any above 0.15 |
| WF-5 | held-out lead 2 (a mechanism the model lacks: copy-stream contention) | `L2` error at most 0.08 in at least 3 of 4 cells (probability 0.4), and the model overstates `L2`'s hidden share when it errs | errors above 0.15 in two or more cells |
| WF-6 | against the screen | the mean absolute error over all held-out cells of the fitted model is below half the copy-count-only model's (probability 0.6) | not below 0.8 times |
| WF-7 | the data fit | the fitted model's mean absolute error on the fit cells (`D` and `P0L`) is at most 0.04 in `H` (probability 0.65) | above 0.08 |

## 5. Decision rules (fixed here)

* **If WF-3 and WF-6 both hold,** the host-aware model replaces the copy-count-only screen in the lane for arms of the `P0L` family (gated sets, oracle), with the measured held-out error stated; it may then be used to screen a proposed arm before a GPU session. **If WF-5 fails by contention,** the model is not used for any arm whose prefetch spans more than one layer's window.
* **If WF-3 fails,** the copy-count-only screen stays with its 0.1 error and no host-aware model is claimed.
* Nothing here predicts any arm on a GPU: it prices arms that were already run.

## 6. Needs Hamed

Nothing (Mac only). The CUDA lane's GPU items stand as in the v247 prompt.
