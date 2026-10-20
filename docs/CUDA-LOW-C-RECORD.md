# CUDA lane: the compute-slowdown slope at a layer time below 0.3 ms, C = 0.15 ms (experiment record; predictions before any build)

Written 2026-10-20 against runtime 0.66.50, **before any code for this session is written or run**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file (`docs/CUDA-LOW-C-RESULTS.md`). A prediction edited after its run is void. Labels follow plan section 0. It is the first open item of `docs/CUDA-THIRD-C-RESULTS.md` section 4: "whether the 1/C form continues below 0.3 ms or above 1.0 ms (a point at 0.15 or 3 ms)". It needs an RTX 5090 instance and Hamed's go (he chose this item on 2026-10-20); nothing here starts one.

## 1. What this asks, and what it cannot claim

The slowdown term says a `P0L` row's compute takes `s` ms longer for each millisecond its window overlaps the copy stream's prefetch batches. Three layer times are measured: the slope on measured overlap is 0.0634 (C = 1.0 ms), 0.0765 (0.6) and 0.1053 (0.3), and a slope affine in 1/C describes all three to within 0.001 (least squares: `s = 0.0461 + 0.01783 / C`). That is an interpolation. At C = 0.15 ms a layer's compute is shorter than one copy (0.196 ms), so the copy stream is saturated by construction and the regime is different in kind: a prefetch batch cannot fit in the previous layer, and the count-only window model (`window_model.py`) puts the hidden share below zero.

**H-continues.** The slope keeps rising along the affine-in-1/C curve below 0.3 ms. The alternatives are a power law (flatter), saturation (a ceiling), or a knee where the copy stream is permanently saturated.

**It cannot claim** why the slope depends on C, anything about another card or instance (the lane has seen the copy rate and slowdown vary by instance), another prefetch family or resident fraction, or the real model's kernels (real layers are longer than 0.15 ms on this card class; the point tests the form, not a deployment). The stand-in compute is a calibrated kernel chain; C is its length per layer. At 0.15 ms the stand-in is about 6 ms a token, and launch overhead matters: whether the instrument can be tuned there is itself a prediction (L-9).

## 2. Baseline (recorded)

* Slope on measured overlap, level 3, `P0L`: 0.0634 (1.0), 0.0765 and 0.0747 (0.6, runs A and B), 0.1053 (0.3). V1 (predicted-overlap) slopes: 0.0624, 0.0767, 0.1122.
* Hidden share of `P0L`, level 4 (measured): 0.423 (1.0), 0.407 (0.6), 0.232 (0.3). The count-only window model at the same C (copy 0.1957 ms): 0.533, 0.480, 0.212 and **-0.085 at 0.15** (overloaded row share 0.0025, 0.051, 0.357, 0.688).
* The unconditional predicted-overlap form is adopted at C = 1.0, 0.6 and 0.3 (hidden-share error 0.0457, 0.0186, 0.0067).
* **Extrapolated slope at C = 0.15** (derived, measured basis): affine in 1/C through all three points 0.1649 (through 1.0 and 0.3 only 0.1652); power law in 1/C (least squares, exponent 0.424) 0.1402; linear in C (least squares) 0.1100. V1 basis, affine in 1/C through three points: 0.1833.
* Demand-only exposure of a token (`D` minus the `N` floor): 9.9 to 10.1 ms whatever the layer time (stage A, fast population).

## 3. Design (fixed here)

* **Trace and arms.** The new multi-lead trace (`n3d-trace.npz`, `n3d-expected.json`); 17 % resident; arms `N`, `D`, `P0L`, `N`, 400 decode tokens a cell, one layer time only: **C = 0.15 ms**.
* **Levels.** L3 with per-row records, L6 (an event after every copy) and L4, as the third-C session. **12 cells a run.** Two runs on the same instance, A then B. Estimate 4 to 6 minutes a run plus calibration and arena allocation; session cap one GPU-hour from Ready, an explicit `--max-seconds` on every run.
* **Gates.** All stage gates stay (event hash, determinism, counts, chain2, readback, calibration, drift, closure, sanity, floor, `rowsum`, `copyevents`). The floor gate checks the `N` floor against layers times C within 25 %; a stand-in that cannot be tuned to 0.15 ms fails it and the run is an instrument finding (L-9), not a measurement.
* **Offline analysis.** `benchmarks/cuda/low_c_score.py` (new) scores L-1 to L-10 on the two runs and lays the slope beside the extrapolated shapes of section 2 and a four-point refit. Rehearsed on fabricated runs before any instance exists.

## 4. Predictions (registered before any build)

All **predicted**; probabilities are my own. "Slope" is the least-squares slope through the origin of the `P0L` row excess busy time on the row's measured overlap with prefetch batches, level 3. Run A is the fit, run B the score.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| L-1 | slope on measured overlap at C = 0.15, run A | between 0.140 and 0.185 (probability 0.55) | outside 0.110 to 0.220 (saturation or a knee: H-continues is then not supported) |
| L-2 | which shape | nearer the affine-in-1/C value 0.165 than the power-law value 0.140: at least 0.153 (probability 0.45) | below 0.125 (power law or flatter fits the four points better) |
| L-3 | replicate | run B's slope within 15 % of run A's (probability 0.65; 2.4 % at C = 0.6) | more than 30 % apart |
| L-4 | measured hidden share, `P0L`, level 4 | between -0.25 and 0.05 (probability 0.55; count model -0.085, prefetch cannot fit a layer) | above 0.12 (prefetch still hides a real share) or below -0.40 |
| L-5 | queue model (V1, run A's parameters, scored on run B): row agreement of predicted and measured overlap | Pearson r at least 0.85 (probability 0.65; 0.998, 0.997, 0.997 at the other C) | below 0.60 |
| L-6 | the same, total overlap | predicted over measured between 0.85 and 1.05 (probability 0.55) | outside 0.65 to 1.25 |
| L-7 | the unconditional form at a layer time outside the fitted range | hidden-share error of the row model plus V1 with its own fitted slope at most 0.07 (probability 0.45; wider than before because the share is extrapolated into negative values) | above 0.12 |
| L-8 | extrapolated slope | the same model with V1's slope extrapolated (affine in 1/C through the three V1 slopes, 0.1833) has a token busy total error at most 0.30 ms against the measured `P0L` minus `D` busy time, run B (probability 0.45) | above 0.80 ms |
| L-9 | the instrument | the calibration and floor gates hold in every cell at C = 0.15 and no cell fails (probability 0.65) | any of these fails: the run is not a measurement of the slope, L-1 to L-8 are not scored, and the minimum layer time of the instrument is reported |
| L-10 | demand exposure | `D` minus the `N` floor, per token, between 9.0 and 11.0 ms (probability 0.65; stage A 9.9 to 10.1 at every layer time) | outside 8.0 to 12.5 |

## 5. Decision rules (fixed here)

* **The 1/C form is described as continuing to 0.15 ms** (four points, one trace, 17 % resident, fast population) if **L-1 is not falsified, L-3 holds and L-9 holds**; the shape nearest the measured value is reported with the other shapes' errors beside it. If L-2 also holds, the slope is described as affine in 1/C over 0.15 to 1.0 ms.
* **The unconditional predicted-overlap form is extended to 0.15** if **L-5 and L-7 hold**; otherwise it stays adopted at 1.0, 0.6 and 0.3, and 0.15 is reported as a failed extension with its error.
* **Extrapolated slopes are usable** one step outside the measured range only if **L-8 holds**.
* If L-1's falsifier fires with L-9 true, H-continues is not supported: the slope is reported at four points with no function fitted, and a fifth C (0.2 ms) is named as the next record, not run.
* If L-9 fails, nothing else is scored; the instrument's minimum layer time is reported and the point is closed as untestable on this harness.
* If L-3's falsifier fires, the slope is treated as instance noise and nothing here is adopted.
* The slow-population rule stands: a median source-record copy of 0.30 ms or more puts the instance in the slow class, L-4 to L-10 are reported but not used for adoption, and a fast-class instance is requested if Hamed agrees.
* Nothing here predicts a new default or a change in `src/`. The fit describes a stand-in compute stream.

## 6. Needs Hamed

An RTX 5090 instance, a time cap of one GPU-hour from Ready, and his go for the named live instance. The agent never runs sudo and runs over SSH only for that instance. His housekeeping from the previous sessions (terminate any instance still up, enter the billed hours, delete the ssh configs and known_hosts of `g1`, `f1`, `f2`, `f3`) stands.
