# CUDA lane: the compute-slowdown slope at a third layer time, C = 0.6 ms (experiment record; predictions before any build)

Written 2026-10-20 against runtime 0.66.48, **before any code for this session is written or run**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file (`docs/CUDA-THIRD-C-RESULTS.md`). A prediction edited after its run is void. Labels follow plan section 0. It is the item `docs/CUDA-PREDICTED-OVERLAP-RESULTS.md` section 4 names: "why the slope doubles from 1.0 to 0.3 ms (a third C at 0.6 ms would give a point on the curve)". It needs an RTX 5090 instance and Hamed's go (he chose this item on 2026-10-20); nothing here starts one.

## 1. What this asks, and what it cannot claim

The slowdown term of `docs/CUDA-SLOWDOWN-RECORD.md` says a `P0L` row's compute takes `s` ms longer for each millisecond its window overlaps the copy stream's prefetch batches. On the two layer times measured so far the slope is 0.0634 at C = 1.0 ms and 0.1053 at C = 0.3 ms on measured overlap (0.0624 and 0.1122 on predicted overlap), so it depends on the stand-in layer time C, and nothing says how. With two points the term can be used only at those two C. A third point at C = 0.6 ms discriminates between smooth shapes (linear in C, affine in 1/C, a power law) and a shape with a knee, and tells whether the term can be interpolated for a layer time that was not measured.

**H-smooth-slope.** The slope is a smooth, monotone function of C between 0.3 and 1.0 ms. Three smooth descriptions fitted through the two known points give different values at 0.6 ms (section 2); the measurement at 0.6 ms says which, if any, is right.

**It cannot claim** why the slope depends on C (the mechanism, memory contention or copy-engine scheduling, is unidentified; a slope measured at three C does not separate them), anything about another card or instance (the lane has seen the copy rate and the slowdown vary by instance), another prefetch family, another resident fraction, or the real model's kernels. The stand-in compute is a calibrated kernel chain; C is its length per layer.

## 2. Baseline (recorded)

* Slope on measured overlap, run A, level 3, `P0L`: 0.0634 (C = 1.0), 0.1053 (C = 0.3); run B 0.0660 and 0.1026. On predicted overlap (V1): 0.0624 and 0.1122.
* Hidden share of `P0L` on the new trace, level 4 (measured): 0.423 at C = 1.0 (row-timing session, 0.433 at level 5), 0.232 at C = 0.3; bridge session 0.407 at C = 0.6 (level 4, a different instance and session).
* The unconditional predicted-overlap form is adopted at C = 1.0 and 0.3 only (hidden-share error 0.0457 and 0.0067).
* **Shapes through the two points** (derived, `s` on measured overlap):
  * linear in C: `s(0.6) = 0.0909`;
  * affine in 1/C: `s = 0.0411 + 0.02134 / C`, so `s(0.6) = 0.0767`;
  * power law, exponent 0.488 in 1/C: `s(0.6) = 0.0801`.
  A knee (a step between 0.3 and 0.6, or between 0.6 and 1.0) would put the value outside the range 0.0767 to 0.0909 or close to one end of the segment.

## 3. Design (fixed here)

* **Trace and arms.** The new multi-lead trace (`n3d-trace.npz`, `n3d-expected.json`); 17 % resident; arms `N`, `D`, `P0L`, `N`, 400 decode tokens a cell, one layer time only: **C = 0.6 ms**. The `N` controls bracket the arms as in every stage.
* **Levels.** L3 with per-row records (profile flag `rows`), L6 (an event after every copy) and L4 (the stage D level that carries the recorded hidden share). **12 cells a run.** Two runs on the same instance, A then B (a replicate, as the row-timing session). Estimate 4 to 6 minutes a run plus calibration and arena allocation; session cap one GPU-hour from Ready, an explicit `--max-seconds` on every run.
* **Gates.** All the stage gates stay (event hash, determinism, counts, chain2, readback, calibration, drift, closure, sanity, floor, `rowsum`, `copyevents`). The floor gate checks the N-control floor against layers times C, so a stand-in tuned to the wrong C is caught (the 0.66.45 defect).
* **Offline analysis.** `benchmarks/cuda/predicted_overlap_fit.py` takes the list of layer times (default 1.0 and 0.3; here 0.6) and scores the registered quantities on the new runs; `benchmarks/cuda/third_c_score.py` (new) puts the slope at 0.6 next to the three shapes, scores G-1 to G-9, and refits the slope as a function of C on the three points. Both are rehearsed on fabricated runs before any instance exists.

## 4. Predictions (registered before any build)

All **predicted**; probabilities are my own. "Slope" is the least-squares slope through the origin of the `P0L` row excess busy time on the row's overlap with prefetch batches, level 3, as in the slowdown record. Run A is the fit, run B the score.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| G-1 | slope on measured overlap at C = 0.6, run A | between 0.070 and 0.095 (probability 0.65) | outside 0.060 to 0.110 (a non-monotone slope, or a knee: H-smooth-slope is then not supported) |
| G-2 | which shape | the slope is nearer the affine-in-1/C value 0.0767 and the power-law value 0.0801 than the linear-in-C value 0.0909: below 0.085 (probability 0.55) | at or above 0.091 (linear in C, or steeper, fits the three points) |
| G-3 | replicate | run B's slope within 15 % of run A's at C = 0.6 (probability 0.70) | more than 30 % apart (the third point is not a point: instance noise) |
| G-4 | measured hidden share, `P0L`, level 4, C = 0.6 | between 0.37 and 0.45 (probability 0.60; bridge 0.407 on another instance, row-timing session 0.423 at C = 1.0) | outside 0.33 to 0.48 |
| G-5 | the predicted-overlap queue model at C = 0.6 (V1, run A's parameters, scored on run B): row agreement of predicted and measured overlap | Pearson r at least 0.85 (probability 0.75; 0.998 and 0.997 at the other two C) | below 0.60 |
| G-6 | the same, total overlap | predicted over measured between 0.85 and 1.05 (probability 0.60; 1.010 and 0.951 at the other two) | outside 0.65 to 1.25 |
| G-7 | the main claim: the unconditional form at a third C | hidden-share error of the row model plus V1 with its own fitted slope, against the level-4 measured share, at most 0.05 (probability 0.55; 0.0457 and 0.0067 at the other two) | above 0.08 |
| G-8 | interpolation | the same model with the V1 slope **interpolated** from the two known C (affine in 1/C through V1's slopes 0.0624 at 1.0 and 0.1122 at 0.3, which gives 0.0411 + 0.0213 / C = 0.0767 at 0.6) has a token busy total error at most 0.15 ms against the measured `P0L` minus `D` busy time, run B (probability 0.50) | above 0.40 ms |
| G-9 | level 6 | V1 with level-6 parameters and the level-3 slope predicts the level-6 `P0L` excess busy per token within 0.15 ms (probability 0.45) | above 0.40 ms |

## 5. Decision rules (fixed here)

* **The slope is described as a function of C on this card** (three points, one trace, 17 % resident, fast population) if **G-1 is not falsified and G-3 holds**. The function reported is the shape nearest the measured value, with the other two shapes' errors beside it, and it is labelled interpolation between 0.3 and 1.0 ms only.
* **The unconditional predicted-overlap form is extended to C = 0.6** if **G-5 and G-7 hold**; if G-7's falsifier fires, the form stays adopted at 1.0 and 0.3 only, and C = 0.6 is reported as a failed extension with its error.
* **Interpolated slopes are usable** (a layer time not measured) only if **G-8 holds**; otherwise the term is used at measured layer times only.
* If G-1's falsifier fires, H-smooth-slope is not supported, the slope is reported at three points with no function fitted, and a fourth C (0.45 or 0.8 ms) is named as the next record, not run.
* If G-3's falsifier fires, the session's slope is treated as instance noise and nothing in this record is adopted.
* The slow-population rule of the lane stands: if the source qualification puts the instance in the slow class (median source-record copy at or above 0.30 ms), G-4 to G-9 are reported but not used for adoption (they were registered for the fast class), and a fast-class instance is requested if Hamed agrees.
* Nothing here predicts a new default or a change in `src/`. The fit describes a stand-in compute stream.

## 6. Needs Hamed

An RTX 5090 instance (the card of every earlier stage), a time cap of one GPU-hour from Ready, and his go for the named live instance. The agent never runs sudo and runs over SSH only for that instance, as in the bridge and row-timing sessions. His housekeeping from the previous sessions (terminate any instance still up, enter the billed hours, delete the ssh configs and known_hosts of `f1`, `f2`, `f3`) stands.
