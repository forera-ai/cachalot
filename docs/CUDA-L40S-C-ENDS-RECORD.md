# CUDA lane: does the RTX 5090's dependence of the compute slowdown slope on layer time exist on an L40S? The ends of the layer-time range (experiment record; predictions before any build)

Written 2026-10-20 against runtime 0.66.54, **before any code for this session is written or run**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file (`docs/CUDA-L40S-C-ENDS-RESULTS.md`). A prediction edited after its run is void. Labels follow plan section 0. It takes the first open item of `docs/CUDA-COPY-COST-RESULTS.md` section 4 ("a record to cover 1.0 and 0.15 ms on this card would say whether the 5090's C dependence exists here at all"). Hamed chose it on 2026-10-20 and created a fresh L40S 48 GB instance (provider name `aquatic-existence-2628`) while this record was still unwritten; nothing is run on it until this record is committed and the harness is built and rehearsed on the Mac. Because the instance is a different one from the earlier L40S (`flat-battle-1078`), the anchors at C = 0.3 and 0.6 ms double as the instance-stability check the earlier results left open.

## 1. What this asks, and what it cannot claim

On the RTX 5090 (fast population) the slope `s` (extra compute milliseconds per millisecond of overlap with prefetch copies, level 3, measured overlap) went 0.0634, 0.0765, 0.1053 and 0.1534 at layer times C = 1.0, 0.6, 0.3 and 0.15 ms. On the first L40S, at C = 0.3 and 0.6 ms and three copy fractions, the slope was 0.092 to 0.105 and flat in C to within 0.002 to 0.005 within a group; at the full copy (f = 1.0) 0.0929 at C = 0.3 and 0.0931 at C = 0.6 (run A), copy time 0.378 ms.

**H-flat.** On the L40S the slope is a constant of about 0.09 to 0.10 at every layer time from 0.15 to 1.0 ms (the 5090's rise at short layers is a property of that card).
**H-rise.** The L40S slope rises as the layer shortens, like the 5090, and the two earlier points (0.3 and 0.6) were a plateau before a rise at 0.15 ms or after a fall toward 1.0 ms.

Only the two ends are new: C = 1.0 and C = 0.15 ms at the full copy (f = 1.0). C = 0.3 and 0.6 are re-measured on the new instance as anchors.

**It cannot claim** why the slope depends (or does not) on C, anything about real expert kernels, another card, another trace or resident fraction, or that the stand-in at 0.15 ms is a real layer (on a card with half the memory bandwidth of the 5090 the stand-in may not be tunable that short: that is prediction J-3). One instance, and the second L40S instance of this card type.

## 2. Baseline (recorded)

* L40S (first instance), f = 1.0, level 3, run A: 0.0929 (C = 0.3), 0.0931 (C = 0.6); run B is in `docs/cuda-results/copy-cost-2026-10-20.json`. Source copy 0.378 ms (run B 0.381), 26.3 GB/s; slow share of source records 0.
* L40S hidden share, level 4, f = 1.0: -0.033 (C = 0.3), 0.223 (C = 0.6) (run A).
* L40S demand exposure `D` minus the `N` floor, f = 1.0, per token: 16.8 ms (-15 % against 52.15 x 0.378 = 19.7) at C = 0.3 and 0.6.
* RTX 5090 fast population: the four slopes above; affine in 1/C fit `s = 0.0498 + 0.01574 / C`; as a function of `r = 0.194 / C`, `g(r) = 0.0498 + 0.0811 r`. Evaluated with the L40S copy time (`r = 0.378 / C`), `g` gives 0.0805 at C = 1.0 and 0.254 at C = 0.15.
* No L40S slope at C = 1.0 or 0.15 exists.

## 3. Design (fixed here)

* **Trace and arms.** The multi-lead trace (`n3d-trace.npz`, `n3d-expected.json`, unchanged); 17 % resident; copy fraction 1.0; arms `N`, `D`, `P0L`, `N`; 400 decode tokens a cell.
* **Cells (profile `j-full`).** One group (`p17j`, one 24 GiB arena), four layer times C = 1.0, 0.6, 0.3, 0.15 ms, each with a level 3 cell set (per-row records) and a level 4 cell set: **32 cells a run**. Level 6 is not run (the source copy comes from the isolated qualification). Smoke (`j-smoke`, 40 tokens): level 3 at C = 1.0 and 0.15 ms, level 4 at 0.15 ms.
* **Runs.** Two runs on the same instance, A then B (A fits, B scores). Estimate 6 to 8 minutes a run (the 48-cell runs took about 10 minutes) plus arena allocation and source qualification; a smoke first, about one minute. Session cap one GPU-hour from Ready, an explicit `--max-seconds` on every run.
* **Gates.** All stage gates stay (event hash, determinism, counts, chain2, readback, calibration, drift, closure, sanity, floor, `rowsum`, `copyevents`).
* **Offline analysis.** `benchmarks/cuda/l40s_c_ends_score.py` (new) scores J-1 to J-10 on the two runs; rehearsed on planted runs before any instance run.
* **Population rule.** The measured source copy of this instance is reported; every ratio uses it.

## 4. Predictions (registered before any build)

All **predicted**; probabilities are my own. "Slope" is the least-squares slope through the origin of the `P0L` row excess busy time on the row's measured overlap with prefetch batches, level 3 (as `third_c_score.slope_of`). Run A is the fit and the first reading, run B the score unless a row says both.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| J-1 | slope at C = 1.0 ms | between 0.075 and 0.110 (probability 0.55; H-flat would say about 0.093) | below 0.060 or above 0.130 |
| J-2 | slope at C = 0.15 ms (if the instrument holds, J-3) | between 0.085 and 0.140 (probability 0.45; H-flat 0.093, the 5090 form 0.254) | above 0.200 |
| J-3 | the instrument at the ends | the calibration and floor gates hold at C = 1.0 and at C = 0.15 in both runs and no cell fails (probability 0.45 for 0.15 ms: this card has half the 5090's memory bandwidth; 0.90 for 1.0) | any fails: that C is not scored and the minimum layer time on this card is reported |
| J-4 | anchors, instance stability | the slopes at C = 0.3 and 0.6 on this instance are each within 15 % of the first instance's 0.093 (probability 0.60) | either more than 30 % off |
| J-5 | source copy | median source-record copy within 0.34 to 0.42 ms (probability 0.60) | outside 0.30 to 0.50 (a different copy-rate class; the earlier L40S result is then read as one instance) |
| J-6 | slope ratio, the headline | s(0.15) / s(1.0) between 0.80 and 1.30 (H-flat; probability 0.45) | above 1.60 in both runs (the 5090's ratio is 2.4) |
| J-7 | direction | s(0.15) is at least s(1.0) in both runs (probability 0.65) | s(0.15) below s(1.0) by more than 0.02 in both runs |
| J-8 | hidden share, level 4 | at C = 1.0 between 0.30 and 0.55 and at C = 0.15 below 0.10 (probability 0.45 for both) | at C = 0.15 above 0.25, or at C = 1.0 below 0.15 |
| J-9 | demand exposure | `D` minus the `N` floor per token lies between 14 and 20 ms at every C (it follows the copy time, not the layer time; probability 0.55) | differs by more than 30 % between any two C values |
| J-10 | replicate | run B's slope within 15 % of run A's at seven or more of the eight points (probability 0.55) | fewer than four of eight within 30 % |

## 5. Decision rules (fixed here)

* **The 5090's C dependence is reported as present on the L40S** if J-3 holds at 0.15 ms and **s(0.15) / s(1.0) is at least 1.30 in both runs** (with J-10 not falsified). The four points are reported with the affine in 1/C fit and its residuals, beside the 5090 form, with no verdict on why.
* **It is reported as absent (H-flat)** if J-3 holds at 0.15 ms and s(0.15) / s(1.0) is below 1.30 in run B and in run A. The slope is then described as a card constant of the stated value and range over 0.15 to 1.0 ms, and the 5090's form as card-specific.
* **If J-3 fails at 0.15 ms**, the ratio is taken between the shortest layer time that holds and 1.0 ms, with that C named; no claim is made about 0.15 ms.
* **If J-4 is falsified**, the slope is not stable between L40S instances: the first instance's result is read as one instance and a third instance becomes the open item.
* **If J-5 is falsified**, the card class changed between instances and no instance-stability statement is made.
* Nothing here predicts a new default or a change in `src/`. The fit describes a stand-in compute stream.

## 6. Needs Hamed

An L40S instance (exists, `aquatic-existence-2628`, set up before the build), a cap of one GPU-hour from Ready, and his go for the named live instance (the agent runs over SSH only after the harness is rehearsed and tested on the Mac; it never runs sudo). The instance accrues time while this record and the build are written; terminate and recreate is his call. Housekeeping from earlier sessions stands: terminate `flat-battle-1078`, `imperfect-duck-7094` and `elegant-things-5691` if still up, enter the billed hours, delete the ssh configs and known_hosts of `i1`, `h2`, `h1`, `g1`, `f1`, `f2`, `f3`.
