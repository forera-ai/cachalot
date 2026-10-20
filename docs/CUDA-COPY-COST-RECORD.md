# CUDA lane: is the compute-slowdown slope a function of copy time over layer time? A copy-cost sweep on one L40S (experiment record; predictions before any build)

Written 2026-10-20 against runtime 0.66.53, **before any code for this session is written or run**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file (`docs/CUDA-COPY-COST-RESULTS.md`). A prediction edited after its run is void. Labels follow plan section 0. It takes the last open confound of `docs/CUDA-LOW-C-FAST-RESULTS.md` section 4 ("the population effect: is the slope proportional to copy cost: a controlled copy-cost sweep on one instance"). Hamed chose this experiment and an **NVIDIA L40S 48 GB** (PCIe 4.0 x16, Ada, 48 MB L2) on 2026-10-20; the instance exists (provider name `flat-battle-1078`) but nothing is run on it until this record is committed and the harness is built and rehearsed on the Mac.

## 1. What this asks, and what it cannot claim

On the RTX 5090 the slope `s` (extra compute milliseconds per millisecond of overlap with prefetch copies) was 0.0634, 0.0765, 0.1053 and 0.1534 at layer times C = 1.0, 0.6, 0.3 and 0.15 ms on fast-population instances (source copy `c` about 0.194 ms), and 0.2234 at C = 0.15 ms on a slow-population instance (`c` 0.349 ms). Writing the fast form with `r = c / C`, the four fast points give `s = 0.0498 + 0.0811 r` (the affine-in-1/C fit of the earlier record, rewritten with `c = 0.194`), and that same line evaluated at the slow instance's ratio (r = 2.37) gives 0.242 against 0.2234 measured: 8 % high. The two effects the lane has seen, a shorter layer and a dearer copy, may be one effect.

**H-ratio.** On one card and one instance the slope is a function of `r = c / C` alone: changing `c` by copying fewer bytes per expert moves the slope the same way as changing C does. The alternatives: the slope follows the layer time alone (copy cost irrelevant); the bytes moved (memory-bandwidth contention, not copy-engine time); or the copy time and the layer time separately.

**The manipulation.** Per cell group the harness copies only the first `f x 9.95 MB` of each expert record (`f` = 1.0, 0.5, 0.25), into the same slot, from the same pinned sources. Slot assignment, hashes, miss and load counts are unchanged by construction (the same bookkeeping runs), so only the cost of a copy moves. `c(f)` is measured per group by the existing source qualification (64 records, median), not assumed.

**It cannot claim** why the slope depends on `r`, anything about a real expert's cost on a real layer (the stand-in compute is a calibrated bandwidth-bound kernel pair and the L40S has a different L2, copy engine and memory system from the RTX 5090), another trace, another resident fraction, or that fewer bytes per copy is a deployable lever (it is an instrument). A within-instance sweep separates copy cost from the instance; it does not separate time from bytes (`c` is nearly proportional to bytes here), which the slow instance of the RTX 5090 did (same bytes, dearer copy, higher slope): that comparison is cited, not re-measured.

## 2. Baseline (recorded)

* RTX 5090, fast population, `P0L` slope on measured overlap, level 3: 0.0634 (C = 1.0), 0.0765 (0.6), 0.1053 (0.3), 0.1534 (0.15); affine in 1/C fit `s = 0.0498 + 0.01574 / C`, rms 0.002; as a function of `r = 0.194 / C`: `g(r) = 0.0498 + 0.0811 r`.
* RTX 5090, slow instance, C = 0.15: 0.2234 at `c` = 0.349 ms (`r` = 2.3); `g` gives 0.242.
* Hidden share of `P0L`, level 4: 0.423 (C = 1.0), 0.407 (0.6), 0.232 (0.3), -0.054 (0.15), fast instances; the count-only window model with `c` = 0.194 ms gave 0.533, 0.480, 0.212, -0.085.
* Demand-only exposure (`D` minus the `N` floor) equals demand misses a token times `c`: 52.15 x 0.194 = 10.1 ms on the fast instances (stage A measured 9.9 to 10.1 ms).
* No L40S number exists.

## 3. Design (fixed here)

* **Trace and arms.** The multi-lead trace (`n3d-trace.npz`, `n3d-expected.json`, unchanged); 17 % resident; arms `N`, `D`, `P0L`, `N`; 400 decode tokens a cell.
* **Cell groups (profile `i-full`).** Three groups, `p17f100`, `p17f50`, `p17f25` (copy fraction `f` = 1.0, 0.5, 0.25). Each group runs, at **C = 0.3 ms and C = 0.6 ms**, a level 3 cell set with per-row records and a level 4 cell set: 16 cells a group, **48 cells a run**. The two C values give equal-`r` pairs (f = 1.0 at C = 0.6 against f = 0.5 at C = 0.3; f = 0.5 at C = 0.6 against f = 0.25 at C = 0.3) at different `c` and C. Level 6 (an event after every copy) is not run: `c(f)` comes from the isolated source qualification, and the level 4 and 3 cells give the hidden share and the slope.
* **Runs.** Two runs on the same instance, A then B (A fits, B scores). Estimate 6 to 9 minutes a run plus per-group arena allocation and source qualification; a smoke (`i-smoke`, 40 tokens, three cell sets) first. Session cap one GPU-hour from Ready, an explicit `--max-seconds` on every run.
* **Gates.** All stage gates stay (event hash, determinism, counts, chain2, readback, calibration, drift, closure, sanity, floor, `rowsum`, `copyevents`). The readback gate compares the copied prefix of a slot (the harness copies only that many bytes; the record is a prefix check by construction). The sanity gate uses the bytes actually copied.
* **Offline analysis.** `benchmarks/cuda/copy_cost_score.py` (new) scores SW-1 to SW-10 on the two runs. Rehearsed on planted runs before any instance run.
* **Population rule.** The L40S has no fast/slow classification from the earlier record; `c(1.0)` is reported as measured and every ratio uses the measured `c` of its own group.

## 4. Predictions (registered before any build)

All **predicted**; probabilities are my own. "Slope" is the least-squares slope through the origin of the `P0L` row excess busy time on the row's measured overlap with prefetch batches, level 3 (as in `third_c_score.slope_of`). `r = c(f) / C`. Run A is the fit and the first reading, run B the score unless a row says both.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| SW-1 | median source-record copy at f = 1.0 on the L40S (9.95 MB, PCIe 4.0 x16) | between 0.36 and 0.50 ms (probability 0.55; about 25 GB/s plus 20 to 40 us fixed) | outside 0.28 to 0.70 ms |
| SW-2 | how copy time follows bytes | c(0.5) / c(1.0) between 0.50 and 0.62 and c(0.25) / c(1.0) between 0.26 and 0.40 (probability 0.5) | c(0.5) / c(1.0) below 0.45 or above 0.70 (copy time is not near-affine in bytes) |
| SW-3 | slope rises with copy cost | at each C, s(f = 1.0) > s(f = 0.5) > s(f = 0.25), run A and run B (probability 0.70) | s(f = 1.0) at most s(f = 0.25) at either C (copy cost does not move the slope) |
| SW-4 | slope is one function of r | across the six (f, C) points the least-squares line `s = a + b r` has R squared at least 0.85 (probability 0.55) | R squared below 0.50 |
| SW-5 | equal-r pairs | in both equal-r pairs (section 3) the two slopes are within 25 % of each other (probability 0.50) | either pair more than 40 % apart (the slope is not a function of r alone) |
| SW-6 | the RTX 5090 fast form carries over | `g(r) = 0.0498 + 0.0811 r` is within 0.03 of the measured slope at four or more of the six points (probability 0.35; the card differs) | mean absolute error of `g` over the six points above 0.06 |
| SW-7 | hidden share | the count-only window model with the measured `c(f)` predicts the measured level 4 hidden share of `P0L` to within 0.10 at four or more of the six points (probability 0.50) | mean absolute error above 0.18 |
| SW-8 | the instrument | the calibration and floor gates hold in every group at both C and no cell fails (probability 0.70) | any fails: the affected groups are not scored, the minimum layer time on this card is reported |
| SW-9 | demand exposure follows c | `D` minus the `N` floor, per token, within 15 % of 52.15 x c(f) at each f and each C, run A (probability 0.55) | more than 35 % off at any point |
| SW-10 | replicate | run B's slope within 15 % of run A's in at least five of the six points (probability 0.60; 0.6 % to 3 % between runs on the 5090) | fewer than three of six within 30 % |

## 5. Decision rules (fixed here)

* **The slope is described as a function of `r = c / C` on this card and instance** if **SW-4 and SW-5 are not falsified, SW-3 holds, SW-10 holds and SW-8 holds**. The fitted line and its residuals are then reported beside `g(r)`, and the slow RTX 5090 point (r = 2.3, s = 0.2234) is placed against the L40S line extrapolated to that ratio, as a derived comparison with no verdict.
* If **SW-4 or SW-5 is falsified with SW-8 true**, the slope is not a function of `r` alone on this card: the six points are reported, no function is fitted, and the three alternatives of section 1 are named as the next record, not run.
* If **SW-3 is falsified**, copy cost does not move the slope here and the slow-instance excess on the RTX 5090 is not explained by `c`; that is reported as the finding.
* If **SW-8 fails** for a group, that group's slope, hidden share and exposure are not scored; the minimum layer time on this card is reported.
* If SW-1 is outside its falsifier the card is reported as a different copy-rate class and nothing is concluded about the earlier population labels.
* The card changes, so SW-6 is a statement about card dependence of the form, not a test of H-ratio; it does not enter the decision rules above.
* Nothing here predicts a new default or a change in `src/`. The fit describes a stand-in compute stream.

## 6. Needs Hamed

An L40S instance (exists), a cap of one GPU-hour from Ready, and his go for the named live instance (the agent runs over SSH only after the harness is rehearsed and tested on the Mac; it never runs sudo). The instance accrues time while it waits for this record and the build; terminate and recreate is his call if the wait is long. His housekeeping from earlier sessions (terminate any older instance, enter the billed hours, delete the ssh configs and known_hosts of `h2`, `h1`, `g1`, `f1`, `f2`, `f3`) stands.
