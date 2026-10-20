# CUDA lane: per-row and per-copy timing of a miss (experiment record; predictions before any build)

Written 2026-10-20 against runtime 0.66.44, **before any per-copy timing code is written or run**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file (`docs/CUDA-ROW-TIMING-RESULTS.md`). A prediction edited after its run is void. Labels follow plan section 0. This record is the one item `docs/CUDA-QUEUE-STATE-RESULTS.md` section 4 names as able to explain the per-miss cost: "a per-row timing instrument would be needed, on a GPU". It needs an RTX 5090 instance and Hamed's go; nothing here starts one.

## 1. What this asks, and what it cannot claim

Three offline records (window fit, queue-state, bridge) found that the misses `P0L` leaves behind cost about 0.22 ms each on the new trace against 0.187 ms for `D`'s misses, and that no row-level count (contended misses, contended rows, burstiness) explains the difference well enough to adopt a model (held-out error 0.0919 against the copy-count screen's 0.0944). The harness measures a token's wall time and the sums, over its rows, of the event pairs it records. It does not record, for one row, how long the compute stream waited behind earlier prefetch copies, how long the row's own demand copies took, or how long any single copy took.

The instrument added here records, for every decode row of a cell, the events the harness already has (a row's `pre`, `start`, `dem_a`, `dem_b`, `pf_a`, `pf_b`) as **per-row values** instead of per-token sums, and, at one new level (**L6**), one CUDA event after every copy the harness enqueues, so that each copy has its own completion time on the copy stream.

**H-queue-wait.** The extra cost of a left-behind miss is not a slower copy but a wait: the row's demand batch is queued on the copy stream behind prefetch copies for the next row that the previous row's compute window did not finish. This is testable per row (the wait is `dem_a - pre`, positive only when earlier copies are still in flight at the row's boundary) and per copy (the copy-time of a demand copy that was issued to an idle queue against one queued behind prefetch).

**It cannot claim** anything about another card or instance (the lane has seen the copy rate vary by instance, 0.19 against 0.35 ms an expert), about the real model, or about a mechanism inside the copy engine: it measures completion times of copies on one stream against one stream of stand-in compute. Per-copy events are themselves an intervention (an event is recorded after each copy); its cost is measured, not assumed (RT-6).

## 2. Baseline (recorded)

* Per-copy time `c`: 0.190 ms (fast-population instance, stage C and bridge; 0.185 in N2 stage 1); 0.35 ms on the slow-population instances (N3 stage B).
* Hidden share of `P0L` on the new trace (bridge, level 4 and 5): 0.433 at 1.0 ms, 0.407 at 0.6, 0.233 at 0.3; old trace 0.339 / 0.305 / 0.098. The row-level copy-count screen overstates the new trace by 0.07 to 0.10; the best queue-state term (T2) lowers that to 0.045 to 0.062.
* Per-miss cost fitted on the new trace: 0.187 ms (`D`), 0.22 ms (misses left behind by `P0L`).
* Per token at 17 % resident on the new trace: about 83.6 prefetch loads and 28.1 demand misses; loads and misses are spread over 28 to 40 rows a token (1.83 misses in a missing row, `layer_miss_profile.py`).

## 3. Design (fixed here)

* **Trace and arms.** The new multi-lead trace (the one stages D, the bridge and the queue-state fit used; `n3d-expected.json`); 17 % resident; arms `N`, `D`, `P0L`, `N` at C = 1.0 ms and C = 0.3 ms. The `N` controls bracket the arms as in every stage.
* **Per-row mode at level L3** (lean bookkeeping, prebuilt slot views; the per-layer events present): per decode row, stored in the raw samples: `n_dem`, `n_pf` (copies enqueued for the row), the compute wait before the row `stall_ms = start - pre`, the demand queue wait `q_ms = max(0, dem_a - pre)` (the part of that wait spent behind copies queued earlier), the demand batch time `d_ms = dem_b - dem_a`, the prefetch batch time `pf_ms = pf_b - pf_a`, and the compute time `busy_ms = end - start`.
* **L6** = L3 plus one event after every enqueued copy (demand and prefetch), read after the token with the rest of the token's events. Per copy: its duration (this event against the previous event on the copy stream: `dem_a`/`pf_a` for the first copy of a batch), its kind (demand or prefetch), its position in the batch, and whether compute was running when it completed (derived from the compute events of the same row).
* **Cells.** Two runs on the same instance, same trace, in the order A, B (a replicate, as the bridge): each run = L3 at C = 1.0 and C = 0.3 (arms `N`, `D`, `P0L`, `N`), L6 at C = 1.0 and C = 0.3 (arms `N`, `D`, `P0L`, `N`), L4 at C = 0.3 (the stage D level, to bridge the new per-row cells to the recorded hidden shares). 20 cells a run, 400 decode tokens each. Estimate 8 to 12 minutes a run plus calibration and arena allocation; session cap one GPU-hour, explicit `--max-seconds`.
* **Gates** (all the stage gates stay: event hash, determinism, counts, chain2, readback, calibration, drift, closure, sanity), plus (h): the per-row records of a token sum to the token's per-token values (`stall`, `busy`, `dem`, `pf`) within float noise; (i): at L6 the number of per-copy events read equals the number of copies the bookkeeping issued.
* **Offline analysis** `benchmarks/cuda/row_timing.py` (Mac): reads the raw samples, scores RT-1 to RT-7, prints the tables of the results file. The same script is rehearsed on fabricated runs.

## 4. Predictions (registered before any build)

All **predicted**. Probabilities are my own. "Idle queue" means the copy stream had no earlier copy in flight when the copy was enqueued (derived from the event order of the same row).

| id | quantity | prediction | falsifier |
|---|---|---|---|
| RT-1 | the cost of one copy to an idle queue | median demand-copy time at L6, `D` arm, C = 1.0 ms: 0.19 ms (0.17 to 0.22 on a fast-population instance; 0.30 to 0.38 if the instance is slow-population, which the source qualification names before the run) (probability 0.7) | outside 0.15 to 0.40, or different between the first and later copies of a batch by more than 0.05 ms |
| RT-2 | the batch overhead | in a batch of two or more demand copies the first copy's time exceeds the median of the later copies by between -0.03 and +0.06 ms (no fixed per-batch cost) (probability 0.6) | outside -0.06 to +0.12 |
| RT-3 | how often a miss waits behind prefetch | share of `P0L` rows with misses whose demand queue wait `q_ms` exceeds 0.02 ms: at most 0.20 at C = 1.0 ms; between 0.40 and 0.75 at C = 0.3 ms (probability 0.5 for both) | share above 0.35 at 1.0 ms, or below 0.25 at 0.3 ms |
| RT-4 | the main claim, H-queue-wait | per left-behind miss, `(q_ms + d_ms) / n_dem` in `P0L` rows with misses exceeds the same quantity for `D` by 10 to 25 % at C = 1.0 ms (the 0.187 to 0.22 ms of the fits), and `q_ms` accounts for at least 70 % of that excess (probability 0.45) | `q_ms` accounts for under 40 % of the excess, or the per-miss cost of `P0L` is within 5 % of `D`'s |
| RT-5 | a copy under compute | the time of a demand copy that completes while a compute kernel runs is between 0 % and +8 % above one completing while compute waits (probability 0.55; S1 and the stage results found compute slowed 0.3 to 1.1 % by copies, the converse is unmeasured) | above +15 %, or below -3 % |
| RT-6 | what per-copy events cost | the median token wall at L6 exceeds L3's (same arm, C, and run) by at most 0.5 ms (about 110 events a token at under 5 microseconds each); closure share at L6 at least 0.90 (probability 0.5) | excess above 1.5 ms (per-copy timing then perturbs the thing measured: all RT-1 to RT-5 are reported as conditional on that) |
| RT-7 | does a queue-aware row model beat the screen | a row model that predicts each row's `q_ms` from the copies still queued at its boundary (`max(0, n_pf_prev * c - C)` with the fitted `c`) correlates with the measured `q_ms` at r of at least 0.8 over `P0L` rows, run B scored on run A's fit; and its token-level `P0L` hidden-share error against the measured L4 shares is at most 0.05 at every C (probability 0.4) | r below 0.6, or token error above 0.08 at any C |

## 5. Decision rules (fixed here)

* **A queue-aware row model is adopted for the `P0L` family** (replacing the copy-count screen for that family) only if **RT-4 and RT-7 both hold** and RT-6's falsifier did not fire. It is then reported with its held-out error, and never for multi-layer prefetch or the oracle without its own test.
* If RT-4's falsifier fires, **H-queue-wait is closed** and the per-miss difference is reported as a per-copy effect (RT-5 or position-dependent), to be tested by a separate record; the screen stays.
* If RT-6's falsifier fires, the per-copy cells (L6) are reported as perturbed, and the per-row cells (L3) carry the conclusion alone.
* The slow-population rule of the lane stands: if the source qualification puts the instance in the slow class (median source-record copy at or above 0.30 ms), RT-1 is scored against its slow interval and RT-3, RT-4 and RT-7 are reported but not used for adoption (they were registered for the fast class); a fast-class instance is then requested if Hamed agrees.
* Nothing here predicts a new default or a change in `src/`.

## 6. Needs Hamed

An RTX 5090 instance (the card every earlier stage used, so the new per-row cells compare with stages C, D and the bridge), a time cap of one GPU-hour from Ready, and his go for the named live instance. The agent never runs sudo and runs over SSH only for that instance, as in the bridge session.

## 7. Erratum, written before any code was built or run (2026-10-20)

Two defects found while designing the harness from this record; neither involves a measurement, and both are fixed here before the build.

1. **RT-5 as worded cannot be measured.** A row's compute waits for its own demand batch (`dem_b`), so a demand copy never completes while that row's compute kernel runs. The comparison is made on **prefetch copies**: the time of a prefetch copy that completes while its row's compute kernels run (between the row's `start` and `end` events) against one that completes after the row's `end` event (compute stream idle, in the gap or in the next row's stall). The interval and falsifier of RT-5 stand unchanged (between 0 % and +8 % slower under compute; falsified above +15 % or below -3 %).
2. **RT-7 needs a level-4 cell at C = 1.0 ms.** Section 3 listed level 4 only at C = 0.3, so no recorded hidden share existed at 1.0 ms to score "at every C" against. The cell list becomes: L3 at C = 1.0 and 0.3, L6 at C = 1.0 and 0.3, L4 at C = 1.0 and 0.3, each with arms `N`, `D`, `P0L`, `N`: **24 cells a run**, 400 decode tokens each (estimate 6 to 8 minutes a run plus calibration; two runs).

## 8. Erratum, written after the first full runs and before the fix run was scored (2026-10-20)

The first full runs (`f-full`, two runs) had a harness defect of mine: the plan returned to C = 1.0 ms after C = 0.3 ms, and the harness re-tuned the stand-in compute only the first time a C appeared. The level-6 and level-4 cells labelled C = 1.0 ran with the stand-in tuned to 0.3 ms (N-control floors 12.6 ms instead of 40 ms); the level-3 cells at both C and the level-6 and level-4 cells at C = 0.3 were valid. No earlier stage revisited a C, and no gate compared the floor with the layers times C. The fix (commit 129119a): re-tune on every change of C, a `floor` gate (the N-control floor within 25 % of layers times C), and the profile `f-fix` (level 6 and level 4 at C = 1.0, one run, run on the same instance). The scoring script takes the fix run's cells in place of the voided ones (`row_timing.py --fix`), so those cells have no replicate. The predictions of section 4 are unchanged.
