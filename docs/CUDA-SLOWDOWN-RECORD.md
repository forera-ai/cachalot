# CUDA lane: a compute-slowdown term for the row model (experiment record; predictions before any fit)

Written 2026-10-20 against runtime 0.66.46, **before any slowdown model is fitted or scored**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file (`docs/CUDA-SLOWDOWN-RESULTS.md`). A prediction edited after its run is void. Labels follow plan section 0. It is the offline follow-up that `docs/CUDA-ROW-TIMING-RESULTS.md` section 5 names. It needs no GPU and starts no instance; it reads the raw samples already copied home (`~/cachalot-private/cuda-results/n3f/` and `n3f-fix/`).

## 1. What this asks, and what it cannot claim

The row-timing session found that the queue-aware row model gets each row's stall right (r 0.945 at 1.0 ms, 0.991 at 0.3 ms) but its token-level hidden share is off by 0.110 and 0.089, against a registered limit of 0.05. The recorded explanation is a hypothesis: the stand-in compute's busy time is 0.607 ms a token longer in `P0L` than in `D` at C = 1.0 ms and 0.77 ms longer at C = 0.3 ms, identical in runs A and B to 0.006 ms, which is about the size of the token error.

**H-overlap-slowdown.** A prefetch copy running on the copy stream while a compute kernel runs stretches that kernel in proportion to the time the two overlap (cross-stream memory contention). The extra compute time of a row is then `s` times the time its compute window overlaps prefetch copy activity, with `s` fitted per C. A row model that adds this term should close most of the token-level error.

**It cannot claim** a mechanism (the fit cannot tell memory contention from copy-engine scheduling), anything about another card or instance (this is one fast-population RTX 5090), or anything about the real model's kernels: the compute is the harness's bandwidth-bound stand-in. It is a description of one instrument's compute stream.

## 2. Baseline (recorded, all from `docs/CUDA-ROW-TIMING-RESULTS.md`)

* Token busy time of `P0L` minus `D` (stand-in compute): +0.607 ms at C = 1.0 (+1.5 %), +0.77 ms at C = 0.3 (+6.3 %), replicate agreement 0.006 ms.
* Queue-aware row model without a slowdown term: hidden share 0.533 against the level-4 measured 0.423 at 1.0 ms (error 0.110), 0.321 against 0.232 at 0.3 ms (error 0.089); per-row correlation 0.945 and 0.991.
* Per token at 17 % resident: about 83.6 prefetch loads (about 17 ms of copy time at 0.2036 ms a copy) and 28.1 demand misses.

## 3. Design (fixed here)

* **Data.** The raw samples of the two full runs (`f-full`, runs A and B) with the fix run's cells in place of the voided C = 1.0 level-6 and level-4 cells (the merge `row_timing.py` already does). Row fields `n_dem`, `n_pf`, `pre`, `start`, `end`, `dem_a`, `dem_b`, `pf_a`, `pf_b`; times are ms since the token's first event. Token 0 of each cell (warm-up) is dropped.
* **Per-row quantities.** `busy = end - start`. The base compute time `b0` per level and C is the median row `busy` of the `D` arm of the same level and C (no prefetch copies exist there). The excess of a `P0L` row is `busy - b0`.
* **Overlap term.** `overlap(row) = sum over rows r' of the same token of |[start, end] intersect [pf_a(r'), pf_b(r')]|`, the time the row's compute window overlaps the copy stream's prefetch batches (a row with no prefetch has no interval). Rows whose events are missing contribute zero.
* **Models**, each fitted by least squares through the origin on run A's `P0L` rows of one level and one C, and scored on run B's `P0L` rows of the same level and C:
  * **M1, overlap:** excess = `s` x overlap.
  * **M2, count:** excess = `a` x (prefetch copies of the row's own batch and of the previous row's batch), the copy-count counterpart.
  * **M3, flat:** excess = `f` per row (a constant).
  * **M0:** excess = 0 (the existing model).
* **Levels used.** Level 3 (lean bookkeeping, per-layer events present) is the primary level and the one the fit and the scoring use. Level 6 (a per-copy event, a different perturbation) at the same C is scored as an out-of-sample replication of the level-3 fit; the fix run's level-6 cells at C = 1.0 are single-run.
* **Token-level check.** The row model of `row_timing.py` (`score_rt7`) with `s` x total overlap per token added to the `P0L` token cost, scored against the level-4 measured hidden share (`H = 1 - excess P0L / excess D`, as RT-7 does), with the fit taken from run A and the score on run B.
* **Statistics.** Held-out R² = 1 minus the sum of squared error of the model's row excess against the measured one, over the sum of squares about the mean measured row excess on run B. Slope uncertainty by a bootstrap over tokens (1,000 draws).

## 4. Predictions (registered before any fit)

All **predicted**. Probabilities are my own. The centres of CS-1 are derived from the aggregates in section 2 (0.61 ms over about 14 ms of overlap at 1.0 ms; 0.77 ms over about 10 ms at 0.3 ms), so they are priors, not blind guesses.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| CS-1 | the fitted slope `s` (ms of extra compute per ms of overlap), level 3, run A | between 0.03 and 0.06 at C = 1.0 ms; between 0.05 and 0.11 at C = 0.3 ms (probability 0.5 for both) | `s` outside 0.01 to 0.15 at either C |
| CS-2 | the replicate | `s` fitted on run B differs from run A's by at most 15 % at each C (probability 0.65) | a difference above 40 % at either C |
| CS-3 | the row-level fit | held-out R² of M1 on run B is at least 0.30 at each C (probability 0.45) | R² below 0.10 at either C (the slowdown is then not a row-level function of overlap) |
| CS-4 | overlap against count | M1's held-out squared error is at least 10 % below M2's at each C (probability 0.5) | M2 is lower than M1 at either C |
| CS-5 | the token total | the predicted `P0L` minus `D` busy time per token is within 0.10 ms of the measured 0.607 ms at C = 1.0 and 0.77 ms at C = 0.3, from run A's `s` on run B (probability 0.6) | an error above 0.25 ms at either C |
| CS-6 | the main claim | the row model plus M1 has a token hidden-share error of at most 0.05 against the level-4 measured share at each C, run A's fit on run B (probability 0.45) | error above 0.08 at either C (today 0.110 and 0.089: failing to move below 0.08 closes the term) |
| CS-7 | the replication at level 6 | M1 fitted at level 3 predicts the level-6 `P0L` excess busy per token within 0.15 ms at each C (probability 0.5) | an error above 0.4 ms at either C (the slowdown is then specific to the bookkeeping level) |

## 5. Decision rules (fixed here)

* **The slowdown term is adopted** for the `P0L` row model, on this card, this trace, 17 % resident and the fast population only, if **CS-3, CS-5 and CS-6 hold and CS-4 is not falsified**. It is then reported with its held-out error and never extended to another prefetch family, another resident fraction, another card or the real model without its own test.
* If CS-3's falsifier fires, **H-overlap-slowdown is closed at the row level**: the extra busy time exists (0.6 and 0.8 ms a token) but is not a function of the row's overlap with prefetch; it is reported as a per-token or per-row constant (M3), which the model may use only as a fitted constant, flagged as such.
* If CS-4's falsifier fires, the count of copies is a better description than the time overlap: the term is reported as a per-copy cost and the mechanism claim (overlap in time) is withdrawn.
* If CS-6's falsifier fires, the term is not adopted whatever the other results, and the copy-count screen stays.
* The fit is a description of a stand-in compute stream. Nothing here predicts a change in `src/`, a default, or a number for the real model.

## 6. Needs Hamed

Nothing. No instance, no key, no sudo, no server. His housekeeping from the previous session (terminate the instance, billed hours, the `f1`, `f2`, `f3` ssh configs) is unchanged.
