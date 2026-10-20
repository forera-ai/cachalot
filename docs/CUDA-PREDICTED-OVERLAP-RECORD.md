# CUDA lane: a predicted-overlap version of the compute-slowdown term (experiment record; predictions before any fit)

Written 2026-10-20 against runtime 0.66.47, **before any predicted-overlap model is built or scored**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file (`docs/CUDA-PREDICTED-OVERLAP-RESULTS.md`). A prediction edited after its run is void. Labels follow plan section 0. It is the offline follow-up that `docs/CUDA-SLOWDOWN-RESULTS.md` section 4 names. It needs no GPU and starts no instance; it reads the raw samples already copied home (`~/cachalot-private/cuda-results/n3f/` and `n3f-fix/`).

## 1. What this asks, and what it cannot claim

The slowdown term of `docs/CUDA-SLOWDOWN-RECORD.md` says the extra compute time of a `P0L` row is `s` times the time the row's compute window overlaps the copy stream's prefetch batches. It was adopted **conditional on measured overlap**: the overlap came from the recorded event times of the very run being scored. Two things make that optimistic. The model cannot be used for a plan whose events have not been recorded, and the measured overlap already contains the stretch (a stretched kernel ends later and so overlaps more). The hidden-share error at C = 1.0 ms was 0.045 against a limit of 0.05.

**H-predicted-overlap.** The overlap can be computed from the queue model alone: the per-row demand and prefetch counts of the trace (known without timing), a per-copy cost, the base compute time and the gap between rows, all fitted on run A. A first-in first-out copy stream (a row's demand batch, then its prefetch batch, behind whatever earlier prefetch work is still queued) gives each row's compute window and each prefetch batch's interval. The overlap of those predicted intervals, with the slope of the slowdown term, should reproduce the measured token-level hidden share nearly as well as the measured overlap did.

**It cannot claim** a mechanism, anything about another card, instance, prefetch family or resident fraction, or anything about the real model's kernels. It is a description of one instrument's compute stream. A model that predicts the overlap from counts still takes `s` from this card's recorded rows.

## 2. Baseline (recorded, from `docs/CUDA-SLOWDOWN-RESULTS.md`)

* Slope on measured overlap (run A, level 3, `P0L`): 0.0634 at C = 1.0 ms, 0.1053 at C = 0.3 ms; run B 0.0660 and 0.1026.
* Hidden-share error against the level-4 measured share: 0.110 and 0.089 without the term; **0.045 and 0.008 with the term and measured overlap** (the numbers this record tries to match without measured events).
* Token total error of the term (predicted minus measured `P0L` minus `D` busy time): -0.022 ms at 1.0, +0.015 ms at 0.3.
* Implied measured overlap per token (measured excess over slope): about 9.6 ms at 1.0 ms and about 7.3 ms at 0.3 ms (derived).
* Row timeline in the recorded rows (read on one token before this record was written): the demand batch starts at the row's `pre`, compute starts when the demand batch ends, and the row's prefetch batch starts at about the same moment (`pf_a` within 0.003 ms of `start` or of `dem_b`), queued behind earlier prefetch work.

## 3. Design (fixed here)

* **Data.** As the slowdown record: runs A and B with the fix run's cells in place of the voided C = 1.0 level-6 and level-4 cells. Token 0 of each cell is dropped.
* **Inputs the predicted model may use on run B:** the per-row counts `n_dem` and `n_pf` of the `P0L` cell (they are the trace's, not timings) and the parameters below fitted on run A. It may **not** use any event time of run B except to score.
* **Parameters, fitted on run A at the same level and C:**
  * `c_pf`: median over `P0L` rows with `n_pf > 0` of `(pf_b - pf_a) / n_pf`.
  * `c_dem`: the demand cost per miss of the `D` arm, as `row_timing.per_miss` already gives it.
  * `b0`: median row busy time of the `D` arm.
  * `g`: median of `pre[r] - end[r-1]` over the `P0L` rows (the gap between rows).
  * `s_pred`: the slope of the slowdown term fitted by least squares through the origin on run A's `P0L` row excess against the **predicted** overlap of run A (so any systematic bias of the predicted overlap is absorbed on the fit side).
* **Simulation of a token** (rows in order, a copy stream free at time `F`, starting at 0): `dem_a = max(pre, F)`, `dem_b = dem_a + n_dem * c_dem`, `start = dem_b`, `pf_a = max(dem_b, F)` and `pf_b = pf_a + n_pf * c_pf` when `n_pf > 0`, `F = pf_b`, `end = start + b0`, next `pre = end + g`. The first row's `pre` is the run A median.
* **Variants**
  * **V1, open loop** (the registered model): `end = start + b0`, no stretch feedback.
  * **V2, closed loop:** `end = start + b0 + s_pred * (overlap of this row's window with the queued prefetch batches known so far)`, iterated to a fixed point per token (at most 8 passes). It tests whether the stretch's own contribution to the overlap matters.
  * **V3, mixed:** V1's predicted overlap with the slope fitted on **measured** overlap on run A (the slowdown record's slope). It tests whether the slope transfers between the two overlap definitions.
* **Scoring on run B** for each variant, level 3, `P0L` against the `D` arm, as in the slowdown record: the row-level held-out R² of the predicted excess against the measured excess; the predicted against the measured overlap per row (Pearson r) and per token (ratio of sums); the token busy total error; the hidden-share error against the level-4 measured share (`H = 1 - excess P0L / excess D`), with the same row model and queue wait as `compute_slowdown_fit.hidden_share` so that only the overlap changes.
* **Level 6** (an event after every copy): parameters fitted on run A's level-6 cells, the slope `s_pred` from level 3, scored on run B's level-6 `P0L` excess busy per token, as the slowdown record's CS-7.

## 4. Predictions (registered before any fit)

All **predicted**; probabilities are my own. The V1 overlap should be biased low against the measured one, because the measured overlap contains the stretch; the size of that bias is not known and is what PO-2 states.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| PO-1 | row-level agreement of V1's predicted overlap with the measured overlap, run B, Pearson r | at least 0.85 at each C (probability 0.60) | r below 0.60 at either C (the queue model then does not place the copies) |
| PO-2 | tokens' total overlap, V1 predicted over measured, run B | between 0.85 and 1.05 at each C (probability 0.55) | outside 0.65 to 1.25 at either C |
| PO-3 | slope `s_pred` of V1 fitted on run A | within 25 % of the measured-overlap slope (0.0634, 0.1053) at each C (probability 0.50); biased high because the predicted overlap is biased low | more than 60 % away at either C |
| PO-4 | token busy total, V1 with `s_pred`, run A's fit on run B | error against the measured `P0L` minus `D` busy time at most 0.10 ms at each C (probability 0.55) | error above 0.25 ms at either C |
| PO-5 | the main claim: hidden-share error of the row model plus V1 | at most 0.05 against the level-4 measured share at each C (probability 0.50; 0.045 at C = 1.0 with measured overlap leaves almost no margin) | error above 0.08 at either C |
| PO-6 | predicted against measured overlap, same slope definition | the V1 hidden-share error is no more than 0.03 above the measured-overlap error (0.045, 0.008) at each C (probability 0.60) | more than 0.06 above at either C |
| PO-7 | the replication at level 6 | V1 with level-6 parameters and the level-3 `s_pred` predicts the level-6 `P0L` excess busy per token within 0.15 ms at each C (probability 0.45) | an error above 0.40 ms at either C |
| PO-8 | the stretch feedback | V2's total overlap per token differs from V1's by at most 8 % at each C (probability 0.70) | more than 20 % at either C (the stretch then matters and V1 is not the model) |
| PO-9 | the slope's transfer | V3 (V1's overlap with the measured-overlap slope) has a token busy total error at most 0.15 ms at each C (probability 0.50) | above 0.40 ms at either C |

## 5. Decision rules (fixed here)

* **The predicted-overlap version is adopted** for the `P0L` row model, on this card, this trace, 17 % resident and the fast population only, as the **unconditional** form of the slowdown term, if **PO-4 and PO-5 hold and PO-1 is not falsified**. The measured-overlap form then remains the reference it was fitted against and the conditional label is dropped for this card and trace only.
* If PO-5's falsifier fires, the term stays **conditional on measured overlap** as adopted on 2026-10-20, and the predicted version is reported as a failed attempt with its error.
* If PO-1's falsifier fires, the queue model does not place the copies and the predicted version is closed whatever the totals show.
* If PO-8's falsifier fires, V2 replaces V1 as the registered form and the scoring is repeated for V2 only, as a second step flagged in the results.
* Whatever the outcome, nothing here predicts a change in `src/`, a default, or a number for the real model. The fit describes a stand-in compute stream.

## 6. Needs Hamed

Nothing. No instance, no key, no sudo, no server. His housekeeping from the previous session (terminate the instance if still up, billed hours, the `f1`, `f2`, `f3` ssh configs) is unchanged.
