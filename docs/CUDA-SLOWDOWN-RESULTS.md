# CUDA lane: a compute-slowdown term for the row model (results; Mac only, no GPU)

Run 2026-10-20 under `docs/CUDA-SLOWDOWN-RECORD.md` (seven predictions committed in 35e2d14 before the script existed; the script and its rehearsal tests were committed before the first real fit). Data: the raw samples of the two full row-timing runs (A: `20261020T112911Z-n3f-f-full-12fdd0`, B: `20261020T113342Z-n3f-f-full-6d1b76`) with the fix run's cells in place of the voided C = 1.0 level-6 and level-4 cells, all from one fast-population RTX 5090 instance. Scoring: `benchmarks/cuda/compute_slowdown_fit.py RUN_A RUN_B --fix FIX`, output `docs/cuda-results/slowdown-fit-2026-10-20.json`. Labels follow plan section 0: **measured** on the recorded rows; fits are **derived**; explanations are **hypothesis**.

## 1. Scoring the registered predictions

All rows are level 3, `P0L` arm, the fit on run A and the score on run B unless stated. Slope `s` is milliseconds of extra compute per millisecond of overlap between a row's compute window and the copy stream's prefetch batches.

| id | measured | verdict |
|---|---|---|
| CS-1 (slope 0.03 to 0.06 at C = 1.0; 0.05 to 0.11 at 0.3) | 0.0634 at C = 1.0; 0.1053 at C = 0.3 | **missed** (the 0.3 half held; the 1.0 half is 0.003 above its interval and well inside the 0.01 to 0.15 falsifier) |
| CS-2 (run B's slope within 15 % of run A's) | 0.0660 against 0.0634 (4.1 %); 0.1026 against 0.1053 (2.5 %) | **held** |
| CS-3 (held-out R² of the overlap model at least 0.30 at each C) | 0.711 at C = 1.0; 0.850 at C = 0.3 | **held** |
| CS-4 (overlap beats the copy-count term by at least 10 % in held-out squared error) | squared error 0.676 against 1.282 (47 % lower) at 1.0; 0.357 against 2.227 (84 % lower) at 0.3 | **held** |
| CS-5 (token total within 0.10 ms of the measured P0L minus D busy time) | predicted 0.613 against measured 0.635 ms (error -0.022) at 1.0; 0.763 against 0.748 (+0.015) at 0.3 | **held** |
| CS-6 (token hidden-share error at most 0.05; today 0.110 and 0.089) | with the term: 0.4679 against the level-4 measured 0.4226 (error 0.045) at 1.0; 0.2397 against 0.2319 (0.008) at 0.3 | **held** (narrowly at 1.0) |
| CS-7 (the level-3 slope predicts the level-6 P0L excess busy time within 0.15 ms) | predicted 0.616 against measured 0.661 ms (error -0.045) at 1.0; 0.764 against 0.749 (+0.015) at 0.3 | **held** |

**6 held, 1 missed, 0 falsified.** The adoption rule (CS-3, CS-5 and CS-6 hold, CS-4 not falsified) is met, with the qualifications of section 3.

## 2. What was measured

* **The extra compute time is a function of the overlap (measured, derived).** The competing descriptions of the same rows: no term (held-out R² -2.5 and -2.7, the rows sit above their base), a constant per row (R² -0.01 and -0.003), the count of copies (0.45 and 0.07), and the time overlap (0.71 and 0.85). The constant per row and the count of copies explain almost nothing at 0.3 ms; the overlap in time does.
* **The slope depends on C (measured, derived).** 0.063 at a 1.0 ms layer and 0.105 at 0.3 ms: a millisecond of overlap stretches a shorter kernel more. This is not explained here.
* **The token error closes (derived).** The row model's hidden share was off by 0.110 and 0.089 without the term; with it, 0.045 and 0.008. The per-token busy-time gap (0.635 and 0.748 ms measured on run B, 0.607 and 0.77 in the earlier section) is reproduced by the term to 0.02 ms.
* **Replication across instrument levels (measured).** The slope fitted on the level-3 cells of run A predicts the level-6 cells (an event after every copy) to 0.045 and 0.015 ms. The level-6 cells at C = 1.0 are the single fix run.
* **Replicate agreement (measured).** The slopes of the two runs agree to 4.1 % and 2.5 %.

## 3. What it means, and does not

* **The overlap is measured, not predicted.** The term uses each row's measured compute window and the measured prefetch batch intervals, so it is conditional on a model of the queue that gives those intervals. The overlap itself contains the stretch (a stretched kernel ends later and overlaps more), which inflates the fit slightly. A model that predicts overlap from the queue state and then adds the term has not been built or scored; the 0.045 at C = 1.0 is thus an optimistic figure for it, and sits close to the limit of 0.05.
* **The bootstrap interval is too narrow to quote.** A bootstrap over tokens gives a 95 % interval of about +-0.0004 on the slope; tokens within a run and the runs on one instance are not independent, so the replicate agreement (2.5 to 4 %) is the honest scale.
* **One instance, one trace, one resident fraction, one stand-in.** Nothing here concerns another card (the copy rate varies by instance), another prefetch family, another C beyond the two used, or the real model's kernels. The cause (memory contention or copy-engine scheduling) is not identified; a time-overlap term is a description.

## 4. Decision

* The slowdown term is adopted for the `P0L` row model on this card, this trace, 17 % resident, fast-population instance, and the two C values, **conditional on measured overlap**. The copy-count screen stays for everything else.
* Open and offline: a predicted-overlap version (the overlap from the queue model's own copy times) scored on the same data, with predictions first. It could turn the 0.045 into a number that is not conditional. It needs no GPU and is Hamed's call.
* Open and needing a GPU and Hamed's go: why the slope doubles from 1.0 to 0.3 ms (a third C at 0.6 ms would give a point on the curve), another card, a native issue path, ROCm, instance population.
