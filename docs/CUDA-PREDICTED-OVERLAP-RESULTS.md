# CUDA lane: a predicted-overlap version of the compute-slowdown term (results; Mac only, no GPU)

Run 2026-10-20 under `docs/CUDA-PREDICTED-OVERLAP-RECORD.md` (nine predictions committed in 2777875 before any model existed; the script and its rehearsal tests were committed in 80cc65a before the first real fit). Data: the raw samples of the two full row-timing runs (A: `20261020T112911Z-n3f-f-full-12fdd0`, B: `20261020T113342Z-n3f-f-full-6d1b76`) with the fix run's cells in place of the voided C = 1.0 level-6 and level-4 cells, all from one fast-population RTX 5090 instance. Scoring: `benchmarks/cuda/predicted_overlap_fit.py RUN_A RUN_B --fix FIX`, output `docs/cuda-results/predicted-overlap-fit-2026-10-20.json`. Labels follow plan section 0: **measured** on the recorded rows; fits are **derived**; explanations are **hypothesis**.

## 1. Scoring the registered predictions

All rows are level 3, `P0L` arm, parameters and slope fitted on run A and scored on run B unless stated. V1 is the registered open-loop model: a first-in first-out copy stream driven only by the trace's per-row demand and prefetch counts, with the per-copy costs, base compute time and row gap fitted on run A. No event time of run B enters the prediction.

| id | measured (C = 1.0 ms / 0.3 ms) | verdict |
|---|---|---|
| PO-1 (row agreement of predicted and measured overlap, r at least 0.85) | 0.9981 / 0.9965 | **held** |
| PO-2 (total overlap, predicted over measured, 0.85 to 1.05) | 1.0104 / 0.9510 | **held** |
| PO-3 (slope on predicted overlap within 25 % of the measured-overlap slope) | 0.0624 against 0.0634 (1.6 %) / 0.1122 against 0.1053 (6.5 %) | **held** |
| PO-4 (token busy total within 0.10 ms of the measured excess) | -0.026 ms (0.609 against 0.635) / +0.025 ms (0.773 against 0.748) | **held** |
| PO-5 (hidden-share error at most 0.05) | 0.0457 (0.468 against the level-4 measured 0.423) / 0.0067 (0.239 against 0.232) | **held**, narrowly at 1.0 ms |
| PO-6 (no more than 0.03 above the measured-overlap error 0.045 and 0.008) | +0.001 / -0.001 | **held** |
| PO-7 (level 6, level-6 parameters and the level-3 slope, within 0.15 ms) | -0.049 ms (0.612 against 0.661) / +0.026 ms (0.775 against 0.749) | **held** |
| PO-8 (closed-loop total overlap within 8 % of open loop) | ratio 1.0006 / 1.0793 | **held**, narrowly at 0.3 ms |
| PO-9 (measured-overlap slope on predicted overlap, token total within 0.15 ms) | -0.016 ms / -0.023 ms | **held** |

**9 held, 0 missed, 0 falsified**, at both layer times. The adoption rule (PO-4 and PO-5 hold, PO-1 not falsified) is met, with the qualifications of section 3.

## 2. What was measured

* **The queue model places the copies (measured).** Per-row predicted overlap correlates with the measured one at 0.998 and 0.997, far above the registered 0.85. The totals agree to 1 % at 1.0 ms and 5 % low at 0.3 ms. At 0.3 ms the shortfall is the stretch: the closed-loop model (V2), which lets a stretched kernel overlap more, raises the total by 7.9 % and lands within 2.6 % of the measured total (7.43 against 7.24 ms).
* **The slope transfers (derived).** Fitted on the predicted overlap it is 0.0624 and 0.1122; fitted on the measured overlap 0.0634 and 0.1053. Using either slope with the predicted overlap gives a token busy total within 0.026 ms of the measured one. At 0.3 ms the slope on predicted overlap is 6.5 % higher, which is what a total that is 5 % low should produce.
* **Predicting the overlap costs nothing at the token level (derived).** The hidden-share error is 0.0457 and 0.0067 with predicted overlap against 0.0450 and 0.0080 with measured. The predicted-overlap form is therefore no more optimistic than the measured-overlap form was, and it can be used for a plan whose events have not been recorded.
* **Out of sample (measured).** Level-6 cells, with their own fitted costs and the level-3 slope, are predicted to 0.049 and 0.026 ms.

## 3. What it means, and does not

* **The remaining 0.0457 at C = 1.0 ms is not the overlap.** The predicted and measured overlap are within 1 % there, and the hidden-share error differs by 0.0007. What is left (the model's hidden share is 0.468 against 0.423 measured at level 4) comes from the rest of the row model: the queue wait fitted on run A and the demand cost per miss. This result does not reduce it and does not claim to; it is still 0.0043 under the 0.05 limit, which is a narrow margin on one instance.
* **The agreement is partly by construction (hypothesis).** The harness's copy stream is a deterministic first-in first-out queue with nearly constant copy times (0.193 to 0.195 ms), so a FIFO model with fitted costs should place it well. An r of 0.998 is evidence that the model matches this instrument, not that it would match a real runtime with variable copy times and several streams.
* **Same limits as before.** One fast-population instance, one trace, 17 % resident, two layer times, a stand-in compute. Nothing here concerns another card, another prefetch family, another resident fraction or the real model's kernels. The mechanism (memory contention or copy-engine scheduling) is still unidentified. No interval is quoted: the scored pair of runs shares one instance, so replicate agreement is the honest scale.
* **V2 is a diagnostic here.** PO-8 held, so the open-loop V1 stays the registered form.

## 4. Decision

* The slowdown term is adopted in its **unconditional** form for the `P0L` row model on this card, this trace, 17 % resident, the fast population and the two layer times: the overlap is computed from the trace's counts and run-A-fitted costs, not from recorded events. The conditional-on-measured-overlap label is dropped for that scope only; the copy-count screen stays for everything else.
* Open and offline: nothing that moves a number. The 0.0457 left at 1.0 ms would need a better queue-wait or demand-cost term in the row model itself, which the row-timing and window-model records already tried without adoption.
* Open and needing a GPU and Hamed's go: why the slope doubles from 1.0 to 0.3 ms (a third C at 0.6 ms would give a point on the curve), another card, a native issue path, ROCm, instance population.
