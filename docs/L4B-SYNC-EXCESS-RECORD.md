# The budget-0 router-eval excess: highest window, fewest misses (record opened 2026-10-12, before the analysis)

Hamed's go: "go, reconcile the budget-0 router-eval excess" (2026-10-12). Offline: the saved raw rows of `benchmarks/results/l4b-raw*/`, no model, nothing in `src/` changes. Follows `docs/L4B-SYNC-RECONCILE-RECORD.md` (HANDOFF 18.142), which left this open.

## Baseline (measured, 18.142)
The router-eval window (SYNC_START to SYNC_END around the router `mx.eval`; GPU work queued since the previous eval) sums per token to: 34 GiB exact 63.1 ms, 28 GiB 65.1, 22 GiB 66.0, 46 GiB 59.5, and **34 GiB with decode miss budget 0: 66.1 ms with the fewest misses (24.2 a token against 43.0)**. Sum of per-layer medians: 64.3 (arm 1) against 60.7 (budget 0); wait-free sum of 10th percentiles 48.9 against 49.2. So the budget-0 excess sits in the upper part of the distribution (mean above median by 5.4 ms, against -1.2 in arm 1), not in the floor. Budget 0 also drops experts (their compute is skipped), changes the text (2,548 tokens against 2,452), and has the highest hidden read share (77 % of read busy time).

## Hypotheses
- H-read: reads in flight while the GPU computes slow the GPU (18.4: "reads slow the GPU part of a token ~25 ms"); budget 0 turns demand waits into overlapped prefetch reads, so more read time falls inside the router-eval window. Prediction: the read time overlapping the sync interval is higher in budget 0 than in arm 1, and across all arms and layer rows sync rises with it.
- H-text: the text, and so the positions and context lengths, differ. Prediction: sync does not depend on position at this scale (arms share 22-130-token prompts).
- H-compute: dropped experts change the GPU work in the window. Prediction: smaller, not larger, GPU work (a dropped expert is not computed), so this cannot explain an excess.

## Prediction (written before the analysis)
1. The mean read time overlapping the router-eval interval per token is higher in the budget-0 arm than in arm 1 by at least 5 ms (budget 0: ~75 % of the token's reads are prefetch and run during compute).
2. Pooled over all arms and layer rows, sync rises with the overlapping read time with slope 0.1 to 0.4 ms per ms; layer type absorbed by a per-layer constant.
3. With that regression, the predicted sync difference between budget 0 and arm 1 is at least 60 % of the measured 3.0 ms mean excess (and the residual excess, measured minus predicted, is under 1.2 ms a token).
4. The tail: rows above 3 ms are a larger share in budget 0 than in arm 1.
5. Position has no effect: the slope of sync on position within a turn is under 0.005 ms per position.

## Falsifier
Overlap not higher in budget 0 (prediction 1), or slope outside 0.05-0.6 (prediction 2), or the predicted difference under 30 % of the excess: H-read is not the explanation and the excess remains unexplained (the text or the compute change is then the candidate). Position slope above 0.02 ms per position: H-text.

## Limits
Observational and pooled across arms, with confounds (misses, text, budget); the regression gives association; a read-throttled arm (`CACHALOT_READ_THROTTLE_GBPS`) would test the cause and is not run here.

## Result (measured offline, 2026-10-12; `benchmarks/sync_excess.py` and the analyses quoted below)

**Overlap.** Read time overlapping the router-eval interval, per token: 28.5 ms (34 GiB exact), **34.5 ms (budget 0)**, 32.0 (28 GiB), 34.8 (22 GiB), 23.3 (46 GiB on), 0 for the prefetch-off arms. Pooled slope of the layer-window length on it: 0.119 ms per ms (layer fixed effects; residual sd 0.33 ms). That predicts +0.70 ms of the measured +3.01 ms budget-0 excess (23 %); residual +2.31.

**Distribution.** Budget 0 minus exact, per layer: the mean rises by 0.02 to 0.19 ms in every one of the 40 layers (sum +3.01; about +0.075 a layer, heavy layers 1, 2, 8, 14 a little more) while the median falls (sum of medians -3.55 ms): the distribution is narrower around a lower middle and right-skewed in every layer. Rows above 3 ms are 1.84 % of rows (0.74 a token, mean 3.4 ms, maximum 5.5, spread over turns and mostly on the heavy attention layers 14, 2, 1, 20, 8 whose means are already 1.6-2.1 ms) against 0.09 % in the exact arm (89 rows, a few outliers up to 39 ms): a broad shift of the upper half, not rare spikes.

**Per turn.** The excess is positive in 19 of 24 turns (mean +1.93, sd 3.45); on the 18 turns with the same token count in both arms it is +2.04: the different text is not the cause. Position slope of the window within a turn: -0.0014 to +0.0013 ms per position in all arms.

**Outside the window.** Server decode time minus router-eval window minus blocked time: 29.8 ms (budget 0) against 29.3 (exact): the rest of the token is the same, so nothing the drop path adds lands outside the window.

**Read activity across arms.** Per token, over the six exact arms the window length follows the read activity: sync = 54.1 + 0.088 x read busy (r = 0.47; within the exact 34 GiB arm the slope is 0.097, r = 0.48), residuals at most about -3.3 and +0.5 ms. **Budget 0 does not follow it: +6.1 ms above that line, and its within-arm slope is -0.012 (r = -0.04).** Adding blocked time to the regression leaves +3.2 ms.

## Against the prediction

1. **Held.** Overlap higher in budget 0 by 6.0 ms (predicted at least 5).
2. **Held.** Slope 0.119 (0.1-0.4; the falsifier range 0.05-0.6).
3. **Missed; the falsifier fired.** The overlap explains 23 % of the excess (predicted at least 60 %, falsified below 30 %) and the residual is +2.31 ms (predicted under 1.2): H-read is not the explanation of the budget-0 excess.
4. **Held in the letter, wrong in the picture.** The share of rows above 3 ms is larger in budget 0 (1.84 % against 0.09 %), but it is a diffuse shift of the upper half, not spikes.
5. **Held.** Position slope under 0.0015 in magnitude (predicted under 0.005); H-text (the falsifier position slope above 0.02) is not supported either; same-length turns show the same excess.

## Explanation (hypothesis, untested)

What is established is a negative: the excess is not read overlap, not the text, not the position, not time outside the window, not rare spikes. The window is higher by about 0.075 ms in every layer, with a lower median and a longer upper half, and it does not vary with read activity inside the budget-0 arm. The one thing budget 0 adds on the layer path is the drop branch (`moe_layer_metal.py`, after `get_many`): `any(e is None ...)`, a rebuilt `weights = mx.array(...)`, `router_weights = weights.tolist()`, and a variable number of experts (0 to 6) into the fused expert kernels, whose launch shapes then vary token to token. A variable expert count can change the queued GPU work or the encoding cost of the next router eval; the tracer does not record how many experts were kept (GET_END's `b` is the requested count), so this cannot be checked from the saved rows. It is a hypothesis.

## Decision

Nothing in `src/` or any default changes. The excess is 3 ms of 112 ms (2.7 %) of a token; the router-eval window is mostly GPU compute (18.142), so it is not a lever. The candidate test is cheap and not run: make GET_END record the number of experts kept (one tracer field, off by default) and record a budget-0 arm, then compare the window of layers by kept count. Ledger row DS-SYNC-EXCESS.
