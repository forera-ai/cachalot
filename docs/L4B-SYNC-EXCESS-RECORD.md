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

# Addendum: the kept-expert count test, opened 2026-10-12 before the run

Hamed's go: "go, record the kept-expert count test" (2026-10-12). Change in `src/` (tracer-only, off by default, no numerics): `moe_layer_metal.py` now writes the number of experts kept (non-None after `get_many`) into GET_END's `b`; GET_START keeps the requested count. With exact decode kept equals requested, so every earlier trace stays comparable; traces from before 0.64.13 carry the requested count in both. `layer_rows.py` rows now have `requested` and `kept`. Estimate: a 34 GiB budget-0 recording of the 24 mixed turns of arm 3 (same recorder, `CACHALOT_DECODE_MISS_BUDGET=0`), about 20 minutes plus a model load of 3, under `guarded_run.sh`. Output `benchmarks/results/l4b-raw-budget0-kept/`. Compare with the saved exact arm 1 and the saved budget-0 arm 3 (same prompts, same text as arm 3: greedy, same budget).

## Baseline
Budget 0's window is +3.01 ms a token above the exact arm (+0.075 ms in every layer, medians lower); about 19.5 experts dropped a token in earlier measurements (18.59), so roughly 0.5 of 6 per layer. The GPU work queued for a layer's router eval is the previous layer's MoE compute (with the kept experts) plus this layer's attention.

## Hypotheses
- H-shape: the variable number of kept experts (0 to 6) into the fused kernels costs more than the work it saves (kernel grid or graph shape variation, encode cost). Prediction: windows following a layer with fewer than 6 kept are higher than windows following a layer with all 6 kept, despite less compute; the budget-0 excess concentrates in windows after layers with kept < 6; windows after full layers match the exact arm.
- H-uniform (not shape): the excess is the same after full and after reduced layers (a property of the budget-0 run, not of the count).
- H-compute: windows after reduced layers are shorter (less work), so the mean excess must come from elsewhere.

## Prediction (written before the run)
1. Share of layers with kept < 6 in the budget-0 recording: 35-55 % (about 19.5 drops a token over 40 layers: 20-30 layers have a miss... the share of layers with at least one drop is the quantity; the 26.4 % in-flight figure of 18.76 is a different quantity).
2. Layer-fixed-effect difference in the next layer's window, after a layer with kept < 6 minus after kept = 6: **+0.15 ms** (planning interval +0.05 to +0.40).
3. Windows after full layers (kept = 6): excess over the exact arm at most 0.04 ms a layer (under half of the average 0.075); windows after reduced layers: at least 0.12 ms a layer.
4. The tally: sum over layers of the excess attributable to reduced layers is at least 60 % of the +3.0 ms.
5. Text identical to arm 3 (greedy, same budget): at least 22 of 24 turns; 0 events dropped; token total within 3 % of arm 3's 2,548.
Registered reading: H-shape if predictions 2-4 hold; H-uniform if the difference in prediction 2 is under 0.05 ms and prediction 3's two numbers are within 0.03 of each other; H-compute if the difference is negative.

## Falsifier
Difference under 0.05 ms or negative, or the full-layer excess above 0.06 ms: H-shape is not the explanation, and the +3 ms stays unexplained (two hypotheses down).

## Limits
Observational within one recording (reduced layers are not randomly assigned: a layer drops experts when it misses, and misses correlate with layer type and token); the contrast is within a recording and layer type is controlled by layer fixed effects; the comparison with the exact arm crosses processes (drift 1-2 %); the tracer's own +0.55 ms is inside; the change to `moe_layer_metal.py` adds one generator expression per layer per traced token.

## Addendum result (measured, 2026-10-12, runtime 0.64.13 tracer)

Budget 0, 34 GiB, 24 mixed turns, prefetch on, recorded as `benchmarks/results/l4b-raw-budget0-kept/` (local), 0 events dropped, 2,533 decode tokens (arm 3: 2,548; the token counts differ in 4 of 24 turns although greedy: with budget 0 what is dropped depends on what is resident or in flight at that moment, so the text is not reproducible run to run), requested 6.00 and **kept 5.31** experts a layer, 27.8 experts dropped a token, 44.3 % of layers with fewer than 6 kept (kept distribution 0: 1.1 %, 1: 0.7, 2: 0.9, 3: 2.5, 4: 9.0, 5: 30.2, 6: 55.7).

| | exact (arm 1) | budget 0 (arm 3) | budget 0 again (this run) |
|---|---|---|---|
| decode (server timing) | 151.6 ms | 112.2 | **105.6** |
| misses a token | 43.0 | 24.2 | 24.5 |
| router-eval window (mean) | 63.1 | 66.1 | **56.8** |
| sum of per-layer medians | 64.3 | 60.7 | 54.1 |
| wait-free (sum of 10th percentiles) | 48.9 | 49.2 | 47.8 |
| blocked on experts | 59.2 | 16.3 | 19.8 |
| read busy | 101.8 | 67.1 | 68.0 |
| hidden share | 43.3 % | 77.1 | 72.3 |

**The window of this run is 6.3 ms below the exact arm, not 3.0 above it.** By the previous layer's kept count the window minus the exact arm's layer mean is -0.126 to -0.161 ms for every count from 0 to 6 (1,141 rows at kept 0, 54,360 at kept 6); the difference between windows after reduced and after full layers is **+0.004 ms** (layer fixed effects). Tally of the excess over the exact arm: full -3.48, reduced -2.76, first layer -0.08 ms a token.

## Against the prediction

1. Share of layers with kept < 6: 44.3 % (35-55): held.
2. The reduced-minus-full difference: +0.004 ms against the predicted +0.15 (falsifier: under 0.05): **falsified**; H-shape (the variable expert count) is not the explanation. No difference at all by kept count (including 0 experts, where `mx.zeros` replaces the expert kernels).
3. The full-layer excess at most 0.04 and the reduced at least 0.12: both are about -0.16 ms; the two numbers are within 0.03 of each other, as the registered reading for H-uniform requires, but the excess itself did not reproduce (it is negative).
4. The attribution of 60 % of +3.0 ms to reduced layers: not testable, there is no excess to attribute.
5. Text identical in at least 22 of 24 turns: 20 (missed); 0 dropped events held; tokens within 3 % of arm 3 (0.6 %) held.

## Explanation

The +3.01 ms was not a property of budget 0. Two recordings of the same configuration (budget 0, 34 GiB, prefetch on, the same prompts) differ by 9.3 ms in the router-eval window mean (66.1 against 56.8), 6.6 ms in decode (112.2 against 105.6, 5.9 %) and 6.6 ms in the sum of medians, while the wait-free floor (sum of 10th percentiles: 49.2 and 47.8) and the miss count (24.2 and 24.5) agree. The upper part of the window distribution moves between processes by amounts larger than the budget-0 effect it was credited with. The cause of that movement (thermal or clock state, background load: the guard's samples list VMware as the largest other process in this run's first seconds, 1.6 GiB resident, after Hamed had quit it earlier; other tools) is not isolated.

## Consequences for earlier records (corrections)

- The "budget-0 router-eval excess" (18.142, 18.146) is withdrawn as a budget-0 property: it is the between-run variation of this metric, here at least 9 ms.
- The drift bound used throughout this series (1-2 % between processes, from one replicate pair: decode +1.3 %) is too small: the same configuration differs by 5.9 % in decode between these two budget-0 runs. Cross-process decode gaps below about 6 % (arm 7's prefetch gap 3.3 %; the 28-34 GiB step of 9.4 % and the 22-28 GiB step of 7.4 %) are not resolved by a single pair of processes; the budget-0 saving (-26 % to -30 %), the prefetch-off penalty at 34 GiB (+10.4 %), the miss counts and the wait-free floor are robust. The alternated one-process design (arm 8b) is the trustworthy prefetch number (+5.5 ms, +0.7 to +9.1).

## Decision

No change in `src/` beyond the tracer field. Budget 0 stays as shipped. The excess question is closed: no mechanism to find. Ledger row DS-SYNC-KEPT, DS-SYNC-EXCESS marked superseded.
