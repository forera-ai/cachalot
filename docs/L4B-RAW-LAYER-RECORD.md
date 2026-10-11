# L4(b) layer level: raw per-layer rows of the DeepSeek decode token (record opened 2026-10-11, before the recorder exists and before any model run)

Charter references: `docs/RESEARCH-DIRECTION.md` section 4.3 (the critical-path spec), 4.6 L4(b)(c)(d)(f), section 7.2. Follows `docs/L4B-OVERLAP-FIT-RECORD.md`, which closed the token-level fit and named this recording as the open item. Hamed's go: "go, record the raw per-layer rows" (2026-10-11), after the estimate "about an hour". Estimate for the run itself: model load about 3 minutes, 4 warm turns plus 24 recorded turns of 128 tokens at about 20 s each, about 15 minutes in all; analysis is offline afterwards.

## Baseline

The saved L2 rows are per-token sums. The tracer's raw columns (timestamp, code, a, b: router sync, `get_many` blocked, read start and end per layer and expert) were never saved, so nothing says which layers wait for experts, which hide reads, or how many reads are in flight when a layer starts. 0.63.4's pooled means: sync 62.4 ms, blocked 62.8, read busy 104.4, hidden 43.1, 66.3 reads, 44.2 misses a token, at 34 GiB.

## Method

New `benchmarks/token_trace_raw.py`: the `Engine.chat` path as in `token_trace_price.py`, tracer recording every decode token (`every=1`), mixed workload, 24 turns (the 12 mixed prompts twice), 128 new tokens, greedy, 34 GiB budget under `guarded_run.sh` (the guard refused 38 on 2026-10-11). The tracer is drained after each turn and the raw columns saved (`np.savez_compressed`, one file per turn) with a manifest of the run. Four unrecorded base turns first as warm-up. Drop count is read from every drain. Analysis (separate, offline, `benchmarks/layer_rows.py`) turns the columns into rows of one decode layer: router sync time, blocked time in `get_many`, reads submitted during the layer, reads completed, and the layer's reads that finished before the layer's `get_many` began.

## Prediction (written before the run)

1. No events dropped (about 300 events a token, a 65,536-event buffer per thread, drained every turn of 128 tokens).
2. Blocked time is concentrated: the 10 layers with the most blocked time in a token carry at least 55 % of that token's blocked time (pooled over tokens, median), because a layer waits only when it has misses and misses cluster.
3. The hidden read share differs by depth: reads that overlap the following layer's sync complete before the next `get_many`, so the median per-layer hidden share of reads submitted by a layer is higher for the first half of the layers (3-22) than for the second half, by at least 5 points.
4. Pooled over the run, the means reproduce 0.63.4 within 10 %: sync 62 ms, blocked 63, read busy 104, 66 reads a token (a replication; the machine is the same budget, 34 GiB).

## Falsifier

If prediction 2 fails (top-10 share under 40 %), blocked time is spread evenly over layers and there is no layer-level lever to look for. If prediction 3 fails with the difference inside +-2 points, depth does not set what hides. If prediction 4 misses by more than 20 %, the run is a different regime than 0.63.4 and its layer rows are not compared with the earlier token rows.

## Limits (written now)

One machine, one budget, one workload family (short prompts, not agent contexts), greedy, 24 turns, the tracer's own cost (+0.55 ms a token, 0.4 %) inside the rows. Layers share cache state and drift across a turn, so intervals are by turn.

## Result (measured, 2026-10-11, runtime 0.64.1)

`benchmarks/token_trace_raw.py`, 24 recorded turns (after 4 unrecorded), 2,452 decode tokens (2,428 with a complete layer sequence), 34 GiB budget under `guarded_run.sh`, 301.7 events a token, 0 events dropped. Misses ran 31-76 a token (hit 68-87 %), decode 4.8-7.2 tok/s. Raw columns kept locally in `benchmarks/results/l4b-raw/` (ignored by git, 3.7 MB); analysis `benchmarks/layer_rows.py`.

- Per-token means over layer rows: router sync 63.1 ms, blocked 59.2, read busy 101.8.
- The ten layers with the most blocked time carry a median 74.2 % of a token's blocked time.
- Mean blocked time per layer is flat apart from the first layers (layer 0: 3.8 ms, layer 1: 2.7, layers 2-39: 0.8-2.5); so the concentration is in which layers miss on a given token, not in a fixed set of layers.
- Share of read busy time hidden: layers 0-19 41.1 %, layers 20-39 46.0 %.

## Against the prediction

1. **Held.** No events dropped.
2. **Held.** Median top-10 share 74.2 % (predicted at least 55 %); across tokens the layers differ, across layers on average they do not (flat means).
3. **Falsified in direction.** Predicted the first half hides more by 5 points; the second half hides 4.9 points more (41.1 against 46.0 %). Depth does set the hidden share a little, the other way: later layers' reads are more often finished by the time the layer's `get_many` starts or are overlapped by the next layer's sync.
4. **Held for the three means** (sync 63.1 against 62.4, blocked 59.2 against 62.8, busy 101.8 against 104.4, all within 6 %); reads a token was not recomputed from the layer rows, so that part of the replication was not checked.

## Explanation (hypothesis)

Blocked time is a property of the token (which layers miss), so no fixed layers are the problem: a lever that prefetches or pins specific layers has little to work with, except the first two layers, which carry about 2x the mean blocked time (3.8 and 2.7 ms; they have no earlier layer to predict them, since the prediction works one layer ahead). The depth effect on the hidden share is small (5 points) and opposite to the guess; the cause is not isolated.

## Decision

Nothing in `src/` or `whatif.py` changes. Candidate leads, none built or priced: (a) the first layers' misses (layer 0 and 1 together are about 6.5 ms of 59 ms blocked per token, 11 %), where no prediction can run ahead; (b) per-token miss clustering as the input for L4(c) and (d). The cliff (c) and prefetch precision (d) need other arms (budgets, prefetch off) and are not answered by this single-regime recording.
