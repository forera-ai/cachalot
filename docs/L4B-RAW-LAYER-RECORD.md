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

# Arm 2: the same recording with decode prefetch off (`CACHALOT_PREDICT_TOPK=0`), opened 2026-10-11 before the run

Hamed's go: "go, record prediction off" (2026-10-11), after the estimate "about 20 minutes each plus load". Same recorder, workload, 24 turns, 128 tokens, 34 GiB, greedy, guard; only the environment variable differs. Output `benchmarks/results/l4b-raw-nopred/`. Compared against arm 1 (above, prediction on, shipped).

## Baseline (arm 1, measured)
Per token: sync 63.1 ms, blocked 59.2, read busy 101.8, hidden share 43.5 % overall (41.1 early, 46.0 late), 302 events, 31-76 misses a token, top-10 blocked share 74.2 %. Earlier ledger rows: only about 35-40 % of predicted loads are used (DS-PRED-USB), prediction nets about 1 ms on the internal drive (DS-PRED-AB).

## Prediction (written before the run)
1. 0 events dropped; events a token fall to 220-270 (no speculative reads).
2. Read busy a token falls to 55-85 ms (demand reads only).
3. The hidden share falls to 5-30 % overall (what hid was mostly prefetch), in both halves.
4. Blocked time a token stays within +-8 % of arm 1 (54-64 ms): fewer reads, but nothing hides them.
5. The top-10 layers' share of blocked time stays at least 60 %; layers 0 and 1 stay about twice the mean (prefetch cannot cover them in either arm).
6. Wall time a decode token (decode seconds over tokens) within +-6 % of arm 1.

## Falsifier
Blocked time more than 15 % lower with prediction off means prefetch costs more than it hides in this regime (its reads compete with demand reads); more than 15 % higher means it does real work here that the 1 ms ledger figure hid. Layers 0 and 1 not elevated would mean the first-layer lead was prediction's doing.

## Limits
Same as arm 1; also two arms in separate processes on one day, so a 5-10 % process drift (DS-DRIFT-AGE) is inside any 6 % comparison. Not alternated.

## Arm 2 result (measured, 2026-10-11, runtime 0.64.2)

Same recorder, 24 recorded turns, 2,452 decode tokens (2,428 with a complete layer sequence), 34 GiB, 0 events dropped, `CACHALOT_PREDICT_TOPK=0`. Misses a token 43.0 in both arms. Raw columns local in `benchmarks/results/l4b-raw-nopred/`.

| per decode token | prediction on (arm 1) | prediction off (arm 2) |
|---|---|---|
| events | 301.7 | 247.0 |
| router sync (ms) | 63.1 | 59.2 |
| blocked on experts (ms) | 59.2 | 84.2 (+42 %) |
| read busy (ms) | 101.8 | 82.5 |
| hidden share of read busy | 43.5 % | 0.0 % |
| decode ms a token (server timing) | 151.6 | 167.3 (+10.4 %) |
| top-10 layers' share of blocked time (median) | 74.2 % | 55.2 % |
| mean blocked, layers 0 / 1 / 2-39 (ms) | 3.79 / 2.69 / 1.39 | 3.08 / 3.17 / 2.05 |

All 24 turns were slower with prediction off (turn ratios 1.03-1.29, mean 1.11; first six turns 1.105, last six 1.137), so a slow drift across the second process does not explain it.

## Against the prediction

1. **Held.** 0 dropped, 247 events a token (predicted 220-270).
2. **Held.** Read busy 82.5 ms (predicted 55-85).
3. **Missed, and the prediction was badly framed.** Hidden share is 0.0 %, not 5-30 %: with no speculative reads every read is a demand read inside a `get_many` wait, so nothing hides by construction.
4. **Falsified, falsifier fired.** Blocked time is 42 % higher with prediction off (predicted within +-8 %; the falsifier was more than 15 % higher): in this regime prefetch does real work.
5. **Partly held.** The top-10 share fell to 55.2 % (predicted at least 60 %: missed narrowly). Layers 0 and 1 are 1.5x the mean of layers 2-39 (2.05 ms) rather than 2.3x: their wait is nearly the same with prediction off (3.1 against 3.5 ms mean on), so prediction does not reach them, and it removes about 0.66 ms a layer elsewhere.
6. **Falsified.** Decode time a token is 10.4 % higher (predicted within +-6 %).

## Explanation (hypothesis)

Prefetch takes about 25 ms off the wait for experts in a token (59.2 against 84.2) because its reads, one layer ahead, are in flight while the previous layer computes: 43.5 % of read time hides behind compute in arm 1, and the spare 20 ms of read busy time (101.8 against 82.5) is the cost of the wasted speculative loads. Net, prediction saves about 16 ms a token (10 %) here. This contradicts the earlier ledger figure for the internal drive (prediction nets about 1 ms, DS-PRED-AB, at 48 GiB and 23.5 misses a token): the difference is the regime, 43 misses a token at 34 GiB against 23.5 at 48 GiB, so the more misses the more prefetch is worth. That reading is a hypothesis; the two budgets were not compared in one session.

## Limits

Two arms in two separate processes, not alternated (a process drift of 5-10 % is possible; the consistent per-turn sign argues against it for a +10 % effect, not for its exact size). One machine, one budget, short mixed prompts. Server timing and the tracer's own cost are inside both arms.

## Decision

No default changes: prediction stays on. Recorded: DS-PRED-LAYER in the ledger. Open: prefetch value as a function of misses a token (the same recording at 48 GiB or with a larger budget) would turn the hypothesis into a curve, and the layers 0 and 1 lead now has a cleaner statement (6.2-6.7 ms of blocked time a token that prediction cannot reach, about 7-11 % of it).
