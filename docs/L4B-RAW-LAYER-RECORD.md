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

# Arm 3: the same recording with the shipped decode miss budget 0 (`CACHALOT_DECODE_MISS_BUDGET=0`), opened 2026-10-11 before the run

Hamed's go: "go, record the shipped decode miss budget 0" (2026-10-11), estimate "about 20 minutes plus load". Same recorder and workload (24 turns, 128 tokens, 34 GiB, greedy, guard), prefetch on (shipped), only `CACHALOT_DECODE_MISS_BUDGET=0` added, as `serve.sh` sets it. Output `benchmarks/results/l4b-raw-budget0/`. **Output-changing:** the generated text differs from arms 1 and 2 (experts missing in a decode layer are dropped, not read), so token counts per turn differ; per-token quantities are compared, not texts. This is a speed and layer-structure recording, not a quality test (quality: HANDOFF 18.59-18.74).

## Baseline
Arm 1 (prefetch on, exact): decode 151.6 ms a token, blocked 59.2, read busy 101.8, hidden share 43.5 %, top-10 share 74.2 %, layers 0/1 at 3.79 / 2.69 ms against 1.39 for layers 2-39. Ledger: budget 0 takes the decode step -22 to -27 % (DS-BUDGET-CURVE, 22k context; 19.5 experts dropped a token), and 26.4 % of decode layers route to an in-flight predicted load (18.76).

## Prediction (written before the run)
1. 0 events dropped; 250-300 events a token (prefetch reads remain, demand reads vanish).
2. Decode time a token 108-125 ms (-18 % to -29 % against arm 1).
3. Blocked time a token 5-25 ms (the layers that route to an in-flight load still wait; demand misses no longer do).
4. Hidden share of read busy 70-95 % (the remaining reads are prefetch reads, outside the blocked windows).
5. The top-10 layers' share of blocked time at least 60 %; layers 0 and 1 no longer elevated (below 1.5x the mean of layers 2-39), since a first-layer miss is dropped like any other.
6. Misses a token as the store counts them (`cache_misses`) fall from 43 to 15-30.

## Falsifier
Decode reduction under 10 % or over 35 % means the ledger's -22 to -27 % does not transfer to this workload and budget; blocked time above 30 ms means demand waits survive budget 0 (an in-flight load wait is larger than priced). Layers 0/1 still above 2x the rest would put the first-layer wait outside the drop rule.

## Limits
As arm 1 and 2 (separate processes, not alternated, one budget, short mixed prompts); output differs, so a turn's tokens and its prompt-dependent miss rate are not the same population as in arm 1 (shorter or longer replies; the per-token means weight turns by token count).

## Arm 3, first attempt: void as a budget arm, kept as a replicate of arm 1 (2026-10-11)

The run above did not apply the budget. `CACHALOT_DECODE_MISS_BUDGET` is read by `cli.py` (`model.set_decode_miss_budget`), not by the model or `Engine`, and `token_trace_raw.py` built the model directly, so the environment variable changed nothing: same 2,452 tokens, 43.0 misses a token, 301.7 events a token as arm 1. My mistake; the recorder is fixed to apply the knob (printing `decode miss budget: N` at start) and the arm is rerun below. The void run is kept in `benchmarks/results/l4b-raw-replicate/` because it is an unplanned replicate of arm 1 in a separate process: decode 153.6 against 151.6 ms a token (+1.3 %), blocked 63.4 against 59.2, busy 105.6 against 101.8, hidden share 41.3 against 43.3 %, top-10 share 73.8 against 74.2 %, layers 0/1 3.96 / 2.78 against 3.79 / 2.69. So two processes on the same configuration differ by about 1-2 % in decode and 4-7 % in blocked time, which bounds the process drift behind arm 2's +10.4 % decode and +42 % blocked (not alternated, but about eight times larger than the replicate's difference).

## Arm 3 result (measured, 2026-10-11, runtime 0.64.3; recorder fixed, budget applied)

`CACHALOT_DECODE_MISS_BUDGET=0`, prefetch on, 24 recorded turns, 34 GiB, 0 events dropped. The text differs from arms 1 and 2 (dropped experts), so there are 2,548 decode tokens (not 2,452) and turns end at different lengths; per-token quantities are compared. Raw columns local in `benchmarks/results/l4b-raw-budget0/`.

| per decode token | arm 1: exact, prefetch on | arm 3: budget 0, prefetch on |
|---|---|---|
| events | 301.7 | 263.8 |
| decode (server timing) | 151.6 ms | 112.2 ms (-26.0 %; all 24 turns faster) |
| router sync | 63.1 ms | 66.1 ms |
| blocked on experts | 59.2 ms | 16.3 ms (-72 %) |
| read busy | 101.8 ms | 67.1 ms |
| hidden share of read busy | 43.3 % | 77.1 % (first half 79.1, second half 74.9) |
| top-10 layers' share of blocked | 74.2 % | 86.5 % |
| misses a token (store counter) | 43.0 | 24.2 |
| mean blocked, layers 0 / 1 / 2-39 | 3.79 / 2.69 / 1.39 ms | 0.04 / 0.06 / 0.43 ms |

## Against the prediction

All six held: (1) 0 dropped, 263.8 events (250-300); (2) 112.2 ms (108-125), -26.0 % (predicted -18 to -29); (3) blocked 16.3 ms (5-25); (4) hidden share 77.1 % (70-95); (5) top-10 share 86.5 % (at least 60), layers 0 and 1 not elevated (0.12x the mean of layers 2-39, below the predicted 1.5x ceiling); (6) 24.2 misses a token (15-30). The falsifier did not fire. The ledger's -22 to -27 % (DS-BUDGET-CURVE, 22k Hermes context) transfers to this short-prompt, 34 GiB workload.

## Explanation (hypothesis)

Budget 0 turns the layers' demand misses into drops, so what is left of the wait (16 ms, 14 % of the token) is layers whose expert is an in-flight prefetch load, not a demand miss; layers 0 and 1, whose misses cannot be prefetched, nearly stop waiting (0.04 and 0.06 ms) because every first-layer miss is simply dropped. [Corrected 2026-10-12, docs/L4B-SYNC-RECONCILE-RECORD.md: the "router sync" window is GPU compute queued before the router eval, not a synchronization cost.] The token is then 66 ms of router sync, 16 of waiting and about 30 of other work: it is close to its all-resident floor (79-85 ms measured with all experts resident; here 112 with 24 misses a token still read as prefetch). The price is output change: the quality evidence for budget 0 is HANDOFF 18.59-18.74 (C# worse in one of three tests, neutral elsewhere), not this recording.

## Limits

Output-changing arm: texts, lengths and therefore the prompt-dependent miss mix differ from arm 1 (the means weight turns by token count); not alternated with arm 1. The void replicate bounds process drift at about 1-2 % in decode, far below 26 %.

## Decision

No default changes (budget 0 is already `serve.sh`'s default; arms 1 and 2 are the exact path). Ledger row DS-BUDGET0-LAYER. What is left for the speed story at this budget is router sync (66 ms of 112) and the remaining 16 ms of waiting behind in-flight loads.

# Arm 4: the same recording at a smaller expert budget, 28 GiB (the cliff, L4(c)), opened 2026-10-11 before the run

Hamed's go: "go, record the memory cliff" (2026-10-11), estimate about 20 minutes plus load (more misses make decode slower, so perhaps 25). Exact path (no miss budget), prefetch on, 24 turns, 128 tokens, greedy, guard, only `--budget-gib 28` differs from arm 1 (34 GiB). Output `benchmarks/results/l4b-raw-28gib/`. The text is identical to arm 1 (exact decode), so token counts match and per-turn pairs are valid. Whether a cliff exists between 28 and 34 GiB is the question; one extra budget gives two points, so it can show a bend only against the ledger's curve (DS-BUDGET-CURVE: 36 / 44 / 48 GiB gave 31.8 / 27.1 / 25.3 misses a token on a 22k Hermes context; this short-prompt workload read 43.0 at 34 GiB).

## Baseline
Arm 1 (34 GiB): decode 151.6 ms a token, 43.0 misses a token, blocked 59.2, read busy 101.8, hidden share 43.3 %, top-10 share 74.2 %, layers 0/1 3.79 / 2.69 ms against 1.39. Earlier wall-on-reads line from the L2 rows: wall = 65.6 + 1.33 ms x reads.

## Prediction (written before the run)
1. 0 events dropped; misses a token 55-75 (+28 % to +75 % against 43.0): a smaller cache loses more of the working set than the ledger's gentle 36-48 GiB slope suggests, but not a collapse.
2. Decode 175-215 ms a token (+15 % to +42 %).
3. Blocked on experts 80-105 ms; read busy 125-165 ms; router sync stays 62-66 ms.
4. Hidden share 35-50 % (prefetch keeps hiding what it hid; its hit rate falls a little).
5. Top-10 layers' share of blocked at least 65 %; layers 0 and 1 stay at least 1.8x the mean of layers 2-39.
6. Pairwise by turn, every one of the 24 turns slower than in arm 1.

## Falsifier
Misses above 86 a token (double) or decode above 240 ms: a cliff (a region, not a slope) lies between 28 and 34 GiB. Misses below 50 or decode below 160 ms: no bend in this range, the slope is the ledger's. Memory pressure or swap growth during the run is recorded from the guard's samples.

## Limits
Two budgets only; one workload (short cold prompts, which the 34 GiB hit rate of 82 % reflects, unlike a 22k agent context); not alternated; drift bound 1-2 % (replicate). A cliff shape cannot be established from two points; this arm can only falsify a smooth slope or show a bend.

## Arm 4 result (measured, 2026-10-11, runtime 0.64.4)

Exact decode, prefetch on, 28 GiB, 24 recorded turns, 2,452 tokens (identical text to arm 1), 0 events dropped. Guard samples: pressure 1 throughout, swap flat at 2,527 MB. Raw columns local in `benchmarks/results/l4b-raw-28gib/`.

| per decode token | 34 GiB (arm 1) | 28 GiB (arm 4) |
|---|---|---|
| events | 301.7 | 324.3 |
| decode (server timing) | 151.6 ms | 165.8 ms (+9.4 %; 24 of 24 turns slower, ratios 1.01-1.20, mean 1.087) |
| misses a token | 43.0 | 51.0 (+18.6 %) |
| router sync | 63.1 ms | 65.1 ms |
| blocked on experts | 59.2 ms | 72.7 ms (+23 %) |
| read busy | 101.8 ms | 119.3 ms |
| hidden share of read busy | 43.3 % | 40.2 % |
| top-10 layers' share of blocked | 74.2 % | 68.2 % |
| mean blocked, layers 0 / 1 / 2-39 | 3.79 / 2.69 / 1.39 ms | 4.61 / 3.16 / 1.71 ms |

## Against the prediction

1. **Partly held.** 0 dropped; misses 51.0 against the predicted 55-75 (below the range, +18.6 % against +28 to +75 %).
2. **Missed low.** Decode 165.8 against 175-215 ms (+9.4 %, predicted +15 to +42 %).
3. **Missed low** on blocked (72.7 against 80-105) and read busy (119.3 against 125-165); router sync held (65.1, predicted 62-66).
4. **Held.** Hidden share 40.2 % (35-50).
5. **Held.** Top-10 share 68.2 % (at least 65), layers 0 and 1 at 2.27x the mean of layers 2-39 (at least 1.8).
6. **Held.** All 24 turns slower.

The falsifier did not fire in either direction: misses are not above 86 and decode not above 240 (no cliff in this range), and misses are not below 50 and decode not below 160 (it is not flat either; 51.0 and 165.8 are close to the lower thresholds).

## Explanation (hypothesis)

Between 28 and 34 GiB the cost is a slope, not a cliff: each GiB removed adds about 1.3 misses a token (8.0 over 6 GiB) and about 2.4 ms a token (14.2 ms over 6 GiB, about 1.8 ms a miss), steeper than the ledger's 0.59 misses per GiB between 36 and 48 GiB (a 22k Hermes context, a different workload) as expected below the working set, but gentler than I predicted. The structure of the wait does not change: the hidden share, the top-10 concentration and the layer 0 and 1 excess all stay close to their 34 GiB values. A cliff, if there is one, lies below 28 GiB or in the pressure/paging region above 48 GiB (DS-GPU-CLIFF), not here.

## Limits

Two budgets, one workload, not alternated (the replicate bounds drift at 1-2 %, much less than the 9.4 % here). Two points cannot show a bend; they show that the slope between 28 and 34 GiB is smooth: about 1.3 misses and 2.4 ms a token per GiB.

## Decision

No default changes. Ledger row DS-BUDGET-SLOPE-28-34. L4(c) as a region remains open below 28 GiB (a third point, for example 22 GiB, would show a bend if one exists; it needs the guard to accept it and Hamed's go).

# Arm 5: the same recording at 22 GiB (a bend below 28 GiB, L4(c)), opened 2026-10-12 before the run

Hamed's go: "go, record 22 GiB" (2026-10-12), estimate about 30 minutes plus load. Exact decode, prefetch on, 24 turns, 128 tokens, greedy, guard, only `--budget-gib 22` differs from arms 1 and 4. Output `benchmarks/results/l4b-raw-22gib/`. Text identical to arms 1 and 4.

## Baseline
34 GiB: 43.0 misses, 151.6 ms. 28 GiB: 51.0 misses, 165.8 ms, blocked 72.7, busy 119.3, hidden share 40.2 %, top-10 share 68.2 %, layers 0/1 2.27x. Slope between them: 1.33 misses and 2.37 ms a token per GiB.

## Prediction (written before the run), from the measured slope, not from ideas
Straight line from 34 and 28 GiB extended 6 GiB: misses 51.0 + 8.0 = 59.0; decode 165.8 + 14.2 = 180.0 ms; blocked 72.7 + 13.5 = 86; busy 119.3 + 17 = 136.
1. 0 events dropped; misses 57-68 (the line plus a small convexity allowance).
2. Decode 176-195 ms.
3. Blocked 80-92 ms; read busy 130-142 ms; router sync 64-67.
4. Hidden share 35-45 %; top-10 share at least 60 %; layers 0/1 at least 1.8x the mean of layers 2-39.
5. All 24 turns slower than at 28 GiB.

## Falsifier
Decode above 205 ms or misses above 72: a bend (cliff) between 22 and 28 GiB. Decode below 172 ms: the slope flattens. Guard samples with pressure above 1 or swap growth would make the arm a paging arm, reported as such, not a pure budget arm.

## Limits
Three budgets now (22, 28, 34): enough to show a bend against the two-point line, not its shape. Same workload and caveats as arms 1-4; not alternated, drift bound 1-2 %.

## Arm 5 result (measured, 2026-10-12, runtime 0.64.5)

Exact decode, prefetch on, 22 GiB, 24 recorded turns, 2,452 tokens (text identical to arms 1 and 4), 0 events dropped. Guard: pressure 1 throughout, swap flat (2,511 to 2,479 MB), peak wired 47.1 GiB. Raw columns local in `benchmarks/results/l4b-raw-22gib/`.

| per decode token | 34 GiB | 28 GiB | 22 GiB |
|---|---|---|---|
| events | 301.7 | 324.3 | 354.4 |
| decode (server timing) | 151.6 ms | 165.8 ms | 178.1 ms |
| misses a token | 43.0 | 51.0 | 62.2 |
| router sync | 63.1 ms | 65.1 | 66.0 |
| blocked on experts | 59.2 ms | 72.7 | 83.3 |
| read busy | 101.8 ms | 119.3 | 134.2 |
| hidden share of read busy | 43.3 % | 40.2 % | 39.0 % |
| top-10 layers' share of blocked | 74.2 % | 68.2 % | 64.0 % |
| mean blocked, layers 0 / 1 / 2-39 | 3.79 / 2.69 / 1.39 | 4.61 / 3.16 / 1.71 | 5.32 / 3.50 / 1.96 |

Per-GiB slopes between points (removing a GiB): misses 1.33 (34 to 28) and 1.87 (28 to 22); decode 2.37 and 2.05 ms; blocked 2.25 and 1.77 ms; busy 2.92 and 2.48 ms. Per added miss: decode 1.78 ms (34 to 28) and 1.10 ms (28 to 22).

## Against the prediction

All held except one: (1) misses 62.2 (57-68); (2) decode 178.1 ms (176-195; the straight line said 180.0); (3) blocked 83.3 (80-92), busy 134.2 (130-142), sync 66.0 (64-67); (4) hidden share 39.0 % (35-45), top-10 share 64.0 % (at least 60), layers 0/1 at 2.25x (at least 1.8); (5) **23 of 24 turns slower than at 28 GiB, not all 24** (mean ratio 1.079). No falsifier fired: decode is not above 205 ms and misses are not above 72 (no bend), decode is not below 172 (no flattening).

## Explanation (hypothesis)

The miss count is mildly convex in the budget (1.33 to 1.87 misses a GiB), but each added miss costs less (1.78 to 1.10 ms): more misses in flight overlap better, so decode time is close to linear in the budget (2.4 then 2.1 ms a GiB). The structure keeps drifting in one direction as the budget falls (hidden share 43 to 40 to 39 %, concentration 74 to 68 to 64 %, router sync 63 to 65 to 66 ms), with no change of regime. A cliff is not in 22-34 GiB at this workload. Memory pressure never rose, so this is not a paging effect either.

## Limits

Three budgets, one workload, not alternated (drift bound 1-2 %, against 7-9 % between neighbours). The 22 GiB point is the smallest tried; below it the expert cache would also stop holding a layer's working set for one token (a prefill chunk's experts alone are several GiB, 18.127), so a limit exists but is not measured here.

## Decision

No default changes. Ledger row DS-BUDGET-SLOPE-22-34. L4(c) at this workload: the cost of the expert budget is a smooth, nearly linear slope of about 2.1-2.4 ms a token per GiB between 22 and 34 GiB; the model line is `decode = 151.6 + 2.2 x (34 - GiB)` ms to within 1 ms at 28 and 22 GiB.

# Arm 6: the prefetch on/off pair at 48 GiB (prefetch value against misses a token), opened 2026-10-12 before the run

Hamed's go: "go, record the prefetch pair at 48 GiB" (2026-10-12). Estimate: two arms of about 20 minutes each plus a model load each (fewer misses make decode faster), about 50 minutes in all, run one after the other (prefetch on first, then off) under `guarded_run.sh`, 24 turns, 128 tokens, greedy, exact decode, only `--budget-gib 48` and `CACHALOT_PREDICT_TOPK=0` for the second differ from arms 1 and 2. Output `benchmarks/results/l4b-raw-48gib-on/` and `-off/`. The guard refused 38 and 48 GiB earlier on 2026-10-11 for lack of available memory; if it refuses again the arm is not run and that is reported.

## Baseline
At 34 GiB (43.0 misses a token) prefetch on against off: decode 151.6 against 167.3 ms (a gap of 15.7 ms, 10.4 %), blocked 59.2 against 84.2, read busy 101.8 against 82.5. The ledger (DS-PRED-AB) had the gap at about 1 ms at 48 GiB with 23.5 misses a token on another workload. Budget slope (DS-BUDGET-SLOPE-22-34): decode about 151.6 + 2.2 x (34 - GiB) ms, which extrapolated to 48 GiB gives 120.8 ms; the slope will flatten above the working set, so this is an upper bound on the improvement.

## Prediction (written before the run)
Linear interpolation of the gap in misses a token between the two known points (43 misses, 15.7 ms; 23.5 misses, about 1 ms) gives 0.75 ms a miss.
1. Misses a token with prefetch on at 48 GiB: 28-38 (the ledger slope of 0.59 a GiB gives 35, our 1.33 a GiB gives 24; the lower slope applies above the working set).
2. Decode with prefetch on: 122-137 ms; with prefetch off: 130-150 ms.
3. The on/off gap: 4-12 ms (the interpolation at about 33 misses gives about 8 ms), clearly smaller than the 15.7 ms at 34 GiB.
4. Hidden share with prefetch on: 40-50 %; blocked on, 35-55 ms; blocked off, 45-70 ms.
5. All turns slower with prefetch off in at least 20 of 24.
6. Zero events dropped; misses a token (store counter) equal between the two arms to within 1.

## Falsifier
Gap above 15 ms (as large as at 34 GiB): prefetch's value does not shrink with the misses, and the ledger's 1 ms was workload-specific. Gap below 0: prefetch hurts at this budget (the 1 ms ledger gain reversed). Misses on at 48 GiB below 20 or above 42: the budget axis behaves unlike 22-34 GiB and the interpolation does not apply.

## Limits
Two arms in two processes, not alternated (drift bound 1-2 %, small against the predicted 4-12 ms gap but not against a gap of 1-2 ms); one workload; one budget above the working set of this workload (whether 48 GiB holds it is exactly what the miss count shows).

## Arm 6 outcome: not run (2026-10-12)

The memory guard refused both arms at preflight (50.8 and 51.5 GiB available against 77 GiB needed for a 48 GiB budget; 2 GiB of stale swap in use). No model was loaded and no data exist. The prediction above stands unmeasured. The arm needs about 77 GiB available (applications closed), or a budget the guard accepts; with the same memory state a 34 GiB run (63 GiB needed) would also be refused.

## Arm 6 amendment, 2026-10-12: run at 46 GiB, not 48 (before the run)

After Hamed quit VMware and ran `sudo purge`, the guard counted 75.3 GiB available against the 77 GiB a 48 GiB budget needs (need = budget + 29 GiB); 46 GiB needs 75 GiB and is accepted. Hamed asked for the pair to run, so it runs at 46 GiB with the guard unchanged (not `--force`). The prediction above is unchanged in form: two more GiB of cache move the expected misses a token up by about 1-3 (0.6-1.3 a GiB), so the ranges 1-6 stand, with the on-arm misses at 29-40. Output folders `l4b-raw-46gib-on/` and `-off/`. A reboot would have allowed 48 but was not needed.

## Arm 6 result (measured, 2026-10-12, runtime 0.64.6; run at 46 GiB, see the amendment above)

Exact decode, 24 recorded turns each, 2,452 tokens each (identical text across all arms), 0 events dropped, 31.4 misses a token in both arms. Guard: pressure 1 throughout, swap flat (1,914 to 1,906 MB), peak wired 69.3 GiB. Prefetch on ran first, off second (separate processes, not alternated). Raw columns local in `benchmarks/results/l4b-raw-46gib-on/` and `-off/`.

| per decode token | 46 GiB prefetch on | 46 GiB prefetch off | 34 GiB on | 34 GiB off |
|---|---|---|---|---|
| events | 268.6 | 223.8 | 301.7 | 247.0 |
| decode (server timing) | 131.8 ms | 141.9 ms | 151.6 | 167.3 |
| misses a token | 31.4 | 31.4 | 43.0 | 43.0 |
| router sync | 59.5 | 56.1 | 63.1 | 59.2 |
| blocked on experts | 45.7 | 62.2 | 59.2 | 84.2 |
| read busy | 80.1 | 60.7 | 101.8 | 82.5 |
| hidden share of read busy | 44.4 % | 0 % | 43.3 % | 0 % |
| top-10 layers' share of blocked | 84.4 % | 62.4 % | 74.2 % | 55.2 % |
| mean blocked, layers 0 / 1 / 2-39 | 2.73 / 2.00 / 1.08 | 2.16 / 2.22 / 1.52 | 3.79 / 2.69 / 1.39 | 3.08 / 3.17 / 2.05 |

The on/off gap in decode: 10.1 ms at 46 GiB (7.7 %; prefetch off slower in 21 of 24 turns, ratio mean 1.076, range 0.88-1.20), 15.7 ms at 34 GiB (10.4 %). Gap per miss: 0.32 ms at 46 GiB, 0.37 ms at 34 GiB. Blocked-time gap: 16.5 ms against 25.0 ms.

## Against the prediction

All six held, no falsifier fired: (1) 31.4 misses (28-38; 29-40 after the amendment); (2) decode 131.8 on (122-137) and 141.9 off (130-150); (3) the gap is 10.1 ms (4-12), clearly under the 15.7 ms at 34 GiB; (4) hidden share 44.4 % (40-50), blocked 45.7 on (35-55) and 62.2 off (45-70); (5) off slower in 21 of 24 (at least 20); (6) 0 dropped, misses equal. The interpolation I used (0.75 ms a miss between the 43-miss and the 23.5-miss points) would have given about 7 ms; the measured gap is 10.1.

## Explanation (hypothesis)

The prefetch gap tracks the misses a token roughly in proportion in this workload (0.32-0.37 ms a miss) rather than the offset line I interpolated. It shrinks as the budget grows but does not vanish at 46 GiB. The ledger's old figure (about 1 ms at 48 GiB and 23.5 misses, DS-PRED-AB) is lower than this proportionality would give (about 8 ms at 23.5 misses) and came from another workload, a 22k Hermes context; I have not measured this pair there. The slope of decode time against budget also flattens above 34 GiB: 1.65 ms a GiB between 34 and 46 (with prefetch on), against 2.1-2.4 below.

## Limits

Two processes per pair, not alternated (drift bound 1-2 %, small against 10 ms); 46 GiB, not 48 (amended before the run); one workload; the other workload (22k agent context) behind the old ledger figure is not re-measured.

## Decision

No default changes (prefetch stays on). Ledger row DS-PRED-LAYER-46 refines DS-PRED-LAYER and flags DS-PRED-AB's 1 ms as workload-specific, not general.

# Arm 7: the prefetch pair on the 22k-token Hermes context (settling the old 1 ms figure), opened 2026-10-12 before the run

Hamed's go: "go, record the prefetch pair on the 22k Hermes context" (2026-10-12). Context: the system prompt and 25 tools of the first body in `benchmarks/results/hermes-live-0.61.1/requests.jsonl` (about 22k tokens); the 12 Hermes user prompts of `benchmarks/server_miss_budget_ab.py` (each turn is system + tools + one user message, so the 22k prefix is prefilled once and reused), cycled twice (24 recorded turns), 160 new tokens, greedy, exact decode, tracer `every=1` as in arms 1-6, `--max-seq-len 32768`. Prefetch on first, then off (`CACHALOT_PREDICT_TOPK=0`), under `guarded_run.sh`. Budget: **42 GiB, not 46**: after the earlier arms the guard saw 71.8 GiB available (budget + 29 needed), so 42 GiB (71 needed) is the largest accepted; the ledger's figure was at 48 GiB with 23.5 misses a token. Estimate: two arms of about 20 minutes each (model load 3, cold 22k prefill about 4, 4 warm turns, 24 turns of about 25 s); about 45 minutes in all. Output `benchmarks/results/l4b-raw-hermes22k-on/` and `-off/`. The bank is the internal verified bank in every arm so far (the ledger's figure was from the same drive era).

## Baseline
Short-prompt arms: gap on/off 15.7 ms at 34 GiB (43.0 misses) and 10.1 ms at 46 GiB (31.4 misses), 0.32-0.37 ms a miss. Ledger DS-PRED-AB: about 1 ms at 48 GiB, 23.5 misses a token, Hermes-shaped context. Budget curve (DS-BUDGET-CURVE, 22k context): 36 / 44 / 48 GiB gave 31.8 / 27.1 / 25.3 misses a token. Context cost (DS-FLOOR-CTX): +5.6 ms of floor at 25k.

## Two hypotheses, to be told apart
H1 (workload-independent): the gap is about 0.32-0.37 ms a miss, so about 9-11 ms at 28 misses. H2 (the old figure stands): the gap is 1-3 ms at the same misses because the 22k context's per-token time is dominated by something prefetch cannot hide (attention at depth), or because the old arm measured something else.

## Prediction (written before the run), H1
1. Misses a token with prefetch on at 42 GiB: 26-33 in both arms (equal to within 1).
2. Decode with prefetch on 135-148 ms; off 145-160 ms (the short-prompt 46 GiB pair plus about 5 ms of context).
3. The gap 6-12 ms, per miss 0.2-0.4 ms, and off slower in at least 18 of 24 turns.
4. Hidden share with prefetch on 38-50 %; 0 dropped; the cold first turn prefills the 22k tokens once per arm and every recorded turn reuses at least 20,000 tokens.
Registered discrimination: gap above 6 ms supports H1; gap below 3 ms supports H2; between is ambiguous (reported as such).

## Falsifier
Gap below 3 ms (H1 falsified, the old figure stands for this workload); gap above 15 ms or below 0; misses outside 22-38 (the budget axis behaves unlike the ledger's curve).

## Limits
Two arms in two processes, not alternated (drift 1-2 %, small against 6-12 ms, not against 1-3 ms); 42 GiB, not 48; one conversation (one dumped session), twelve prompts twice; the tracer's own +0.55 ms is inside both arms.

## Arm 7 result (measured, 2026-10-12, runtime 0.64.9)

The 22k-token Hermes context (system prompt and 25 tools of the first body of the dumped session, prefilled once per arm in 232 s, then reused: 27 of 28 requests in each arm reused at least 22,407 tokens), 24 recorded turns of 160 tokens at most (2,824 tokens, identical token counts per turn in both arms), 42 GiB, exact decode, 33.5 misses a token in both arms, 0 events dropped. Guard: pressure 1, swap 1,906 to 2,037 MB (+131 MB, under the 512 MB kill line), peak wired 73.2 GiB. Prefetch on first, then off, separate processes. Raw columns local in `benchmarks/results/l4b-raw-hermes22k-{on,off}/`.

| per decode token | 22k Hermes, 42 GiB, prefetch on | prefetch off | 46 GiB short prompts: on | off |
|---|---|---|---|---|
| decode (server timing) | 139.8 ms | 144.4 ms | 131.8 | 141.9 |
| gap (token-weighted, 95 % bootstrap over turns) | **4.62 ms (+1.68 to +7.30)** | | 10.06 (+7.01 to +12.91) | |
| misses a token | 33.5 | 33.5 | 31.4 | 31.4 |
| router-eval window | 67.4 | 62.3 | 59.5 | 56.1 |
| blocked on experts | 43.0 | 57.3 | 45.7 | 62.2 |
| read busy | 79.8 | 55.6 | 80.1 | 60.7 |
| hidden share of read busy | 47.9 % | 0 % | 44.4 % | 0 % |
| top-10 layers' share of blocked | 87.6 % | 64.9 % | 84.4 | 62.4 |
| mean blocked, layers 0 / 1 / 2-39 | 2.61 / 1.71 / 1.02 | 2.06 / 1.80 / 1.41 | 2.73 / 2.00 / 1.08 | 2.16 / 2.22 / 1.52 |

Off slower in 17 of 24 turns (per-turn ratio mean 1.027, range 0.88 to 1.15; per-turn gap sd 8.25 ms). Gap per miss 0.14 ms (short prompts 0.32).

## Against the prediction

1. Misses 33.5 against 26-33: **missed narrowly**; equal in both arms: held.
2. Decode on 139.8 (135-148): held; off 144.4 (145-160): **missed narrowly**.
3. Gap 4.6 ms against 6-12: **missed**; per miss 0.14 against 0.2-0.4: **missed**; off slower in 17 of 24 against at least 18: **missed narrowly**.
4. Hidden share on 47.9 % (38-50), 0 dropped, reuse of the 22k prefix on every recorded turn: held.

By the discrimination rule fixed beforehand (above 6 ms supports H1, below 3 ms supports H2, between is ambiguous) the result is **ambiguous**: 4.6 ms, with a 95 % interval of +1.7 to +7.3 that excludes both the ledger's 1 ms and the short-prompt pattern's 9-11 ms. Neither hypothesis holds as stated; the gap in this workload is real (the interval excludes zero) and smaller than proportional scaling from the short prompts would give.

## Explanation (hypothesis)

Prefetch still hides what it hid (blocked time falls 14.3 ms, against 16.5 ms in the 46 GiB short-prompt pair), but at a 22k context its price is larger: the router-eval window it adds is +5.1 ms (+3.4 ms with short prompts), because the predicted-set arrays are computed against a 22k attention context; and the remaining part of the token (attention at depth, which does not overlap with reads) is larger. So the net gap shrinks from about 10 ms to about 4.6 ms. This is consistent with the ledger's 1 ms figure being the same effect at a lower miss count and a faster drive era, but the two arms do not reproduce it (1.7 ms is its interval's low end). Not tested: the same pair at 48 GiB and 23.5 misses.

## Limits

42 GiB (the guard), not 48; one dumped conversation, twelve prompts twice, 160 tokens; separate processes, not alternated (the per-turn gap sd of 8.25 ms is above the drift bound of 1-2 %, so the interval is the honest width); the tracer (+0.55 ms) is inside both arms.

## Decision

No default change (prefetch stays on: the gap is positive and its interval excludes zero). The ledger's DS-PRED-AB is refined, not refuted: the prefetch gap depends on workload and context (about 10 ms on short prompts at 31 misses, about 4.6 ms on the 22k Hermes context at 33.5 misses). Ledger row DS-PRED-LAYER-HERMES22K.

# Arm 8: the alternated prefetch pair on the 22k Hermes context, opened 2026-10-12 before the run

Hamed's go: "go, record the alternated prefetch pair" (2026-10-12), after the arm 7 estimate that an alternated design would narrow the interval. Estimate: model load 3 minutes, one cold 22k prefill about 4, 4 warm turns, then 24 prompt pairs of two turns of about 20 s: about 25-30 minutes in all, 42 GiB (the guard accepted it with 74.7 GiB available), exact decode, under `guarded_run.sh`. One process: `benchmarks/prefetch_alternated.py` flips `moe_layer_metal.PREDICT_TOPK` (6 on, 0 off; read at every MoE layer call) between the two runs of each prompt, the order alternating by prompt; tracer `every=1` on both sides as in arm 7. The 12 Hermes prompts are cycled twice (24 pairs), 160 tokens at most.

## Baseline
Arm 7 (22k Hermes, 42 GiB, separate processes, not alternated): on 139.8 ms, off 144.4, gap 4.62 ms (95 % bootstrap over turns +1.68 to +7.30), 17 of 24 turns slower with it off, per-turn gap sd 8.25 ms; blocked 43.0 against 57.3; router-eval window 67.4 against 62.3.

## Prediction (written before the run)
1. Position-matched gap (off minus on), token-weighted over 24 pairs: **+4.6 ms** with a planning interval +2.5 to +7.0 (arm 7's centre; an alternated pair removes between-process drift, so the centre should hold and the interval shrink).
2. 95 % bootstrap half-width at most 2.0 ms (arm 7: 2.8).
3. Prefetch on faster in at least 18 of 24 pairs; per-pair gap sd at most 5 ms.
4. Order effect (gap of on-first pairs minus gap of off-first pairs) smaller than 3 ms in absolute value.
5. The flip took effect: hidden share of read time about 48 % on and 0.0 % off; events about 277 on and 228 off; blocked about 43 on and 57 off; misses a token equal between the sides to within 1.5.
6. Text identical in 24 of 24 pairs; 0 events dropped; the prefix reused on every pair turn.

## Falsifier
Gap interval including 0: prefetch brings nothing on this context. Gap above 8 ms: arm 7 understated it (drift). Hidden share of the off turns above 5 %: the flip did not take effect and the arm is void. Order effect above 6 ms: carryover of cache state between the two runs of a pair biases the design.

## Limits
Back-to-back runs of the same prompt share a cache: the second run starts with the first run's resident set and its in-flight speculative loads (prefetch on loads extra experts), so the pair is not two independent measurements; order alternation balances the average and the order effect is reported. 42 GiB, one dumped conversation, 12 prompts twice, the tracer's +0.55 ms inside both sides. The shipped path is one process serving prefetch on; flipping the constant is an instrument, not a supported switch.

## Arm 8 result (measured, 2026-10-12): the back-to-back design is void

`benchmarks/prefetch_alternated.py`, 24 prompt pairs, 42 GiB, 22k Hermes context, 48 turns (2,824 tokens a side), text identical 24 of 24, 0 events dropped, prefix reused on every turn (at least 22,407 tokens), pressure 1, swap flat, peak wired 69.5 GiB. Saved as `benchmarks/results/prefetch-alt/run-backtoback.json` (local).

| | prefetch on | prefetch off |
|---|---|---|
| wall a token (stamp gaps) | 137.0 ms | 133.9 |
| router-eval window | 70.4 | 63.3 |
| blocked on experts | 38.4 | 46.4 |
| read busy | 73.6 | 45.0 |
| hidden share | 49.7 % | 0.0 % |
| events a token | 259 | 204 |
| misses a token | 28.5 | 22.7 |

Position-matched gap (off minus on): **-3.09 ms (95 % bootstrap over pairs -14.18 to +6.01)**, on faster in 15 of 24 pairs, per-pair sd 31.7 ms. **Pairs with on first: -23.29 ms; pairs with off first: +11.08 ms**: an order effect of 34 ms.

## Against the prediction

The flip took effect (prediction 5, first half: hidden share 49.7 % on and 0.0 % off, events 259 and 204; text and drops held). Everything the design was for failed: the gap interval is 20 ms wide (predicted half-width at most 2.0; the arm 7 interval was 5.6 wide); **the order effect is 34 ms (predicted below 3; the falsifier's 6 ms fired)**; misses a token differ between the sides, 28.5 against 22.7 (predicted equal within 1.5). The centre (-3.1) says nothing: the interval spans zero and arm 7's +4.6.

## Explanation

The second run of the same prompt, started straight after the first, finds its experts already resident (the first run read them), so it takes far fewer misses and is much faster whichever side it is on (the side that ran second is faster by 11 to 23 ms). Order alternation balances the average but not the variance, and the carryover is larger than the effect. The back-to-back design cannot price a 5 ms effect. My prediction was written without the size of this carryover in mind; the design note in the record (a "limit": the second run starts with the first's resident set) named the mechanism but I underestimated it.

## Decision

Arm 8 is void as a gap measurement and kept as a lesson. The corrected design (arm 8b below) spreads the two runs of a prompt 12 turns apart, the way `server_miss_budget_ab.py` does (HANDOFF 18.59): the cache state seen by a prompt is then determined by the other prompts, not by its own twin.

# Arm 8b: the corrected alternated pair (prompts interleaved, twins a pass apart), opened 2026-10-12 before the run

The design of `server_miss_budget_ab.py` (HANDOFF 18.59) applied to prefetch: `--design interleave` runs the 12 Hermes prompts four times, as two (on, off) pairs per prompt. Pass 0 runs prefetch on for odd prompts and off for even ones; pass 1 swaps; pass 2 is pass 0 again; pass 3 is pass 1 again. A prompt's two runs of a pair are 12 turns apart, with ten or eleven different prompts between them, so a run no longer inherits its own twin's resident experts. Same setting as arm 7: 22k Hermes context, 42 GiB, exact decode, 160 tokens at most, tracer on every token, one process. Estimate: load 3 minutes, cold 22k prefill about 4, 4 warm turns, 48 turns of about 20 s (16 minutes): about 27 minutes. The interval is a bootstrap over the 12 prompts (whole prompts resampled; the four runs of a prompt are not independent).

## Baseline
Arm 7 (separate processes, not alternated): gap 4.62 ms (+1.68 to +7.30), per-turn gap sd 8.25 ms. Arm 8 (back-to-back, void): order effect 34 ms, per-pair sd 31.7 ms, misses 28.5 on against 22.7 off.

## Prediction (written before the run)
1. The flip works: hidden share about 48 % on and 0.0 % off; events about 277 on and 228 off.
2. Position-matched gap (off minus on), token-weighted: **+4.6 ms**, planning interval +1.5 to +8.0 (arm 7's centre; the design removes between-process drift but keeps carryover from the preceding turn).
3. Bootstrap half-width over the 12 prompts at most 3.5 ms (arm 7: 2.8 over turns, which is not comparable: prompts are the unit here).
4. Order effect (gap of pairs whose on-run came first minus pairs whose off-run came first) under 5 ms in absolute value.
5. Misses a token equal between the sides within 2.0; text identical in all 24 pairs; 0 events dropped; the 22k prefix reused on every turn.
6. Prefetch on faster in at least 16 of 24 pairs.

## Falsifier
Gap interval including 0; gap above 9 ms; hidden share of the off turns above 5 % (flip void); misses differing by more than 3 or an order effect above 8 ms (carryover not removed, the design void again).

## Limits
Carryover from the previous turn remains (an on turn follows an off turn and the reverse, so each side has the same mix); 42 GiB, one conversation, 12 prompts, four runs each; the flip of the module constant is an instrument; the tracer's +0.55 ms is inside both sides.

## Arm 8b result (measured, 2026-10-12, runtime 0.64.10)

`benchmarks/prefetch_alternated.py --design interleave`, one process, 22k Hermes context, 42 GiB, exact decode, 48 turns (2,824 tokens a side), text identical in 24 of 24 pairs, 0 events dropped, the 22,407-token prefix reused on every turn, pressure 1, swap +7 MB, peak wired 68.4 GiB. Saved as `benchmarks/results/prefetch-alt/run-interleave.json` (local).

| | prefetch on | prefetch off |
|---|---|---|
| wall a token (stamp gaps) | 134.2 ms | 139.7 |
| router-eval window | 64.9 | 59.4 |
| blocked on experts | 42.4 | 57.0 |
| read busy | 77.0 | 55.5 |
| hidden share | 46.7 % | 0.0 % |
| events a token | 265 | 215 |
| misses a token | 35.3 | 34.3 |

Position-matched gap (off minus on), token-weighted over 24 pairs: **+5.50 ms (95 % bootstrap over the 12 prompts +0.73 to +9.06)**; server timing +5.49 (+0.74 to +9.02); on faster in 18 of 24 pairs; per-pair sd 11.4 ms. Pairs whose on-run came first +7.06 ms, off first +3.27 (order effect 3.8 ms). First block of pairs (passes 0, 1) +4.89, second block (passes 2, 3) +6.11. Per-prompt gap (mean of its two pairs): prompts 1-7 and 9-11 between +3.1 and +15.8 ms; prompt 0 -20.9 and prompt 8 -3.0.

## Against the prediction

Held: (1) the flip took effect (hidden share 46.7 % on and 0.0 % off; events 265 and 215, a little under the predicted 277 and 228); (2) the gap +5.50 ms inside the planning interval +1.5 to +8.0 around the centre +4.6; (4) the order effect 3.8 ms (under 5); (5) misses 35.3 against 34.3 (within 2.0), text identical, 0 dropped, prefix reused; (6) 18 of 24 pairs (at least 16). **Missed:** (3) the bootstrap half-width is 4.2 ms (predicted at most 3.5). No falsifier fired: the interval excludes zero, the gap is under 9 ms, the off turns' hidden share is 0.0 %, misses differ by 1.0 and the order effect is 3.8.

## Explanation (hypothesis)

The alternated design agrees with arm 7: decode prefetch is worth about 5 ms a token (about 4 %) on the 22k Hermes context at 42 GiB; the two estimates (+4.62, +1.68 to +7.30 across processes; +5.50, +0.73 to +9.06 alternated in one process) are consistent and the second is not narrower, because the unit that carries the variance is the prompt (per-prompt gaps run from +3 to +16 ms, with prompt 0 at -21 ms, a first-in-pass prompt that follows the warm turns; its sign flips the mean by about 2 ms). Prefetch saves 14.6 ms of waiting (57.0 to 42.4) and costs 5.5 ms in the router-eval window and part of the rest, netting about 5.5 ms. Compared with the ledger's older 1 ms, arm 7's interval excludes it and this interval (+0.73 to +9.06) does not; together they favour about 5 ms, not 1 ms.

## Limits

One conversation and 12 prompts (the interval is over prompts, 12 clusters); 42 GiB, not 48; carryover from the previous turn remains but is balanced by the alternation (order effect 3.8 ms); the flip of the module constant is an instrument; the tracer's +0.55 ms is inside both sides. Prompt 0 is post hoc interesting, not excluded here.

## Decision

No default change. The best statement of prefetch's value on the 22k Hermes context is about +5 ms a token (+0.7 to +9.1, alternated; +1.7 to +7.3, across processes), about half the short-prompt pair's 10 ms. Ledger row DS-PRED-LAYER-ALT. Open: the same alternated pair at 48 GiB; whether prompt 0's negative gap is a position-in-pass effect (a pass-0 first turn after the warm turns) or the prompt.

# Arm 9: the alternated prefetch pair on a second prompt set, opened 2026-10-13 before the run

Hamed's go: "Prefetch pair, second conversation", with twelve new user prompts and a `sudo purge` run by him first (2026-10-13). The only other dump (`hermes-live-glm-0.61.12`) has a system block within six characters of the first and the same 25 tools, so it would not be a second workload; instead `benchmarks/prefetch_alternated.py` gained `--prompts FILE`, and `benchmarks/prefetch_prompts_b.json` holds twelve prompts of different topics and lengths (git history, Go, dependency audit, a Java stack trace, SQL, JSON to YAML, cron, mutex versus semaphore, a TypeScript race, a database migration plan, a shell one-liner, pagination). Everything else is as arm 8b: the 22k Hermes system block and tools of `benchmarks/results/hermes-live-0.61.1/requests.jsonl`, `--design interleave` (four passes, a prompt's twins 12 turns apart), 42 GiB, exact decode, 160 tokens at most, tracer on every token, one process, under `guarded_run.sh`. Estimate: model load 3 minutes, cold 22k prefill about 4, 4 warm turns, 48 turns of about 20 s: about 27-30 minutes.

**Baseline.** Arm 7 (two processes) +4.62 ms (+1.68 to +7.30); arm 8b (one process, the built-in prompts) +5.50 ms (+0.73 to +9.06), order effect 3.8 ms, 18 of 24 pairs, hidden share 46.7 % on and 0.0 % off, misses 35.3 against 34.3.

**Prediction (written first).** Gap +5.0 ms (planning +1.5 to +9.0), half-width at most 4.5 over the twelve prompts, on faster in at least 16 of 24, order effect under 5 ms, misses a token equal within 2.0 and between 25 and 45, hidden share 40 to 52 % on and 0 % off, text identical in each pair. The prompt set differs, so I expect the per-prompt gaps to span a similar range (a few prompts near zero or negative). Agreement with arm 8b means the two gaps differ by less than 4 ms.

**Falsifier.** The interval includes zero; the gap is above 9 or below 1.5; misses differ by more than 3; the order effect is above 8 ms; the flip fails (hidden share off above 2 %); a pair's texts differ.

## Arm 9 result (measured, 2026-10-13, runtime 0.64.14)

Run: `benchmarks/prefetch_alternated.py --prompts benchmarks/prefetch_prompts_b.json`, 42 GiB, guard accepted with 72.0 GiB available after Hamed closed VMware and other applications (the first attempt was refused at 67.2); output `benchmarks/results/prefetch-alt/run-promptsb.json` (local). Hamed also ran `sudo purge` while the model was loading, which empties the file cache; the first pair's off run (146.2 ms) was not slow, so no sign of an effect, but it is a disclosed disturbance. 48 turns, 5,584 pair tokens, 0 dropped events, text identical in all 24 pairs.

On 145.5 ms, off 154.8; **gap +9.26 ms a token (95 % bootstrap over prompts +5.88 to +12.01)**, server timing +9.25 (+5.88 to +11.98); on faster in 22 of 24 pairs; median +10.61, sd 8.87. Order effect: on first +10.74, off first +6.64, difference 4.1 ms. Misses a token 44.1 on, 43.7 off. Blocked 51.0 on against 67.4 off; sync 65.9 against 62.4; read busy 89.5 against 65.6; hidden share 44.7 % against 0.0 %; events 273 against 215. Per-prompt gaps (mean of the two pairs): 0 -3.2, 1 +10.8, 2 -3.4, 3 +13.0, 4 +12.4, 5 +14.2, 6 +14.2, 7 +8.4, 8 +22.0, 9 +6.2, 10 +6.3, 11 +13.2.

**Against the prediction.** Held: half-width 3.07 (at most 4.5), 22 of 24 (at least 16), order effect 4.1 (under 5), misses equal within 0.4 and inside 25-45 (44.1), hidden share 44.7 %, texts identical, agreement with arm 8b (the gaps differ by 3.76 ms, under 4). Missed: the gap +9.26 against +5.0 (planning +1.5 to +9.0), and the **registered falsifier "gap above 9" fired**, by 0.26 ms.

**Explanation (hypothesis).** This prompt set costs 44.1 misses a token against 35.3 in arm 8b, and prefetch saves about 0.21 ms per miss here against 0.16 there, so a gap of 9 follows from the miss count more than from a different prefetch mechanism: across the two prompt sets the gap tracks misses a token. Prompts 0 and 2 are slightly negative; prompt 0 was also the one negative prompt of arm 8b (-20.9) although the prompts are different, so a position effect (prompt 0 is the first pair turn, right after four prefetch-on warm turns) is now more likely than a property of that prompt. Not tested here.

**Limits.** One system block and tool set (the second conversation is a second prompt set, not a second agent session); 42 GiB; the module-constant flip is an instrument; twelve prompts.
