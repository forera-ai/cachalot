# Experiment record: the DeepSeek-V4.1-Flash-1017 server floor and hit rate at 48 GiB (runbook step 8)

Written 2026-11-02 before the first arm. Runtime 0.72.5 plus the uncommitted-at-writing default switch (serve.sh, chat.sh and
`pareto.serve_defaults` default to the 1017 checkpoint), committed together with this record. Hamed: "Yes, step 8", then "Go, run it".

## 1. Question

Runbook step 8 says to re-record the DeepSeek floor and hit rate at 48 GiB after the switch, because the routing changed. What are
the all-resident server token, the cost of a miss, and the hit rate on 1017 with its own hotlist, exact and under the shipped budget 0?

## 2. Baseline (recorded, not reproduced today)

- 0.60.10, HANDOFF 18.76, first release, internal bank, old hotlist, same driver, 48 GiB:
  - A exact: floor repeats 1-2 80.3 / 79.2 ms (87.8 after the mixed prompts); mixed 126.2 ms at 23.5 misses a token; end-of-run hit rate 0.896.
  - C budget 0: floor 81.9 / 81.3 (85.0 after); mixed 93.3 ms, 16.5 dropped a token; hit rate 0.923.
- Since then the FP8 GEMV decode went on by default (0.66.0, ~3.3 ms microbenchmark) and MLX moved to 0.32.4 (0.68.0). The newest exact
  all-resident server token is 75.3-76.0 ms (0.69.3, HANDOFF 18.231, E-1, first release).
- 1017's hotlist (0.72.4, HANDOFF 18.246) covers about twice the old hotlist's routing mass on 1017's own sessions.
- No same-day arm on the first release: its bank now lives only on the X10Pro (internal copy deleted 2026-11-01), so a same-day arm would be
  read-bound and not comparable. The comparison is against the recorded numbers only, and any difference mixes checkpoint, hotlist and the
  runtime changes since 0.60.10.

## 3. Predictions (fixed before the run)

- F-1. A, floor repeats 1-2 (exact, all-resident): 73-80 ms. Falsified below 71 or above 83. (Same architecture and kernels as the
  first release; 18.231 measured 75.3-76.0.)
- F-2. A, mixed prompts: 14-24 misses a token (fewer than the first release's 23.5 because the 1017 hotlist covers more mass); end-of-run
  hit rate 0.89-0.95. Falsified above 27 or below 10 misses a token.
- F-3. A, cost of a miss = (mean mixed token - mean floor repeats 1-2) / mean mixed misses a token: 1.6-2.4 ms. Falsified outside 1.3-2.8.
- F-4. C (budget 0): floor repeats 1-2 within 0 to +5 ms of A's; mixed token 82-98 ms with 9-18 dropped experts a token. Falsified if the
  mixed token is above 105 or below 78.
- F-5. Drift: the floor prompt after the twelve mixed prompts reads at most 10 ms above repeats 1-2 in each arm (0.60.10: +8.0 and +3.4).

Stop rules: the guard's own (pressure, swap, wall clock 2400 s an arm); one attempt per arm; if the guard kills an arm, report it and
do not retry without Hamed.

## 4. Method

`docs/v1017-floor/run-all.sh`: quiet gate (no other runtime, 5-minute load under 2.5), arm A, `settle.sh --budget-gib 48 --timeout 600`,
quiet gate, arm C. Each arm (`docs/v1017-floor/arm.sh`): a manifest (`run_manifest.py`, workload W1:floor-server), a fresh `./serve.sh`
under `guarded_run.sh --budget-gib 48 --max-seconds 2400 --force` (the force skips only the 77 GiB preflight arithmetic, as in 18.76;
2026-11-01's settle found 72 GiB available), `CACHALOT_SNAPSHOT_DIR=` (no agent snapshots), model 1017 with its internal bank, the
1017 hotlist serve.sh picks, `CACHALOT_DECODE_MISS_BUDGET=off` for A and `0` for C. Driver (`docs/v1017-floor/driver.py`, byte-identical
to 0.60.10's): warm-up, the floor prompt three times, twelve mixed prompts, the floor prompt twice more; greedy, 120 tokens, non-streaming;
`/v1/stats` after the warm-up and at the end. Workload class W1 only. Raw output in `benchmarks/results/floor-server-0.73.0-1017/`
(gitignored); the result table goes in HANDOFF.

## Erratum (2026-11-02, before any arm ran)

The first launch waited 30 minutes at the quiet gate without starting an arm: the 5-minute load stayed at 3.2-3.3, held there by
`hybridsearchd` (the macOS search indexer, ~87 % of one core). The 0.60.10 baseline's gate read the 1-minute field by mistake, and its
arm B started at a 5-minute load of 2.93 (`run-all.log`: `{ 2.30 2.93 2.95 }`), so 2.5 on the correct field was stricter than the
baseline's real condition. The gate now waits for a 5-minute load under 3.5. Nothing else changed; no prediction changed.
