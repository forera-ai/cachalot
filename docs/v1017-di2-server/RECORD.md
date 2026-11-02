# Experiment record: DI-2 keep 140 against DI-1's exact tail, prefill time through `serve.sh` (ABBA)

Written 2026-11-02 before the first arm, runtime 0.74.0 (29f89d3). Hamed: "Push, then run A/B" (the server speed step after the DI-2
quality curve, HANDOFF 18.251).

## 1. Question

On the long gate keep 140 was inside noise and its arms ran ~11 s shorter per 8,192-token prefill than DI-1 (derived from arm wall time).
How much prefill time does `CACHALOT_PREFILL_DECODER_KEEP=140` save on the shipped server, on a real cold Hermes block and on a mid-size
prompt, where the saving's share is derived to be largest?

## 2. Baseline (recorded)

- DI-1 on the server (0.71.3, 0.72.1, §18.249 today): the cold 16,040-token Hermes block prefills in 102.4-105.4 s at 48 GiB.
- §18.251: keep 140 against DI-1, three 8,192-token prefills: 242 against 276 s of arm wall time (about 11 s a prompt, derived).
- Mechanism: keep 140 runs the decoder half (layers 20-39) over 140 rows instead of 2,541, a fixed 2,401 rows fewer whatever the prompt
  length (once the prompt is longer than 2,541 rows and more than one 4,096-token call).

## 3. Predictions (fixed before the run)

- S-1. Exact (DI-1) arms, cold Hermes block (16,040 tokens): 98-112 s.
- S-2. Hermes block, keep 140 minus exact (means of two arms): -5 to -15 s (about -5 to -14 %). Falsified if the saving is under 2 s or
  over 20 s.
- S-3. Mid prompt (4,273 tokens, two calls), keep 140 minus exact: -5 to -14 s (about -15 to -45 %). Falsified if the saving is under 2 s.
- S-4. The two arms of one condition agree within 5 % on each prompt.
- S-5. The DI-2 startup line appears in the keep 140 arms and not in the exact ones.

Stop rules: the guard's own; one attempt per arm; report a guard kill, do not retry without Hamed.

## 4. Method

`docs/v1017-di2-server/run-all.sh`: ABBA (A1 exact, B1 keep 140, B2 keep 140, A2 exact); quiet gate (no other runtime, 5-minute load
under 3.5) before each arm, `settle.sh --budget-gib 40 --timeout 300` between. Each arm (`arm.sh`): a manifest, a fresh `./serve.sh` with
the shipped 1017 defaults (internal bank, DI-1 on, budget 0, wired 80, the 1017 hotlist) under `guarded_run.sh --budget-gib 48
--max-seconds 1500 --force`, `CACHALOT_SNAPSHOT_DIR=` (both prompts cold), B arms `CACHALOT_PREFILL_DECODER_KEEP=140`. Driver
(`driver.py`): request 0 of `~/cachalot-dump-1017.jsonl` (system block, 21 tools, the first user message), then `mid_prompt.txt` (the first
13,1xx characters of `docs/SPEED-RESEARCH-2026-10-03.md` under a one-line instruction, pinned here, 4,273 tokens), greedy, non-streaming,
16 new tokens each. Scored by `analyse.py` from the server's `[request]` prefill seconds. Estimate: about 3 minutes an arm plus settle,
15-25 minutes in all. Raw output in `benchmarks/results/di2-server/` (gitignored).
