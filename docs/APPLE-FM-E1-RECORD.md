# E-1: DeepSeek decode through the server, with and without the system model working beside it (experiment record; predictions before any build or run)

Written 2026-10-29 against runtime 0.69.1, **before the driver for this run is written or anything is run**. Charter
section 7.2 fields 1 to 3 are fixed here; results go in `docs/APPLE-FM-E1-RESULTS.md`. A prediction edited after its run
is void. Hamed asked for it on 2026-10-29: "Go: run E-1 (serve.sh DeepSeek 48 GiB, repeated greedy prompt, 12 replies
an arm, ABBA, with and without concurrent auxiliary classification calls). Give the estimate first."

## 1. Baseline

* **Stand-in (0.69.0, `docs/APPLE-FM-RESULTS.md` section 4, measured):** an MLX fp16 GEMV loop lost 2.1 % beside the
  system model generating continuously (659.4 to 645.8 GB/s), a GEMM 0.5 %; the system model lost 2.3 % beside the GEMV.
* **The real decode (dated, measured):** DeepSeek's server token all-resident at 48 GiB is 79-80 ms (0.60.10,
  DS-FLOOR-SRV), of which ~53 ms is GPU work (0.60.9: 55.5 ms inside `mx.eval`, 17.9 ms CPU between 44 host syncs);
  attention, the routed and shared experts and the FP8 GEMVs are bandwidth-bound like the stand-in.
* **The auxiliary side (measured 0.69.0):** a classification call 0.82 s median alone (structured output, ~64 output
  tokens at most; the system model processes the prompt at ~1,000 tok/s and writes ~52 tok/s).

## 2. Hypothesis

**H-e1:** the system model takes about the same memory bandwidth from DeepSeek's decode as from the stand-in, so a
token slows by roughly 2 % of its bandwidth-bound part: about 0.5-2 ms of an ~80 ms token. The CPU side of the
token (17.9 ms between host syncs) may also pay for the auxiliary client's and the system service's CPU work.
Classification calls are short bursts (prompt processing, then ~20-60 output tokens), so the load is less continuous
than the stand-in's generation loop.

## 3. Method

* One fresh `./serve.sh` (DeepSeek, 48 GiB expert budget, internal 2-bit bank, shipped defaults including decode miss
  budget 0) under `benchmarks/guarded_run.sh --budget-gib 48 --force`, with `CACHALOT_SNAPSHOT_DIR=` (no saved agent
  blocks loaded). `benchmarks/settle.sh` first. sysctl 88064, screensaver off, no other runtime process.
* Driver (`benchmarks/apple_fm_e1.py drive`, system `python3`, standard library only): `/health`, two warm-up replies,
  then 24 replies in the order A B B A repeated six times (A: auxiliary idle; B: auxiliary load active), each the same
  request: one user message asking for a ~150-word explanation, `temperature 0`, `thinking: false`,
  `max_tokens 128`, not streamed. Per reply: `usage.cachalot.decode_seconds`, `completion_tokens`, prefill seconds,
  reused prefix tokens, the reply text's hash; afterwards the server's `[request]` lines (misses a token).
* Auxiliary load (`benchmarks/apple_fm_e1.py load`, the runtime venv's python, inline provider, started after the
  server is up): classification calls back to back over the 40 routing fixtures while a control file says "on"; it
  reports when it is idle, and the driver starts an A reply only after it is idle. Every call's start, end and latency
  logged. Aux-alone baselines: 30 s of the same load before the server starts and after it stops.
* Workload class W7 (concurrent requests) on W1's short greedy decode.
* Primary statistic: decode ms a token, per reply; the arm difference B - A as the median of the six ABBA cycles'
  differences (each cycle's mean B minus mean A), with the range over cycles.

## 4. Predictions

All **predicted**; probabilities mine.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| E1-1 | text identity | all 24 measured replies byte-identical (greedy; the auxiliary never touches the target) (0.90) | any two differ |
| E1-2 | arm A decode | 78-90 ms a token median (0.65) | below 74 or above 100 ms |
| E1-3 | B - A, decode ms a token | +0.3 to +2.5 ms (0.50) | above +6 ms (more than ~7 %), or below -1 ms |
| E1-4 | consistency | B slower than A in at least 5 of 6 cycles (0.50) | B slower in 2 or fewer |
| E1-5 | misses a token | equal between arms within 0.2 (budget 0, all resident after warm-up) (0.80) | differ by more than 1 |
| E1-6 | auxiliary beside decode | classification latency median within +0 to +15 % of the aux-alone baselines (0.55) | more than +30 % |
| E1-7 | completion | the run finishes without a guardian kill, swap growth under 512 MB, pressure normal (0.75) | a kill or pressure warning sustained |

## 5. What each outcome means

* **E1-3 held:** the stand-in transfers; auxiliary work during decode costs DeepSeek about 1-3 %, and a scheduler is not
  worth building to avoid it. **Above +6 ms:** auxiliary work must be kept out of decode (between turns or during tool
  execution), and the stand-in under-predicts the real decode (likely through the CPU side or host syncs).
* Nothing here is about GLM, MiniMax, long contexts, a read-bound decode or any default.

## 6. Stop rules and cost

Estimated wall clock: settle 1-3 min, aux baseline 0.5 min, server start ~1 min, 26 replies x ~13 s ≈ 6 min,
shutdown and the second baseline ~1 min: **~10-12 minutes, up to ~25 if the guardian kills an arm and it is rerun once.**
Stop and report on a guardian kill, a server error, or a text mismatch between replies of the same arm.

## Addendum 1: the exact-path replicate (written 2026-10-29 after the first run, before the replicate)

Hamed: "Go: run the E-1 exact-path replicate with CACHALOT_DECODE_MISS_BUDGET=off (~9 min)." Same script and settings as
the first run (`docs/APPLE-FM-E1-RESULTS.md`), except `CACHALOT_DECODE_MISS_BUDGET=off` in the server's environment, so a
miss is read, not dropped. Baseline: the first run (budget 0): A 78.1 ms, B - A +0.57 ms, 24 distinct texts, misses
1.2-1.4 a token; 18.76's exact all-resident server token 79.2-80.3 ms; ~2.0 ms a read miss.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| E1R-1 | text identity | all 24 measured replies byte-identical (0.75) | any two differ |
| E1R-2 | misses a token, measured replies | median at most 0.3 in both arms (the prompt goes all-resident once its experts are read) (0.65) | median above 1.5 in either arm |
| E1R-3 | arm A decode | 78-84 ms a token median (0.60) | below 75 or above 90 |
| E1R-4 | B - A | +0.2 to +2.0 ms a token, median of cycles (0.55) | above +5 or below -1 |
| E1R-5 | consistency | B slower in at least 5 of 6 cycles (0.55) | 2 or fewer |
| E1R-6 | completion | no guardian kill, swap flat, pressure normal (0.80) | a kill |
