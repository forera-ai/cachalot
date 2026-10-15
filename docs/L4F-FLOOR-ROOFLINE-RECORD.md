# L4(f): the DeepSeek floor roofline (experiment record)

Written 2026-10-15 against runtime 0.64.17, before any analysis or run. Status: **step 1 (offline) done 2026-10-15; step 2 (machine, compressed-attention ceilings) not run, waiting for Hamed's go.** Predictions above are unchanged from the version written before the analysis.

## Goal

`benchmarks/whatif.py` answers "what if GPU memory bandwidth doubled" with a range of 51-78 ms (LEDGER WI-FLOOR-BW), because the share of the 52.2 ms of chained GPU kernel time that is bandwidth-bound was never measured. Replace the range with a number and an interval by splitting each GPU row of the all-resident token into a bandwidth-bound part and a remainder (dequantization compute, launch cost, latency).

## Baseline (measured, from the repo)

- All-resident DeepSeek token through the server: 79-80 ms (DS-FLOOR-SRV); in-process 75.4 ms = 55.5 ms inside `mx.eval` + 17.9 ms CPU graph building (DS-FLOOR-SPLIT-0609, §18.75).
- Chained GPU rows sum to 52.2 ms: compressed reuse attention 15.9, compressed source attention 8.7, sliding-window attention 1.0, routed experts 10.1, shared expert 6.0, hyper-connection glue 4.9, source compressor/indexer 2.3, head 1.6, router 0.9, Engram 0.8.
- Bytes a token about 7.5 GB (trunk 5.14 GB, 240 expert reads of 9.95 MB), which is 9.2 ms at the full 819 GB/s (DS-BYTES-TOKEN, derived).
- Size-matched achievable rates (MC-READ-SIZE, MC-READ-SIZE-Q): 2-bit quantized matmul 56 % of 819 GB/s at 3 MiB, 68-72 % at 8 MiB and above; FP8 GEMV 37 % at 3 MiB, 56-65 % at 8 MiB and above; bf16 GEMV 77-90 %.
- Measured rows against those rates: routed experts 6 x 9.49 MiB in 0.252 ms (about 28 % of 819 GB/s); shared expert (three FP8 GEMVs) 0.150 ms.

## Method

1. Offline, no machine: for each chained GPU row, derive the bytes it reads (weights from the checkpoint shapes, KV and compressed state from the 17-position context of the saved run; the 22k context as a second column) and divide by the size-matched achievable rate of its kernel class, giving `t_bw` for the row. Bandwidth-bound time of the floor is the sum of `min(t_bw, t_row)`; the remainder is not bandwidth-bound. Doubling bandwidth saves half of the bandwidth-bound sum. New `benchmarks/floor_roofline.py` with tests; reads the saved rows in `benchmarks/results/floor-profile-0.60.7/gpu.out`.
2. On the machine, only where a row has no matched ceiling (the compressed attention rows): the existing `micro_*_roofline.py` instruments at the row's byte count, random buffers, under `guarded_run.sh` with a small budget. No model load, no server.
3. Replace `BW_BOUND_SHARE_RANGE` in `whatif.py` and the LEDGER row WI-FLOOR-BW with the result and its interval.

## Estimate

About one hour offline (script, tests, bytes per row, write-up) and 20-30 minutes of machine time for step 2 (about 10 minutes a missing ceiling, arms sequential, `settle.sh` between). No `sudo`, no server, no change to `src/`. Needs Hamed's go for step 2 only; the machine gate must be green (the 48 GiB memory problem does not matter: step 2 allocates buffers of tens of MiB).

## Prediction (written first)

- P1: the bandwidth-bound sum `t_bw` over all GPU rows is **11-15 ms** at 17 positions (the 9.2 ms of whole-token bytes at full rate, divided by the 60-80 % that quantized and FP8 kernels reach at these sizes), which is 21-29 % of the 52.2 ms of chained GPU time.
- P2: doubling memory bandwidth therefore lowers the server floor by **5.5-7.5 ms (7-9 % of 80 ms)**, to 72.5-74.5 ms; the 51-78 ms range in `whatif.py` becomes about 72-75 ms.
- P3: the routed-expert and shared-expert rows are the largest `t_bw` contributors by bytes but run at 41-45 % of their own size-matched ceiling (0.252 ms against 0.104 ms and 0.150 ms against 0.067 ms), so most of their time is kernel inefficiency, not bandwidth.
- P4: at a 22k context the compressed attention rows grow and the bandwidth-bound share rises by at most 3 points.

## Falsifier

- P1 fails if the sum of `t_bw` is below 6 ms or above 25 ms. Above 25 ms means the floor is substantially bandwidth-bound and the "kernels run at 29-45 % of peak" reading (LEDGER section 5) was wrong; below 6 ms means almost no row is bandwidth-bound.
- P2 fails if the saving at 2x is below 3 ms or above 12 ms.
- P3 fails if either expert row is within 15 % of its size-matched ceiling.
- If the per-row bytes cannot be derived to within 15 % (checked against the 7.5 GB token total), the analysis is void and reported as such.

## Limits

Derived from published kernel rates and byte counts, not from changing the hardware: it prices a bandwidth multiplier for the model's own kernels and says nothing about a faster GPU's clock or cache. The 17-position context is the saved run; the 22k column is derived from the same byte formulas. The CPU graph-building time (17.9 ms) and GPU idle (27 %) are untouched by bandwidth and stay in the floor.

## Result (step 1, offline; derived, not measured on a machine)

Instrument: `benchmarks/floor_roofline.py` (7 tests in `tests/test_floor_roofline.py`). Bytes come from the safetensors headers of layers 0, 2, 5 and 25 and the config; one finding changed the byte count: the runtime reads `wo_a` as bf16 (`wo_a_bf16`, 67.1 MB a layer) though the checkpoint stores it as FP8 (33.5 MB).

- Bytes a token: **11.94 GB** (attention projections 6.4 GB, routed experts 2.39 GB, shared experts 1.42 GB, head 1.32 GB, router gate and hyper-connection matrices 0.3 GB, compressors and indexers the rest). The 7.5 GB of LEDGER DS-BYTES-TOKEN left out the shared experts, the head, the bf16 `wo_a` and the compressors.
- Chained GPU rows in the saved run: 49.9 ms (the ledger's 52.2 includes 2.3 ms of derived compressor and indexer time not in the parsed rows).
- **Bandwidth-bound time: 20.8 ms (42 % of the GPU rows).** Largest parts: compressed reuse attention 8.4, routed experts 4.2, shared expert 3.0, source and index-only attention 2.4, head 1.6 (the head runs at 99 % of 819 GB/s: it is bandwidth-bound by itself).
- **Doubling bandwidth saves 10.4 ms of the 80 ms server floor (13.0 %), to 69.6 ms.** Sensitivity to the kernel rate tables: rates 20 % lower give 12.8 ms, 20 % higher give 8.9 ms.
- Rows against their own size-matched ceiling: routed experts 41 % (0.252 ms against 0.104), shared expert 50 % (0.150 against 0.075), reuse attention 53 %, source attention 21-35 %: the expert kernels spend most of their time on dequantization and launch cost, not on memory.
- At 22,000 tokens the KV and indexer-scan bytes add 0.06 GB a token (0.5 %); the bandwidth-bound share moves from 42.4 % to 42.5 %. The row times at 22k were not measured, so this is a byte statement only.

### Against the predictions

- P1 (11-15 ms): **missed**, 20.8 ms. It rested on the stale 7.5 GB; the falsifier (above 25 ms) did not fire, though the 20 %-lower rate case reaches 25.5.
- P2 (5.5-7.5 ms saved): **missed**, 10.4 ms; inside the falsifier band (3-12), but the sensitivity case at -20 % rates (12.8) is outside it.
- P3 held: expert rows at 41 % and 50 % of ceiling (predicted 41-45 % and the falsifier was 15 %). Most of their time is not bandwidth.
- P4 held: +0.1 point, derived only.
- The byte-reconciliation clause: bytes differ from the 7.5 GB reference by 59 %, so the clause as written fires. The reference is the stale one; the 11.94 GB is built from header sizes, passes the no-row-above-peak test (the head reaches 99 %), and the explanation above accounts for the gap. I treat the clause as having caught a stale ledger figure, not as voiding the analysis.

### Limits and what step 2 would add

The rate tables (MC-READ-SIZE-Q) were measured on random buffers with the library kernels; the attention projections run through the project's own kernels, so their ceiling is an assumption. Source and index-only rows assume the compressor and indexer sit inside them. Step 2 (the existing `micro_*_roofline.py` at the attention projections' byte counts, about 20-30 minutes on the machine) would replace that assumption with measured ceilings. `whatif.py` and LEDGER WI-FLOOR-BW are **not** changed yet: the 10.4 ms is a derived estimate with a 9-13 ms sensitivity band, not a measured number.

## Step 2 (machine): prediction written before the run, 2026-10-15

Instrument: new `benchmarks/micro_attn_projection_roofline.py` (random buffers, no model, 40 distinct layers chained per arm, `mx.sum` over the same bytes as the machine's ceiling at that size). Arms, in order: `wq_a`, `wkv`, `wq_b`, `wo_b` (each `fp8_linear` at its real shape), `wo_a` (bf16 grouped matmul, 67 MB a layer), then `layer` (all five chained per layer); each with its `sum` ceiling. Two processes (A then B) under `guarded_run.sh --budget-gib 4`, `settle.sh` between; about 10-12 minutes in all.

Baseline from HANDOFF 7.1.10 (before the wq_a/wkv fusion): FP8 GEMV family wq_a 171 GB/s, wkv 87, wo_b 383, wq_b 456; the table-based estimate in step 1 used 65 % of peak (530 GB/s) for a 40 MB FP8 launch.

- S2-a: `wq_a` and `wkv` reach at most 40 % and 25 % of 819 GB/s (few rows, occupancy); `wq_b` and `wo_b` reach 45-60 %; the bf16 `wo_a` 60-85 %.
- S2-b: the five projections chained take **0.36-0.46 ms a layer**, at least 70 % of the 0.53 ms reuse-attention row.
- S2-c: `mx.sum` reaches at least 85 % of 819 GB/s on every buffer of 30 MB or more.
- S2-d: three bounds on the bandwidth-bound time of the attention rows: lower (bytes at the sum ceiling) 0.20-0.24 ms a layer, central (step 1's table) 0.281, upper (the measured projection time) 0.36-0.46; whole-floor bandwidth-bound time 15-17 ms (lower), 20.8 (central, unchanged), 24-28 ms (upper).

Falsifiers: S2-b fails below 0.30 or above 0.50 ms a layer. If `wq_b`, `wo_b` and `wo_a` each reach 85 % or more of the sum ceiling, step 1's 65 % FP8 rate was too low and the central estimate rises by more than 2 ms. If A and B disagree on a layer time by more than 5 %, the arms are reported as drifting and the best of the two is not used as one number.

## Result (step 2, measured on the machine 2026-10-15, two processes A then B, `settle.sh` between)

Raw: `benchmarks/results/l4f/attn-roofline-A.json`, `-B.json`; logs under `benchmarks/results/guarded/attn-roofline-*`. Per layer, best of 9 reps, 40 distinct layers chained (A / B): `wq_a` 6.6 MB 0.0392 / 0.0345 ms (167 / 190 GB/s, 20-23 % of 819); `wkv` 2.6 MB 0.0276 / 0.0279 (95 / 94, 11.5 %); `wq_b` 42 MB 0.1036 / 0.1022 (405 / 411, 50 %); `wo_b` 42 MB 0.1154 / 0.1144 (364 / 367, 44.6 %); `wo_a` bf16 67 MB 0.1066 / 0.1056 (629 / 635, 77 %); **all five chained 0.3798 / 0.3776 ms (422-424 GB/s, 51.5-51.8 %)**, 0.6 % apart. `mx.sum` over the same bytes: 206-213 GB/s at 6.6 MB, 125 at 2.6, 485 at 42 MB (59 %), 544-550 at 67 MB (66-67 %).

Three bounds on the bandwidth-bound time of the whole floor (`floor_roofline.py --measured`): low 19.6 ms (all projection bytes at the best rate any projection reached, 77 % of peak), step 1's table 20.8, centre 22.8 (each projection's bytes at the `mx.sum` rate over the same bytes, capped by its own time), high 25.1 (the projections' measured time). **Doubling bandwidth saves 9.8-12.5 ms of the 80 ms floor (centre 11.4, 14 %).** `whatif.py` and LEDGER WI-FLOOR-BW now carry that instead of the 51-78 ms range.

### Against the step 2 predictions

- S2-a: `wq_a` (20-23 %), `wkv` (11.5 %), `wq_b` (50 %) and `wo_a` (77 %) in their ranges; `wo_b` 44.6 % just below its 45-60 % (missed by 0.4 points).
- S2-b held: 0.378-0.380 ms a layer (0.36-0.46), 71 % of the 0.53 ms row.
- S2-c **falsified**: `mx.sum` reaches 59 % of 819 GB/s at 42 MB and 66-67 % at 67 MB, not 85 %. I mis-applied the project's own table, which says 63 % at 8 MiB and 93 % only from 128 MiB; the ceiling at attention's launch sizes is 480-550 GB/s. The bf16 `wo_a` matmul beats `mx.sum` (115 % of it).
- S2-d: lower bound 0.253 ms a layer (predicted 0.20-0.24, missed, because the ceiling premise was wrong); upper 0.389 ms (predicted 0.36-0.46, held); floor bandwidth-bound low 19.6 (predicted 15-17, missed), centre 22.8 (predicted 20.8 unchanged, missed by 2.0), high 25.1 (predicted 24-28, held). No falsifier fired (`wq_b`, `wo_b` not at 85 % of the sum ceiling together; A and B within 0.6 %).

### What this changes

The floor is more bandwidth-bound than step 1 said, but still not mostly: about 25-31 % of the 80 ms token (20-25 ms of GPU rows) would speed up with more bandwidth, and about 55-60 ms would not (CPU graph building 18 ms, GPU idle, kernel dequantization and launch cost). The best attention kernel (`wo_a`, bf16) reaches 77 % of peak; the FP8 projections reach 45-50 % at 42 MB and run at 76-92 % of a plain sum over the same bytes, so FP8 kernel work could still gain 10-25 % on projections that are 0.38 of the 0.53 ms attention row. That is a lever for the next kernel session (`wo_b` first: 0.115 ms against 0.087 ms for a sum), unpriced beyond this arithmetic: at most about (0.115-0.087) + (0.104-0.087) = 0.045 ms a layer, 1.8 ms a token.
