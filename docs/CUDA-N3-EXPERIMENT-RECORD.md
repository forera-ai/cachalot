# CUDA lane N3 stage A: experiment record (layer-ahead prefetch of predicted experts against demand fetch, with a stand-in compute; RTX 5090)

Written 2026-10-19 against runtime 0.66.28, **before any N3 harness code and before any instance exists**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void (plan section 12). Labels follow plan section 0. This follows plan section N3, `docs/CUDA-N2S1-RESULTS.md`, `docs/CUDA-BACKING-RESULTS.md`, `docs/CUDA-HUGEPAGE-RETEST-RESULTS.md` and plan sections 17.5 to 17.7. Nothing here starts an instance: that needs Hamed's yes, and he creates it.

## 1. What this stage asks, and what it cannot claim

N2 stage 1 measured the **exposed demand-fetch copy time** a decode token pays when an expert arena (a fixed slot arena in VRAM) misses and the copy is on the critical path: 16.7 / 12.8 / 9.4 ms at 8 / 12 / 17 % resident on a fast-buffer instance, with no compute and no prefetch. This stage adds the two things stage 1 left out: a **compute stand-in** that the copies can overlap with, and **layer-ahead prefetch** using the runtime's own recorded next-layer predictions. The question (CUDA-Q6, partly): how much of the copy time can the copy engine hide behind compute on a discrete GPU, when is the hiding limited by the compute window, by the copy lane's own capacity, or by wasted prefetch copies, and what do concurrent copies cost the compute (CUDA-H7)?

It **cannot** claim: tokens per second of DeepSeek or any model on NVIDIA hardware (the compute is a stand-in, a bandwidth-bound read kernel, not the model); anything about a V100 or another provider's host; anything about CUDA-H8's floor (no router, launch chain or host driver loop of a real decode is present beyond the per-layer synchronization emulated here). It claims overlap mechanics: exposed time against compute window, copy-lane saturation and interference, on this card, for this trace.

## 2. Baseline (recorded)

* **N2 stage 1 (RTX 5090, fast-buffer instance, LRU, `routing-trace-deepseek-long`, 1,972 decode tokens):** demand misses a token 90.84 / 69.64 / 51.85 at 8 / 12 / 17 % resident; exposed copy time 16.74 / 12.83 / 9.43 ms; 0.1835 to 0.1853 ms an extra miss.
* **Copy cost is a property of the buffer and the instance (S1 to B2):** 0.182 to 0.185 ms a 9.95 MB pinned copy for fast buffers, 0.359 ms for the slow class; the B1 and B2 instances gave 48 of 48 slow buffers in a fresh process, no allocation technique tried (allocator, page size, populate, order) made a slow buffer fast. This session's per-copy time **c** is therefore *measured at the start* by qualifying the 64 source records (each copied alone) and every absolute prediction below is stated in terms of **c**, not as a fixed number.
* **Interference (S1, S2):** a compute kernel ran 21 to 26 % (S1 instance) and 19 to 22 % (S2) slower under a *continuous* copy stream, with weights larger than L2; the slowdown varies by instance.
* **Recorded predictor coverage (derived, offline, `benchmarks/cuda/prefetch_coverage.py`, tested; the semantics in section 3):** at 17 % resident the runtime's recorded next-layer predicted set (all six predicted experts of the target layer, loaded one layer ahead) cuts demand misses a token from **52.15 to 28.09** (covered 27.41, prefetch loads 55.50, **wasted 28.09**), so copies a token rise from 52.15 to **83.59** (+60 %); at 12 %: 69.92 to 36.06 (loads 73.58, wasted 35.54, copies 109.63); at 8 %: 91.05 to 45.02 (loads 95.24, wasted 43.97, copies 140.26). An oracle (the layer's real experts known one layer ahead) leaves 0.01 to 0.04 demand misses and wastes nothing: 17 %: 52.15 copies a token, all prefetched. The "none" lines (52.15 / 69.92 / 91.05 at 17 / 12 / 8 %) are 0.6 to 0.8 % above N2's 51.85 / 69.64 / 90.84 because the offline model replays 2,028 decode tokens against N2's 1,972, a difference in what each counts as a token, not in the policy; the harness's gate (a) below settles it.
* **Mac (cited):** 41 to 47 % of drive-read time hidden (HANDOFF 18.4; charter ledger), a different mechanism (drive reads, unified memory), and a reference for the order of magnitude only.

Working hypotheses: **H-window:** prefetch hides copies only up to the compute window available when the prediction is known; **H-lane:** copy-lane saturation (copies a layer times c above the compute time of a layer) turns prefetch into a net loss; **H-serial:** the demand misses that remain after prefetch are serialized on the copy stream and exposed in full, so the best a recorded predictor can hide at 17 % is about 1 - 28.09 / 52.15 = 0.46 of the demand-fetch exposure (derived); **H-interference:** intermittent copies slow the compute lane by about their duty cycle times the continuous-copy slowdown.

## 3. Design (profile `n3`; `benchmarks/cuda/overlap_arena.py`, to be built after this record is committed)

**Stand-in compute.** Per layer, one kernel on the compute stream that reads **W bytes of resident weights** (a bandwidth-bound reduction over a tensor larger than L2, so copy interference is realistic as in S2's larger-than-L2 cells), with W chosen by a calibration pass so the kernel alone takes **C** ms: C in {0.3, 1.0, 3.0}. Derived anchors: 0.30 GB of a DeepSeek token's bytes per layer (11.94 GB over 40 layers) takes 0.17 ms at the card's vendor bandwidth, so C = 0.3 is near the bandwidth floor of a real layer and C = 3.0 is the Mac's order (2 ms a layer). The calibration is recorded and must land within 5 % of C or the cell is rejected.

**Per token, per layer** (the same walk as stage 1; 40 layers; the first 400 decode tokens after the same prefill): compute stream runs the stand-in; at the layer boundary the host performs a **synchronization** (`event.synchronize` on the layer's end event, emulating the router sync; its cost is part of the floor) and then issues the next layer's copies. Copies go on a separate copy stream; the compute of layer l waits (an event dependency) for the copies of layer l's demand misses.

**Policies (arms).**
| arm | what is copied, and when |
|---|---|
| `N` | nothing (all-resident floor): stand-in plus the per-layer synchronization only; run **before and after** each C group (drift control) |
| `D` | demand fetch only: layer l's missing experts are copied after layer l-1's sync, compute l waits for them (stage 1's behaviour with a compute stand-in) |
| `P0` | layer-ahead prefetch with the recorded predicted set for layer l+1 issued **at the start of layer l's compute** (best case: the prediction is available at once); demand copies for what was not prefetched are issued at the boundary |
| `P40` | as `P0` but issued **40 % into** layer l's compute (the runtime's predictor runs on the layer's own input, so it is available only after attention: a later window) |
| `O` | oracle: layer l+1's real experts are issued at the start of layer l's compute (an upper bound on any predictor) |

**Prefetch bookkeeping (fixed here).** Predicted keys not resident are loaded as most recently used, evicting from the LRU end, marked pending; a pending key is *covered* when the layer's real experts use it, and *wasted* when the layer ends without use. Hits refresh recency. Arena slot assignment and eviction are the stage 1 `ArenaLRU` with a prefetch insertion method. Prefill is as in stage 1.

**Cells.** 17 % resident (2,611 slots, 24.2 GiB; fits the card with the stand-in weights, W at most 4.5 GB): arms `N`, `D`, `P0`, `O` at C = 0.3, 1.0, 3.0 (12 cells), `P40` at C = 1.0 (1 cell, plus 2 more at 0.3 and 3.0: 3 cells), the closing `N` control per C (3 cells); and 8 % resident (1,229 slots): `D` and `P0` at C = 1.0 (2 cells). Total 20 cells. Estimated GPU time 12 to 20 minutes (the 3.0 ms cells dominate: 400 tokens at about 120 ms plus exposed time).

**Qualified sources.** The 64 source records are qualified at the start (each copied alone, 15 repeats) as in stage 1; `c` is the median per-copy time and the slow-record share is reported. Records are indexed modulo 64 as in stage 1 (the stage 1 limitation stands: an emulated host tier).

**Gates.** (a) the arena's event sequence equals an **independent** reference implementation of the prefetch bookkeeping, event for event, on the Mac (tests) and on the host (it refuses to time a cell whose hash differs), and its demand-miss, prefetch-load, covered and wasted counts equal `benchmarks/cuda/prefetch_coverage.py` on the same tokens; (b) bytes survive a round trip, checked right after the copy of the sampled slots (every N copies and on every demand copy of the first token), with K above 1 in the rehearsal tests; (c) the calibration of the stand-in is within 5 % of C; (d) the `N` control's start and end token times differ by at most 3 %, or the C group is flagged and not scored; (e) no copy rate above the link's sanity bound; (f) token accounting closes (N3-P10). Every rehearsal test asserts every gate the GPU run checks (the B1 lesson).

**Measurements.** Per token: host wall time from the token's first event to its last synchronization (a measured wall time, never a sum of lanes); compute-lane busy time (sum of the stand-in event durations); copy-lane busy time (sum of copy event durations, a **resource** time and not a wall time); demand misses, prefetch loads, covered, wasted; per-layer compute stall (event-measured wait of the compute stream on copies). **Exposed copy time** of an arm X at one C is `E_X = wall_X - wall_N` (derived, p50 over tokens, the `N` control bracketing the group). **Hidden share** `H_X = 1 - E_X / E_D` (derived). Post-processing reports the critical path per token as the compute busy time plus the compute-stream stalls plus the host gap, and its closure against the measured wall.

## 4. Predictions (registered before any code)

All **predicted**. `c` is this session's measured median per-copy time of the qualified records; `E_D` is the demand arm's exposed time at the same C. Probabilities are my own.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| N3-P1 | `D` arm identity with stage 1 | `E_D` at 17 % lies between 0.95 and 1.25 times 52.15 x c, at every C (probability 0.7: the synchronization and event overheads add to serial copies) | outside 0.85 to 1.40 times at any C |
| N3-P2 | `P0`, window not limiting (C = 3.0) | hidden share `H_P0` between 0.30 and 0.50 at 17 % (centre 0.42; the best case under H-serial is 0.46) (probability 0.55) | below 0.20, or above 0.52 (the serial model of what remains is wrong) |
| N3-P3 | `P0`, C = 1.0 | if copies a layer times c (83.59 / 40 x c) is at most 0.6 x C: `H_P0` between 0.30 and 0.48; otherwise between 0.10 and 0.40 (probability 0.5) | `H_P0` outside the applicable interval by more than 0.10 |
| N3-P4 | `P0`, C = 0.3: copy-lane saturation | `H_P0` between -0.30 and 0.20 (83.59 / 40 x c is at least 0.39 ms, above C = 0.3 ms) (probability 0.55) | `H_P0` at least 0.35 (prefetch hides more than the copy lane can carry: a timing or accounting error until shown otherwise) |
| N3-P5 | oracle `O` | `H_O` at least 0.85 at C = 3.0, and at C = 1.0 when 52.15 / 40 x c is at most 0.5 x C; between 0.30 and 0.80 at C = 0.3 (probability 0.6) | `H_O` below 0.70 at C = 3.0 (serialization and synchronization overhead eat what an oracle should hide) |
| N3-P6 | window: `P40` against `P0` at C = 1.0 | `H_P40` lower than `H_P0` by 0.03 to 0.20 (probability 0.5) | equal within 0.02 (the window does not limit) or lower by more than 0.30 |
| N3-P7 | interference (CUDA-H7) | the compute-lane busy time a token in `P0` is 4 to 20 % above `N` at C = 1.0 and at most 10 % above at C = 3.0 (probability 0.55: the copy duty cycle times the 19 to 26 % continuous-copy slowdown of S1 and S2) | below 2 % at C = 1.0 (intermittent copies cost nothing) or above 30 % |
| N3-P8 | prefetch pays | token wall time in `P0` is below `D`'s at C = 1.0 and C = 3.0 (probability 0.8) | `P0` slower than `D` at C = 1.0 or 3.0 (wasted copies cost more than hiding saves) |
| N3-P9 | 8 % resident, C = 1.0 | `H_P0` between 0.25 and 0.50 (the demand share left is 45.02 / 91.05 = 0.49, best case 0.51; copies a layer 3.5 x c) (probability 0.5) | outside 0.10 to 0.60 |
| N3-P10 | the accounting closes | for at least 90 % of tokens, the measured wall time lies within 5 % of compute busy plus compute-stream stalls plus host gap, in every cell (probability 0.6) | fewer than 80 % of tokens close in any cell: the decomposition is rejected for that cell and the cell is not scored |
| N3-P11 | host orchestration | the host time a token spends issuing copies, events and bookkeeping is under 1.5 ms (stage 1 read 0.12 to 0.16 ms for the bookkeeping alone; the extra is the per-layer synchronization and event issue) (probability 0.6) | 3 ms or more |
| N3-P12 | completion | all 20 cells complete inside 25 minutes of GPU time, gates (a) to (f) hold, no failed cell (probability 0.65) | any gate false, a failed cell or truncation |

## 5. What each outcome means

* **N3-P2 holds with N3-P8 holding:** layer-ahead prefetch with the recorded predictor hides about 40 to 46 % of the demand exposure at 17 % resident on a discrete card, as on the Mac (41 to 47 %), and the remaining exposure is demand misses on the critical path: the next lever is predictor coverage, not the copy engine. **N3-P4 holding:** at a compute window below the copy lane's capacity the prefetch's wasted copies (+60 % copies at 17 %) turn the hiding into a loss; a gating rule (the weight gate the Mac priced offline) is then the lever. **N3-P5 holding with N3-P2 near its best case:** the ceiling is the predictor, not the hardware.
* **N3-P1 failing:** the stand-in or the synchronization changes the demand path, and the stage 1 exposed copy time is not additive with compute; every later number must be read against `E_D` measured here, not stage 1's.
* **N3-P10 failing in a cell:** that cell's overlap claims are withdrawn (no overlapping durations are ever summed into a wall time).
* Nothing here is a statement about the V100, about a real model's decode, about CUDA-H8's floor, or about another provider's hosts.

## 6. Stop rules and cost

Stop and bring the files home on a failed cell, a false gate, truncation, a drift flag on the first `N` pair (the instance is unstable), an event-hash mismatch (a harness bug: fix on the Mac), or 80 % of the cap before the first scored cell. Session cap **1 GPU-hour** from Ready. The instance takes about seven minutes to initialize and is not the same buffer population as the earlier ones; the qualification pass says which population it is, and the results state it.

## 7. Needs Hamed

A go and an RTX 5090 instance when the agent says it is ready (same listing and key as S2, N2, B1 and B2); then the usual SSH arrangement for that one instance, no sudo. The 11 MB trace is copied to the host as in N2 (it is not in the repository). Housekeeping stands (delete the S1, S2, V1, N2, B1 and B2 ssh configs, known_hosts files and the key).
