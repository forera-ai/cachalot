# CUDA lane N3 stage B: experiment record (what the host issue time between layers is made of, and whether the recorded predictor rises toward its bound when it falls; RTX 5090)

Written 2026-10-19 against runtime 0.66.31, **before any stage B code and before any instance exists**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void (plan section 12). Labels follow plan section 0. This follows `docs/CUDA-N3-RESULTS.md`, `docs/CUDA-N3-EXPERIMENT-RECORD.md` (and its erratum) and plan section 17.10. Nothing here starts an instance: that needs Hamed's yes, and he creates it.

## 1. What this stage asks, and what it cannot claim

N3 stage A found that with the copies hidden by an oracle, 2.5 ms a token of exposure remains at a 1 ms layer time and 2.0 ms of it is **GPU idle time between layers while the host issues work**, and it called that the host's issue loop. Reading my own harness afterwards shows that claim is not yet earned: the timed loop of stage A also computes a SHA-256 digest of every bookkeeping event (about 480 a token for the oracle), builds per-event tuples, records eight CUDA events a layer, creates a tensor view for every copy, and launches the stand-in as two eager kernels. **Part or most of the 2.0 ms may be my instrumentation and my Python, not what any runtime must pay.** This stage separates them, level by level, so the number that remains is the closest this harness gets to the cost of issuing the work, and says what the recorded predictor's hidden share becomes when that cost is lowered.

It **cannot** claim: the host cost of a real runtime (still a Python loop, still torch's dispatcher, no native code, no CUDA-graph copy nodes); anything about a real model's decode; anything about a V100 or another provider's host; a result for the slow-copy instance population (the session's own qualified `c` says which population it gets).

## 2. Baseline (recorded, from `docs/CUDA-N3-RESULTS.md`; fast-population instance, c = 0.194 ms; 17 % resident; 400 decode tokens; C = 1.0 ms a layer; p50 per token, ms)

| arm | wall | compute stall | host gap | host issue | exposed `E` | hidden share `H` |
|---|---:|---:|---:|---:|---:|---:|
| `N` (floor) | 40.076 | 0.015 | 0.196 | 0.84 | 0 | |
| `D` | 50.072 | 8.690 | 1.620 | 2.42 | 9.997 | |
| `P0` | 47.654 | 4.900 | 2.405 | 3.18 | 7.579 | 0.242 |
| `P40` | 47.218 | 5.354 | 1.623 | 3.19 | 7.143 | 0.286 |
| `O` (oracle) | 42.589 | 0.223 | 2.079 | 2.89 | 2.513 | 0.749 |

At C = 0.3 ms: `E_D` 9.903, `E_P0` 9.151 (`H` 0.076), `E_P40` 10.901 (`H` -0.101), `E_O` 3.441 (`H` 0.653), host issue 0.80 / 2.23 / 2.99 / 3.02 / 2.62 for `N D P0 P40 O`. The serial model's bound for the recorded predictor on this window is `H` = 0.365 (erratum of the N3 record).

**Measured cost of the instrumentation on the Mac (derived, a microbenchmark, not the GPU host's CPU):** one `_digest_feed` of a bookkeeping event costs 0.31 to 0.66 microseconds on this Mac; the whole oracle bookkeeping of 400 tokens with the digests costs 0.57 ms a token (prefill walks included). The GPU host's CPU is slower by an unknown factor, so the digest and bookkeeping may be 0.2 to 1.5 ms of the 2.05 ms a token that `O` adds to `N`'s host issue time (2.89 against 0.84); I did not measure the split on the host.

**Working hypotheses.** **H-instr:** the per-event digest, the per-event tuples and the eight events a layer are a large part of the host term. **H-python:** what remains is Python dispatch of about 64 `copy_` calls and a per-layer launch pair, which a native issue path would remove; this stage can only show that the remainder is not instrumentation, not that a native path would remove it. **H-confound:** the early prefetch arm (`P0`, copies issued before the layer's launch) loses to the late arm (`P40`) at 1 ms and above because of host latency before the launch, not because of the window. **H-HOL:** at small windows the recorded predictor is held down by the copy lane (demand copies queued behind prefetch copies on one stream), which no host lever moves.

## 3. Design (profile `n3b`; `benchmarks/cuda/overlap_arena.py`, to be extended after this record is committed)

**Levels** (cumulative; the stand-in, the trace window, the arena, the bookkeeping decisions and the slot assignments are identical at every level):

| level | what changes against the previous level |
|---|---|
| `L0` | stage A as run (digest per event in the timed loop, eight events a layer, a view per copy, two eager kernels): a replicate |
| `L1` | **no digest** in the timed loop (the bookkeeping decisions are unchanged; the event hash is checked in an untimed pass, as before) |
| `L2` | **lean bookkeeping**: the same LRU decisions without per-event tuples, per-token dictionaries or digests; a rolling integer chain (`chain2`) over every demand miss and every prefetch load (token, layer, expert, slot) stands in for the hash |
| `L3` | **pre-built views**: the arena's slot views are created once, before timing, and the copies use them |
| `L4` | **bare**: no per-layer instrumentation events except the ones correctness needs (the demand batch event the compute waits for, the layer-end event the host synchronizes on); token wall from the host clock only; no stall, gap or lane breakdown at this level and above |
| `L5` | `L4` plus the stand-in layer as **one CUDA-graph replay** (captured after calibration; the pair's time is re-calibrated against C; `unavailable` is a valid outcome if capture fails or the calibration misses 5 %) |

**Arms.** `N` floor, `D` demand, `P0` recorded predictor with the prefetch copies issued **before** the layer's compute launch (stage A's `P0`), **`P0L`** recorded predictor issued **after** the launch with no wait (the window starts with the layer; this separates issue order from window), `P40` (stage A's, after 40 % of the layer through an event wait; only at `L0`), `O` oracle.

**Cells (43).** C = 1.0 ms, 17 % resident: `L0`: `N0 D O P0 P0L P40 N1` (7); `L1`, `L2`, `L3`: `N0 O P0L N1` (4 each, 12); `L4`: `N0 D O P0 P0L N1` (6); `L5`: `N0 D O P0L N1` (5); C = 0.3 ms: `L0`: the seven; `L4`: the six (13). `N0` and `N1` bracket every (level, C) group: the floor differs by level, so each level has its own pair. Estimated GPU time 12 to 18 minutes.

**Gates.** (a) at every level the event hash of the bookkeeping, checked in an untimed pass against the independent reference (as stage A); (a2) at `L2` and above the lean book's counts and `chain2` equal the reference's `chain2` for the same cell (the reference `PfBook` gains `chain2` with identical updates); (b) readback between tokens; (c) calibration within 5 % of C (and the graph's at `L5`); (d) each group's `N` pair differs by at most 3 %; (e) sanity; (f) closure of the accounting **at `L0` to `L3` only** (`L4` and `L5` record no per-layer events and say so). Every rehearsal test asserts every gate the GPU run checks, runs the real path, and checks the stream order of the new arm on a recording arena.

**Measurements.** As stage A (wall, compute busy, stall, host gap, copy-lane busy, host issue time) where the level records events; wall, host issue time, counts and `chain2` everywhere. `E_X` and `H_X` as stage A (wall of X minus the mean of its group's two `N` controls; `H_X = 1 - E_X / E_D` within the group; `L1` to `L3` have no `D`, so their `H` is reported against the `L0` group's `E_D` and flagged, and their `E` is the primary number).

## 4. Predictions (registered before any code)

All **predicted**, all at C = 1.0 ms and 17 % resident unless stated, against the baseline in section 2 (same arm, same instance population; if the session's qualified `c` is not within 0.17 to 0.22 ms the scoring is restricted to the within-session comparisons between levels, and the results say so). Probabilities are my own.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| B-P1 | digest off (`L1` against `L0`), `O` arm | `host_gap_ms_token` falls by 0.15 to 0.60 ms (probability 0.5) | falls by less than 0.08 ms or by more than 1.0 ms |
| B-P2 | lean bookkeeping (`L2` against `L1`), `O` | `host_gap_ms_token` falls further by 0.15 to 0.50 ms (0.45) | falls by less than 0.05 ms, or rises |
| B-P3 | pre-built views (`L3` against `L2`), `O` | `host_gap_ms_token` falls further by 0.10 to 0.45 ms (0.45) | falls by less than 0.03 ms, or rises |
| B-P4 | bare floor | `host_issue_ms_token` of `N` at `L4` is at most 0.50 ms (`L0`: 0.84) (0.5) | at least 0.70 ms |
| B-P5 | the main question | `E_O` at `L4` is between 1.0 and 2.0 ms (`L0`: 2.51) (0.45) | above 2.3 ms (the exposure is not instrumentation or avoidable Python overhead of these kinds) or below 0.5 ms (it was almost all instrumentation) |
| B-P6 | demand exposure | `E_D` at `L4` is between 8.8 and 9.9 ms (`L0`: 10.00) (0.55) | below 8.3 or above 10.2 ms |
| B-P7 | the predictor and its bound | `H_P0L` at `L4` between 0.26 and 0.365 (`L0` `P0`: 0.242) (0.4) | below 0.22 (the host cost is not what holds the predictor below its bound; head-of-line or stall is) |
| B-P8a | issue order against window, `L0`, C = 1.0 | `H_P0L - H_P0` between +0.02 and +0.09 (0.55) | below +0.005 |
| B-P8b | issue order against window, `L0`, C = 1.0 | `|H_P0L - H_P40|` at most 0.04 (0.5) | above 0.07 (the window itself matters at 1 ms) |
| B-P8c | window at a small layer time, `L0`, C = 0.3 | `H_P0L - H_P40` at least 0.10 (0.55) | below 0.04 |
| B-P9 | graph | the `N` wall at `L5` is below `L4`'s by 0.10 to 0.50 ms, or `L5` is `unavailable` (probability of the in-range outcome 0.4) | a reduction under 0.03 ms or a rise over 0.2 ms |
| B-P10 | small window, host levers do not rescue the predictor | `H_P0L` at `L4`, C = 0.3 is at most 0.25 (0.55) | at least 0.35 |
| B-P11 | completion | all 43 cells complete inside 20 minutes of GPU time, gates (a) to (f) hold where they apply, no failed cell (0.6) | any gate false, a failed cell or truncation |

## 5. What each outcome means

* **B-P5 holds** (E_O falls into 1.0 to 2.0 ms): roughly half the stage A host term was my instrumentation and Python overhead, the other half is what this harness cannot remove; the lane's "host orchestration is the largest remaining term" statement is restated at the lower value. **B-P5 below 0.5:** stage A's reading of CUDA-H8 was an artefact and is withdrawn. **B-P5 above 2.3:** the issue time is the cost of launching and of `copy_` dispatch in this stack, and only native code or graphs could move it; the next lever is a different issue mechanism, not trimming.
* **B-P7 holds:** the predictor's gap to its bound was host time and a lower host cost recovers it; **B-P7 fails (below 0.22):** the gap is stall (head-of-line behind prefetch copies on one stream, or the 32 demand copies that remain) and the next lever is stream structure or a predictor gate.
* **B-P8a with B-P8b holding:** the stage A `P0` against `P40` difference was an issue-order effect, as suspected; **B-P8b failing:** the window matters at 1 ms and the confound was not the whole story.
* Nothing here is a statement about a real runtime's host cost, a V100, or a real model.

## 6. Stop rules and cost

Stop and bring the files home on a failed cell, a false gate, truncation, a drift flag on the first `N` pair, an event-hash or `chain2` mismatch (a harness bug: fix on the Mac), or 80 % of the cap before the first scored cell. Run the session with an explicit `--max-seconds` under the remaining cap (stage A's lesson). A VM with no `/dev/nvidia*` or no outbound network is replaced or rebooted by Hamed from the console; the agent runs no sudo. Session cap **1 GPU-hour** from Ready.

## 7. Needs Hamed

A go and an RTX 5090 instance when the agent says it is ready (same listing and key); the usual SSH arrangement for that one instance, no sudo; the 11 MB trace is copied to the host as in N2 and N3. Housekeeping stands (delete the S1, S2, V1, N2, B1, B2 and N3 ssh configs, known_hosts files and the key; the N3 billed hours).
