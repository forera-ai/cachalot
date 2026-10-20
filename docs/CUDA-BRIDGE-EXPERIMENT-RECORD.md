# CUDA lane: the bridge session (is the higher P0L hidden share the trace or the instance? RTX 5090)

Written 2026-10-20 against runtime 0.66.40, **before any bridge profile code and before any instance exists**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void. Labels follow plan section 0. Nothing here starts an instance: that needs Hamed's yes.

## 1. What this asks, and what it cannot claim

Stage D's lead-1 arm `P0L` hid more than stages B and C's did: at C = 0.3 ms (17 % resident, level L4) 0.233 against 0.096 to 0.100, at C = 1.0 ms (L5) 0.440 against 0.339. Stage D used a different trace (the multi-lead recording, whose serial bound is 0.535 against 0.365 on its first 400 tokens) and a different instance, on a different day. Two explanations: **H-trace** (the trace's row-level copy pattern lets the same predictor hide more) or **H-session** (this instance, host or time of day does). The bridge runs **both traces in one session on one instance**, in the order old, new, new, old, so the trace effect is separated from the instance and any drift is visible. It cannot claim anything about a real model, other conversations, other cards, or why a trace differs beyond the row-level copy counts below.

## 2. Baseline and an offline price (derived)

* Measured hidden share of `P0L` (17 % resident, first 400 decode tokens): old trace, stages B and C, C = 0.3 ms L4 0.096 and 0.100, C = 1.0 ms L5 0.339; new trace, stage D, C = 0.3 ms 0.233, C = 0.6 ms 0.408, C = 1.0 ms 0.440. Measured `E_D`: 9.6 ms (old, 0.3 ms) and 8.84 ms (new).
* **A row-level window model** (`benchmarks/cuda/window_model.py`, Mac only): per decode row it counts the prefetch loads issued during the previous layer (`n_pf`) and the demand misses (`n_dem`) from the stage A/B bookkeeping; with a copy time c = 0.194 ms and layer time C it exposes `max(0, n_pf c - C) + n_dem c` a row, and the demand-only arm `n_dem c` a row. No host, event or interference term. Its hidden shares: **old trace** 0.3 ms: -0.001 (36 to 42 % of rows overloaded), 0.6 ms: 0.303, 1.0 ms: 0.363; **new trace** 0.3 ms: 0.216, 0.6 ms: 0.481, 1.0 ms: 0.533. Model `E_D`: 9.79 (old) and 9.16 ms (new). So the row-level copy counts alone predict the new trace hides **0.22 more at 0.3 ms and 0.17 more at 1.0 ms**; the measured cross-session gaps are 0.135 and 0.101. The model overstates the new trace at 0.6 and 1.0 ms by 0.07 to 0.09 and understates the old one at 0.3 ms by 0.10.

## 3. Design (profile `e`; built on the Mac after this record is committed)

`overlap_arena.py` gains the profiles `e-smoke` and `e-full` (no new arm; the harness is unchanged): 17 % resident, 400 decode tokens, groups (C = 1.0 ms level L5), (0.6 ms L4), (0.3 ms L4), each with arms `N`, `D`, `P0L`, `N` (12 cells a run). `session.sh n3ebridge` runs `e-full` four times in the fixed order **old, new, new, old** (old trace `n3-trace.npz` with `n3c-expected.json`, new trace `n3d-trace.npz` with `n3d-expected.json`; both are already registered files); a smoke (`n3esmoke`, 40 tokens, both traces) comes first. 48 cells, estimated 10 to 15 minutes plus four arena allocations and calibrations; session cap **30 minutes** of GPU time from Ready. The trace is identified in each run by its sha256 in the run manifest.

**Gates:** the stage B/C gates (hash, determinism, counts, `chain2`, readback, calibration, drift, closure, sanity); gate (g) not needed (no gated or lead-2 arm).

## 4. Predictions (registered before any code)

All **predicted**. Probabilities are my own. `H` = hidden share of `P0L` within its group; "difference" = new minus old, each averaged over the session's two runs of that trace.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| BR-1 | the trace effect at 0.3 ms | old `H_P0L` between 0.04 and 0.16, new between 0.19 and 0.28, difference between 0.10 and 0.25 (the model says 0.217, the cross-session measurement 0.135) (probability 0.6) | difference under 0.05 (the gap is the session, not the trace) |
| BR-2 | at 1.0 ms (L5) | old `H_P0L` between 0.30 and 0.38, new between 0.40 and 0.48, difference between 0.06 and 0.18 (probability 0.6) | difference under 0.03 |
| BR-3 | at 0.6 ms (L4) | old between 0.22 and 0.36, new between 0.36 and 0.48, difference between 0.08 and 0.22 (probability 0.5) | difference under 0.03 |
| BR-4 | the row-level model | the model's `H` is within 0.10 of the measured `H` in at least 4 of the 6 (trace, C) cells (probability 0.6) | within 0.10 in at most 2 cells |
| BR-5 | replication | for each (trace, C) the two runs' `H_P0L` differ by at most 0.04 (probability 0.8) | any pair differs by more than 0.08 |
| BR-6 | the demand-only exposure | `E_D` between 9.2 and 9.9 ms (old) and between 8.4 and 9.2 ms (new) in every group (probability 0.7) | outside 8.0 to 10.3 ms |
| BR-7 | completion | 48 cells inside 25 minutes of cell time, every gate true, no failed cell (probability 0.7) | any gate false, a failed cell or truncation |

## 5. What each outcome means

* **BR-1 to BR-3 hold (differences clearly above the falsifiers):** the higher hidden share is the trace, quantitatively close to what row-level copy counts predict; stage D's `P0L` baseline is right for its trace, the stage B/C numbers are right for theirs, and no instance effect needs explaining. **BR-1 falsified:** the same predictor on the same card hid the same amount on both traces within a session: the earlier gap was the instance or the day, and cross-session comparisons in this lane need a bridge cell each time.
* **BR-5 failing:** run-to-run variation within a session is larger than the effects discussed; the stage conclusions that rest on differences below it are weakened.
* **BR-4 failing:** the row-level counts are not what sets the hidden share; the host, event and interference terms matter at these layer times.
* Nothing here is a statement about a real model, another trace or another provider's hosts.

## 6. Stop rules and cost

As stage C and D records: stop on a failed cell, a false gate, truncation, a drift flag on a group's first `N` pair, an event-hash mismatch, or 80 % of the cap before the first scored run. Check `nvidia-smi` and DNS on a new VM before uploading anything; a GPU that does not initialize is replaced, and the replacement is created **before** the faulty instance is terminated. Billing: Hamed's private cost file.

## 7. Needs Hamed

A go for the Mac build (given), and later a go and an RTX 5090 instance. Housekeeping stands.
