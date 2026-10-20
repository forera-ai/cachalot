# CUDA lane N3 stage D: experiment record (a lead-2 prediction arm; the window hypothesis; RTX 5090)

Written 2026-10-20 against runtime 0.66.38, **before any stage D harness code and before any instance exists**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void (plan section 12). Labels follow plan section 0. **This is an exception to a pre-registered rule, chosen by Hamed** ("Yes: design a lead-2 GPU arm as an exception", 2026-10-20): `docs/CUDA-LEAD-TRACE-RESULTS.md` section 4 says the serial bound closes the lead lever offline (lead 2 needed 0.442 and has 0.366); this record tests the one thing that bound does not price, the extra window. It does not reinterpret that rule: the rule stands as the offline verdict and this is a new, separately registered test. Nothing here starts an instance: that needs Hamed's yes.

## 1. What this stage asks, and what it cannot claim

A lead-2 prediction (the router of layer l+2 applied to layer l's input) is available one full layer earlier than the lead-1 prediction the earlier stages used, so its copies can start two layers before they are needed. It loses coverage (serial bound 0.412 against 0.535 on the new trace, first 400 tokens, 17 % resident) and adds copies (80.2 against 72.3 a token). Stage B showed the window decides the hidden share at short layer times (`P40`, issued later in the layer, was 0.17 below `P0L` at 0.3 ms), and stage C showed fewer copies do not help. **H-win:** where the lead-1 predictor is window-limited, the extra layer of window lifts the hidden share by more than the lost coverage costs. The question is whether any (layer time, resident share) cell shows `L2` above `P0L`.

It cannot claim: a real model's decode on NVIDIA hardware (the compute is a stand-in), a native runtime, another trace (one conversation), or a router-score predictor of another kind; the lead-2 sets are the runtime's own, recorded by Hamed on 2026-10-20 (`routing-trace-deepseek-leads`, `CACHALOT_PREDICT_AHEAD=3`, `CACHALOT_PREDICT_SUBMIT=0`).

## 2. Baseline (recorded and derived)

* **Stage B/C, old trace, 17 % resident, RTX 5090 fast population (c = 0.194 ms):** `H_P0L` = 0.339 at C = 1.0 ms (L5; 93 % of the 0.365 serial bound) and 0.096 to 0.100 at C = 0.3 ms (L4); the oracle 0.85 and 0.71.
* **Offline price on the new trace (derived, `prefetch_coverage.py --timed`, first 400 decode tokens):** 17 % resident: no prefetch 47.20 demand misses a token; lead 1: 21.96 misses, 50.35 loads, 72.31 copies, bound **0.535**; lead 2 (issued two rows ahead, so its loads enter the arena before the next layer's demand): 27.73 misses, 52.41 loads, 80.15 copies, bound **0.412**; lead 3: 0.320. 8 % resident: no prefetch 84.12; lead 1 bound 0.556 (126.1 copies), lead 2 0.440 (137.4 copies). Lead-1 timed equals the untimed model to the printed digits on both traces.
* A copy lane that carries 72 copies a token at 0.194 ms is busy 14.0 ms a token; a 0.3 ms layer gives 12.4 ms of compute, so the lane is near saturation there in this model; stage C's gate result (16 % fewer copies saved almost nothing at 0.3 ms) is the evidence that it is not the whole story.

## 3. Design (profile `d`; built on the Mac after this record is committed)

`benchmarks/cuda/overlap_arena.py` gains the arm **`L2`**: the `P0L` order (issued after the layer's launch, no wait) with the recorded **lead-2** set of the row two layers ahead of the same token, bookkept right after the demand of the current row (so its loads enter the arena before the next layer's demand and can evict what it needs), and a row whose lead-2 set was not issued that way (layers 0 and 1, and the first rows after a prefill) issued just before its own demand. The bookkeeping equals an independent reference and `prefetch_coverage.py --timed` (gate (g)). Lead-1 arms and all existing hashes must not move. The run uses the **new** trace (copied to the host, 39.8 MB; the host refuses a different sha256).

**Cells (17 % unless said; 400 decode tokens, token 0 a warm-up; every group has its own `D` arm so `H` is defined):** C = 1.0 ms at level L5: `N`, `D`, `P0L`, `L2`, `O`, `N` (6); C = 0.6 ms at L4: the same six (6); C = 0.3 ms at L4: the same six (6); 8 % resident, C = 0.3 ms at L4: `N`, `D`, `P0L`, `L2`, `N` (5). **23 cells**, estimated 6 to 9 minutes of cells plus qualification and calibration; session cap **30 minutes** of GPU time from Ready. Levels follow the stage C erratum (L4 records no compute-busy rows; no interference prediction is made here).

**Gates:** the stage B/C gates plus (g) as above.

## 4. Predictions (registered before any code)

All **predicted**. Probabilities are my own. `H` is within its group.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| D-P1 | `P0L` baseline on the new trace | `H_P0L` at C = 1.0 ms between 0.42 and 0.54 (stage B's 93 % of the bound 0.535 gives 0.50) (probability 0.65) | outside 0.35 to 0.60 |
| D-P2 | lead 2 at a layer time that is not window-limited | `H_L2` below `H_P0L` by 0.05 to 0.18 at C = 1.0 ms (the bound is 0.123 lower) (probability 0.6) | `H_L2` within 0.02 of `H_P0L` or above |
| D-P3 | the window hypothesis at 0.3 ms | `H_L2 - H_P0L` between -0.05 and +0.04 at C = 0.3 ms, 17 % resident: H-win not supported (probability 0.6) | `H_L2 - H_P0L` at least +0.05 (H-win supported) |
| D-P4 | between | at C = 0.6 ms, `H_L2 - H_P0L` between -0.12 and +0.02 (probability 0.55) | at least +0.05 |
| D-P5 | smaller arena | at 8 % resident, C = 0.3 ms, `H_L2 - H_P0L` at most +0.04 (probability 0.6) | at least +0.07 |
| D-P6 | the main claim | no cell shows `H_L2` at least 0.05 above `H_P0L` (probability 0.7) | any cell does |
| D-P7 | the counts equal the offline model | `L2` demand misses and loads a token equal `coverage_timed(lead 2)` to within 0.01 at 17 % and 8 % (probability 0.9) | any differs by more than 0.05 |
| D-P8 | host issue time | host issue time a token of `L2` within 0.20 ms of `P0L`'s at C = 1.0 ms (the copy counts differ by 8 a token, at about 6 microseconds a copy) (probability 0.7) | differs by more than 0.50 ms |
| D-P9 | the oracle reproduces | `H_O` at C = 1.0 ms between 0.80 and 0.90 (probability 0.7) | below 0.75 |
| D-P10 | completion | all 23 cells complete inside 15 minutes of cell time, every gate true, no failed cell (probability 0.7) | any gate false, a failed cell or truncation |

## 5. What each outcome means

* **D-P6 holds:** the lead lever is closed on the card as well as on the bound; no further lead work, and the open predictor lever is information the runtime does not produce. **D-P3, D-P4 or D-P5 falsified (a cell with `L2` at least 0.05 above `P0L`):** H-win is supported in that regime; the next stage is a sweep of C and resident share to find where, and an adaptive rule (lead by layer time), with predictions first.
* **D-P1 failing:** the new trace changes the stage B baseline; every comparison is read against this session's `P0L`.
* **D-P7 failing:** a harness bug; fix on the Mac.
* Nothing here is a statement about a real model, another trace or another provider's hosts.

## 6. Stop rules and cost

As the stage C record: stop on a failed cell, a false gate, truncation, a drift flag on the first `N` pair, an event-hash mismatch, or 80 % of the cap before the first scored cell. Check `nvidia-smi` and DNS on a new VM before uploading anything; a GPU whose open-driver probe says "no GSP" is replaced, not debugged. Billing: Hamed's private cost file.

## 7. Needs Hamed

A go for the Mac build (given), and later a go and an RTX 5090 instance. Housekeeping stands (delete the S1, S2, V1, N2, B1, B2, N3 and N3-C ssh configs, known_hosts files and the key; enter the N3, N3-B and N3-C billed hours).
