# CUDA lane N3 stage C: experiment record (a router-weight gate on the recorded predictor; RTX 5090)

Written 2026-10-19 against runtime 0.66.33, **before any stage C harness code and before any instance exists**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void (plan section 12). Labels follow plan section 0. This follows `docs/CUDA-N3B-RESULTS.md` (v238 First jobs, candidate 1) and plan section 17.12. Nothing here starts an instance: that needs Hamed's yes.

## 1. What this stage asks, and what it cannot claim

Stage B showed the recorded predictor reaches 0.339 of the serial model's 0.365 bound once the host cost is low, and that about 60 % of its prefetch loads are wasted (17 % resident, first 400 decode tokens: 57.43 loads, 23.00 covered, 34.42 wasted a token). The question: **does loading only the predicted experts whose recorded router weight is at least a threshold, which cuts wasted copies, lift the hidden share where copies are the limit (a short layer time) at an acceptable loss of coverage where they are not?** The offline price on the Mac says a gate lowers the bound (fewer experts covered) and lowers the copy count; whether the lower copy count buys back more than the coverage costs depends on the host issue cost (about 19 microseconds a copy) and on copy-lane interference, which only the card can measure.

It cannot claim: anything about a real model's decode on NVIDIA hardware (the compute is the stage B stand-in), a native issue path, the V100 or AMD, other instance populations, or whether the gate threshold generalizes beyond this one trace (`routing-trace-deepseek-long`, Mac workload mix). The gate uses weights the runtime already records; the live runtime has no CUDA path.

## 2. Baseline (recorded and derived)

* **Stage B (RTX 5090, fast population, c = 0.1937 ms, level L5 = one graph replay a layer, 17 % resident, 400 decode tokens, p50):** floor 39.949 ms; `E_D` 9.460; `P0L` exposure 6.254, `H_P0L` **0.339** at C = 1.0 ms; at C = 0.3 ms (L4) `H_P0L` **0.100**, `E_P0L` 8.655, compute lane +5.8 % busy over the floor; host issue time of the oracle at L5 1.210 ms for 50.4 copies (about 19 microseconds a copy).
* **Offline gate price (derived; `benchmarks/cuda/prefetch_coverage.py --gates`, first 400 decode tokens, per token):**

| resident | gate | demand misses | covered | prefetch loads | wasted | copies | serial bound `1 - misses / ungated-demand` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 17 % | none (demand only) | 50.45 | 0 | 0 | 0 | 50.45 | 0 |
| 17 % | none (recorded predictor) | 32.03 | 23.00 | 57.43 | 34.42 | 89.46 | 0.365 |
| 17 % | 0.10 | 31.86 | 22.82 | 56.55 | 33.73 | 88.41 | 0.368 |
| 17 % | 0.15 | 32.70 | 21.88 | 53.42 | 31.54 | 86.12 | 0.352 |
| 17 % | 0.20 | 35.84 | 17.50 | 39.27 | 21.78 | 75.12 | 0.290 |
| 17 % | 0.30 | 46.16 | 4.97 | 8.11 | 3.14 | 54.27 | 0.085 |
| 8 % | none (recorded predictor) | 54.44 | 46.18 | 102.30 | 56.12 | 156.74 | 0.406 |
| 8 % | 0.20 | 62.24 | 36.02 | 72.09 | 36.08 | 134.34 | 0.321 |

  A gate of 0.20 removes 14.3 copies a token (16 %) and 7.2 covered misses (a loss of 0.075 in the bound); a gate of 0.10 changes almost nothing (the predicted weights below 0.10 are rare); 0.30 gives up almost all prefetch.
* Copy cost is a property of the instance's buffers (S1 to B2); every absolute number below is stated against this session's measured **c** and the same session's ungated `P0L`, never against the stage B numbers directly.

Working hypotheses: **H-gate-lane:** where copies a layer times c exceed the compute window (C = 0.3 ms: 89.46 / 40 x c = 0.43 ms against 0.3), every wasted copy delays a useful one, and a gate that removes 16 % of copies raises the hidden share even though its serial bound is lower. **H-gate-host:** the 14.3 fewer copies save about 14.3 x 19 us = 0.27 ms of host issue time a token, a gain of the same order as the coverage loss at C = 1.0. **H-gate-serial:** where the lane is not saturated (C = 1.0), the hidden share follows the serial bound, so a gate that lowers the bound lowers the hidden share.

## 3. Design (profile `n3c`; built on the Mac after this record is committed)

The stage B harness `benchmarks/cuda/overlap_arena.py` at level **L5** (lean bookkeeping, pre-built views, bare events, a CUDA graph a layer; the best arm). One new arm family: **`G<t>`**, the `P0L` arm (prefetch issued after the layer's launch, no wait) with the recorded predicted set filtered to experts whose recorded weight is at least `t`; `G0` is `P0L` by construction (a gate: the `G0` event sequence equals `P0L`'s, event for event). Prefetch bookkeeping as in the N3 record section 3: gated-out predicted keys are never loaded, so they cannot be covered or wasted; their experts that the layer uses are demand misses.

**Cells (17 % resident, 400 decode tokens, token 0 a warm-up, 2,611 slots):** C = 1.0 ms: `N`, `P0L`, `G0.10`, `G0.20`, `O`, closing `N` (6); C = 0.3 ms: the same six (6); 8 % resident (1,229 slots), C = 1.0: `N`, `P0L`, `G0.20`, closing `N` (4). **16 cells**, estimated 5 to 7 minutes of cell time plus qualification and calibration (stage B: 43 cells in 680 s). Session cap **30 minutes** of GPU time from Ready.

**Gates.** The stage B gates (event hash against an independent reference, determinism, counts, `chain2`, readback, calibration within 5 %, control drift, closure, sanity) plus: (g) each `G<t>` arm's demand misses, prefetch loads, covered and wasted counts equal `prefetch_coverage.py --gates` on the same tokens (a mismatch is a harness bug: fix on the Mac). Measurements and the definition of `E_X` and `H_X` as in the N3 record section 3.

## 4. Predictions (registered before any code)

All **predicted**. Probabilities are my own. `H` values are within their C group, at 17 % unless said.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| C-P1 | counts equal the offline model | demand misses, loads, covered, wasted of `G0.10` and `G0.20` equal the table in section 2 to within 0.01 a token (probability 0.9) | any differs by more than 0.05 |
| C-P2 | `G0.10` is inert | `H_G0.10` within 0.02 of `H_P0L` at C = 1.0 and at C = 0.3 (probability 0.7) | differs by more than 0.04 at either C |
| C-P3 | the gate loses where the lane is not saturated (C = 1.0) | `H_G0.20` lower than `H_P0L` by 0.02 to 0.10 (centre 0.05: the bound falls 0.075 and the host saving gives some back) (probability 0.55) | `H_G0.20` at or above `H_P0L` |
| C-P4 | the gate helps where the lane is saturated (C = 0.3) | `H_G0.20` above `H_P0L` by 0.02 to 0.10, i.e. between 0.12 and 0.20 (probability 0.45) | `H_G0.20` at or below `H_P0L` (the copy lane is not the limit at 0.3 ms either) |
| C-P5 | host issue time | the host issue time a token of `G0.20` is lower than `P0L`'s by 0.15 to 0.40 ms at C = 1.0 (14.3 fewer copies at about 19 us) (probability 0.5) | lower by less than 0.05 ms, or by more than 0.60 |
| C-P6 | interference | the compute-lane busy time at C = 0.3 for `G0.20` lies between 2.5 % and 5.5 % above the floor, below `P0L`'s (probability 0.5) | at or above `P0L`'s |
| C-P7 | 8 % resident | `H_G0.20` below `H_P0L` by 0.02 to 0.12 (bound falls 0.085) (probability 0.5) | `H_G0.20` at or above `H_P0L` |
| C-P8 | no gated arm beats the oracle | `H_O` at C = 1.0 between 0.80 and 0.88 (stage B 0.857) and above every gated arm by at least 0.4 (probability 0.8) | `H_O` below 0.75 (the stage B instance is not reproduced) |
| C-P9 | completion | all 16 cells complete inside 15 minutes of cell time, every gate true, no failed cell (probability 0.7) | any gate false, a failed cell or truncation |

## 5. What each outcome means

* **C-P4 holds (gate helps at 0.3 ms) and C-P3 holds (it costs at 1.0 ms):** the lever is conditional on the window; the useful rule is to gate when the layer time is short, an adaptive rule not measured here. **C-P3 and C-P4 both fail with the gate losing:** the weight gate is closed on this card; the remaining predictor lever is better coverage, not fewer copies. **The gate wins at both C:** its threshold should be swept (0.15 to 0.25) in a following stage.
* **C-P5 failing low:** the 19 microseconds a copy is not proportional to the copy count (a fixed per-layer part dominates), which weakens candidate 2 (fewer, larger copies).
* **C-P1 failing:** harness bug; nothing from the card is scored until fixed on the Mac.
* Nothing here is a statement about the V100, a real model's decode, a native runtime, or another provider's hosts.

## 6. Stop rules and cost

Stop and bring the files home on a failed cell, a false gate, truncation, a drift flag on the first `N` pair, an event-hash mismatch (a harness bug), or 80 % of the cap before the first scored cell. Check `nvidia-smi` and DNS on a new VM before uploading anything; a GPU whose open-driver probe says "no GSP" is replaced, not debugged (v238 rules). Billing: Hamed's private cost file; the agent does not hold figures.

## 7. Needs Hamed

A go for the stage C harness build on the Mac (rehearsal tests asserting every gate) and, afterwards, an RTX 5090 instance. Housekeeping stands (delete the S1, S2, V1, N2, B1, B2, N3 and N3-B ssh configs, known_hosts files and the key; enter the N3 and N3-B billed hours in the cost file).

## 8. Erratum, written 2026-10-19 while building the harness, before any instance and before any GPU run

The predictions in section 4 and the design in section 3 are **not edited**; this section corrects the cell list and says which level each cell uses.

* **Each (level, C) group also runs a `D` arm.** Section 3 listed `N`, `P0L`, `G0.10`, `G0.20`, `O`, `N` without `D`, but every prediction that is a hidden share (C-P2 to C-P4, C-P7) defines `H_X = 1 - E_X / E_D` within the group, so the group needs its own `E_D`. The cells are therefore 19, not 16: C = 1.0 ms at 17 % resident, level L5: `N`, `D`, `P0L`, `G0.10`, `G0.20`, `O`, `N` (7); C = 0.3 ms at 17 %, level L4: the same seven (7); 8 % resident, C = 1.0 ms, level L5: `N`, `D`, `P0L`, `G0.20`, `N` (5). The estimate is 6 to 9 minutes of cells plus qualification and calibration; the 30-minute session cap and C-P9's 15 minutes of cell time stand.
* **C = 0.3 ms cells use level L4, not L5.** Stage B measured C = 0.3 ms only at L0 and L4, so the baseline `H_P0L` = 0.100 quoted in section 2 is an L4 number and a C = 0.3 ms graph replay was never calibrated; C = 1.0 ms cells use L5 (stage B baseline 0.339). Both levels were the best available for their C. If the L5 graph is unavailable on the card the C = 1.0 ms group is reported as such and is not re-run at another level inside the session.
* **Names in the harness:** the arms are `G10` and `G20` (thresholds 0.10 and 0.20); `G00` (threshold 0, equal to `P0L` event for event) exists for the Mac tests only and is not run on the card. Gate (g) is checked twice: when `n3c-expected.json` is written (the Mac refuses to write it on a mismatch) and in every gated cell on the host (gate `coverage`).
