# CUDA lane: the bridge session (results; the higher P0L hidden share is the trace; RTX 5090)

Run 2026-10-20 under `docs/CUDA-BRIDGE-EXPERIMENT-RECORD.md` (seven predictions committed in baf6a88 before any bridge code). Harness: `benchmarks/cuda/overlap_arena.py` at scaffold commit 635f4b6 (`REVISION` dirty=0), runtime 0.66.41, the profiles `e-smoke` and `e-full`. Instance: the stage D virtual machine (ID `32db98a4`, still up after stage D, GPU idle at the start: RTX 5090, driver 580.159.03, torch 2.7.1+cu128, PCIe 5.0 x16). Raw rows, manifests, samples and the scoring output: `docs/cuda-results/bridge-rtx5090-2026-10-20/` (`score.json` from `bridge_score.py`). Labels follow plan section 0: measured on this one instance unless it says derived or hypothesis. The compute is a stand-in and the host a Python loop over torch.

## 1. What ran

`check`, `doctor`, `smoke` clean; `n3esmoke` on both traces (old `f961e9e7`, new `b68aefa4`) clean; `n3ebridge`: four runs of `e-full` (12 cells each, 17 % resident, 400 decode tokens; arms `N`, `D`, `P0L`, `N`; C = 1.0 ms at L5, 0.6 and 0.3 ms at L4) in the order **old, new, new, old**: 157, 170, 170, 157 s, 48 cells in 653 s, **every gate true in every run**, no failed cell, none truncated. Source records c = 0.194 ms. The old trace in this session also gives stage B and C's baseline on a different day and instance, so it is a bridge to those stages.

## 2. Results (hidden share of `P0L`, `H = 1 - E_P0L / E_D`; the two runs of each trace; p50 per token)

| C, level | old trace runs | old mean | new trace runs | new mean | new minus old | row-level model old / new |
|---|---|---:|---|---:|---:|---|
| 1.0 ms, L5 | 0.339, 0.340 | **0.339** | 0.430, 0.435 | **0.433** | **+0.093** | 0.363 / 0.533 |
| 0.6 ms, L4 | 0.308, 0.303 | **0.305** | 0.403, 0.410 | **0.407** | **+0.101** | 0.303 / 0.481 |
| 0.3 ms, L4 | 0.102, 0.094 | **0.098** | 0.234, 0.232 | **0.233** | **+0.135** | -0.001 / 0.216 |

`E_D` (demand-only exposure): old 9.54 to 9.68 ms, new 8.78 to 8.89 ms in every group. The two runs of a trace in a group differ by at most 0.008 in `H` (the order old, new, new, old shows no drift).

## 3. Scoring the registered predictions

| id | outcome |
|---|---|
| BR-1 (0.3 ms: old 0.04 to 0.16, new 0.19 to 0.28, difference 0.10 to 0.25) | **held**: 0.098, 0.233, +0.135 |
| BR-2 (1.0 ms: old 0.30 to 0.38, new 0.40 to 0.48, difference 0.06 to 0.18) | **held**: 0.339, 0.433, +0.093 |
| BR-3 (0.6 ms: old 0.22 to 0.36, new 0.36 to 0.48, difference 0.08 to 0.22) | **held**: 0.305, 0.407, +0.101 |
| BR-4 (the model within 0.10 of measured in at least 4 of 6 cells) | **held**: 5 of 6 (0.024, 0.100 [new, 1.0 ms, 0.1004 against the 0.10 line: out], 0.002, 0.074, 0.0997, 0.017) |
| BR-5 (the two runs of a cell within 0.04) | **held**: at most 0.008 |
| BR-6 (`E_D` old 9.2 to 9.9, new 8.4 to 9.2 ms) | **held**: 9.54 to 9.68 and 8.78 to 8.89 |
| BR-7 (48 cells inside 25 minutes, gates true) | **held**: 653 s |

**7 held, 0 missed, 0 falsified.** (BR-4 holds by one cell on the margin.)

## 4. What the result is, and is not

* **Measured: the difference is the trace, not the instance or the day.** On one instance in one session the same predictor on the same card hides 0.09 to 0.14 more on the multi-lead trace than on the old one, at every layer time, with run-to-run spread under 0.01. The old trace reproduces stages B and C's numbers on a different instance and day (1.0 ms: 0.339 against 0.339; 0.3 ms: 0.098 against 0.096 to 0.100), so **the harness reproduces across sessions to about 0.005 at these cells** and stage D's baseline was right for its trace. Stage D's lead-2 conclusion is unaffected (its arms ran on one trace).
* **Derived: copy counts explain the sign and most of the size.** The row-level window model (copy counts only, c = 0.194 ms, no host term) gets the direction at all three layer times and the old trace's values within 0.024 at two of three cells; it **overstates the new trace's gain at 1.0 and 0.6 ms** (0.533 against 0.433, 0.481 against 0.407) and puts the 0.3 ms old-trace value at -0.001 against 0.098. So row-level copy counts are a good first screen for a trace, not an exact predictor; the missing terms (host issue, event overhead, interference) are about 0.07 to 0.10 at 0.6 to 1.0 ms and about 0.10 the other way for the old trace at 0.3 ms (hypothesis; not separated).
* **Not shown.** Why the new conversation's routing is friendlier (fewer demand misses a token, 47.2 against 50.5, and a higher recorded-predictor bound 0.535 against 0.365: it is a different conversation and different generated text); another conversation; a real model.
* **Consequence for the lane.** Cross-trace hidden-share comparisons carry the trace; comparisons inside one trace and one session (stage C's gate, stage D's lead 2) stand. The row-level model is kept as a cheap screen with its error stated. The rule added in v246 stands: a stage that changes the trace carries a bridge cell.

## 5. Cost and housekeeping

GPU time on the instrument: smokes 41 s, the bridge 653 s plus setup; the VM was reused from stage D (up since then). Billed hours are Hamed's to enter. He terminates instances (this one is still up); the ssh configs and known_hosts files for every earlier instance, and the key, are his to delete.
