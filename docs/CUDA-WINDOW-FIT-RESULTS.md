# CUDA lane: a host-aware row-level window model, fitted and validated (results; Mac only, no GPU)

Run 2026-10-20 under `docs/CUDA-WINDOW-FIT-RECORD.md` (seven predictions committed in 5e59faa before any fitting code was run). Script `benchmarks/cuda/window_fit.py` (3 tests), raw output `docs/cuda-results/window-fit-2026-10-20.json`. Data: the recorded cells of stage C (old trace), stage D (new trace) and the bridge (both traces, two runs each): 19 `P0L` group-runs (and as many `D`) and 14 held-out arm cells. Labels follow plan section 0: everything here is **derived** from a model fitted to measured cells; no GPU was used. Two corrections to the record, both of the record's own counting: it said four oracle cells and 12 bridge group-runs; there are five oracle cells (stage C 1.0 and 0.3 ms, stage D 1.0, 0.6, 0.3 ms) and 12 bridge group-runs plus 7 in stages C and D, 19 `P0L` group-runs in all.

## 1. The fit

Step 1 (the `D` arm, 19 group-runs): `c` = **0.190 ms**, `L_dem` = **0.0 ms** (the grid's lower edge). Step 2 (the `P0L` arm): `L_pf` = **-0.025 ms**. So the two host terms the model was built around fit to about zero: the fitted model is the copy-count model with `c` = 0.190 and a slightly longer effective window. Mean absolute error in hidden share on the fit cells: 0.026.

## 2. Results (error = `|H_model - H_measured|`, measured `E_D` in the denominator)

| group of cells | cells | fitted model: mean error | copy-count-only screen: mean error |
|---|---:|---:|---:|
| fit cells (`D`, `P0L`) | 38 | **0.026** | (not fitted) |
| held out, gated (`G10`, `G20`) | 5 | **0.026** (all at most 0.044) | 0.038 (one 0.111) |
| held out, oracle `O` | 5 | 0.100 (0.148, 0.037, 0.142, 0.147, 0.027) | 0.094 |
| held out, lead 2 `L2` | 4 | **0.187** (0.058, 0.106, 0.274, 0.308) | 0.166 |
| held out, all | 14 | **0.0986** | **0.0944** |

`P0L` errors above 0.05: six of 19 group-runs, all on the new trace at C = 1.0 and 0.6 ms (0.073 to 0.094; the model overstates the new trace) and stage D's own two (0.077, 0.079); the old trace's `P0L` is within 0.047 everywhere.

## 3. Scoring the registered predictions

| id | outcome |
|---|---|
| WF-1 (`c` 0.17 to 0.23, `L_dem` 0 to 0.30, `E_D` residual at most 0.35 ms) | **held**: 0.190, 0.0, largest residual 0.30 ms |
| WF-2 (`L_pf` 0 to 0.25; `P0L` error at most 0.05 in 80 % of group-runs) | **missed**: -0.025 (inside the falsifier range) and 13 of 19 (68 %) within 0.05 (falsifier at most 50 %: not fired) |
| WF-3 (gated arms: at least 75 % of cells within 0.05) | **held**: 5 of 5 |
| WF-4 (oracle: at least 3 of 4 cells within 0.08) | **missed**: 2 of 5; the falsifier (any above 0.15) did not fire (largest 0.148) |
| WF-5 (lead 2: at least 3 of 4 cells within 0.08; the model overstates when it errs) | **falsified**: 1 of 4 within 0.08 and two errors above 0.15 (0.274, 0.308); the direction (overstating) held |
| WF-6 (held-out mean error below half the screen's) | **falsified**: 0.0986 against 0.0944 (not below 0.8 times) |
| WF-7 (fit-cell mean error at most 0.04) | **held**: 0.026 |

**3 held, 2 missed, 2 falsified.**

## 4. Decision rules (fixed in the record)

The rule to replace the copy-count-only screen required WF-3 **and** WF-6; WF-6 fails, so **the screen stays** with its error of about 0.1 and no host-aware model is adopted. The second rule applies: the model is **not used for an arm whose prefetch spans more than one layer's window** (`L2`: its errors are 0.06 to 0.31 and grow as the layer time shrinks). The fitted model is better than the screen on the gated `P0L`-family arms (0.026 against 0.038 on five cells), which the record's rule does not count; this is reported as an observation, not an adoption.

## 5. What the result is, and is not

* **Derived: the fixed host terms are not what is missing.** `L_dem` fits to 0 and `L_pf` to about 0: a constant per row (a launch or event latency) is not visible in the exposure. The errors are structural. On the new trace at 0.6 and 1.0 ms the measured exposure of `P0L` is 0.7 to 0.8 ms above the model's (4.9 against 4.2 ms), i.e. the demand misses that remain after prefetch cost about 0.22 ms each against 0.187 ms for `D`'s misses on the same trace (old trace: 0.197 against 0.189): the misses prefetch leaves behind are dearer than the ones it never saw (hypothesis: they cluster in rows whose copy queue is already busy, or arrive behind in-flight prefetch copies; not isolated).
* **Derived: the oracle's floor is not in the model.** At 0.6 ms and above the model hides almost all of the oracle's copies (error 0.14 to 0.15), while measured exposure keeps 1.2 to 1.7 ms (the router synchronization and host issue the oracle still pays every layer). That floor, not copy time, limits `O`.
* **Derived: copy-stream contention across layers is what breaks `L2`.** The model gives `L2` a window of two layers; the measured arm is far worse at short layers, as in stage D (hidden share 0.079 at 0.3 ms against the model's 0.35).
* **Consequence for the lane.** Keep the copy-count-only screen for trace comparisons within the `P0L` family, with an error of about 0.1; do not use either model for multi-layer prefetch or for the oracle; a GPU stage is still the only test of an arm whose prefetch spans layers. The structural term to model next (a per-miss cost that depends on queue state) is a candidate for a later offline record, not a GPU stage.
* **Not shown.** Any other card, host CPU, stand-in or native issue path; whether a queue-state term would explain the new trace's per-miss cost.
