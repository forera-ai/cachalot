# CUDA lane: the compute-slowdown slope at C = 0.15 ms (results)

Run 2026-10-20 on one RTX 5090 VM (driver 580.159.03, torch 2.7.1+cu128, PCIe 5.0 with a **x8** link width reported by `check`; earlier sessions' link width was not recorded in this form) under `docs/CUDA-LOW-C-RECORD.md` (ten predictions committed in 823aa63 before any code). `n3hsmoke` (8 cells, 40 tokens, 24 s) then `n3hfull` twice (run A `20261020T201101Z-n3h-h-full-9abd8a`, run B `20261020T201257Z-n3h-h-full-191ed2`; 12 cells each, 400 decode tokens, 110 s a run). Every gate true in all three runs (`event_hash`, `determinism`, `counts`, `chain2`, `readback`, `calibration`, `drift`, `closure`, `sanity`, `coverage`, `floor`, `rowsum`, `copyevents`), no failed cell, nothing truncated; the stand-in calibrated to 0.15 ms within 1.6 % (221 MB of weights, not capped). Raw samples (hash-verified copy, 5.2 MB archive, sha256 prefix a0f62535f43f33c9) are under `~/cachalot-private/cuda-results/n3h/`; scoring output `docs/cuda-results/low-c-2026-10-20.json` from `benchmarks/cuda/low_c_score.py`. Labels follow plan section 0: **measured** on the recorded rows; fits are **derived**; explanations are **hypothesis**.

**This instance is in the slow population.** The median source-record copy was 0.349 ms in both runs (record section 5 puts the line at 0.30 ms; the fast population of the earlier sessions reads 0.194), and the fitted per-copy cost is 0.355 ms (demand) and 0.346 ms (prefetch) against 0.195 to 0.200 at the third-C session. The record's rule applies: L-4 to L-10 are reported but not used for adoption, and a fast-class instance is to be requested if Hamed agrees. L-1 to L-3 are not covered by that clause, but they were measured on a card whose copies cost 1.8 times as much as the card behind the three earlier slopes, so what they say about H-continues is confounded with the instance.

## 1. Scoring the registered predictions

Run A fits, run B scores. All rows level 3, `P0L` against `D`, unless stated.

| id | measured | registered | verdict |
|---|---|---|---|
| L-1 slope on measured overlap, C = 0.15, run A | 0.2234 | 0.140 to 0.185 (falsified outside 0.110 to 0.220) | **falsified** (by 0.003) |
| L-2 which shape | 0.2234, at least 0.153 | at least 0.153 (falsified below 0.125) | **held** (above every shape: affine in 1/C 0.165, power law 0.140, linear in C 0.110) |
| L-3 replicate | run B 0.2205, 1.3 % from run A | within 15 % (falsified beyond 30 %) | **held** |
| L-4 measured hidden share, `P0L`, level 4 | -0.279 | -0.25 to 0.05 (falsified above 0.12 or below -0.40) | **missed narrowly**, inside the falsifier |
| L-5 queue model row agreement, run B | r = 0.9946 | at least 0.85 | **held** |
| L-6 total overlap, predicted over measured | 0.799 (3.97 against 4.98 ms a token) | 0.85 to 1.05 (falsified outside 0.65 to 1.25) | **missed narrowly**, inside the falsifier |
| L-7 hidden-share error, row model with its own slope | 0.0607 (-0.218 predicted against -0.279) | at most 0.07 (falsified above 0.12) | **held** |
| L-8 extrapolated V1 slope 0.1833, token busy total error, run B | -0.329 ms (0.728 against 1.058) | at most 0.30 ms (falsified above 0.80) | **missed narrowly**, inside the falsifier |
| L-9 calibration and floor gates at 0.15 ms | both true in all three runs, calibration error 1.6 % | hold in every cell | **held** |
| L-10 demand exposure, `D` minus the `N` floor | 15.53 ms a token (run B 15.52) | 9.0 to 11.0 (falsified outside 8.0 to 12.5) | **falsified** |

**5 held, 3 missed narrowly, 2 falsified.**

## 2. What was measured

* **The harness can be tuned to 0.15 ms (measured).** Calibration within 1.6 %, floor gate true, both runs; L-9, the instrument prediction, held.
* **The slope at 0.15 ms is 0.223, well above the affine-in-1/C value 0.165 (measured; derived comparison).** A four-point affine refit through all four slopes has a largest residual 0.0166 (at 0.3 ms) against 0.0007 for the three-point fit; the curve no longer describes the points. Registered as a falsification of L-1 and, under the record's decision rule, H-continues is **not supported on this instance**.
* **The slow instance explains most of it, post hoc (derived, hypothesis).** The count-only window model (`window_model.py`) run with this instance's copy time (0.355 ms) puts the hidden share at **-0.286** at C = 0.15, against the measured -0.279; with the fast-population copy time (0.1957 ms) it gives -0.085. Demand exposure scales the same way: 15.5 ms measured, 16.8 ms from the model with 0.355 ms copies, against about 10 ms on the fast instances. The model was not run with this copy time before the measurement; the registered predictions assumed the fast population, and the hidden-share and exposure misses are the copy time, not the form.
* **The queue model's structure holds (measured).** Row agreement 0.9946 on an instance and layer time it was not built for; its total overlap is 20 % low (0.799) and the level-6 total is 0.057 ms off (the third-C session 0.019).
* **A slow copy raises the slope or the layer time lowers it, and two points are confounded (hypothesis).** Both the instance class and C moved at once; this session cannot say how much of 0.223 against 0.165 is the shorter layer and how much is the 1.8 times slower copy.

## 3. What it means, and does not

* The 1/C form is **not** established below 0.3 ms. Neither is it refuted: the one point is on a slower population of the card, whose copies were not the ones the three earlier slopes were measured with.
* It does not identify the mechanism. The measured slope rising above the curve is compatible with the copy stream being permanently saturated (every copy overlaps the compute) and with slow-population copies each occupying memory longer.
* **Limits.** One instance, slow population, x8 link width, one trace, 17 % resident, a stand-in compute; two runs on the same instance (replicate spread 1.3 %). Nothing concerns the real model's kernels or another card.

## 4. Decision (record section 5)

* L-1 falsified with L-9 true: **H-continues is not supported as registered; the slope is reported at four points with no function fitted** (0.0634, 0.0765, 0.1053 on fast instances; 0.2234 on a slow one). The slow-population rule makes L-4 to L-10 informational, not adopted.
* **The unconditional predicted-overlap form is not extended to 0.15 ms** (L-6 and L-8 missed; L-5 and L-7 held, but under the slow rule they are not used). It stays adopted at 1.0, 0.6 and 0.3 ms.
* Extrapolated slopes are not usable one step outside the range (L-8 missed).
* Nothing changes in `src/` and no default moves.
* **Open, needs a GPU and Hamed's go:** repeat the point on a **fast-class** instance (median source copy below 0.25 ms), since the question this record asked is only answered by a fast-class slope at 0.15 ms; the fifth C (0.2 ms) named by the record for a falsified L-1 is better chosen after that. Housekeeping unchanged (terminate the instance, billed hours, delete the `h1` and earlier ssh configs and known_hosts).
