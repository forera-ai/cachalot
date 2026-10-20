# CUDA lane: the compute-slowdown slope at a third layer time, C = 0.6 ms (results)

Run 2026-10-20 on one RTX 5090 VM (driver 580.159.03, PCIe 5.0 x16, torch 2.7.1+cu128, fast population: source-record copy 0.194 ms) under `docs/CUDA-THIRD-C-RECORD.md` (nine predictions committed in 32bd9e2 before any code; erratum in its section 7 before any run). `n3gsmoke` (8 cells, 40 tokens, 25.6 s) then `n3gfull` twice (run A `20261020T171039Z-n3g-g-full-4129c5`, run B `20261020T171329Z-n3g-g-full-120c92`; 12 cells each, 400 decode tokens, about 165 s a run). Every gate true in all three runs (`event_hash`, `determinism`, `counts`, `chain2`, `readback`, `calibration`, `drift`, `closure`, `sanity`, `coverage`, `floor`, `rowsum`, `copyevents`), no failed cell, nothing truncated; the N-control floor was 24.18 ms against 40 layers x 0.6 ms. Raw samples (hash-verified copy, 5.4 MB archive) are under `~/cachalot-private/cuda-results/n3g/`; scoring output `docs/cuda-results/third-c-2026-10-20.json` from `benchmarks/cuda/third_c_score.py`. Labels follow plan section 0: **measured** on the recorded rows; fits are **derived**; explanations are **hypothesis**.

## 1. Scoring the registered predictions

Run A fits, run B scores. All rows level 3, `P0L` against `D`, unless stated.

| id | measured | registered | verdict |
|---|---|---|---|
| G-1 slope on measured overlap, C = 0.6, run A | 0.0765 | 0.070 to 0.095 (falsified outside 0.060 to 0.110) | **held** |
| G-2 which shape | 0.0765, below 0.085 | below 0.085 (falsified at or above 0.091) | **held** |
| G-3 replicate | run B 0.0747, 2.4 % from run A | within 15 % (falsified beyond 30 %) | **held** |
| G-4 measured level-4 hidden share, `P0L` | 0.4065 | 0.37 to 0.45 (falsified outside 0.33 to 0.48) | **held** |
| G-5 row agreement of predicted and measured overlap, run B | r = 0.9974 | at least 0.85 | **held** |
| G-6 total overlap, predicted over measured | 0.9974 (9.27 against 9.30 ms a token) | 0.85 to 1.05 | **held** |
| G-7 hidden-share error, queue model with its own fitted slope (0.0767) | 0.0186 (0.425 predicted against 0.407 measured) | at most 0.05 (falsified above 0.08) | **held** |
| G-8 interpolated V1 slope (0.0767), token busy total error, run B | +0.017 ms (0.711 against 0.693) | at most 0.15 ms | **held** |
| G-9 level 6, level-6 parameters and the level-3 slope | +0.019 ms (0.716 against 0.696) | within 0.15 ms | **held** |

**9 held, 0 missed, 0 falsified.**

## 2. What was measured

* **The slope at 0.6 ms sits on the 1/C curve (derived).** Through the slopes 0.0634 (1.0 ms) and 0.1053 (0.3 ms) the three shapes give, on the measured basis, 0.0873 (linear in C), 0.0786 (power law, exponent 0.42) and 0.0753 (affine in 1/C). The measured 0.0765 (run B 0.0747) is 1.6 % above the affine-in-1/C value, 2.7 % below the power law and 12 % below linear in C. Linear in C is the one shape the measurement separates from the others; the power law and the 1/C form are not separated by a point 3 % apart on an interval of about 2.4 % replicate spread. The same holds on the V1 basis: V1's slope at 0.6 is 0.0767 against the interpolated 0.0767.
* **The unconditional predicted-overlap form holds at a layer time it was not built on (measured).** Row agreement 0.997, overlap total 0.997 of measured, hidden-share error 0.0186 (the other two layer times: 0.0457 and 0.0067), closed-loop total 1.011 times open-loop.
* **An interpolated slope is usable between 0.3 and 1.0 ms on this card (derived).** The slope affine in 1/C through V1's two known values predicts the 0.6 ms token total to +0.017 ms.
* **The hidden share against C (measured).** Level-4 shares 0.423 (1.0), 0.407 (0.6), 0.232 (0.3): it falls slowly from 1.0 to 0.6 and fast below, on this instance and on the bridge session's instance (0.407).

## 3. What it means, and does not

* **It does not explain why the slope depends on C (hypothesis).** A slope affine in 1/C is what a fixed extra time per overlapped copy burst would give (extra compute per unit of overlapped window growing as the window shortens); the record's two mechanisms (memory contention, copy-engine scheduling) are not separated by three points, and no test here distinguishes them.
* **The agreement is partly by construction.** The harness's copy stream is a deterministic first-in first-out queue with nearly constant copy times (0.196 ms), so a FIFO model with fitted costs should place it well: r 0.997 is evidence the model matches this instrument, as in `docs/CUDA-PREDICTED-OVERLAP-RESULTS.md`.
* **Limits.** One instance (fast population), one trace, 17 % resident, a stand-in compute, three layer times; two runs on the same instance, so replicate agreement (2.4 %) is the scale, not an interval. Nothing concerns another card, another prefetch family, another resident fraction, or the real model's kernels.

## 4. Decision (record section 5)

* G-1 not falsified and G-3 held: **the slope is described as a function of C** on this card, trace and 17 % resident, fast population: `s = a + b / C`, a = 0.0454, b = 0.0180 on measured overlap (a = 0.0411, b = 0.0213 on V1's), fitted through 1.0 and 0.3 ms and confirmed at 0.6 ms to 1.6 %. Labelled interpolation between 0.3 and 1.0 ms only.
* G-5 and G-7 held: **the unconditional predicted-overlap form is extended to C = 0.6 ms.**
* G-8 held: **interpolated slopes are usable** between 0.3 and 1.0 ms on this card.
* Nothing changes in `src/` and no default moves; the fit describes a stand-in compute stream.
* Open and needing a GPU and Hamed's go: whether the 1/C form continues below 0.3 ms or above 1.0 ms (a point at 0.15 or 3 ms), another card, fewer and larger copies, a native issue path, ROCm, instance population. Offline: nothing that moves a number.
