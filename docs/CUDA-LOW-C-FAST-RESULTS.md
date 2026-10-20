# CUDA lane: the compute-slowdown slope at C = 0.15 ms on a fast-population instance (results)

Run 2026-10-20 on a second RTX 5090 VM (`imperfect-duck-7094`; driver 580.159.03, PCIe 5.0 x16, torch 2.7.1+cu128), the repeat Hamed asked for after `docs/CUDA-LOW-C-RESULTS.md` found the first instance in the slow population. Same record (`docs/CUDA-LOW-C-RECORD.md`, ten predictions committed in 823aa63 before any code, unchanged), same code (commit ec7df8f, `dirty=0`). `n3hsmoke` (20 s) then `n3hfull` twice (run A `20261020T202441Z-n3h-h-full-9a0a9d`, run B `20261020T202610Z-n3h-h-full-b0798f`; 12 cells each, 400 decode tokens, 84 s a run). Every gate true in all three runs, no failed cell, nothing truncated; calibration error -1.1 % in the smoke. **Median source-record copy 0.194 ms in both runs: fast population**, so the record's assumptions hold and L-4 to L-10 are scored for adoption. Raw samples (hash-verified copy, sha256 prefix 632c4c9cce3fb849) are under `~/cachalot-private/cuda-results/n3h2/`; scoring output `docs/cuda-results/low-c-fast-2026-10-20.json`. Labels follow plan section 0.

## 1. Scoring the registered predictions

Run A fits, run B scores. All rows level 3, `P0L` against `D`, unless stated.

| id | measured | registered | verdict |
|---|---|---|---|
| L-1 slope on measured overlap, C = 0.15, run A | 0.1534 | 0.140 to 0.185 (falsified outside 0.110 to 0.220) | **held** |
| L-2 which shape | 0.1534 | at least 0.153 (falsified below 0.125) | **held**, by 0.0004: the slope is midway between the affine-in-1/C value 0.165 and the power-law value 0.140 |
| L-3 replicate | run B 0.1525, 0.6 % from run A | within 15 % (falsified beyond 30 %) | **held** |
| L-4 measured hidden share, `P0L`, level 4 | -0.054 | -0.25 to 0.05 (falsified above 0.12 or below -0.40) | **held** |
| L-5 queue model row agreement, run B | r = 0.9936 | at least 0.85 | **held** |
| L-6 total overlap, predicted over measured | 0.888 (3.98 against 4.48 ms a token) | 0.85 to 1.05 | **held** |
| L-7 hidden-share error, row model with its own slope | 0.0434 (-0.010 predicted against -0.054) | at most 0.07 (falsified above 0.12) | **held** |
| L-8 extrapolated V1 slope 0.1833, token busy total error, run B | +0.050 ms (0.729 against 0.679) | at most 0.30 ms | **held** |
| L-9 calibration and floor gates at 0.15 ms | both true in all three runs | hold in every cell | **held** |
| L-10 demand exposure, `D` minus the `N` floor | 8.84 ms a token (run B 8.84) | 9.0 to 11.0 (falsified outside 8.0 to 12.5) | **missed narrowly** (low), inside the falsifier |

**9 held, 1 missed narrowly, 0 falsified.**

## 2. What was measured

* **The slope at 0.15 ms is 0.153 on a fast instance (measured).** Four points on fast instances: 0.0634, 0.0765, 0.1053, 0.1534 at C = 1.0, 0.6, 0.3, 0.15 ms.
* **The 1/C form describes all four points to about 0.003 (derived).** Least-squares affine in 1/C through all four points: `s = 0.0498 + 0.01574 / C`, largest residual 0.0030 (rms 0.0020). A power law in 1/C (exponent 0.468) has rms 0.0027, residuals up to 0.0037; the two are not separated by four points and the replicate spread. **Linear in C is rejected** (rms 0.0151, residuals up to 0.019). The three-point form extrapolated to 0.15 ms overshoots by 0.011 (7 %): the form continues, with a slightly flatter slope than the three-point fit gave.
* **The instance class changes the slope (measured, derived comparison).** On the slow instance (copies 1.8 times dearer) the slope at the same layer time was 0.2234 (`docs/CUDA-LOW-C-RESULTS.md`), 1.46 times this one. The earlier slopes at 1.0, 0.6 and 0.3 ms were measured on fast instances; a slow one would have shifted them too, so the slope is a property of the card population as much as of C.
* **Prefetch no longer hides anything at 0.15 ms (measured).** Hidden share -0.054 at level 4: a layer shorter than one copy cannot absorb a prefetch batch; the count-only model with the fast copy time said -0.085, the slow instance -0.279 (model -0.286).
* **The unconditional predicted-overlap form holds at a layer time outside its fitted range (measured).** Row agreement 0.9936, hidden-share error 0.0434, level-6 total 0.012 ms off, an extrapolated slope within 0.050 ms of the token total; the total overlap is 11 % low (0.888), inside the registered interval.
* **Demand exposure was lower than registered (measured).** 8.84 ms a token against the 9.9 to 10.1 ms of stage A at every layer time; at 0.15 ms the `N` floor is 6.15 ms (40 layers x 0.15), and the demand arm's wall is shorter by about 1 ms than the earlier layer times' exposures. Not explained; one instance.

## 3. What it means, and does not

* **H-continues is supported on the fast population down to 0.15 ms** by the registered rule (L-1 not falsified, L-3 and L-9 held), with the shape between affine in 1/C and a power law of exponent about 0.47; linear in C is out.
* It does not identify the mechanism; it says the slope keeps rising as the layer shortens and the copy stream becomes saturated, and that it depends on the card population.
* **Limits.** One instance of the fast population plus one of the slow; one trace, 17 % resident, a stand-in compute whose layer is shorter than a real layer on this card; replicate spread 0.6 % (same instance). Nothing concerns another card or the real kernels.

## 4. Decision (record section 5)

* L-1 not falsified, L-3 and L-9 held: **the 1/C form is described as continuing to 0.15 ms** on this card, trace and 17 % resident, fast population. L-2 held, so the form reported is affine in 1/C over 0.15 to 1.0 ms: `s = 0.0498 + 0.01574 / C` (least squares, four points, rms 0.002), with the power law (exponent 0.47) as an equally good description.
* L-5 and L-7 held: **the unconditional predicted-overlap form is extended to C = 0.15 ms** (fast population).
* L-8 held: **extrapolated slopes are usable one step outside the measured range** on this card (error 0.05 ms), with the three-point extrapolation 7 % high at the slope itself.
* Population rule: a slow instance gives a different slope; the form is reported for the fast population only.
* Nothing changes in `src/` and no default moves. The fit describes a stand-in compute stream.
* **Open, needs a GPU and Hamed's go:** a point above 1.0 ms (3 ms); another card; fewer and larger copies; a native issue path; ROCm; the population effect (is the slope proportional to copy cost: a controlled copy-cost sweep on one instance).
