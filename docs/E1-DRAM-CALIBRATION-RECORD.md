# E1 record: DRAM energy against bytes moved (charter L5b, `docs/POWER-ACCOUNTING-PLAN.md` step E1)

Fields 1-3 written 2026-10-10, before the run, against runtime 0.62.31. Result and explanation are appended after the run (HANDOFF 18.119).

## 1. Baseline

No model. Idle-ish machine (display on, apps open): IOReport `DRAM0_0 + DRAM0_1` read 1.17-1.25 W, `DCS0_n` 1.6-1.7 W, `AMCC0_n` 2.0-2.3 W, with a root `powermetrics` sampling beside (HANDOFF 18.118, 5 s means, one reading each). SMC `PSTR` 22-40 W, noisy. Workload class: none of W1-W7; a synthetic streaming read (GPU `mx.sum` over distinct 1 GiB bf16 buffers, 4 GiB in all, so no launch reads what the last left in the system-level cache).

## 2. Hypothesis and prediction (tags: prediction)

Hypothesis: DRAM energy is roughly linear in bytes moved. The M3 Ultra uses LPDDR5X; typical interface-plus-array energy is 4-6 pJ per bit, i.e. 32-48 pJ per byte.

Predictions, with an interval:
- Slope of `DRAM0_0 + DRAM0_1` watts against achieved GB/s: **30-50 pJ per byte** (0.030-0.050 W per GB/s). At the ~350-400 GB/s a 100 % duty stream reaches (micro_read_size_roofline: `mx.sum` 385 GB/s at 34 MiB), DRAM power rises by about 11-20 W above idle.
- `AMCC` and `DCS` scale with bytes too (each slope above 5 pJ per byte); if their sum with DRAM is the total memory-subsystem energy, the three together give 50-100 pJ per byte. Whether they are inside or beside DRAM cannot be decided from this: only whether each moves with bytes.
- Cross-check: (`PSTR` rise) minus (CPU + GPU + ANE rise) lies within 30 % of the DRAM + DCS + AMCC rise.
- Linearity: R-squared of DRAM watts on GB/s at least 0.95 over the nine duty points.

## 3. Falsifier

Any of: DRAM slope below 10 or above 100 pJ per byte; R-squared below 0.9 (a non-linear or saturating curve); AMCC and DCS flat against bytes (slope within noise of zero), which would say they measure something else; the PSTR cross-check off by more than 50 %; the counters not advancing (the sampler was not running), which voids the run, not the hypothesis.

## 4. Intervention

None to the runtime. New instrument `benchmarks/dram_calibration.py`: duty-cycled GPU streaming at 0, 10, 25, 50, 75, 100 % in two passes (ascending-descending order swapped) of 30 s a step, the first 5 s of each step excluded; IOReport counters and `PSTR` read once a second by a user-level reader. Needs Hamed's `sudo powermetrics` sampling for the whole run (about 10 minutes).

## 5. Result (measured, 2026-10-10, run 01:44-01:50, `benchmarks/results/energy/e1-dram.jsonl`, sampler `e1-sampler.txt`)

Twelve steps (two passes, order swapped), means over the last 25 s of each 30 s step. Achieved rates: 0, 70, 176, 361, 543, 735 GB/s (the full stream reaches 735 GB/s, not the 350-400 predicted; the paired steps agree to within 1 %).

| duty | GB/s | dram W | dcs W | amcc W | cpu+gpu+ane W | PSTR W |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (A, B) | 0 | 1.28, 1.84 | 1.77, 2.82 | 2.21, 3.81 | 2.7, 4.8 | 26.1, 32.8 |
| 0.10 | 70 | 5.2-5.4 | 3.2-3.6 | 5.0-5.8 | 7.0-9.8 | 39-45 |
| 0.25 | 176 | 10.3-10.5 | 4.5-4.8 | 8.1-8.8 | 13.5-13.6 | 55-57 |
| 0.50 | 361 | 19.4 | 6.7-6.9 | 14.5-14.9 | 25.8-26.6 | 80-83 |
| 0.75 | 543 | 27.9-28.1 | 8.8 | 20.3-20.4 | 37.8-38.0 | 105-106 |
| 1.00 | 735 | 38.8 | 11.3 | 27.2 | 52.3-52.8 | 133-134 |

Fits of watts on GB/s: **dram 49.9 pJ per byte** (intercept 1.55 W, R^2 0.999); **dcs 11.95 pJ/B** (R^2 0.993); **amcc 32.6 pJ/B** (R^2 0.997); dram+dcs+amcc **94.5 pJ/B** (R^2 0.998); cpu+gpu+ane 65.8 pJ/B (that is the GPU doing the streaming); PSTR 138.9 pJ/B (R^2 0.996).

Against the prediction: slope 30-50 held (49.9, at the edge); AMCC and DCS above 5 held; the sum 50-100 held (94.5); R^2 at least 0.95 held; the PSTR cross-check held (PSTR slope minus the SoC slope is 73.1 pJ/B against 94.5, 23 % low, inside 30 %); the sustained rate of 350-400 GB/s was **missed by a factor of two** (735), so the predicted DRAM rise of 11-20 W was also missed (it is 37.5 W above idle) while the slope, which does not depend on the rate, held.

Coverage (sum of the IOReport components cpu, gpu, ane, dram, dcs, amcc, display against PSTR) rises with load: 0.32-0.42 at idle, 0.53-0.56 at 70 GB/s, 0.67 at 176, 0.83 at 361, 0.90 at 543, 0.97-0.98 at 735. The part of PSTR the components do not explain falls from 18-19 W at idle to 2.5-3.5 W at the full stream.

## 6. Explanation

DRAM energy is linear in bytes moved over 0-735 GB/s on this Mac: about 50 pJ per byte (6.2 pJ per bit), the upper end of the LPDDR5X range assumed in the hypothesis. The two memory-controller channels also scale (AMCC 33 and DCS 12 pJ/B), so memory traffic costs about 94 pJ/B counting all three; whether DCS and AMCC are inside or beside `DRAM0_n` is **still not decided** by this run (all three scale; their sum is a plausible total, but a sum with an inside channel would double count). The shrinking unexplained residual is **not explained**. A constant remainder (SSD, fans, PSU losses) cannot shrink with load, so one source is nonlinear or two channels overlap at load; candidates are PSTR lagging or saturating, the GPU channel overlapping DCS or AMCC, or the DC-input rail including a term that falls as the SoC draws more. It was not tested.

## 7. Generalization

Hardware-specific (M3 Ultra, two dies, LPDDR5X, one OS build) and workload-specific (a sequential GPU read of a 4 GiB buffer; the streaming kernel reaches 735 GB/s, far above any decode kernel's 29-45 % of peak). The transferable statement is only the form: memory energy per byte moved is constant over the rate range, so a token's DRAM energy is its bytes moved times about 50 pJ (about 94 pJ with the controllers), not a fixed share of package power. Whether a random-access or small-read pattern costs the same per byte was not measured.

## 8. Bottleneck after

Unchanged for speed. For the energy lane the missing component is the SSD (E2) and the unexplained, load-dependent residual of section 6.

## 9. Kind

Research (predicted, measured, explained in part): the slope, linearity, controller scaling and cross-check predictions held; the rate and watts prediction missed by 2x and is recorded as missed.
