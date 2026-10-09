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

## 5. Result, explanation, generalization, bottleneck after, kind

To be written after the run.
