# E3 record: joules a token with every component (charter L5b, `docs/POWER-ACCOUNTING-PLAN.md` step E3, DeepSeek arm)

Fields 1-4 written 2026-10-10, before the run, against runtime 0.62.34. Result and explanation are appended after the run.

## 1. Baseline

The 0.62.29 arms (HANDOFF 18.116, LEDGER ENERGY-DECODE), DeepSeek exact decode, internal bank (`~/DeepSeek-V4.1-Flash-q2g128`, 48 GiB budget), CPU+GPU+ANE package power only: all-resident (one prompt repeated, about 82 ms a token) 20.94 W (GPU 16.28, CPU 4.66), 1.726 J a token; read-bound (fresh prompts, about 23 misses a token, about 138 ms a token) 18.29 W (GPU 10.94, CPU 7.35), 2.524 J a token; ratio 1.41-1.51 (1.46 raw). Idle server 3.2-5.0 W. Since then: DRAM is 49.9 pJ per byte (18.119), system power `PSTR = 18.4 W + 1.27 cpu + 1.24 gpu + 0.34 dram + 1.04 (dcs+amcc)` (18.120), the X10Pro adds 2.71 W reading at 1 GB/s (18.121), the internal drive is unmeasured.

## 2. Hypothesis and prediction (tags: prediction)

Hypothesis: the E1b fit, made on synthetic loads, transfers to a real decode, so system energy per token is measured system power times token time, and the package-only ratio of 1.46 between a read-bound and an all-resident token understates the system ratio, because the 18 W baseline and the memory terms scale with time and bytes, not with package power.

Predictions (arms as in 0.62.29: A idle server 90 s, B all-resident 120 s, C read-bound 150 s, D idle 90 s; the sampler beside):
- Mean system power (SMC `PSTR`): B **55 W (45-65)**, C **50 W (42-60)**; idle phases 26 W (20-33).
- System energy a token (PSTR times seconds a token): B **4.5 J (3.7-5.4)**, C **7.0 J (5.7-8.5)**.
- Ratio C over B **1.55 (1.4-1.7)**, against 1.46 for package power alone.
- Transfer of the fit: the E1b fit evaluated on the measured IOReport components of a phase predicts that phase's PSTR within **4 W** (7 %).
- The internal drive: PSTR minus the fit in C, less the same in B, is the drive's mean power in the read-bound phase; predicted between **-1 and +4 W** (the fit's noise is 1.3-1.4 W, so it may not be resolved).

## 3. Falsifier

Any of: the fit's prediction of a phase's PSTR off by more than 8 W (15 %), so the fit does not transfer to decode; the ratio C over B outside 1.3-1.8; a system energy a token outside the ranges above by more than 25 %; a sampler not running during the run (void).

## 4. Intervention

None to the runtime. New: `benchmarks/energy_logger.py` (a user-level logger of IOReport energy by component, `PSTR`, and the server's `/v1/stats` counters, once a second, started after the server is up), `benchmarks/energy_e3.py` (per-phase analysis against the driver's phase clock times), and `benchmarks/energy_arms.sh` starts and stops the logger. Needs Hamed's `sudo powermetrics` sampling for the whole run (about 15 minutes; the run is about 13).

## 5. Result, explanation, generalization, bottleneck after, kind

To be written after the run.
