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

## 5. Result (measured; 2026-10-10, run 02:37:50-02:46:19, Cachalot Lab closed; `benchmarks/results/energy/e3-log.jsonl`, `arms-driver.jsonl`, sampler `e3-sampler.txt`)

Per phase (IOReport watts, SMC `PSTR`, the E1b fit's prediction `fit`, `err = PSTR - fit`, energy a token = PSTR times seconds a token):

| phase | s | tokens | ms/token | cpu W | gpu W | dram W | dcs+amcc W | PSTR W (sd) | fit W | err W | system J/token | cpu+gpu+ane J/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A idle server | 90 | 0 | - | 1.9 | 0.4 | 1.1 | 4.0 | 25.3 (6.4) | 25.9 | -0.6 | - | - |
| B all-resident | 124 | 1440 | 86 | 4.7 | 14.8 | 10.2 | 14.9 | 74.5 (6.5) | 61.7 | +12.8 | **6.42** | 1.68 |
| C read-bound | 152 | 1080 | 141 | 9.1 | 11.1 | 8.3 | 14.8 | 76.2 (6.6) | 61.9 | +14.3 | **10.73** | 2.84 |
| D idle server | 90 | 0 | - | 3.0 | 0.9 | 1.6 | 5.3 | 30.4 (8.3) | 29.4 | +1.0 | - | - |

Store counters: B read 1.82 GB/s from the internal drive with 0.0 misses a token (speculative prefetch loads); C read 3.56 GB/s with 33.7 misses a token. System energy a token split by the fit's terms: B 6.42 J = baseline 1.58 + cpu 0.52 + gpu 1.58 + dram 0.30 + dcs+amcc 1.33 + unfitted 1.10 (the fit's 12.8 W error times 86 ms); C 10.73 J = baseline 2.59 + cpu 1.63 + gpu 1.93 + dram 0.40 + dcs+amcc 2.17 + unfitted 2.02. Ratio C over B: **1.67** for system energy, 1.68 for cpu+gpu+ane (0.62.29 measured 1.46 for the latter, with C at 18.3 W and 2.52 J a token; this run's C read 20.2 W and 2.84 J a token, the CPU 9.1 W against 7.35 W).

Against the predictions: system power B 55 W (45-65) **missed** (74.5); C 50 W (42-60) **missed** (76.2); energy a token B 4.5 J (3.7-5.4) **missed** (6.42, +43 %); C 7.0 J (5.7-8.5) **missed** (10.73, +53 %); ratio 1.55 (1.4-1.7) held at the upper end (1.67); **the fit's prediction of a phase's PSTR within 4 W: falsified** (error +12.8 and +14.3 W; the falsifier was 8 W); the internal drive (C minus B of the error) +1.5 W, inside -1 to +4 W but under 1.4 W of noise, so not resolved. The falsifier "fit error above 8 W" fired, and a second one (system energy a token more than 25 % outside its range) fired for both phases.

## 6. Explanation

The E1b fit holds at idle (errors -0.6 and +1.0 W) and on its synthetic loads (rms 1.3 W) but under-predicts the real decode by 13-14 W, in the all-resident phase as well as the read-bound one, so the missing power is not the read-bound drive. Not explained. Candidates, none tested: (a) the IOReport channels used by the fit do not capture some real-decode draw (the "Energy Model" group has other GPU channels, `GPU0_0` and `GPU CS0_0` among them, that the fit did not use; the logger kept only the aggregates, so this cannot be refit from this run); (b) the fit's DRAM coefficient of 0.34, learned on streaming loads, does not hold for decode's access pattern (here the DRAM counter reads 10.2 W, and a coefficient near 1.6 instead of 0.34 would close the gap); (c) the internal drive is active in both phases (1.8 and 3.6 GB/s) and the Apple SSD's NAND draws more than the 2.7 W an external drive adds, but a drive would not give the same 12.8 W with 0.0 misses a token unless prefetch reads cost that much, which a 1.5 W difference for a 1.7 GB/s difference argues against; (d) a mix of GPU compute and memory traffic at once has a different loss than either alone (E1b ran them apart, except for one GPU-plus-CPU step). The measured quantity, PSTR times seconds a token, does not depend on the fit.

Package-only ratio: 1.68 here against 1.46 in 0.62.29 is a run-to-run difference of C's package power (20.2 against 18.3 W); the conditions differ in what else ran (this run: Lab closed, the logger and a different sampler file), not isolated.

## 7. Generalization

Hardware-specific and workload-specific (DeepSeek exact decode, internal bank, 48 GiB budget, a repeated prompt and ten fresh ones). The transferable form: a fit made on single-source loads does not extrapolate to a mixed real workload; a joule total must be measured at the system rail, and a decomposition by component needs the real workload in its fit.

## 8. Bottleneck after

Speed: unchanged. Energy: the 13-14 W the fit does not explain under real decode, and therefore any split of a token's energy by component beyond the measured system total. The next step is to log every Energy Model channel, not only the aggregates, and refit with decode phases included.

## 9. Kind

Research (predicted, measured, partly explained); several predictions missed and one fired a falsifier, recorded as such.
