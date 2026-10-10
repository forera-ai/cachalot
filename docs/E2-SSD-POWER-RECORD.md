# E2 record: what the X10Pro costs in system power when it reads (charter L5b, `docs/POWER-ACCOUNTING-PLAN.md` step E2, first drive)

Fields 1-4 written 2026-10-10, before the run, against runtime 0.62.33. Result and explanation are appended after the run. The internal drive is a separate run (it needs a temporary 48 GiB file); this record covers the X10Pro.

## 1. Baseline

No model. E1b (HANDOFF 18.120): `PSTR = 18.4 W + 1.27 cpu + 1.24 gpu + 0.34 dram + 1.04 (dcs+amcc)`, rms 1.3-1.4 W over 16 steps; within a 25 s step PSTR has a standard deviation of 2-7 W. The X10Pro is a bus-powered USB drive (assumed, to verify: it has no wall adapter), so its draw should be inside PSTR; the IOReport components do not see it (no SSD channel, 18.118). Recorded read rate of the drive: about 0.96-1.0 GB/s (LEDGER ST-X10-QD), queue depth two or more.

## 2. Hypothesis and prediction (tags: prediction)

Hypothesis: the drive adds a measurable, mostly rate-independent active power to PSTR: a USB NVMe enclosure pays for its controller, bridge chip and link as soon as it is busy, and the NAND energy per byte is small next to that.

Predictions, with intervals, for reads of 8 MiB blocks at queue depth 2, duty (fraction of each second the drive is busy) 0.25, 0.5 and 1.0 against idle steps between them:
- PSTR minus the fitted component terms rises by **3.5 W (2.0-5.5) at duty 1.0** (about 0.96 GB/s).
- The rise at duty 0.25 is **at least 40 % and at most 90 % of the rise at duty 1.0** (a large fixed part).
- The rise at duty 1.0 is larger than twice its standard error from the within-step spread (n = 25 samples a step, two repeats).

## 3. Falsifier

Any of: the rise at duty 1.0 below 1.0 W (the drive is not detectable in PSTR, so its power stays unmeasurable without a meter) or above 8 W; the rise at duty 0.25 under 40 % of the full-duty rise (a mostly rate-proportional power) or above 90 %; the rise not exceeding twice its standard error (below detectability whatever its size); an achieved rate over 1.2 GB/s at duty 1.0 (a page-cache hit, the run is void); a sampler not running (void).

## 4. Intervention

None to the runtime. New instrument `benchmarks/ssd_power_test.py`: `pread` of 8 MiB blocks with `F_NOCACHE` by two threads from the 40 shards of `/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q2g128` (142 GB, read-only), block order a shuffled list used without replacement across the whole run (no block is read twice); thirteen 30 s steps alternating idle and load (idle, 25 %, idle, 50 %, idle, 100 %, idle, 100 %, idle, 50 %, idle, 25 %, idle), the first 5 s of each excluded; each load step is compared with the mean of its two neighbouring idle steps and corrected with the E1b coefficients for the changes in the IOReport components. Needs Hamed's root `powermetrics` sampling for the run (about 8 minutes of load, 105 GB read).

## 5. Result (measured, derived by the E1b correction; 2026-10-10, 02:19-02:26, `benchmarks/results/energy/e2-x10pro.jsonl`, sampler `e2-sampler.txt`)

Thirteen 30 s steps, 25 s of each used. Each load step against the mean of its two idle neighbours; "corrected" removes the change in the IOReport components with the E1b coefficients.

| step | duty | GB/s | PSTR W | rise in PSTR | corrected rise (se) |
|---|---:|---:|---:|---:|---:|
| 2 | 0.25 | 0.68 | 27.4 | +3.20 | +1.38 (1.54) |
| 4 | 0.50 | 0.50 | 29.1 | +4.67 | +1.79 (1.35) |
| 6 | 1.00 | 0.99 | 33.1 | +7.86 | +2.19 (1.21) |
| 8 | 1.00 | 0.99 | 33.7 | +8.65 | +3.24 (1.19) |
| 10 | 0.50 | 0.50 | 28.4 | +3.71 | +1.03 (1.35) |
| 12 | 0.25 | 0.25 | 27.7 | +2.46 | +1.13 (1.67) |

By level: duty 1.00 (0.99 GB/s) **+2.71 W** (se 0.85); duty 0.50 (0.50 GB/s) +1.41 W (se 0.95); duty 0.25 (mean rate 0.47 GB/s, because step 2 read 0.68 GB/s for reasons not found) +1.25 W (se 1.13). Ratio of the duty-0.25 rise to the full-duty rise: 0.46. The raw rise at full duty is +8.25 W; the corrections remove 5.5 W of it (the reader's own CPU, DRAM and controller activity), which is why the drive is a small remainder of a large correction. Top achieved rate 0.99 GB/s; no cache hit.

Against the prediction: +3.5 W (2.0-5.5) at full duty held at the low end (2.71); a duty-0.25 rise of 40-90 % of the full-duty rise held, near the edge (0.46); the full-duty rise above twice its standard error held (2.71 against 1.70, about 3.2 sigma). No falsifier fired.

## 6. Explanation

The X10Pro adds about 2.7 W to the system while it reads at its wall of 1 GB/s, and the rise is about 1.25-1.4 W at 0.5 GB/s and 2.7 W at the full rate. Whether that is a fixed active power or a cost proportional to the rate cannot be told apart at these standard errors (0.85-1.13 W a level): a least-squares line through all six load steps has an intercept of 0.3 W and a slope of 2.3 W per GB/s, each uncertain by about as much as its own size. At the full rate it is 2.7 nJ a byte, about 50 times the 50 pJ/B of DRAM (18.119). That PSTR responds at all supports the assumption that the drive is bus-powered. The unexplained 0.68 GB/s first step: not found (a start-up effect of the first burst against a cold drive is a guess).

## 7. Generalization

Hardware-specific (this X10Pro, this Mac port, one session). The transferable form is only the measured one, about 2.7 W at 1 GB/s. Whether the drive's power is fixed while active or proportional to the rate is undetermined here, so a lever that cuts read time and one that cuts read bytes cannot yet be priced separately. The internal drive was not measured (no large file; a 48 GiB temporary file is Hamed's call).

## 8. Bottleneck after

Speed unchanged. Energy: the sum for a token is now measurable from three derived or measured terms (SoC by IOReport, DRAM at 49.9 pJ/B, the X10Pro at 2.7 W while busy) plus the 18 W baseline; what remains is E3 (a joules-per-token run with all of them) and the internal drive. The SSD figure has about +-0.9 W of statistical and about +-0.8 W of systematic (correction coefficient) uncertainty: an inline meter would replace it with a measurement.

## 9. Kind

Research (predicted, measured, partly explained): every prediction held, narrowly; the unexplained first step is recorded.
