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

## 5. Result, explanation, generalization, bottleneck after, kind

To be written after the run.
