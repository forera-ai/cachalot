# E2i record: what the internal SSD costs in system power when it reads (charter L5b, E2 second drive; tests hypothesis (a) of `docs/E3C-GATHER-RECORD.md`)

Fields 1-4 written 2026-10-10, before the run, against runtime 0.62.37. Result and explanation are appended after the run.

## 1. Baseline

E2 (HANDOFF 18.121): the X10Pro adds +2.71 W (se 0.85, derived) reading at 0.99 GB/s, +1.4 W at 0.5 GB/s. E3 (18.122) and E3b (18.123): real DeepSeek decode draws 12.6 W (all-resident) and 13.9 W (read-bound) above the component fit; the internal drive was reading 1.82 GB/s in the all-resident phase (speculative prefetch loads, 0.0 misses a token) and 3.56 GB/s in the read-bound one; the idle server phases without reads showed no excess (+1.0 W, +5.7 W after thermal soak). E3c (18.124): scattered small GPU reads and synchronized groups reproduce at most +5 W, so a read pattern on the GPU does not explain it. No Energy Model channel reports the internal drive.

## 2. Hypothesis and predictions (tags: prediction)

Hypothesis H-internal-SSD: the Apple internal SSD (controller on the SoC, NAND on the board) draws a large, mostly fixed power once it is reading, and that power is what the component fit leaves out in real decode. Probability I give it that the drive accounts for at least half of the 13 W: **30 %**.

Predictions for reads of 8 MiB blocks with `F_NOCACHE` from the internal copy of the DeepSeek bank (`~/DeepSeek-V4.1-Flash-q2g128`, 142 GB, read-only; the order is a shuffled list cycled once the bank is exhausted, so a block repeats only after 142 GB of other reads, far more than the machine's RAM), four threads, GPU idle, in 100 ms windows so the rate is steady rather than bursty, duty 0.1, 0.25, 0.5, 1.0 (the drive's wall is about 6.8 GB/s):
- At duty 0.25 (about 1.7 GB/s, E3's all-resident prefetch rate) the corrected rise in `PSTR` is **+4 W (1.5-9)**.
- At duty 1.0 (about 6.8 GB/s) it is **+7 W (2-14)**.
- The rise at duty 0.25 is **at least 40 %** of the rise at duty 1.0 (a large fixed part, as for the X10Pro).
- The rate at duty 1.0 does not exceed **7.5 GB/s** (above it would be a page-cache hit and void the run).

## 3. Falsifier

H-internal-SSD (the drive accounts for at least half of the 13 W) is falsified if the corrected rise at 1.7 GB/s is **below 3 W**: at E3's own read rate the drive then explains under a quarter of the excess. It is strongly supported if the rise at 1.7 GB/s is **6.5 W or more** (half). A rate above 7.5 GB/s, or a sampler not running, voids the run. A rise inside the fit's noise (about 1.4 W rms) is reported as unresolved, not as zero.

## 4. Intervention

None to the runtime. `benchmarks/ssd_power_test.py` gains `--period` (the window length of the duty cycle) and `--wrap` (cycle the block list instead of stopping when it is exhausted); otherwise as for the X10Pro run. Needs Hamed's root `powermetrics` beside it for the run (about 12 minutes), Lab closed. The run reads about 700 GB in all.

## 5. Result, explanation, generalization, bottleneck after, kind

To be written after the run.
