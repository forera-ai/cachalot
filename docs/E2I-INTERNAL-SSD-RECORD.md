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

## 5. Result (measured, derived by the E1b correction; 2026-10-10, 17:58:38-18:07, Cachalot Lab closed; `benchmarks/results/energy/e2-internal.jsonl` and `.header.json`, sampler `e2i-sampler.txt`)

Seventeen 30 s steps (idle, then duty 0.1, 0.25, 0.5, 1.0, 1.0, 0.5, 0.25, 0.1 with an idle step between each), 100 ms windows, four threads, 8 MiB `F_NOCACHE` blocks from the internal copy of the DeepSeek bank, the list cycled (a block repeated only after 142 GB). Each load step against the mean of its two idle neighbours, corrected with the E1b coefficients:

| duty | first pass: GB/s, raw rise W, corrected rise W (se) | second pass: GB/s, raw, corrected (se) |
|---:|---|---|
| 0.10 | 1.29, +0.37, +0.63 (1.38) | 0.75, +5.18, +2.73 (1.60) |
| 0.25 | 1.92, +1.03, +1.10 (1.61) | 1.78, +12.98, +5.08 (1.46) |
| 0.50 | 3.41, +9.72, +3.84 (1.36) | 3.39, +16.79, +5.28 (1.77) |
| 1.00 | 6.33, +17.99, +7.74 (1.38) | 6.46, +15.25, +7.39 (1.31) |

By level (mean of the two passes): 1.02 GB/s **+1.68 W** (se 1.06); 1.85 GB/s **+3.09 W** (1.09); 3.40 GB/s **+4.56 W** (1.12); 6.39 GB/s **+7.56 W** (0.95). A line through the four levels: about 1.1 W per GB/s with an intercept near 0.7 W, i.e. roughly proportional to the read rate (about 1.1 nJ a byte). The raw rise at full rate is +16.6 W; the correction (the reader's own CPU, DRAM and controller activity) removes 9 W of it. The two passes differ by up to 4 W at the low duties (duty 0.25: +1.10 then +5.08 W; duty 0.10: +0.63 then +2.73), the second pass higher at every low duty; the standard errors do not cover that, and a warm-up of the drive or the machine between the passes is a candidate (not checked). Top achieved rate 6.46 GB/s.

Against the predictions: the rise at 1.7 GB/s +4 W (1.5-9): **held** (+3.09 mean); at 6.8 GB/s +7 W (2-14): **held** (+7.56 at 6.39 GB/s); the duty-0.25 rise at least 40 % of the full-duty rise: **held, barely** (0.41); the rate not above 7.5 GB/s: **held** (6.46). The hypothesis H-internal-SSD, that the drive accounts for at least half of the 13 W (6.5 W at 1.7 GB/s): **not supported**: the mean is +3.09 W and the larger of the two repeats is +5.08 W, both below 6.5 W. The falsifier (a rise below 3 W at 1.7 GB/s) was not met by the letter (+3.09 W against 3 W, inside the noise); the hypothesis is not strongly supported either.

## 6. Explanation

The internal SSD adds about 3 W (range 1-5 W across the passes) at E3's all-resident prefetch rate (1.82 GB/s) and about 4.7 W at its read-bound rate (3.56 GB/s, read off the line: 0.7 + 1.1 x 3.56), roughly proportional to the rate. Against the unexplained 12.6 W (all-resident) and 13.9 W (read-bound) of real decode that leaves **about 9.8 W and 9.2 W unexplained**, almost the same in both phases although the read rate doubled: the remaining excess does not depend on the read rate. Derived split of the all-resident phase (74.5 W): baseline 18.4 + cpu terms 6.0 + gpu terms 18.4 + dram 3.5 + memory controllers 15.5 (the E1b fit, 61.7 W) + internal SSD about 2.8 + unexplained about 10 W. Candidates for the rest, none tested: the real quantized matrix-multiply and attention kernels, the server's CPU-side work, 67 GB of resident, wired memory. An internal SSD at 3 W for 1.8 GB/s is also cheaper per byte (about 1.1 nJ) than the X10Pro (2.7 nJ at 1 GB/s).

## 7. Generalization

Hardware-specific (this Mac's internal 1 TB SSD, one session). The transferable form: the drive's power is close to proportional to the read rate here, so a lever that cuts bytes read saves drive energy in proportion, unlike the X10Pro run, where a fixed part could not be excluded.

## 8. Bottleneck after

Speed: unchanged. Energy: about 10 W of real decode that no tested synthetic load or drive reproduces, the same in both phases. Next candidates: decode with the prefetch and reads switched off (the drive part would drop by about 3 W and the rest stay), and the model's kernels run in-process without the server.

## 9. Kind

Research (predicted, measured, explained in part): every numeric prediction held; the headline hypothesis was not supported; the pass-to-pass drift is recorded and unexplained.
