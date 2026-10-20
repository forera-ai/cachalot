# CUDA lane: the copy-cost sweep on an L40S (results)

Run 2026-10-20 on one NVIDIA L40S 48 GB VM (provider name `flat-battle-1078`; driver 580.159.03, PCIe 4.0 x16, torch 2.7.1+cu128, Python 3.12.3), the experiment Hamed chose after `docs/CUDA-LOW-C-FAST-RESULTS.md`. Record `docs/CUDA-COPY-COST-RECORD.md` (ten predictions SW-1 to SW-10, committed in 3e88a1b before any code, unchanged). Code: commit f9f7f12 (`dirty=0`; the harness was built in 8ba42a1 and f9f7f12 only changed the setup arch gate, below). `n3ismoke` (49 s) then `n3ifull` twice (run A `20261020T205130Z-n3i-i-full-b03f3a`, run B `20261020T210154Z-n3i-i-full-a49d85`; 48 cells each, 400 decode tokens, about 616 s and 623 s). Every gate true in the smoke and in both runs (event hash, determinism, counts, chain2, readback, calibration, drift, closure, sanity, floor, rowsum, copyevents, coverage), no failed cell, nothing truncated. Raw samples (hash-verified copy, sha256 prefix 98ef2afb466e745c) are under `~/cachalot-private/cuda-results/n3i/`; scoring output `docs/cuda-results/copy-cost-2026-10-20.json`. Labels follow plan section 0.

Two operational notes. The setup arch gate stopped on this card (cu128 lists `sm_86` but no `sm_89`); a real matmul and pinned copy ran, so the gate was corrected to accept a lower-minor cubin of the same major (f9f7f12, test added). Separately, my wait loop used `pgrep -f` inside the ssh command, which matched its own shell and kept reporting the run as up for about 13 minutes after both runs had finished; Hamed's message caught it. The instance was billed for that wait.

## 1. Scoring the registered predictions

Run A reads SW-1, SW-2 and SW-9, run B SW-4 to SW-7, both runs SW-3 and SW-10, as registered. Slopes are on measured overlap, level 3.

| id | measured | registered | verdict |
|---|---|---|---|
| SW-1 median source copy at f = 1.0 | 0.378 ms (run B 0.381) | 0.36 to 0.50 (falsified outside 0.28 to 0.70) | **held** |
| SW-2 copy time against bytes | c(0.5) / c(1.0) = 0.507, c(0.25) / c(1.0) = 0.275 (0.1917, 0.1041 ms) | 0.50 to 0.62 and 0.26 to 0.40 | **held** (both near the lower edge: a small fixed cost) |
| SW-3 slope rises with copy cost | in all four (run, C) comparisons the order is reversed: s(f = 1.0) is **below** s(f = 0.5), which is below s(f = 0.25) (0.0929, 0.0976, 0.1008 at C = 0.3, run A) | s(1.0) > s(0.5) > s(0.25) (falsified if s(1.0) at most s(0.25)) | **falsified**: the slope falls slightly as copies get dearer |
| SW-4 slope is one function of r | line `s = 0.111 - 0.0157 r`, R squared 0.205 | R squared at least 0.85 (falsified below 0.50) | **falsified** |
| SW-5 equal-r pairs | (f = 1.0, C = 0.6) 0.0920 against (0.5, 0.3) 0.0982: 6.3 %; (0.5, 0.6) 0.0951 against (0.25, 0.3) 0.1284: 26.0 % | both within 25 % (falsified beyond 40 %) | **missed narrowly** (the second pair, by one outlier cell, below) |
| SW-6 the 5090 form carries over | `g(r)` within 0.03 at 3 of 6 points; mean absolute error 0.031 | at least 4 of 6 (falsified above 0.06) | **missed narrowly** |
| SW-7 count-only window model | hidden-share error within 0.10 at 3 of 6 points; errors 0.034, 0.002, 0.002, 0.113, 0.154, 0.312; mean 0.103 | at least 4 of 6 (falsified above 0.18) | **missed narrowly** |
| SW-8 the instrument | calibration and floor true at both C in every group, no failed cell | hold | **held** |
| SW-9 demand exposure | `D` minus the `N` floor is 6 % to 15 % below 52.15 x c(f) at all six points (largest -14.7 %, run A) | within 15 % (falsified beyond 35 %) | **held** (narrowly; always low) |
| SW-10 replicate | run B within 15 % of run A at 5 of 6 points (0.6 % to 1.6 % at five; the f = 0.25, C = 0.3 cell 27.5 %) | at least 5 of 6 (falsified below 3) | **held** |

**5 held, 3 missed narrowly, 2 falsified (SW-3, SW-4).**

## 2. What was measured

* **On this L40S the slope does not follow the copy time over the layer time (measured).** Across six points with r from 0.17 to 1.27 the slope sits at 0.092 to 0.105 (one cell, run B f = 0.25 at C = 0.3, reads 0.128), mean 0.100 over both runs. Within a group the slope is the same at C = 0.3 and C = 0.6 to within 0.002 to 0.005 (run A: 0.0929 and 0.0931 at f = 1.0, 0.0976 and 0.0964 at f = 0.5, 0.1008 and 0.1045 at f = 0.25). Between groups it falls weakly as copies get dearer: about 0.103 at c = 0.10 ms, 0.097 at 0.19 and 0.093 at 0.38.
* **The 5090 pattern (a slope that rises as C falls or copies get dearer) is absent here (measured).** On the RTX 5090 the slope went 0.0634 to 0.1053 between C = 1.0 and 0.3 and 0.2234 at a dear-copy instance; here C = 0.3 against 0.6 changes it by under 5 %, and 4x more copy time changes it by about 10 % in the other direction. The sweep's own question, whether copy cost moves the slope within one instance, has the answer no on this card; the 5090 slow-instance excess is therefore not shown to be a copy-cost effect (it may still be on that card; the card is a second difference, not controlled here).
* **A copy is near-affine in bytes (measured).** 9.95 MB 0.378 ms (26.3 GB/s), 4.98 MB 0.192 ms, 2.49 MB 0.104 ms: about 0.02 ms fixed plus 0.036 ms per MB; the slow share of source records is 0 at every fraction on this instance.
* **Hidden share (measured).** f = 1.0: -0.033 (C = 0.3), 0.223 (C = 0.6); f = 0.5: 0.227, 0.360; f = 0.25: 0.328, 0.347 (run A; run B within 0.01 except 0.221 at f = 0.25, C = 0.6). The count-only window model with the measured copy time is within 0.04 where the layer is long compared with a prefetch batch (r from 0.63) and overstates the hidden share by 0.11 to 0.31 where copies are cheap (r below 0.35): at small copies the share saturates near 0.33 to 0.37, which the count model does not.
* **Demand exposure follows the copy time with a factor below one (measured).** 16.8 ms a token at f = 1.0 against 52.15 x 0.378 = 19.7 predicted (-15 %); 8.8 against 10.0; 4.8 to 5.1 against 5.4. The fast-population 5090 had 52.15 x c exactly; here each batch of misses is 6 % to 15 % cheaper than isolated copies times the count (the batch overlaps its own issue; not examined further).
* **One outlier cell.** Run B f = 0.25, C = 0.3, 0.1284 against 0.1008 in run A (27.5 %). It alone moves the second equal-r pair outside 25 % (SW-5) and the f = 0.25 line point; run A and the other five replicate pairs agree to 1.6 %. Not explained (one run); the registered SW-10 count tolerates it.

## 3. What it means, and does not

* **H-ratio is not supported on this card and instance.** SW-3 and SW-4 are falsified. The slope is close to a constant of about 0.10 ms of compute per ms of overlap for layer times 0.3 to 0.6 ms and copy times 0.10 to 0.38 ms.
* It does not say why the 5090 slope depended on C; that remains hypothesis (derived: the affine 1/C form fits four 5090 points to 0.003). It says the form does not transfer to the L40S (SW-6 mean absolute error 0.031, 3 of 6 within 0.03 only because the L40S constant sits near the middle of the 5090 range), and that a copy-cost explanation of the 5090 slow instance is not supported by a within-instance sweep on a different card.
* **Limits.** One L40S instance, one trace, 17 % resident, a stand-in compute stream (a bandwidth-bound kernel pair; the L40S has 864 GB/s against the 5090's 1,792 and 48 MB of L2 against 96 MB), two layer times only (0.3 and 0.6 ms, no 0.15 or 1.0), copies shortened by bytes (time and bytes move together, so time against bytes is not separated here). No claim about another card, about real expert kernels or about the 5090 population effect.

## 4. Decision (record section 5)

* SW-3 falsified: **copy cost does not move the slope here; the slow-instance excess on the RTX 5090 is not explained by `c`** (stated as the registered finding, card caveat above).
* SW-4 falsified: **the slope is not a function of r on this card;** the six points are reported with no function fitted. The three alternatives of record section 1 (layer time alone, bytes, time and layer separately) are not tested further here: on the L40S the slope is flat in C over 0.3 to 0.6 and mildly decreasing in c; a record to cover 1.0 and 0.15 ms on this card would say whether the 5090's C dependence exists here at all.
* SW-8 held: the instrument is tunable to 0.3 and 0.6 ms on this card.
* Nothing changes in `src/` and no default moves. The fit describes a stand-in compute stream.
* **Open, needs a GPU and Hamed's go:** the L40S at C = 1.0 and 0.15 ms (does the 5090's C dependence exist on this card at all); a second L40S instance (is the run-B outlier noise, and is the slope instance-stable); an RTX 4090 or A100 for a card-dependence picture; fewer and larger copies; a native issue path; ROCm.
