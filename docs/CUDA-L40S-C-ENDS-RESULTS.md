# CUDA lane: the layer-time ends on an L40S (results)

Run 2026-10-20 on one NVIDIA L40S 48 GB VM (provider name `aquatic-existence-2628`; driver 580.159.03, PCIe 4.0 x16, torch 2.7.1+cu128, Python 3.12.3), a different instance from the copy-cost sweep's `flat-battle-1078`. Record `docs/CUDA-L40S-C-ENDS-RECORD.md` (ten predictions J-1 to J-10, committed in 04a172c before any code; section 7 corrects the J-10 count before any run; section 8, committed in 992553f before any 0.2 ms run, replaces C = 0.15 ms by 0.2 ms and re-registers J-2, J-3, J-6, J-7 and J-8). Code: commit 992553f (`dirty=0`) for the runs, a3cb44d for the first smoke. Raw samples (hash-verified copies) are under `~/cachalot-private/cuda-results/n3j/`; scoring output `docs/cuda-results/l40s-c-ends-2026-10-20.json`. Labels follow plan section 0.

Sequence. (1) `n3jsmoke` at C = 1.0 and 0.15 ms (a3cb44d): every gate true except calibration at 0.15 ms (0.498 ms reached, error +2.32); source copy 0.3776 ms. A standalone probe on the instance found why: the stand-in's time is 0.044 to 0.047 ms up to 96 MiB (L2-resident), 0.106 ms at 100 MiB, 0.169 ms at 104 MiB, then linear at about 1.4 microseconds per MiB, so the shortest stand-in that streams from DRAM is about 0.17 ms. (2) Hamed chose to reshape; 0.15 ms was replaced by 0.2 ms. (3) `n3jsmoke` again (51 s): every gate true, source copy 0.3782 ms. (4) `n3jfull` twice: run A `20261020T215150Z-n3j-j-full-3ef156` (502 s, every gate true, no failed cell) and run B `20261020T220020Z-n3j-j-full-fb8eb4` (496 s, one failed cell: `p17j.L3.C0.2`, N control drift -0.035 above the 0.03 limit; every other gate true). The wait was polled with `ps aux | grep '[o]verlap_arena'`, not `pgrep -f`.

## 1. Scoring the registered predictions

Slopes are on measured overlap, level 3. Run B scores J-1, J-2, J-4 and J-8, run A J-5 and J-9, both runs J-3, J-6, J-7 and J-10, as registered.

| id | measured | registered | verdict |
|---|---|---|---|
| J-1 slope at C = 1.0 | 0.0906 (run A 0.0862) | 0.075 to 0.110 | **held** |
| J-2 slope at C = 0.2 | 0.1051 (run A 0.1077) | 0.085 to 0.140 (falsified above 0.200) | **held** (run B's cell failed its drift gate, below; run A's clean cell agrees) |
| J-3 instrument at 0.2 and 1.0 | calibration and floor true at all four layer times in both runs (128 MiB tuned to 0.2078 ms, +3.9 %); run B's 0.2 ms level 3 cell failed the drift gate (N control drift -0.035 against 0.03) | no cell fails | **falsified** by its registered text (a cell failed); the cell's number is reported with that flag, run A's 0.2 ms cell is clean |
| J-4 anchors on a second instance | run B 0.0930 (C = 0.3) and 0.0915 (0.6) against the first instance's 0.093: 0.0 % and 1.6 % (run A 0.097 and 0.0914) | within 15 % | **held** |
| J-5 source copy | 0.3803 ms (run A), 0.3804 (run B); smokes 0.3776, 0.3782 | 0.34 to 0.42 | **held** |
| J-6 slope ratio | s(0.2) / s(1.0) = 1.249 (run A), 1.160 (run B) | 0.80 to 1.30 (falsified above 1.60 in both) | **held** |
| J-7 direction | s(0.2) above s(1.0) in both runs (0.1077 / 0.0862, 0.1051 / 0.0906) | at least in both | **held** |
| J-8 hidden share | at C = 1.0 0.360, at C = 0.2 -0.220 (run A 0.361, -0.208) | 0.30 to 0.55 and below 0.10 | **held** |
| J-9 demand exposure | 16.65 to 17.27 ms a token at all four C (run A) | 14 to 20, within 30 % across C | **held** |
| J-10 replicate | run B against run A: 5.1 %, 0.1 %, 4.1 %, 2.4 % at C = 1.0, 0.6, 0.3, 0.2 | within 15 % at three of four | **held** (four of four) |

**9 held, 1 falsified (J-3).**

## 2. What was measured

* **On this L40S the slope is nearly flat in the layer time (measured).** Run A: 0.0862, 0.0914, 0.0970, 0.1077 at C = 1.0, 0.6, 0.3, 0.2; run B: 0.0906, 0.0915, 0.0930, 0.1051. The ratio of the shortest to the longest layer time is 1.25 and 1.16 against the RTX 5090's 2.42 (0.0634 at 1.0 ms to 0.1534 at 0.15 ms, fast population). A mild rise at the shortest layer is visible in both runs (0.105 to 0.108 against 0.093 to 0.097 at 0.3 ms).
* **The affine-in-1/C form fits the four points, with small coefficients (derived).** Run A `s = 0.0816 + 0.0051 / C`, run B `s = 0.0856 + 0.0034 / C` (residuals at most 0.004), against the 5090's `0.0498 + 0.01574 / C`: the L40S coefficient on 1/C is 3 to 4.6 times smaller and the constant 1.6 to 1.7 times larger. The 5090 fast form evaluated with this card's copy time (`g(r) = 0.0498 + 0.0811 r`, r = 0.38 / C; run B) is 0.010 below the measured slope at C = 1.0, 0.010 above it at 0.6, and 0.060 and 0.099 above it at 0.3 and 0.2 ms.
* **The instance is stable (measured).** Slopes at 0.3 and 0.6 ms on this instance (0.093 to 0.097, 0.091 to 0.092) equal the first L40S instance's 0.093; copy 0.3803 against 0.378. Two L40S instances agree; the RTX 5090's slow and fast populations did not.
* **The hidden share is not flat (measured).** 0.36 at C = 1.0, 0.19 to 0.23 at 0.6, -0.04 to -0.06 at 0.3, -0.21 to -0.22 at 0.2 (both runs within 0.04): as on the 5090, prefetch stops hiding copy time and then costs time once the layer is shorter than a copy (0.38 ms). The slope does not follow it.
* **Demand exposure follows the copy time, not the layer time (measured).** 16.7 to 17.3 ms a token at every C against 52.15 x 0.3803 = 19.8 (-12 % to -16 %, as in the copy-cost sweep).
* **The stand-in's cache cliff (measured, side probe).** The 0.15 ms smoke failure is a property of the stand-in on this card (L2 residency below about 100 MiB), not of the slope.

## 3. What it means, and does not

* **Decision (record section 5, with 0.2 in place of 0.15): the 5090's C dependence is reported as absent (H-flat) on this L40S**, with a mild residual rise: s(0.2) / s(1.0) is 1.25 and 1.16, below 1.30 in both runs. The slope is described as a card constant of about 0.09 to 0.10 for layer times 0.3 to 1.0 ms, rising to about 0.105 at 0.2 ms. Because J-3 was falsified by run B's drift-failed 0.2 ms cell, run A's clean cell is the reading for that point; it gives the same side of the rule (1.249), and run B's number agrees with it to 2.4 %.
* It does not say why the 5090's slope depends on C and the L40S's barely does. Candidates (hypothesis, untested): the 5090's 1,792 GB/s against this card's 864 GB/s and 48 MB against 96 MB of L2 change how a copy stream and the stand-in share memory; the stand-in streams at about 0.7 TB/s here. A third card would say whether this is the L40S or the 5090.
* The comparison with the 5090 is not at equal layer time (0.2 against 0.15 ms) and not at equal copy-to-layer ratio (1.9 against 1.3); both favour a rise on the L40S, and it is small.
* **Limits.** Two instances of one card, one trace, 17 % resident, a stand-in compute stream (shortest streaming layer on this card about 0.17 ms; 0.15 ms is not reachable), one run B cell failed its drift gate, no claim about real expert kernels, another card or any default.
* Nothing changes in `src/` and no default moves.

## 4. Open, needs a GPU and Hamed's go

A third card (RTX 4090, A100, H100) to see whether the 5090's rise is the card; a native issue path; fewer and larger copies; ROCm. No offline CUDA job remains.
