# CUDA lane N3 stage A: results (layer-ahead prefetch against demand fetch, with a stand-in compute; RTX 5090)

Run 2026-10-19 under `docs/CUDA-N3-EXPERIMENT-RECORD.md` (12 predictions committed in c2305a6 before any harness code; erratum 2db00e5, written before any run, which moved two derived anchors and edited no prediction). Harness: `benchmarks/cuda/overlap_arena.py` at scaffold commit 9390ea5 (`REVISION` dirty=0), runtime 0.66.30; trace `routing-trace-deepseek-long` (sha256 prefix `f961e9e7a8971317`, the same file as N2). Instance: a sixth rented RTX 5090 virtual machine of the S2 to B2 class (the second N3 VM; the first was unusable, section 1) (driver 580.159.03, torch 2.7.1+cu128, 7 vCPUs, 24 GiB RAM, PCIe 5.0 x16, kernel 6.8). Raw rows, manifests and per-token samples: `docs/cuda-results/n3-rtx5090-2026-10-19/`. Labels follow plan section 0: measured on this one instance unless it says derived or hypothesis. The compute is a stand-in (a bandwidth-bound read kernel), not a model; nothing here says anything about a real decode's token time, a V100, or another provider's hosts.

## 1. What ran

A first N3 instance had no NVIDIA kernel module loaded (the packages and the DKMS module were installed, the PCI device was present, a console reboot did not change it; loading it needs root and I ran no sudo) and was replaced. The second instance had a working driver but no outbound network until a console reboot (the S2 fault again); after it `check`, `setup`, `doctor`, `smoke` and `n3smoke` were clean and `n3full` ran as 22 cells in 559.6 s against the 1,320 s cap I set (the script's own 2,400 s would have overrun the 1 GPU-hour cap). **All gates true in the smoke and the full run, no failed cell, status complete**: event hash against the independent reference, determinism, bookkeeping counts, readback, calibration (errors -0.4 % to +2.3 %), control drift (|N end / N start - 1| at most 0.013 %), closure of the accounting (see 4), sanity. The stream code (events, `wait_event`, the copy stream, the stand-in kernels, the calibration) ran correctly on the GPU on its first run; the Mac tests had checked its order on a recording arena.

Population of this instance's source records (the 64 pinned records, each copied alone): median **c = 0.1936 ms**, fastest 0.1933, slowest 0.3625, **3 of 64 slow (4.7 %)**: an N2-like fast population, unlike B1's and B2's fresh-process buffers (all slow). Every absolute time below is therefore at c = 0.194 ms.

## 2. Results (p50 per decode token over 399 tokens after the warm-up; ms)

| cell | wall | compute busy | compute stall | host gap | copy busy: demand | copy busy: prefetch | host issue | demand misses | prefetch loads | covered | wasted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **C = 0.3 ms a layer, 17 % resident** | | | | | | | | | | | |
| N (floor, mean of N0, N1) | 12.509 | 12.299 | 0.015 | 0.190 | 0 | 0 | 0.80 | | | | |
| D | 22.412 | 12.181 | 8.724 | 1.520 | 9.476 | 0 | 2.23 | 50.45 | 0 | 0 | 0 |
| P0 | 21.660 | 12.994 | 6.323 | 2.363 | 5.498 | 10.696 | 2.99 | 32.03 | 57.43 | 23.01 | 34.42 |
| P40 | 23.410 | 12.486 | 9.281 | 1.610 | 5.447 | 10.502 | 3.02 | 32.03 | 57.43 | 23.01 | 34.42 |
| O | 15.950 | 12.969 | 1.050 | 1.959 | 0.312 | 9.187 | 2.62 | 0.01 | 50.45 | 50.45 | 0 |
| **C = 1.0 ms a layer, 17 %** | | | | | | | | | | | |
| N | 40.076 | 39.858 | 0.015 | 0.196 | 0 | 0 | 0.84 | | | | |
| D | 50.072 | 39.734 | 8.690 | 1.620 | 9.537 | 0 | 2.42 | 50.45 | 0 | 0 | 0 |
| P0 | 47.654 | 40.313 | 4.900 | 2.405 | 5.705 | 10.698 | 3.18 | 32.03 | 57.43 | 23.01 | 34.42 |
| P40 | 47.218 | 40.247 | 5.354 | 1.623 | 5.626 | 10.577 | 3.19 | 32.03 | 57.43 | 23.01 | 34.42 |
| O | 42.589 | 40.292 | 0.223 | 2.079 | 0.432 | 9.229 | 2.89 | 0.01 | 50.45 | 50.45 | 0 |
| **C = 3.0 ms a layer, 17 %** | | | | | | | | | | | |
| N | 120.178 | 119.956 | 0.015 | 0.200 | 0 | 0 | 1.03 | | | | |
| D | 130.262 | 119.817 | 8.730 | 1.708 | 9.595 | 0 | 2.74 | 50.45 | 0 | 0 | 0 |
| P0 | 127.819 | 120.315 | 4.898 | 2.496 | 5.739 | 10.707 | 3.50 | 32.03 | 57.43 | 23.01 | 34.42 |
| P40 | 127.163 | 120.316 | 5.072 | 1.663 | 5.650 | 10.581 | 3.50 | 32.03 | 57.43 | 23.01 | 34.42 |
| O | 122.715 | 120.330 | 0.222 | 2.158 | 0.453 | 9.254 | 3.20 | 0.01 | 50.45 | 50.45 | 0 |
| **C = 1.0 ms, 8 % resident** | | | | | | | | | | | |
| N | 40.091 | 39.874 | 0.015 | 0.196 | 0 | 0 | 0.84 | | | | |
| D | 58.207 | 39.664 | 16.547 | 1.996 | 17.646 | 0 | 2.76 | 91.72 | 0 | 0 | 0 |
| P0 | 52.058 | 40.799 | 8.222 | 2.935 | 9.425 | 19.514 | 3.71 | 54.44 | 102.30 | 46.18 | 56.12 |

**Exposed time and hidden share** (derived: `E_X` = wall of X minus the mean of the two `N` controls; `H_X = 1 - E_X / E_D`):

| group | `E_D` | `E_P0` / `H_P0` | `E_P40` / `H_P40` | `E_O` / `H_O` | N drift |
|---|---:|---|---|---|---:|
| 17 %, C = 0.3 | 9.903 | 9.151 / **0.076** | 10.901 / **-0.101** | 3.441 / **0.653** | 0.00013 |
| 17 %, C = 1.0 | 9.997 | 7.579 / **0.242** | 7.143 / **0.286** | 2.513 / **0.749** | 0.00002 |
| 17 %, C = 3.0 | 10.084 | 7.641 / **0.242** | 6.985 / **0.307** | 2.537 / **0.748** | 0.00008 |
| 8 %, C = 1.0 | 18.117 | 11.968 / **0.339** | | | 0.0001 |

Compute-lane busy time against the `N` controls (interference, derived): `P0` +5.7 % at C = 0.3, +1.1 % at C = 1.0, +0.3 % at C = 3.0; `O` +5.4 %, +1.1 %, +0.3 %; `P40` +1.5 %, +1.0 %, +0.3 %; `D` -1.0 %, -0.3 %, -0.1 %.

## 3. Scoring the registered predictions

| id | outcome |
|---|---|
| N3-P1 (`E_D` at 17 % within 0.95 to 1.25 x 52.15 x c) | **held** at every C: 9.90, 10.00, 10.08 ms against the band 9.59 to 12.62 (the erratum's anchor 50.45 x c = 9.77: ratios 1.014, 1.023, 1.032) |
| N3-P2 (`H_P0` 0.30 to 0.50 at C = 3.0; falsifier below 0.20 or above 0.52) | **missed low**: 0.242; the falsifier did not fire. The best case under the serial model on this window was 0.365 (erratum); the run reached 66 % of it |
| N3-P3 (`H_P0` at C = 1.0 in 0.30 to 0.48 when copies a layer x c is at most 0.6 C: here 0.43 ms against 0.6) | **missed low**: 0.242, 0.058 below the interval; the falsifier (more than 0.10 outside) did not fire |
| N3-P4 (`H_P0` -0.30 to 0.20 at C = 0.3) | **held**: 0.076 (the copy lane carries 16.2 ms of copies in a 21.7 ms token) |
| N3-P5 (`H_O` at least 0.85 at C = 3.0 and at C = 1.0; 0.30 to 0.80 at C = 0.3) | **missed low** at C = 3.0 and 1.0 (0.748, 0.749); the falsifier (below 0.70 at C = 3.0) did not fire; the C = 0.3 clause **held** (0.653) |
| N3-P6 (`H_P40` below `H_P0` by 0.03 to 0.20 at C = 1.0) | **falsified in direction**: `H_P40` is **higher** by 0.044 (0.286 against 0.242); neither registered falsifier text (equal within 0.02, or lower by more than 0.30) fired literally, and the prediction is scored as missed in the wrong direction. At C = 0.3 `P40` is 0.177 lower (the window matters there), at C = 3.0 higher by 0.065 |
| N3-P7 (`P0` compute busy +4 to +20 % at C = 1.0, at most +10 % at C = 3.0) | **falsified at C = 1.0** (+1.1 %, the registered falsifier "below 2 %" fired); the C = 3.0 clause held (+0.3 %) |
| N3-P8 (`P0` wall below `D`'s at C = 1.0 and 3.0) | **held**: 47.65 against 50.07 and 127.82 against 130.26 |
| N3-P9 (8 %, C = 1.0: `H_P0` 0.25 to 0.50) | **held**: 0.339 (best case 0.406) |
| N3-P10 (at least 90 % of tokens close within 5 %) | **held**: 100 % in all 22 cells, median error 0.01 to 0.06 % |
| N3-P11 (host issue time a token under 1.5 ms; falsifier 3 ms or more) | **falsified**: 2.2 to 2.8 ms in `D`, 3.0 to 3.7 ms in `P0` and `P40` (3.0, 3.2, 3.5 and 3.7 ms in four cells, the falsifier fired), 2.6 to 3.2 ms in `O`; the `N` arm alone (launch, events and the per-layer synchronization) is 0.8 to 1.0 ms |
| N3-P12 (22 cells inside 25 min, all gates, no failed cell) | **held**: 559.6 s, all eight gates true, 0 failed cells |

Of 12: 6 held (P1, P4, P8, P9, P10, P12), 3 missed low with the falsifier not fired (P2, P3, P5), 3 falsified (P6 in direction, P7 at C = 1.0, P11).

## 4. What the result is, and is not

- **Measured.** On this instance (c = 0.194 ms, the fast population) layer-ahead prefetch with the recorded predictor hides **24 % of the demand-fetch exposure at 17 % resident for any compute window of 1 ms a layer or more** (0.242 at C = 1.0 and 3.0), 34 % at 8 %, and almost nothing at C = 0.3 (0.08, and -0.10 for the late prefetch): the copy lane is then the limit (16.2 ms of copies in a 21.7 ms token). An oracle that knew the next layer's experts hides **75 %** at C of 1 ms and above. Demand-fetch exposure is 9.9 to 10.1 ms a token whatever the compute window.
- **Derived: where the rest goes.** For `P0` at C = 3.0 the exposure of 7.64 ms is compute stall 4.88 (waiting for the 32 demand copies that remain) plus host gap +2.30 plus compute busy +0.36. For the oracle (no real misses) the 2.54 ms that remain are **host gap +1.96**, stall +0.21, compute busy +0.37: with the copies hidden, what is left is the **host's issue time** (a Python loop issuing 50 to 90 copies, events and bookkeeping a token, 2.4 to 3.7 ms of host work against 0.8 for launches alone), about 2 ms a token of idle GPU time between layers. This is the CUDA counterpart of the Mac's per-layer host cost (CUDA-H8's direction): after the copy is hidden, host orchestration is the largest remaining term of the exposure on this card, 80 % of the oracle's 2.5 ms; it is 5 % of a 40 ms token at C = 1.0 and 16 % of a 12.5 ms token at C = 0.3. It is a property of my Python harness; a native loop or a CUDA graph would change it, and nothing here measures that.
- **Measured.** The serial model's best case (0.365) was not reached because the late demand copies also pay host issue time; the prefetch arms also issue about 57 more copies a token (89.5 against 50.5, +77 %) at a host cost, and 34.4 of the 57.4 prefetched experts a token are wasted (60 %).
- **Measured, against the prediction.** The late prefetch (`P40`) was not worse than the early one at C of 1 ms and above: issuing the prefetch copies after the layer's compute is launched (`P40`) instead of before (`P0`) takes about 0.8 ms of host gap out of the critical path (host gap 1.62 against 2.41 ms at C = 1.0), which outweighs the window it loses. The window effect appears only at C = 0.3 (P40 is 0.177 lower). So the `P0` against `P40` comparison here mixes a window effect with an issue-order effect; they were not separated (hypothesis).
- **Interference (CUDA-H7).** Copies slowed the compute lane by +0.3 to +1.1 % at C of 1 ms and above and by +5.7 % (`P0`) and +5.4 % (`O`) at C = 0.3, where the copy duty cycle is highest; the plan's H7 ("under 5 %") holds for C of 1 ms and above and is at its edge at 0.3 (5.7 % against a 5 % bound, one card, one instance).
- **Not shown.** What a real decode's compute does to these numbers (the stand-in reads weights sequentially; a real layer is many kernels with a different memory pattern); the 400-token window against the whole trace (the all-token demand misses are 51.85 a token, the window's 50.45); the fast-population instance against the slow one (on B1's and B2's instances c would be 0.359 ms and every exposure about 1.85 times larger, derived from c alone and not measured here); any V100 behaviour.
- **Consequence for the lane (derived).** At a layer time of 1 ms or more, a recorded predictor recovers about a quarter of the exposed copy time at 17 % resident, an oracle three quarters; the gap between them (0.24 against 0.75) is predictor coverage, and the wasted 34 prefetches a token are the price of the 23 covered. The next levers are a better or gated predictor (the weight gate priced offline on the Mac), fewer and cheaper copies to issue (the host term), and a native issue loop.

## 5. Cost and housekeeping

GPU time on the instrument: the smoke 13 s, the full run 560 s. Two instances were used, the first for about 10 minutes of diagnosis (no driver). The billed hours of both are Hamed's to enter in the private cost file (he said he will); the agent did not guess them. The instance is terminated (his word). Housekeeping stands: delete the S1, S2, V1, N2, B1, B2 and N3 ssh configs, known_hosts files and the key (nothing was deleted here).
