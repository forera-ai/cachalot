# CUDA lane N3 stage C: results (a router-weight gate on the recorded predictor; RTX 5090)

Run 2026-10-19 under `docs/CUDA-N3C-EXPERIMENT-RECORD.md` (nine predictions committed in c72cea2 before any stage C code; erratum in its section 8, before any run). Harness: `benchmarks/cuda/overlap_arena.py` at scaffold commit b7f8aa5 (`REVISION` dirty=0), runtime 0.66.34; trace `routing-trace-deepseek-long`. Instance: an RTX 5090 virtual machine (driver 580.159.03, torch 2.7.1+cu128, 7 vCPUs, 24 GiB RAM, PCIe 5.0 x16); it worked from its first login (`nvidia-smi` and DNS checked before uploading). Raw rows, manifests and per-token samples: `docs/cuda-results/n3c-rtx5090-2026-10-19/`. Labels follow plan section 0: measured on this one instance unless it says derived or hypothesis. The compute is a stand-in and the host a Python loop over torch; nothing here says anything about a native runtime, a real model, a V100 or another provider's hosts.

## 1. What ran

`check`, `setup`, `doctor`, `smoke` clean; `n3csmoke` (10 cells, 20.7 s) clean; `n3cfull` (19 cells, 301.2 s against the 900 s cap): **every gate true** (event hash, determinism, counts, `chain2`, `coverage` (gate g), readback, calibration, control drift, closure, sanity), no failed cell, status complete. The L5 graph built and calibrated for C = 1.0 ms. This instance's source records: median **c = 0.194 ms**, 3 to 5 % slow (the fast population, as in N2, N3 and N3-B). Control drift of every group's two `N` cells: below 0.001.

## 2. Results (400 decode tokens, token 0 a warm-up; p50 per token, ms)

`E` = wall minus the group's floor; `H` = `1 - E / E_D` within the group.

| group | floor | `E_D` | `E_P0L` | `E_G10` | `E_G20` | `E_O` | `H_P0L` | `H_G10` | `H_G20` | `H_O` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 %, C = 1.0 ms, L5 | 39.837 | 9.538 | 6.307 | 6.292 | 7.237 | 1.433 | **0.339** | 0.340 | **0.241** | 0.850 |
| 17 %, C = 0.3 ms, L4 | 12.443 | 9.602 | 8.678 | 8.601 | 8.538 | 2.780 | 0.096 | 0.104 | **0.111** | 0.710 |
| 8 %, C = 1.0 ms, L5 | 39.797 | 17.608 | 10.572 | | 12.111 | | **0.400** | | **0.312** | |

Counts a token at 17 % (identical at both C): `P0L` 32.03 demand misses, 57.43 prefetch loads, 23.00 covered, 34.42 wasted; `G10` 31.87 / 56.55 / 22.82 / 33.73; `G20` 35.84 / 39.27 / 17.50 / 21.78; at 8 %: `P0L` 54.44 / 102.30, `G20` 62.24 / 72.09. These equal the offline model to the printed digits (gate g). Host issue time a token (host clock) at C = 1.0 ms: `D` 0.967, `P0L` 1.351, `G10` 1.355, `G20` 1.264, `O` 1.190; at C = 0.3 ms (L4): `P0L` 2.079, `G10` 2.084, `G20` 1.987. Wall time a token at C = 0.3 ms: `P0L` 21.121, `G10` 21.044, `G20` 20.981 (`G20` 0.14 ms, 0.7 %, below `P0L`).

## 3. Scoring the registered predictions

| id | outcome |
|---|---|
| C-P1 (counts equal the offline table within 0.01) | **held**: all within 0.005 |
| C-P2 (`G0.10` within 0.02 of `P0L`) | **held**: +0.001 at C = 1.0, +0.008 at C = 0.3 |
| C-P3 (`G0.20` below `P0L` by 0.02 to 0.10 at C = 1.0) | **held**, at the edge: -0.098 (0.241 against 0.339) |
| C-P4 (`G0.20` above `P0L` by 0.02 to 0.10 at C = 0.3) | **missed low**: +0.015 (0.111 against 0.096); the falsifier (at or below `P0L`) did not fire |
| C-P5 (host issue time lower by 0.15 to 0.40 ms at C = 1.0) | **missed low**: -0.087 ms (1.264 against 1.351); the falsifier (under 0.05) did not fire |
| C-P6 (compute-lane busy at C = 0.3 for `G0.20` between 2.5 and 5.5 % above the floor, below `P0L`'s) | **not testable**: level 4 records no per-layer events, so the harness produced no compute-busy rows at C = 0.3 ms (my design error: the stage B interference numbers at 0.3 ms came from L0) |
| C-P7 (8 %: `G0.20` below `P0L` by 0.02 to 0.12) | **held**: -0.088 |
| C-P8 (`H_O` 0.80 to 0.88 at C = 1.0 and 0.4 above every gated arm) | **held**: 0.850; 0.51 above `G10` |
| C-P9 (19 cells inside 15 minutes, gates hold) | **held**: 301 s, all gates true |

Of the nine ids: **6 held, 2 missed low without the falsifier firing (C-P4, C-P5), 1 not testable (C-P6), 0 falsified.**

## 4. What the result is, and is not

* **Measured: a weight gate does not pay at 1 ms layers.** `G10` is `P0L` (0.340 against 0.339; it drops only 1.6 % of loads); `G20` loses 0.098 of hidden share at 17 % and 0.088 at 8 %. Its 14.3 fewer copies a token (16 %) saved only 0.087 ms of host issue time, not the 0.27 ms priced: about 6 microseconds a removed copy, not 19, so the host cost of a copy is not proportional to the copy count (the fixed per-layer part is large). The serial-model bound fell 0.075 and the measured hidden share fell 0.098: the host saving gave back almost nothing.
* **Measured: at 0.3 ms the gate is within noise of a win.** `G20` is 0.015 above `P0L` in hidden share and 0.14 ms (0.7 %) faster in wall time; the registered band (0.12 to 0.20) was too optimistic. Whether 0.015 is real at one run is not shown (the pair was run once; the group's two controls agree to below 0.1 %). `H_O` there is 0.71: at a 0.3 ms window the predictor's coverage, not wasted copies, is what separates it from the oracle.
* **Derived: the lane is not the limit at 0.3 ms.** If wasted copies saturated the copy lane (H-gate-lane), cutting 16 % of copies would have lifted the hidden share well above 0.015; it did not. So the loss at 0.3 ms comes from the window (copies cannot start until the prediction is known) and coverage, not from the copy count (hypothesis).
* **Not shown.** Any other threshold (0.15 was not run), any adaptive rule by layer time, a better predictor than the recorded one, the compute-lane interference at 0.3 ms at level 4 (C-P6), this instance's population only, a native runtime.
* **Consequence for the lane (derived).** The weight gate as registered is closed on this card for 1 ms layers and is worth at most a fraction of a percent at 0.3 ms. Of the open levers, fewer and larger copies is now less attractive than it looked (the per-copy host cost shrank by only 6 microseconds per removed copy), a native issue path is untested, and predictor coverage (the oracle's 0.85 against 0.34) remains the large unexplored term.

## 5. Cost and housekeeping

GPU time on the instrument: smoke runs 23 s, the full run 301 s; the instance was up well under an hour. Its billed hours are Hamed's to enter in the private cost file. The instance should be terminated by Hamed; the ssh config, known_hosts file and key for the S1, S2, V1, N2, B1, B2, N3 and N3-C instances are Hamed's to delete (nothing was deleted here).
