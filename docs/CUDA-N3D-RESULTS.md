# CUDA lane N3 stage D: results (a lead-2 prediction arm; the window hypothesis; RTX 5090)

Run 2026-10-20 under `docs/CUDA-N3D-EXPERIMENT-RECORD.md` (ten predictions committed in a2335b8 before any stage D harness code; an exception, chosen by Hamed, to the offline lead rule). Harness: `benchmarks/cuda/overlap_arena.py` at scaffold commit 70515c1 (`REVISION` dirty=0), runtime 0.66.39; trace `routing-trace-deepseek-leads` (the multi-lead recording, sha256 prefix `b68aefa4adef11a1`). Instance: an RTX 5090 virtual machine (driver 580.159.03, torch 2.7.1+cu128, 7 vCPUs, 24 GiB RAM, PCIe 5.0 x16) in the provider's Korea region. Raw rows, manifests and per-token samples: `docs/cuda-results/n3d-rtx5090-2026-10-20/`. Labels follow plan section 0: measured on this one instance unless it says derived or hypothesis. The compute is a stand-in and the host a Python loop over torch.

## 1. What ran

Two earlier instances (IDs `42b42b9e` and `6c8aea92`, the second on the same address) had a GPU that did not initialize: the device was on the PCI bus (`10de:2b85`), `nvidia-smi` failed, no `/dev/nvidia*`; per the N3-B rule they were replaced, not debugged, and nothing was uploaded to them. Hamed created a third instance **without** terminating the faulty one first, which got a different node and worked at its first login (`nvidia-smi`, DNS fine). On it `check`, `setup`, `doctor`, `smoke` and `n3dsmoke` (10 cells, 35 s) were clean and `n3dfull` (23 cells, **322.8 s** against the 900 s cap) ran with **every gate true** (event hash, determinism, counts, `chain2`, `coverage` (gate g), readback, calibration, control drift, closure, sanity), no failed cell. Source records: c = 0.194 ms, 4.7 % slow (the fast population). Control drift below 0.001 in every group.

## 2. Results (400 decode tokens, token 0 a warm-up; p50 per token, ms)

`E` = wall minus the group's floor; `H` = `1 - E / E_D` within the group.

| group | floor | `E_D` | `E_P0L` | `E_L2` | `E_O` | `H_P0L` | `H_L2` | `H_L2 - H_P0L` | `H_O` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 %, C = 1.0 ms, L5 | 39.981 | 8.669 | 4.855 | 5.774 | 1.244 | **0.440** | **0.334** | -0.106 | 0.857 |
| 17 %, C = 0.6 ms, L4 | 23.891 | 8.921 | 5.279 | 6.217 | 1.709 | 0.408 | 0.303 | -0.105 | 0.808 |
| 17 %, C = 0.3 ms, L4 | 12.447 | 8.841 | 6.782 | 8.139 | 2.461 | **0.233** | **0.079** | **-0.154** | 0.722 |
| 8 %, C = 0.3 ms, L4 | 12.455 | 16.118 | 13.910 | 15.806 | | 0.137 | 0.019 | -0.118 | |

Counts a token at 17 %: `P0L` 21.96 demand misses, 50.36 loads, 23.08 wasted; `L2` 27.74 / 52.42 / 29.43; at 8 %: `P0L` 37.32 / 88.82, `L2` 47.12 / 90.32. These equal the offline model (`prefetch_coverage.py --timed`) to the printed digits (gate g). Host issue time a token at C = 1.0 ms: `D` 0.964, `P0L` 1.295, `L2` 1.383, `O` 1.167. Wall time at C = 0.3 ms (17 %): `P0L` 19.228, `L2` 20.586.

## 3. Scoring the registered predictions

| id | outcome |
|---|---|
| D-P1 (`H_P0L` 0.42 to 0.54 at C = 1.0 ms) | **held**: 0.440 |
| D-P2 (`H_L2` below `H_P0L` by 0.05 to 0.18 at 1.0 ms) | **held**: -0.106 |
| D-P3 (`H_L2 - H_P0L` between -0.05 and +0.04 at 0.3 ms) | **missed low**: -0.154; the falsifier (at least +0.05) did not fire |
| D-P4 (between -0.12 and +0.02 at 0.6 ms) | **held**: -0.105 |
| D-P5 (at most +0.04 at 8 %, 0.3 ms) | **held**: -0.118 |
| D-P6 (no cell shows `L2` at least 0.05 above `P0L`) | **held**: every cell is 0.105 to 0.154 below |
| D-P7 (counts equal the offline model within 0.01) | **held**: all within 0.005 |
| D-P8 (`L2` host issue within 0.20 ms of `P0L`'s at 1.0 ms) | **held**: +0.088 |
| D-P9 (`H_O` 0.80 to 0.90 at 1.0 ms) | **held**: 0.857 |
| D-P10 (23 cells inside 15 minutes, gates hold) | **held**: 323 s, all gates true |

**9 held, 1 missed low (D-P3), 0 falsified.**

## 4. What the result is, and is not

* **Measured: the window hypothesis is not supported.** A lead-2 prediction, issued a full layer earlier, is below the lead-1 predictor in every cell, and by **more** at short layers (-0.154 at 0.3 ms) than at 1 ms (-0.106), the opposite of what a window effect would give. At 0.3 ms `L2`'s wall time is 1.36 ms above `P0L`'s. The offline serial bound (0.535 against 0.412, a gap of 0.123) predicted the 1 ms gap well (0.106) and under-predicted the 0.3 ms one.
* **Derived: at short layers copies and coverage both hurt.** `L2` has 7.8 more copies a token than `P0L` (80.2 against 72.3) and 5.8 more demand misses; at 0.3 ms the lane carries 80 copies at 0.194 ms (15.5 ms) against 12.4 ms of compute, and the earlier stage C found fewer copies alone bought almost nothing, so the larger loss is more likely coverage that arrives too early and is evicted before use than copy count (hypothesis; the extra 0.048 of wasted share is consistent but not isolated).
* **Measured, unexplained: the lead-1 predictor hides more on this trace than on the old one.** `H_P0L` at 0.3 ms is 0.233 here against 0.096 to 0.100 on the old trace (stages B and C), and 0.440 against 0.339 at 1 ms. The new trace's serial bound is higher (0.535 against 0.365), which covers 1 ms (0.82 of the bound against 0.93 earlier) but not 0.3 ms (0.44 of the bound against 0.27). Same card population (c = 0.194 ms), same harness level; I do not know why and did not test it.
* **Consequence for the lane.** The lead lever is closed on the card as well as on the bound: leads 2 and 3 lose coverage, the union adds copies, and the window does not rescue them. Predictor work now needs information the runtime does not produce.
* **Not shown.** A real model's decode, another conversation, another card population, any native issue path, the cause of the higher hidden share on the new trace.

## 5. Cost and housekeeping

GPU time on the instrument: smoke runs 37 s, the full run 323 s; the instance was up well under an hour. Billed hours are Hamed's to enter in the private cost file (three instances this stage, two diagnosed and unused). He terminates instances; the ssh configs and known_hosts files for S1, S2, V1, N2, B1, B2, N3, N3-C and the stage D instances, and the key, are his to delete (nothing deleted here, except the agent's own stale known_hosts file for the reused address).
