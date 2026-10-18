# CUDA lane session V1: results (Tesla V100-SXM3 32 GB: the PCIe 3 contrast)

Run 2026-10-18 under `docs/CUDA-V1-EXPERIMENT-RECORD.md` (17 predictions committed in d507e14 before any instance existed). Instrument: `benchmarks/cuda/` at scaffold commit 4a6efdc (`REVISION` dirty=0), runtime 0.66.20. Every number is **measured** unless marked **derived**, and describes one rented virtual machine on one day (plan section 9; S2 showed that two instances of one class can differ two-fold). Sanitized manifests and rows are in `docs/cuda-results/v1-v100-2026-10-18/`; raw samples stay local. Billed time: to be filled from the console's transactions page (Hamed logs it in the private cost file). No hostname, address, key or amount appears in this repository.

## 1. The machine (N0b inventory)

| item | value |
|---|---|
| GPU | Tesla V100-SXM3-32GB, compute capability 7.0, 32,494 MiB total and 32,183 MiB free to the framework, 6 MiB L2 (framework property) |
| link | PCIe generation 3, width 16, negotiated equal to the maximum |
| host | KVM virtual machine, 4 vCPUs (Xeon Platinum 8168 class), 83.5 GiB RAM, locked-memory limit about 10.4 GiB |
| software | driver 580.178.04, Ubuntu 24.04, torch **2.7.1+cu126**, arch list `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90` (the cu128 wheel of S1 and S2 has no `sm_70`); the new arch gate in `session.sh setup` passed |

## 2. Results by field name (baseline, n = 30 a cell, CUDA events; the S2 profile three times)

Pinned H2D is 11.4 GB/s at 3 MiB, 11.6 GB/s at the DeepSeek record (9.95 MB, 0.86 ms), 11.7 at 14.16 MB and 21.1 MiB, 11.9 at 64 and 256 MiB: a flat plateau from 3 MiB up, 75 % of the PCIe 3.0 x16 theoretical 15.75 GB/s (**derived**). Pageable H2D is 4.3 to 5.0 GB/s at every size from 3 MiB; pageable D2H 4.3 to 5.2; pinned D2H 12.7 to 13.1 GB/s. The 1 MiB pinned cell and everything under 1 MiB are below the instrument's resolution (timer overhead 16.4 us).

| quantity | value |
|---|---|
| device-to-device peak (read plus write, 512 MiB) | 883 GB/s (98 % of the vendor 900; **derived** share) |
| launch (framework, python and dispatch) | 6.9 us |
| router chain, 40 layers (kernel, 4-byte D2H, host wait) | 0.81 ms (20.2 us a layer) |
| timer overhead (event pair) | 16.4 us |
| fp16 expert-shaped cell, m = 128: single projection | 36.0 TFLOP/s (its event time is below resolution) |
| same, a batched launch of 16 projections | 72.8 TFLOP/s (loop of 16: 46.6) |
| batched against loop at g = 8, m = 128 | 56.0 against 45.8 TFLOP/s, 1.22 times |
| overlap cell of the baseline (`overlap.copy_vs_compute`, S1's cell) | 7.2 % slowdown |

**S2 profile** (3 runs, 9.9 to 12.3 s each, all gates true): `compute_slowdown_with_copy`, `overlap.resident.h2d` -1.3 to +0.2 %, `overlap.large.h2d` 8.9 to 12.3 %, `overlap.resident.d2d` 218 to 219 %, `overlap.large.d2d` 203 to 220 %. Batched copy, `per_expert_ms`: `k1.single` 0.844 to 0.860; `k2.single` 0.842 to 0.848; `k4.single` 0.839 to 0.844; `k8.single` 0.838 to 0.840; `loop` at k = 2 / 4 / 8: 0.850 to 0.863, 0.848 to 0.856, 0.846 to 0.858. `control_drift` 0.5, 0.6 and 1.3 %. The baseline's own `control_drift` 0.9 %.

## 3. Scoring the registered predictions

| id | outcome |
|---|---|
| V1-P1 (cu126 lists `sm_70`) | **held** |
| V1-P2 (capability 7.0; 31 to 32 GiB total, 30 GiB or more free) | **held** (31.7 GiB total, 31.4 free) |
| V1-P3 (PCIe 3.0 x16, probability 0.65) | **PCIe 3.0 x16** |
| V1-P4 (CUDA-H1: 0.70 to 0.88 L at 64 MiB; 9.95 MB at least 85 % of that rate) | **held**: 0.75 L (11.87 GB/s); the record size reaches 98 % of the 64 MiB rate (the 5090 reached 54 %) |
| V1-P5 (CUDA-H2: pageable 0.60 to 0.95 of pinned at 9.95 MB) | **missed low, falsifier fired**: 0.43 (4.98 against 11.62 GB/s); pageable lags pinned strongly here |
| V1-P6 (D2H within 15 % of H2D at 64 MiB) | **held** (+10.6 %) |
| V1-P7 (device-to-device 650 to 840 GB/s) | **missed high**: 883 GB/s (falsifier was below 550 or above 900) |
| V1-P8 (CUDA-H3: single 40 to 80 % of peak; batched g = 8 at least 1.5 times the loop) | single **missed low** (about 32 %, **derived** from 36 TFLOP/s; the time row itself is below resolution); batched **missed** (1.22 times at m = 128, 1.33 at m = 1); the falsifier (gain under 10 %) did not fire |
| V1-P9 (35 to 85 TFLOP/s at m = 128) | **held** (36.0 to 72.8) |
| V1-P10 (launch 3 to 10 us; 40-layer chain 0.2 to 1.0 ms) | **held** (6.9 us; 0.81 ms) |
| V1-P11 (timer overhead 1 to 6 us) | **missed high, falsifier fired**: 16.4 us (above the 10 us line), which puts many small cells below resolution |
| V1-P12 (CUDA-H7: 2 to 20 % slowdown in each of resident and large) | resident **missed low** (-1.3 to +0.2 %; the baseline's S1 cell read 7.2 %); large **held** (8.9 to 12.3 %); falsifiers (above 30 %, or resident above large by more than 5 points) did not fire |
| V1-P13 (saturating d2d 200 to 450 %) | **held**, at the bottom (203 to 220 %) |
| V1-P14 (k = 1: 0.80 to 1.10 ms; k = 8: 0.75 to 0.95; batching gains under 20 %) | **held**: 0.84 to 0.86 and 0.84; k = 8 is within 2 % of k = 1 |
| V1-P15 (loop within 10 % of single at k = 8) | **held** (within 2 %) |
| V1-P16 (control drift under 8 %) | **held** (0.5 to 1.3 %) |
| V1-P17 (gates true, no failed cell, smoke 2 to 6 s, baseline under 30 s, S2 profile under 30 s) | gates and completion **held** in all five runs; two timing sub-intervals **missed** (smoke 7.4 s, baseline 33.8 s); S2 profile 10 to 12 s held |

## 4. Findings (post hoc where marked)

1. **The expert-sized copy over PCIe 3 costs 0.84 ms on this V100 (11.8 GB/s) against 0.35 ms on the RTX 5090s (derived ratio 2.4).** The curve is flat from 3 MiB: no fixed-cost ramp like the 5090's (54 % of its plateau at the record size). CUDA-H1 holds here and fails on the 5090, so the ramp is a property of that card or its link, not of the workload.
2. **Batching buys nothing on this link.** Every `k` costs 0.84 ms an expert, for one transfer or a queue (within 2 %); the link is the limit. On the 5090 batching and queueing recovered up to 40 % (S2). The two cards disagree about whether batching matters, as the plan's comparability rules (R2, R3) require stating.
3. **H9, derived and assumption-bound.** The Mac's 2.0 ms a decode miss against 0.84 ms here is a cost ratio of 2.4 (the 5090: 5.7). With the N0a slope of about 3.0 misses a token a GiB at 8 to 17 % resident (two traces, no GPU), the host tier's value is about 2.5 ms a GiB here against 1.7 to 2.2 on the Mac: **on this PCIe 3 card the value of a resident GiB is about the same as, not smaller than, the Mac's**, because the slow link offsets the cheaper miss. Caveats: pinned host memory as the tier (83 GiB RAM here), the slope from traces recorded on the Mac, nothing here is a token time.
4. **Compute under copies: not reproduced on the V100 for the reused-matrix cell, present for the large-weights cell.** -1.3 to +0.2 % for `overlap.resident.h2d` in the S2 profile but 7.2 % for the same cell in the baseline run minutes earlier (n = 30 each): the within-session spread on one VM is as large as most effects here. With 6 MiB of L2 neither the 23.6 MB reused matrix nor the large set fits the cache, so **this card cannot test the L2 hypothesis at all** (the 5090's 96 MiB could, and S2 did not support it). The 9 to 12 % with eight distinct matrices is a candidate for a copy effect on a memory-bound cell (the cell reads 189 MB in 0.52 ms, about 360 GB/s, **derived**); one session, three runs, not established. The device-to-device interferer again starves the matmul (203 to 220 %).
5. **The V100 is faster than predicted as a memory device and has a coarser timer:** 883 GB/s of copy bandwidth is 98 % of the vendor figure, and event pairs cost 16.4 us (3 times the 5090's), so every cell under about 160 us is below resolution there, including the single-projection time that H3 needs. A single-projection claim on the V100 would need a longer-running variant or a host timer over many launches.
6. **Pageable lags pinned by 2.3 to 2.8 times at every size from 3 MiB** (4.3 to 5.2 against 11.4 to 13.1 GB/s), against 0.85 on the 5090 at the record size and 0.47 at 64 MiB; a pageable staging path would roughly double the cost of an expert copy here (derived).
7. **Operations.** The new arch gate passed on a real torch and the cu126 wheel carries `sm_70`; no cell failed; the instance networked at first boot (S2's first VM did not); the S2 profile took 10 to 12 s here against 3 s on the 5090.

## 5. What this does not establish

One V100 SXM3 VM, one day, 74 seconds of GPU run time in total. No model, no token time, no Cachalot code, no fp8 or bf16 cell (gated off by the capability, as registered). Nothing transfers to other cards (plan section 11, R10); the L2 question is untested on the V100; the H9 reading is derived from Mac-recorded traces; H4, H5 and H6 are untested on the GPU.

## 6. Smallest next experiment

The cheapest informative step is offline: re-state CUDA-H9 with the measured copy costs of both cards (0.35 and 0.84 ms) and the N0a slopes, and carry the restated form into N2 stage 1 (a synthetic and recorded-trace replay with a real slot arena and real H2D copies on one card). That needs a design record and Hamed's go for more GPU time; no further microbenchmark session is needed to start it.
