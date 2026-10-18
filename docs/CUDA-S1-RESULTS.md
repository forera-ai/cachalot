# CUDA lane session S1: results (RTX 5090, inventory, smoke, baseline)

> **Correction (added 2026-10-18 after N2 stage 1, `docs/CUDA-N2S1-RESULTS.md` section 4):** the 9.95 MB pinned host-to-device cost reported below (0.35 ms, 28 GB/s, "a fixed cost of about 160 us") and the CUDA-H1 falsification at the record size depend on which pinned host buffer the cell got: a fast-backed buffer of that size copies in 0.182 ms (54.7 GB/s, the plateau). Findings 1, 2 and 4 and the H1 and H9 readings below are to be read with that correction; the measurements themselves are as recorded.

Run 2026-10-18 under `docs/CUDA-S1-EXPERIMENT-RECORD.md` (predictions committed in f15ef2a before any instance existed). Scaffold commit af4ed4b (`REVISION` file), runtime 0.66.15. Instrument: `benchmarks/cuda/` over SSH from the Mac. Labels follow plan section 0: every number below is **measured** unless marked **derived**, and all of them describe one rented virtual machine on one day (plan section 9: a shared host drifts, one run is one run). Sanitized manifests and result rows are in `docs/cuda-results/s1-rtx5090-2026-10-18/`; raw samples stay local. Billed time: 0.36 GPU-hours (21 min 22 s, from the provider's billing page; currency stays in the private log). The measuring itself took about three minutes of GPU run time (6 runs, 2 to 9 s each), the rest was setup and copy-back. No hostname, address, instance identifier, account or amount appears in this repository.

## 1. The machine (N0b inventory, measured)

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, 170 SMs, 32,607 MiB reported by the driver (31,602 MiB free to the framework after warm-up) |
| virtualization | KVM virtual machine, 7 vCPUs (AMD Ryzen 7 7800X3D class), 23.5 GiB host RAM, locked-memory limit 2.9 GiB |
| link | PCIe generation 5, width 16, negotiated equal to the maximum |
| software | driver 580.159.03 (CUDA 13.0 capable), Ubuntu 24.04, torch 2.7.1+cu128 whose arch list includes `sm_120` and `compute_120`; no `nvcc` on the host (not needed: no custom kernels yet) |
| profiler | CUDA events only; Nsight not probed; GPU clock and some power fields read implausibly inside the VM (recorded, not used) |

The host RAM (23.5 GiB) is small next to the 142 GiB DeepSeek bank and to a 32 GB card: the second tier of this machine cannot hold more than a fraction of it (plan risk table, R6).

## 2. Results (baseline profile, 30 timed iterations a cell, CUDA events; one run of four is shown, the repeats are in section 4)

Copy bandwidth in GB/s (p50 time in microseconds in brackets); the pinned plateau is about 52 GB/s, 83 % of the link's theoretical 63 GB/s (**derived**):

| bytes | pageable H2D | pageable D2H | pinned H2D | pinned D2H |
|---:|---|---|---|---|
| 1 MiB | 14.7 (72) | 14.0 (75) | below resolution | below resolution |
| 3 MiB (a read) | 20.8 (151) | 17.0 (185) | 27.1 (116) | 54.9 (57) |
| 9.95 MB (DeepSeek expert) | 24.3 (410) | 18.3 (545) | **28.4 (350)** | 55.1 (181) |
| 14.16 MB (GLM expert) | 24.8 (570) | 18.2 (777) | 30.4 (467) | 55.1 (257) |
| 21.1 MiB (MiniMax expert) | 22.6 (978) | 18.5 (1194) | 41.0 (539) | 55.1 (402) |
| 64 MiB | 24.5 (2739) | 16.3 (4111) | 52.4 (1281) | 55.0 (1219) |
| 256 MiB | 20.8 (12914) | 18.2 (14736) | 52.2 (5141) | 54.8 (4894) |

Other measured quantities (the first baseline run): device-to-device copy 1,517 GB/s (read plus write, a 512 MiB buffer; 85 % of the vendor 1,792, **derived**); framework kernel launch 2.96 us; a 40-layer chain of (kernel, 4-byte D2H, host wait) 7.5 us a layer, so 0.30 ms for 40 layers; event-pair timer overhead 5.1 us; fp16 GEMM at an expert-like 4096 x 2880 projection 100 to 135 TFLOP/s at 128 rows. A batched launch of 8 projections (one row) reaches 97 % of the measured copy peak where a loop of 8 launches reaches 41 %; with 16 projections the batched share exceeds 100 %, which only says that a pure read can beat the read-plus-write copy figure used as the divisor (the divisor is the plan's chosen peak, not a vendor limit). Compute slowed by 21 to 26 % (six runs) while pinned 64 MiB H2D copies covered its window.

All gates held in all six runs: copy round trips byte-equal, the matmul matches the float32 CPU reference, no sanity violation, no failed cell; the baseline took 9 s of its 300 s cap and the smoke run 2 s of 45 s.

## 3. Scoring the registered predictions

| id | prediction | outcome |
|---|---|---|
| S1-P1 | capability 12.0, `sm_120` in the arch list, driver CUDA 12.8 or newer | **held** (12.0; `sm_120` present; driver 580) |
| S1-P2 | VRAM 31.5 to 32.0 GiB total, 30.5 GiB or more free | **held** (31.8 GiB total by the driver, 30.9 GiB free to the framework) |
| S1-P3 | negotiated link: gen5 x16 with probability 0.55 | gen5 x16 |
| S1-P4 (CUDA-H1) | 0.70 to 0.92 x L at 64 MiB; and 9.95 MB at least 85 % of the 64 MiB rate | first part **held** (52.4 GB/s = 0.83 x 63); **second part missed**: 28.4 against 52.4 is 54 %, the falsifier fired. The copy ramps with size: about 160 us of excess time at the expert size against the plateau |
| S1-P5 (CUDA-H2) | pageable at 9.95 MB is 0.25 to 0.60 of pinned | **missed, falsifier fired**: 24.3 against 28.4 GB/s is 0.85 (within 20 % of pinned). At 64 MiB it is 0.47 and at 21.1 MiB 0.55, so H2 holds at large sizes and fails at the DeepSeek record size |
| S1-P6 | pinned D2H within 15 % of pinned H2D at 64 MiB | **held** (55.0 against 52.4) |
| S1-P7 | device-to-device peak 1,350 to 1,700 GB/s | **held** (1,517) |
| S1-P8 (CUDA-H3) | single projection 40 to 75 % of peak; batched g = 8 at 70 to 90 % | the single-launch time row was below resolution in every run (a 37 us kernel against a 5 us timer), so the share is only indicative (about 42 % from the TFLOP/s row, **derived**, inside the interval); the batched g = 8 share is 97 %, **above** its interval. The H3 falsifiers (single launch at or above 80 %; grouping gain under 10 %) did not fire: grouping is 2.4 times faster than the loop at g = 8 |
| S1-P9 | launch 5 to 20 us, a layer chain 15 to 60 us | **missed low**: 2.96 us and 7.5 us. The falsifier (a chain above 100 us) did not fire. Idle synchronize was below resolution |
| S1-P10 | timer overhead 1 to 5 us | **missed by 0.1 us** (5.1 us) |
| S1-P11 | the 1 MiB pinned H2D cell is below resolution (probability 0.5) | **yes**, in every run |
| S1-P12 (CUDA-H7) | compute slowdown under 5 % while copies run (probability 0.7) | **missed, falsifier fired**: 21 to 26 % in six runs. See finding 3 |
| S1-P13 | control drift under 5 % (probability 0.8) | **partly missed**: 3.4, 7.6, 1.0, 3.3, 1.5 and 5.7 % over the six runs; two of six reach 5 % |
| S1-P14, P15 | smoke and baseline complete inside their caps | **held** (2 s of 45 s; 9 s of 300 s) |
| S1-P16 | at least one cell fails with an exception on the real GPU (probability 0.4) | **did not happen** (the CPU rehearsal before the session did its job) |
| S1-P17 | all gates hold on a clean run | **held** in all six runs |

## 4. Run-to-run variation (four baseline runs and two smoke runs, same instance, minutes apart)

Pinned H2D at 9.95 MB: 28.4, 28.2, 28.1, 27.7 GB/s over the four baselines (spread 1 %), 28.6 and 27.7 in the two smoke runs. Pinned H2D at 64 MiB: 52.4, 52.4, 49.8, 52.3. Pageable H2D at 9.95 MB: 24.3, 24.4, 24.3, 23.8. Pinned D2H at 9.95 MB: 55.0 to 55.1. Device-to-device: 1,516.5 to 1,517.4. Launch 2.96 to 3.0 us, layer chain 7.44 to 7.58 us. Compute slowdown 0.209 to 0.262. The numbers are stable within a session, which says nothing about another instance or another day.

## 5. Findings (post hoc where marked)

1. **The cost of one expert-sized copy is measured: 9.95 MB over pinned host memory takes 0.35 ms (28.4 GB/s)**, 0.47 ms for 14.16 MB and 0.54 ms for 21.1 MiB. This replaces the N0a record's assumed 0.40 to 1.0 ms; it is slightly below the low end. A copy at the expert size reaches only 54 % of the plateau because a fixed cost of about 160 us rides on every transfer, so batching experts into larger transfers would raise the effective rate (not tested).
2. **CUDA-H9 as stated is falsified on this machine (derived).** H9 expected the value of a GiB of fast memory to be at least 5 times smaller on a PCIe host tier than on the Mac's SSD tier. The Mac's 2.0 ms a miss against this machine's 0.35 ms is a cost ratio of 5.7. Multiplying the N0a slopes (about 3.0 misses a token a GiB in the 8 to 17 % region; 0.83 in the Mac's 25 to 34 % region) gives about 1.1 ms a GiB here against 1.7 to 2.2 ms a GiB on the Mac: a ratio of 1.5 to 2, nowhere near 5. The cheaper miss is largely offset by the steeper small-fraction curve, as N0a suggested. Caveats: the miss source here is pinned host memory, and this VM's host RAM is 23.5 GiB, so a host tier of that size holds a fraction of a bank; the slopes come from two traces; nothing here is a token time.
3. **Concurrent copies slow compute by 21 to 26 %, against a predicted under 5 % (H7 falsified as registered).** The copy traffic is about 3.5 % of device bandwidth, so bandwidth contention does not explain it. A hypothesis, not tested: the compute in that cell re-reads a 23.6 MB weight matrix that fits the 5090's large L2 cache when it runs alone, and a 64 MiB copy streaming through the device evicts it; the matmul cells flush L2 deliberately and this one does not. The next session should add a flushed or a larger-weights variant before the slowdown is called a property of copy engines.
4. **Pinned memory is not generally faster than pageable here.** At the DeepSeek record size pinned H2D is only 17 % faster than pageable (28.4 against 24.3 GB/s); the advantage appears only above about 20 MB (52 against 24.5 at 64 MiB). Pageable D2H is 3 to 3.4 times slower than pinned D2H at the same sizes (16 to 18.5 against 55 GB/s). H2D and D2H are asymmetric for pinned memory at expert sizes (28.4 against 55.1 GB/s at 9.95 MB).
5. **The router-sync cost is small on this host: 7.5 us a layer, 0.30 ms for 40 layers** (a lower bound: a real router also has top-k and the next layer's launch). The Mac's 40 router syncs set about 9 to 12 ms of its 80 ms floor. CUDA-H4 (sync at least 10 % of an all-resident token) cannot be tested without a token; it would need a layer loop and a model.
6. **Defect found:** none in the cell code. One operational defect: the host lacked the Python venv package, so `session.sh setup` stopped. The agent was not given sudo; pip was bootstrapped into a venv without it (a hash-verified pip 25.1.1 wheel from PyPI, no system change), which `session.sh` now does itself. The first-run failure the record gave a 0.4 chance of did not occur, and the cell logic had been exercised on the CPU first.

## 6. What this does not establish

One VM, one day, one session; the other GPU (V100) and the PCIe 3 contrast are untested. No token time, no model, no Cachalot code ran on the GPU: these are microbenchmarks, and working microbenchmarks do not make a CUDA Cachalot. Hardware-specific limits stay with the RTX 5090: no statement here transfers to a V100, to a datacenter card or to other hosts. CUDA-H1 and CUDA-H2 failed their registered falsifiers at the record size; H3 survived; H7 failed; H9 as stated failed (derived); H4, H5 (the CUDA replay) and H6 are untested on the GPU.

## 7. Smallest next experiment

A second, short RTX 5090 session (or the V100 session D3 planned) that (a) re-measures the overlap with a flushed-L2 and a larger-weights variant, to separate L2 eviction from copy-engine interference; (b) adds a batched-copy cell (several expert records in one transfer) to test the fixed-cost explanation of finding 1; (c) repeats the control cell across a second day. The predictions for it are written first, in a new record, before any instance is started.
