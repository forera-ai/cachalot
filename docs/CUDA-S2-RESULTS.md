# CUDA lane session S2: results (RTX 5090: overlap variants and batched expert copies)

Run 2026-10-18 under `docs/CUDA-S2-EXPERIMENT-RECORD.md` (13 predictions committed in 3d044d4 before any cell code or instance). Instrument: `benchmarks/cuda/` at scaffold commit 810c538 (first run) and 01bf195 (three later runs, after the race fix below); runtime 0.66.18. Instance: a second rented RTX 5090 virtual machine of the same class as S1 (driver 580.159.03, torch 2.7.1+cu128, 7 vCPUs, 23 GiB RAM, PCIe 5.0 x16; 96 MiB L2 reported by the framework). Every number is **measured** unless marked **derived**, and describes one VM on one day (plan section 9). Sanitized manifests and rows: `docs/cuda-results/s2-rtx5090-2026-10-18/`; raw samples stay local. The `region` field in the manifests reads "asia", typed by the operator from the provider's earlier listing and not read from the instance; it is unverified. No hostname, address, key or amount appears in this repository. Billed time: 0.486 GPU-hours (29 min 9 s, from the provider's transactions page; currency stays in the private cost log); the measuring itself took about 25 seconds of GPU run time, the rest was setup, a network fault and copy-back. Cumulative with S1: 0.842 GPU-hours.

## 1. What ran

One clean smoke run, then the `s2` profile four times (about 3 s of GPU run time each, minutes apart). **Run 1 failed a gate (`copy_checksum`) because of a defect in my new cell**: in the two device-to-device overlap variants the interferer's source buffer was generated on the default stream and read by the copy stream without a synchronize, so the byte check could see stale data. The CPU rehearsal has no streams and could not catch it. The fix (one `self._sync()`) is commit 01bf195; runs 2 to 4 are clean (all gates true, no failed cell, not truncated). Run 1's two `d2d` overlap cells produced no timing rows by design (a failed gate emits none); its other cells are kept as a first replicate and labelled. Before setup the first VM had no outbound network (DNS and IP unreachable); a console reboot fixed it, which cost about ten minutes of billed time and is not part of any number. One earlier ssh attempt failed on the user name and key attachment; no remote command ran until the connection worked.

## 2. Results by field name (p50 of n = 30 a cell; ranges are over the runs shown)

**Overlap** (compute = 8 fp16 matmuls of 128 x 4096 by 4096 x 2880, alone 0.203 ms; the interferer queues two copies, so it covers the compute window; slowdown = p50 with interferer / alone - 1):

| cell (weights, interferer) | `compute_slowdown_with_copy` | runs |
|---|---|---|
| `overlap.resident.h2d` (one 23.6 MB matrix reused; pinned H2D 64 MiB) | **12.3 to 13.3 %** | 4 |
| `overlap.large.h2d` (8 distinct matrices, 189 MB, above the 96 MiB L2; same H2D) | **18.8 to 22.1 %** | 4 |
| `overlap.resident.d2d` (device copy of 256 MiB) | 361 to 364 % | 3 |
| `overlap.large.d2d` | 320 to 323 % | 3 |

**Batched copy** (DeepSeek expert record 9.95 MB, pinned H2D, event time; `per_expert_ms` = p50 / k):

| cell | `per_expert_ms` over 4 runs | `bandwidth_gb_s` (events) |
|---|---|---|
| `k1.single` | 0.340 to 0.362 | 27.5 to 29.3 |
| `k2.single` | 0.321 to 0.361 | 27.6 to 31.0 |
| `k2.loop` | 0.354 to 0.368 | 27.1 to 28.1 |
| `k4.single` | 0.249 to 0.271 | 36.7 to 40.0 |
| `k4.loop` | 0.326 to 0.333 | 29.9 to 30.5 |
| `k8.single` | 0.201 to 0.242 | 41.1 to 49.6 |
| `k8.loop` | 0.219 to 0.287 | 34.7 to 45.5 |

`control_drift` (first and last pinned H2D 9.95 MB): 18.0, 6.9, 7.1 and 2.2 % over the four runs (the first control read 27.5 GB/s each time, the last 28.2 to 33.5). `l2_cache_bytes` reported by the framework: 100,663,296 (96 MiB). All four runs: `kernel_reference` and `sanity` true; `copy_checksum` false in run 1 only.

## 3. Scoring the registered predictions

| id | outcome |
|---|---|
| S2-P1 (large weights + H2D slowdown below 8 %) | **missed, falsifier fired**: 18.8 to 22.1 % |
| S2-P2 (resident + H2D reproduces S1 at 15 to 32 %) | **missed low**: 12.3 to 13.3 % (S1 on the first instance: 21 to 26 %). The same cell on a second instance of the same class reads about half |
| S2-P3 (resident + d2d at 10 % or more) | **held, trivially**: 361 to 364 % |
| S2-P4 (large + d2d below 8 %) | **missed, falsifier fired**: 320 to 323 % |
| S2-P5 (verdict rule) | **H-L2 is not supported; the registered rule picks H-copy-engine** (the large-weights cell is at 15 % or more). See finding 1 for why this is weaker than it reads |
| S2-P6 (`single` per expert at k = 2 / 4 / 8: 0.25-0.30 / 0.21-0.26 / 0.19-0.23 ms) | **missed at k = 2** (0.32 to 0.36), **partly held at k = 4** (two of four runs inside; 0.249 to 0.271) and **k = 8** (three of four inside; 0.201 to 0.242). The falsifier (k = 8 above 0.30 ms) did not fire |
| S2-P7 (`loop` at k = 8 between 0.25 and 0.35; at least 15 % cheaper than 0.35 ms: probability 0.5) | the interval **held in two of four runs** (0.219 and 0.287 are outside); the 15 % saving **held** (0.219 to 0.287, i.e. 18 to 37 % below 0.35) |
| S2-P8 (k = 8 single at 47 to 53 GB/s) | **missed low** in three of four runs (41.1, 45.7, 46.7 and 49.6) |
| S2-P9 (control drift under 5 %) | **missed in three of four runs** (18.0, 6.9, 7.1; held at 2.2) |
| S2-P10 (all gates hold on a clean run) | **held on the three post-fix runs; failed on run 1 through my stream race** |
| S2-P11 (no cell fails with an exception) | **held** |
| S2-P12 (profile inside 120 s) | **held** (about 3 s) |
| S2-P13 (framework reports the L2 size, 96 MiB or close) | **held**: 96 MiB exactly |

## 4. Findings (post hoc where marked)

1. **The 96 MiB L2 hypothesis (H-L2) is not supported by the registered test, but the test has a limit.** Making the weights larger than the cache did not remove the slowdown; it made it larger (19 to 22 % against 12 to 13 %). The alone times are equal (0.2028 against 0.2030 ms), so at this size the matmuls are not faster when the weights fit in L2, and the cell is probably compute-bound (24 GFLOP in 0.2 ms is about 120 TFLOP/s, **derived**), which is why reusing a cached weight matrix does not help it. Eviction of a weight matrix therefore cannot be the mechanism for a compute-bound cell. What the data do not show is the actual cause: copy traffic slows a compute-bound kernel by 12 to 22 %, and this note does not say whether that is a clock or power effect of the DMA engines, memory-controller arbitration or something about the VM. That is a hypothesis list, not a result, and **CUDA-H7 stays falsified (predicted under 5 %)** with a size that varies by cell and by instance.
2. **The device-to-device interferer was a design flaw for its purpose.** A 256 MiB device copy runs the memory system at full rate, so the 3 to 4 times slowdown (320 to 364 %) is bandwidth contention, not cache eviction, and says nothing about L2. S2-C and S2-D cannot separate the two explanations; the registered verdict rule used only the H2D cells, so it stands, but no conclusion should be drawn from the d2d numbers beyond "a saturating device copy starves a concurrent matmul". A cleaner L2 test would use a small, bounded eviction load (for example a stream of reads sized just over the cache, at a fixed low rate) or the profiler's L2 counters if the VM grants them (not probed).
3. **Instance-to-instance spread is large.** The S1 cell read 21 to 26 % on the first instance and 12 to 13 % here, same class, same software, same cell code, another day. One instance's slowdown is not a property of the GPU; any claim about copy and compute interference needs more than one instance. The control cell moved 2 to 18 % in minutes, so within-session drift is also larger than the 5 % I predicted (the end control reads higher than the start one in every run: a warm-up or clock-state effect is plausible, untested).
4. **Queued separate copies get most of the benefit of one batched transfer.** Eight expert-sized pinned transfers queued back to back (`k8.loop`) take 0.219 to 0.287 ms an expert (35 to 45 GB/s) against 0.34 to 0.36 for one at a time; one contiguous 79.6 MB transfer (`k8.single`) takes 0.201 to 0.242 (41 to 50 GB/s). So the fixed cost per copy overlaps when copies are queued; making the experts contiguous saves a further 10 to 15 % at k = 8, **derived from overlapping ranges** and not a tested contrast. The curve is not smooth: `k4.loop` (0.326 to 0.333) is no better than a single copy, and `k2.single` (about 29 GB/s at 19.9 MB) does not match S1's 41 GB/s at 22.1 MB; these are unexplained and are not claimed as a ramp model. A real expert-gather path does not give k contiguous experts for a cache-miss pattern, so no Cachalot claim follows.
5. **Defect and operations.** The d2d race (above) was a cell bug caught by a gate, fixed on the Mac, and re-run in minutes; the CPU rehearsal cannot test stream ordering, so any cell with two streams needs a GPU smoke before it counts. The fresh VM had no outbound network until a console reboot.

## 5. What this does not establish

One new VM, one day, 3 seconds of GPU time a run, four runs. No model, no token time, no Cachalot code. The L2 test is inconclusive in the sense of finding 2; CUDA-H7's cause is open. H4, H5 and H6 are untested on the GPU; the V100 session (D3) has not happened.

## 6. Smallest next experiment

If more RTX 5090 time is wanted: a profiler run (Nsight Compute or CUPTI L2 counters if permitted) on the compute cell with and without the H2D stream, plus the same overlap cell on a third instance on another day, to separate an instance effect from a mechanism; the predictions are written first. A V100 session (D3) is the alternative and answers a different question (PCIe 3 and no fp8). Both need Hamed's go.
