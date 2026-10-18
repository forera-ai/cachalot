# CUDA lane session S2: experiment record (RTX 5090: what makes compute slow under copies, and does batching experts cut the fixed copy cost)

Written 2026-10-18 against runtime 0.66.16, **before any instance exists, before any S2 cell is written, and before any S2 measurement**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void (plan section 12). Labels follow plan section 0. This follows `docs/CUDA-S1-RESULTS.md` section 7 ("smallest next experiment"). Nothing here starts an instance: that needs Hamed's yes (plan section 2, rule 1).

## 1. Baseline (recorded, S1, one RTX 5090 virtual machine, one day)

* Pinned H2D: 9.95 MB in 350 us (28.4 GB/s), 64 MiB in 1,281 us (52.4 GB/s). A two-parameter reading of those two points is a fixed cost of about 160 us plus a plateau of about 52 GB/s (**derived**; the S1 results call the 160 us "excess time at the expert size").
* Compute (8 repeated fp16 matmuls, 128 x 4096 by 4096 x 2880, one 23.6 MB weight matrix reused) slowed 21 to 26 % in six runs while pinned 64 MiB H2D copies covered its window. Copy traffic is about 3.5 % of device bandwidth (derived), so bandwidth contention does not explain it.
* Device-to-device copy 1,517 GB/s; an RTX 5090's L2 cache is large (vendor figure about 96 MB, **derived input**, the doctor does not read it; S2 records `l2_cache_size` from the device properties if the framework reports it).
* Unexplained, **hypothesis H-L2** from S1 finding 3: the single 23.6 MB weight matrix is re-read from L2 when the compute runs alone, and a streaming copy evicts it, so the slowdown is a cache effect of the cell design, not a property of copy engines.
* Competing hypothesis **H-copy-engine**: concurrent H2D traffic (the PCIe DMA writes into device memory through the memory controller) slows memory-bound kernels by a fixed fraction regardless of the cache.

## 2. Questions and cells (the S2 profile)

All cells reuse the S1 instrument (`benchmarks/cuda/suite.py`), CUDA events, n = 30 timed iterations, warm-up 5. New cells are listed with the arm names the results will use.

| id | cell | what varies |
|---|---|---|
| S2-A | `overlap.resident.h2d` (the S1 cell, repeated) | weights: one 23.6 MB matrix reused; interferer: pinned H2D 64 MiB stream |
| S2-B | `overlap.large.h2d` | weights: 8 distinct 23.6 MB matrices (189 MB, above the L2 size), each used once per window; interferer: the same H2D stream |
| S2-C | `overlap.resident.d2d` | weights as S2-A; interferer: a device-to-device copy of a 256 MiB buffer on a second stream (evicts L2, uses no PCIe) |
| S2-D | `overlap.large.d2d` | weights as S2-B; interferer as S2-C |
| S2-E | `batchcopy.k{1,2,4,8}.{single,loop}` | k DeepSeek experts (9.95 MB each) as one pinned H2D transfer of k x 9.95 MB (`single`), or k queued transfers of one expert (`loop`, one event pair around all k); per-expert time derived |
| S2-F | control: pinned H2D 9.95 MB, first and last | drift bound within the session |

## 3. Predictions (registered before the session)

All **predicted**. "Slowdown" is p50(compute with interferer) / p50(compute alone) - 1.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| S2-P1 (H-L2 core) | slowdown S2-B (large weights, H2D interferer) | below 8 % (probability 0.45) | 8 % or more: then the weight cache is not the cause of S1's 21 to 26 % |
| S2-P2 | slowdown S2-A (S1 cell, repeated) | 15 to 32 % (reproduces S1; probability 0.8) | outside 15 to 32 % |
| S2-P3 (separates cache from PCIe) | slowdown S2-C (resident weights, D2D interferer) | 10 % or more (probability 0.6): a cache-evicting load with no PCIe slows the resident-weight compute too | below 10 %: eviction alone does not slow it, so the S2-A slowdown is not an L2 effect |
| S2-P4 | slowdown S2-D (large weights, D2D interferer) | below 8 % (probability 0.45) | 8 % or more: a device-memory-bandwidth effect that the cache does not explain |
| S2-P5 | rule for the verdict | H-L2 is **supported** if S2-A at least 15 % and S2-B below 8 %; **H-copy-engine is supported** if S2-B at least 15 %; anything between is reported as mixed | n/a (rule, fixed now) |
| S2-P6 | batched single transfer, per-expert time at k = 2 / 4 / 8 | 0.25 to 0.30 / 0.21 to 0.26 / 0.19 to 0.23 ms (the two-parameter model of section 1, 160 us fixed + 52 GB/s, gives 0.27 / 0.23 / 0.21) | k = 8 per-expert time above 0.30 ms: the fixed-cost explanation of S1 finding 1 is wrong |
| S2-P7 | `loop` of k queued transfers, per-expert time at k = 8 | 0.25 to 0.35 ms, i.e. between the batched figure and S1's isolated 0.35 (queued transfers overlap their launch costs but not their device-side setup; probability that the loop is at least 15 % cheaper per expert than 0.35 ms: 0.5) | n/a (scored by outcome) |
| S2-P8 | pinned H2D bandwidth at k = 8 (79.6 MB) | 47 to 53 GB/s | outside |
| S2-P9 | control drift (first and last S2-F) | under 5 % (probability 0.75) | 5 % or more |
| S2-P10 | all gates hold on a clean run: copied bytes equal, matmul against the float32 reference, no sanity violation | probability 0.85 | any false |
| S2-P11 | no cell fails with an exception (the new cell code is rehearsed on the CPU first) | probability 0.8 | any failed cell |
| S2-P12 | the S2 profile finishes inside 120 s of GPU run time (estimated upper bound from the plan; the wall-clock cap is the bound) | probability 0.85 | truncated |
| S2-P13 | the framework reports an L2 cache size for the device | 96 MiB or close (probability 0.6; may be absent) | n/a (scored by outcome) |

## 4. Stop rules and what each outcome means

* Stop and bring the files home (no further cells) on: a failed cell, a false gate, a truncated run, the session reaching 80 % of its cap before the S2 profile starts. A defect found on the instance is fixed on the Mac, not on the billed clock. Session cap: **1 GPU-hour** (S1 used 0.36; the S2 profile is about two minutes of run time plus setup).
* If S2-A, S2-B, S2-C and S2-D all land inside the 8 to 15 % band, the verdict is mixed and is reported as such; no hypothesis is claimed.
* CUDA-H7 is restated only from this session's rows, by field name (never by column position: S1's first summary mislabelled columns).
* A batched transfer that wins here is a statement about this card's PCIe fixed cost; it says nothing about a real expert-gather path (a real batch needs the k experts contiguous in pinned memory, which a cache-miss pattern does not give). No Cachalot CUDA claim follows from S2.
* The V100 session (D3) and a second-day repeat are not part of S2; a second day is only a control repeat and is Hamed's scheduling call.

## 5. Needs Hamed

A yes to provision one RTX 5090 instance for a session of at most 1 GPU-hour; the provider SSH key from S1 deleted and the pasted API key rotated (v221). The agent does not start an instance and runs over SSH only after Hamed instructs it for a specific live instance.
