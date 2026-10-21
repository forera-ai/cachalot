# CUDA lane session K: results (the pinning route, the issue-loop synchronisation guard, the committed copy-population class, and the follow-up on allocation order)

Written 2026-10-21 against runtime 0.66.58 plus the follow-up's code (not yet released). Records: `docs/CUDA-PIN-ROUTE-RECORD.md` (K-1 to K-12, committed first in 735efd8; clarifications in 3bc8d3d before any run) and `docs/CUDA-PIN-ROUTE-FOLLOWUP-RECORD.md` (L-1 to L-9, committed first in ae41d43 before any follow-up code or run). Labels follow plan section 0. Scoring is by `benchmarks/cuda/pin_route_score.py` and `pin_route_followup_score.py`; the scored JSON is `docs/cuda-results/pin-route-2026-10-21.json` and `docs/cuda-results/pin-route-followup-2026-10-21.json`. Raw run directories are home under `~/cachalot-private/cuda-results/k1/` (archive sha256 `4735b8c3…81cf1`, verified after the copy) and `k1-followup/` (sha256 `cb7c2aa4…33c14`, verified).

## 1. What ran

**Instance.** A first RTX 5090 instance (`giant-scarf-3095`) came up with the GPU on the PCI bus and **no NVIDIA kernel module and no `/proc/driver/nvidia`** (`nvidia-smi` could not reach the driver, at 5 minutes of uptime, twice); no cell ran on it and the agent did not use sudo. The replacement `hellish-cry-4960` (RTX 5090 32 GB, driver 580.159.03, PCIe 5.0 x16 by `nvidia-smi`, 24 GB RAM, 7 CPUs, torch 2.7.1+cu128) ran everything below: `smoke` (clean), `pinroute` (six fresh processes, 1.0 to 1.4 s each), `n3ksmoke` (clean, all gates true including `guard_sees_planted`), `n3kfull` (27 cells, 415 s, all gates true, no failed cell, complete), then the follow-up `pinroute2` (nine fresh processes, 0.9 to 1.0 s each). Code at commit `080b34d` for the first run and `cf9c203` for the follow-up (`REVISION dirty=0` both times).

**The trace** `n3d-trace.npz` (sha256 checked by the run against `n3d-expected.json`).

## 2. Scores of K-1 to K-12

"In interval", "outside interval" and "past falsifier" are the scorer's words; the explanation column is mine and a hypothesis unless stated.

| id | result | score | note |
|---|---|---|---|
| K-1 | class of `k0` and `k5`: slow and slow | in interval | stable across the session |
| K-2 | plateau 41.3 GB/s (`k0`), 40.4 (`k5`) | outside interval (not past the falsifier of 40) | see section 4: the 64 MiB buffers were allocated in the same pattern that later proved slow |
| K-3 | the two rules agree on 24 of 24 buffers | in interval | trivially: every fresh-process buffer was slow by both |
| K-4 | device copy 1,525.5 and 1,518.2 GB/s (read plus write), change 0.5 % | in interval | |
| K-5 | route slow shares by the committed rule: `k1` 0.00, `k2` 0.08, `k3` 0.42, control `k4` 1.00 | **past falsifier** | confounded by the plateau; the absolute rule gives 0.75, 0.75, 0.96 and 1.00 (section 3) |
| K-6 | flag 0 against portable: 34.5 against 33.8 GB/s, 2 % | in interval | |
| K-7 | ordinary memory 31.3 GB/s against the control's 27.5: 13.8 % faster | outside interval (the falsifier is 30 %) | |
| K-8 | **zero** synchronising calls in 16,000 guarded rows (400 tokens of 40 rows) in each of the six guarded `D` and `P0L` cells at levels 3 and 4 | in interval | H-clean supported (this torch, 2.7.1, one trace) |
| K-9 | the guard raised the median host time of `P0L` by 2.1 % (level 4, C = 1.0 and 0.3) | in interval | the guarded cells are never used for timing; the cost is small |
| K-10 | the planted `.item()` was counted in 400 of 400 tokens, all from one source line | in interval | the guard sees a synchronisation on this build |
| K-11 | link generation 5, width 16 in all 38 samples (`k0` 14, `k4` 10, `k5` 14) | in interval | **H-link is not supported** (the slow class is not a lower link generation at 10 Hz sampling) |
| K-12 | every gate true, no failed cell, `pinroute` 6.9 s of GPU run time, guard runs 22.6 s and 415 s | in interval | |

**The population of the instance.** By the committed rule `k0` and `k5` read the instance as **slow** (12 of 12 buffers at 0.362 ms, 27.5 GB/s, within 0.001 ms of each other), but the arena process in the same session measured its 64 pinned source records at **0.194 ms** (the `source_copy_ms_median` row, slow share 0). The same card, in the same hour, therefore has both rates; section 4 says why.

## 3. DC-3: the pinning route

By rate (the absolute rule's view, 24 buffers a cell, interleaved allocation, small destination, default stream): `cudaHostAlloc` 0.362 ms (27.5 GB/s, all slow); a self-made mapping registered with flag 1 or flag 0 0.27 to 0.31 ms (33.8 and 34.5 GB/s; 25 % of the buffers in a third class at 0.20 to 0.28 ms); ordinary allocator memory registered with flag 0 0.27 to 0.34 ms (31.3 GB/s); none reached the arena's 0.194. No registered buffer was huge-page backed (smaps `AnonHugePages` 0 for every cell). The flag changes nothing measurable (K-6). So on this instance **registering your own memory beats `cudaHostAlloc` by about 14 to 26 % in the interleaved pattern**, and the route is a smaller effect than the one in section 4. The scorer's K-5 verdict (past the falsifier) is the registered rule's reading and is reported as scored; it is inflated by the plateau confound (the 64 MiB plateau came from degraded buffers, 1.6 to 2.1 ms), which is why the relative slow share of `k1` reads 0.00 while its rate is 33.8 GB/s against an arena rate of 51 to 52.

## 4. Follow-up L: the slow class came from how the buffers were allocated and timed

Nine fresh processes, reversed route order, 24 buffers a cell, `cudaHostAlloc` for the four context cells:

| cell | allocation | destination | stream | median ms | GB/s | slow share |
|---|---|---|---|---|---|---|
| `k4` first | interleaved | 9.95 MB | default | 0.362 | 27.46 | 1.00 |
| `k3` ordinary, flag 0 | interleaved | 9.95 MB | default | 0.335 | 29.7 | 1.00 |
| `k2` registered, flag 0 | interleaved | 9.95 MB | default | 0.300 | 33.2 | 0.83 |
| `k1` registered, flag 1 | interleaved | 9.95 MB | default | 0.300 | 33.2 | 1.00 |
| **`l1`** | **all 24 first, then timed** | 9.95 MB | default | **0.192** | **51.8** | **0.00** |
| `l2` | interleaved | slot 0 of a 24 GiB buffer | default | 0.362 | 27.5 | 1.00 |
| `l3` | interleaved | 9.95 MB | non-default | 0.362 | 27.5 | 1.00 |
| **`l4`** (all three) | all 24 first, then timed | 24 GiB slot | non-default | **0.191** | **52.0** | **0.00** |
| `k4` last | interleaved | 9.95 MB | default | 0.362 | 27.46 | 1.00 |

Scores: L-1 in interval (the positive control reproduced the arena: `l4` 0.191 ms, slow share 0); **L-2 in interval and L-3 past its falsifier** (allocation order alone makes all 24 buffers fast; the destination alone does nothing); L-4 in interval (the stream does nothing); L-5 in interval (exactly one factor, batch allocation, is fast alone); L-6 in interval (the reversed-order routes are within 2.0, 3.9 and 5.0 % of the first run's rates: the route result is not an order effect); L-7 in interval (the two controls 0.362 and 0.362 ms); L-8 in interval (the ranking `k1`, `k2` above `k3` above `k4` held); L-9 in interval (all gates true, nine processes 9.0 s of run time in all).

**What this says (measured).** On this instance a 9.95 MB pinned buffer that is **allocated and then copied before the next buffer is allocated** copies at 27.5 GB/s, and the same buffers allocated **all before the first copy** copy at 51.8 GB/s. The destination (a slice of a 24 GiB device allocation or a small buffer) and the stream do not matter. **What it does not say:** why. Candidates, none tested: a pinned allocation made after a host-to-device copy has run is mapped differently (page pinning or IOMMU mapping under load); the first copy changes a link or driver state that applies to later allocations; the allocator's behaviour after a context has done DMA. A follow-up that allocates all buffers first, copies one, then allocates more would separate the first two.

**Consequences for earlier readings (derived, to be checked, not corrections of numbers).** Every earlier cell that timed a buffer right after allocating it used the interleaved pattern: B1's `alloc_order` (48 of 48 slow), B2's `h7` and `h1` to `h5` (the "huge-page buffers are slow" result), the first run's `k0` to `k5`. The arena (N2, N3 and later) allocated all its sources first and was fast. So the **"slow class is a property of the buffer, stable per buffer, set below the guest" reading of S1, B1 and B2 is better described as "set by the allocate-then-copy order of the harness's own cells"**; B2's falsification of huge-page backing was measured only in that pattern and is **not** tested in the batch-first pattern. One earlier cell does not fit the story simply: B1's curve process allocated all buffers of a size before timing any and still read 17 to 31 % slow at 9.95 MB on a different instance, so the pattern is at most one of the causes there (the instances differ and nothing here tests that cell). The committed classification rule (DC-6) as built (a fresh process, interleaved) labels any instance slow, because the pattern makes it so; the rule needs the batch-first pattern before it is a population class. The N3 and later results used the arena's pattern and are not affected.

## 5. DC-5 (the guard)

The issue part of every row (bookkeeping, copy issue, event records, waits and stand-in launches, from the row's start to just before the router synchronisation) contains **no synchronising call that torch 2.7.1 reports**, at level 3 with C = 1.0 and level 4 with C = 1.0 and 0.3, for the `D` and `P0L` arms. The host issue time is 2.0 to 2.5 ms a token (`P0L`), as in stage B. The guard's cost is 2.1 %. The guard's own control (a planted `.item()` per token) was counted in every token and attributed to a single line (`overlap_arena.py:849`, `plant_sync`). **Limits:** the hook sees only the calls torch routes through it; a driver-level block inside an asynchronous launch (a full queue) is invisible to it; one torch build, one trace, one instance.

## 6. What this does not claim

Nothing about the real model's kernels, another card, DeepSeek, MiniMax or GLM on NVIDIA hardware, or why the allocation order matters; one instance of one card, one day. `src/` is unchanged.

## 7. Open

A registered cell that separates "allocated after a copy has run" from "allocated while other buffers are timed" (allocate half, copy once, allocate the rest); the batch-first pattern for the huge-page question (B2) and for the registered routes; re-specifying DC-6's rule on the batch-first pattern; the instance `hellish-cry-4960` stays up until Hamed terminates it.
