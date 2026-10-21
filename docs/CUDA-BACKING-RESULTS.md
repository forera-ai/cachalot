# CUDA lane session B1: results (what sets the host-to-device rate of a pinned 9.95 MB buffer; the S1 copy curve with qualified buffers; RTX 5090)

> **Note added 2026-10-21 (0.66.59).** Every slow reading in this file was taken in the interleaved pattern (each buffer allocated and then copied before the next was allocated). Session K's follow-up (`docs/CUDA-PIN-ROUTE-RESULTS.md` section 4) found that on an RTX 5090 the same buffers allocated all before the first copy copy at 51.8 GB/s instead of 27.5 GB/s, whatever the destination or stream. Read the "slow class is a property of the buffer" and the huge-page conclusions here as measured in the interleaved pattern only; the huge-page question is untested in the batch-first pattern.

Run 2026-10-19 under `docs/CUDA-BACKING-EXPERIMENT-RECORD.md` (15 predictions committed in bc72198 before any instrument code or instance). Instrument: `benchmarks/cuda/backing_probe.py` at scaffold commit 450f42c (`REVISION` dirty=0) for the first four processes and at 4f54685 for the curve rerun (section 1). Runtime 0.66.24. Instance: a fourth rented RTX 5090 virtual machine of the S2 and N2 class (driver 580.159.03, torch 2.7.1+cu128, 7 vCPUs, 24 GiB RAM, PCIe 5.0 x16, a KVM guest, transparent huge pages `enabled=madvise` and `defrag=madvise`, `shmem_enabled=never`). Raw rows, manifests and per-buffer samples: `docs/cuda-results/b1-rtx5090-2026-10-19/`. Labels follow plan section 0: everything below is measured on this one instance unless it says derived or hypothesis. Nothing here says anything about a V100, another provider's hosts, or the models on NVIDIA hardware.

## 1. What ran

Over SSH to the one instance Hamed named, no sudo: `check`, `setup`, `doctor`, `smoke` (clean), then `session.sh backing`: four fresh processes in order, `b1` twice (48 `cudaHostAlloc` buffers of 9.95 MB, each timed alone), `rest` (huge-page advice cells, reroll, the 256 MiB slab), `curve` (the qualified copy curve). Process 1 to 3 were clean (gates true). **Process 4 failed its round-trip gate**, which stopped the session as registered. The cause was an instrument defect and not data: the curve cell copied K different pinned buffers through one device buffer and then compared the first two buffers' bytes with the device buffer, which holds only the last buffer's content, so the check could never pass for K above 1. The CPU rehearsal passed because its test never asserted the gate. Fixed on the Mac (commit 4f54685: each of the first two buffers is checked right after its own copy; the test now asserts the gate, and fails on the old code), repacked from a clean commit, and the curve cell was run again in a fresh process on the same instance. The first curve run's timings are within noise of the rerun (section 4) and are kept in the results directory; the rerun is the scored one. All gates true in every other process and in the rerun. Wall time on the host: the four processes took 2.0, 2.0, 2.9 and 14.1 s of GPU time against the 150 s bound.

## 2. Results by field name

**Allocation order (`alloc_order`, two fresh processes, 48 buffers each).** `slow_share` **1.0** in both; every buffer read 0.3584 to 0.3592 ms (27.7 GB/s); `third_mode_share` 0; `order_gap` undefined (one class); lag-1 agreement 1.0 equals its independent expectation 1.0. There was no fast buffer in either process.

**Huge-page advice (`rest`, 16 buffers each of 9.95 MB registered with `cudaHostRegister` on a 2 MiB aligned anonymous mapping, advised before the first touch).** `huge_verified_count` **0 of 16 in all three arms** (`mmap_thp`, `mmap_nothp`, `mmap_default`), `huge_fraction_median` 0.0; the advice and registration calls returned no error (0 errors). `slow_share` 0.875 (`thp`), 0.875 (`nothp`), 1.0 (`default`); copy time 0.272 to 0.354 ms (a band 0.276 to 0.33, lower than the 0.359 of the `cudaHostAlloc` buffers); `third_mode_share` 0.125 in the two advised arms. `/proc/meminfo` read `AnonHugePages: 0 kB` on the host after the session.

**Reroll (`reroll_default`, `reroll_thp`, 12 allocate, time and free cycles).** `slow_share` 1.0 in both (0.32 to 0.356 ms), no huge pages; lag-1 agreement equals its expectation (one class).

**Slab (`slab_chunks`, one 256 MiB `cudaHostAlloc` slab in 4 MiB chunks).** `slow_chunk_share` 0.1406 (9 of 64), mean slow run length 3.0, fast chunks 53.65 GB/s median, slow chunks 32.07; `whole_slab_gbps` 50.52 against 48.49 implied by the chunks.

**The qualified copy curve (rerun, K buffers each timed alone; GB/s):**

| bytes | K | fastest | median | slowest | slow share | D2H (fastest buffer) | pageable median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 16 | 26.54 | 26.53 | 26.49 | 0 | 52.51 | 17.12 |
| 3,145,728 | 16 | 29.14 | 27.47 | 27.40 | 0 | 55.01 | 22.52 |
| 9,950,000 | 16 | 54.64 | 54.49 | 27.78 | 0.3125 | 54.96 | 25.27 |
| 14,160,000 | 12 | 54.80 | 54.57 | 27.95 | 0.25 | 55.11 | 25.59 |
| 22,124,953 | 12 | 54.97 | 54.93 | 31.59 | 0.1667 | 54.98 | 25.70 |
| 67,108,864 | 8 | 55.16 | 55.12 | 55.05 | 0 | 54.98 | 24.92 |
| 268,435,456 | 3 | 55.10 | 55.06 | 55.05 | 0 | 54.99 | 22.78 |

The first curve run (round-trip gate false because of the defect above) read the same shape: 9.95 MB slow share 0.375, 14.16 MB 0.4167, 21.1 MiB 0.1667, 64 MiB 0.125 (slowest 42.07 GB/s), 256 MiB 0 (slowest 51.10).

## 3. Scoring the registered predictions

| id | outcome |
|---|---|
| B-P1 (48 fresh-process buffers bimodal, slow share 0.05 to 0.35, fast 0.178 to 0.195 ms) | **falsified**: slow share 1.0, one class at 0.359 ms, no fast buffer |
| B-P2 (slow buffers allocated later than fast ones by 8 or more) | **not testable**: one class only (the falsifier needs both classes) |
| B-P3 (two fresh processes within 0.15 on slow share) | **held**, trivially: 1.0 and 1.0 |
| B-P4 (at least 14 of 16 buffers show 90 % huge pages) | **falsified**: 0 of 16 in every arm; B2 and the thp reroll cannot say anything about H-backing, as the record said |
| B-P5 (the causal test: every huge-page-verified buffer is fast) | **not testable**: no huge-page-verified buffer (the falsifier needs at least 8) |
| B-P6 (`nothp` slow share at least 0.5, `default` within 0.2 of it) | **held**: 0.875, and 1.0 (difference 0.125); but the advised arm is as slow (0.875), because no arm got huge pages, so this says nothing about small pages causing slowness |
| B-P7 (lag-1 agreement within 0.15 of the independent expectation) | **held**: reroll 1.0 against 1.0, `mmap_thp` 0.733 against 0.781, `mmap_nothp` 0.867 against 0.781; uninformative where one class holds all buffers |
| B-P8 (slab: slow chunks 0.05 to 0.40 of chunks in runs of 2 or more; fast 45 to 56, slow 25 to 35 GB/s) | **held**: 0.1406, mean run 3.0, 53.65 and 32.07 |
| B-P9 (whole slab 40 to 54 GB/s, within 10 % of the chunk-weighted rate) | **held**: 50.52 against 48.49 (4.2 %) |
| B-P10 (fastest buffer at least 50 GB/s from 3 MiB up, 45 to 55 at 3 MiB) | **falsified at 3 MiB only**: 29.14 GB/s; 9.95 MB and up read 54.6 to 55.2 |
| B-P11 (slow share: 9.95 MB and 21.1 MiB 0.1 to 0.35; 64 and 256 MiB under 0.15) | **held** on the rerun: 0.3125, 0.1667, 0 and 0 (the first run read 0.375 at 9.95 MB, 0.025 over) |
| B-P12 (S1's single-buffer curve lies between this session's slowest and fastest at every size) | **missed narrowly** on the scored run: at 3 MiB S1's 27.1 is 0.3 GB/s below the slowest 27.40, and at 64 MiB S1's 52.4 is below the slowest 55.05 (inside the first run's range, 42.07 to 55.1); at 9.95 MB, 14.16 MB and 21.1 MiB S1's points (28.4, 30.4, 41.0) lie inside the range, which is the claim that mattered |
| B-P13 (pageable H2D 4 to 26 GB/s, within 15 % of S1 at 9.95 MB) | **held**: 17.1 to 25.7 GB/s, 25.27 at 9.95 MB |
| B-P14 (pinned D2H 50 to 56 GB/s from 3 MiB up) | **held**: 54.96 to 55.11 |
| B-P15 (inside 150 s, all gates true, no failed cell, advice and registration succeed) | **held with an exception**: GPU time 21 s in all, no failed cell, advice and registration returned no error; the round-trip gate was false once, from the instrument defect in section 1, and true in every rerun |

## 4. What the result is, and is not

- **Measured.** On this instance a fresh process's first 48 pinned 9.95 MB `cudaHostAlloc` buffers were all slow (0.359 ms, 27.7 GB/s), in two separate processes. In the curve process, after smaller buffers had been allocated, 9.95 MB, 14.16 MB and 21.1 MiB buffers were slow 31 %, 25 % and 17 % of the time and 64 MiB and larger never (in the rerun; 12.5 % at 64 MiB in the first run). So "the class of a buffer" is not a fixed share of an instance: it varied from 0 to 100 % for one size with the order and the process, on one instance.
- **Measured.** The fast rate at every size from 9.95 MB up is the link plateau, 54.6 to 55.2 GB/s, D2H the same: for a fast buffer there is no per-copy fixed cost at expert size. Below that the curve is not at the plateau on this card at all: 1 MiB 26.5 and 3 MiB 27.4 to 29.1 GB/s for every buffer (0.04 ms and 0.11 ms). B-P10's "no fixed-cost ramp from 3 MiB" fell at 3 MiB: below about 4 MiB a copy runs at about half the plateau whatever the buffer (this session's buffers at those sizes were all one class). Whether that is a fixed cost, a smaller-transfer effect or the same slow class is not separated (slow-class buffers read 27.4 to 28 GB/s at 3 MiB and at 9.95 MB, the same rate).
- **Measured.** The huge-page route was never tested: `madvise` and `cudaHostRegister` succeeded, but the kernel reported 0 of 16 buffers backed by huge pages in every arm, with a fresh instance (7 minutes up, 23 GiB available) and `enabled=madvise`, `defrag=madvise`. The advice before the first touch did not produce huge pages. **Not established:** why (a guest that cannot allocate 2 MiB pages, the touch pattern of my instrument, `cudaHostRegister` splitting the mapping, or something under the guest). H-backing and H-host-side are both still open; B-P5 gave neither support.
- **Not shown.** That slow buffers are made by small pages, by allocation order, or by freshly mapped guest memory: the second fresh process reused memory the first had just freed and was still entirely slow, which does not fit "never-touched guest memory is slow" as the whole story (hypothesis, derived from this one pair).
- **Consequence for the lane (derived).** Any pinned expert store's copy cost has to be quoted per qualified buffer and with the allocation context; a store of expert-sized buffers allocated in one fresh process could be entirely at 0.359 ms (this instance) or mostly 0.185 ms (N2's instance, 19 of 24 fast, 3 or 4 of 64 slow in the replay). Qualify-and-replace (the production remedy B-P7 was meant to test) cannot be assessed from these data, because the reroll arms had no fast buffer to reroll into.

## 5. Cost and housekeeping

GPU time on the instrument: 21 s of runs; the host's uptime read 7 minutes at the last command and the console's uptime was 9 minutes when the agent started, so the session was far inside the 1 GPU-hour cap. The instance is terminated by Hamed (the agent cannot). Cost-file entry written without money figures in chat. Housekeeping from earlier sessions (log the N2 hours; delete the S1, S2, V1, N2 and this session's ssh configs, known_hosts files and the key, which are in `~/cachalot-private/` and `~/.ssh/`) is Hamed's yes; nothing was deleted here.

## 6. Erratum found on offline review the same day (2026-10-19)

Reading `MmapBuffer` again while designing the retest: it maps with `mmap.mmap(-1, size)`, whose default flags are `MAP_SHARED` (Python's documented default), so every "mmap" buffer in `rest` was a **shared anonymous** mapping. Transparent huge pages for shared memory are governed by `shmem_enabled`, which was `never` on this host (read at the end of the session), and smaps reports shared pages as `ShmemPmdMapped`, not `AnonHugePages`. So the 0 of 16 in section 2 is what that code is expected to read on that host whatever a private mapping could do, and **B-P4 was falsified by an instrument defect as much as by the guest**; B-P4, B-P5, B-P6 and B-P7's mmap arms must not be read as evidence about huge pages in this guest. Section 4's third bullet ("the huge-page route was never tested") stands, with the cause no longer open-ended: the most likely cause is the shared mapping (derived from the code and the host setting, not tested on Linux here). Section 4's list of possible causes (a guest that cannot allocate 2 MiB pages, the touch pattern, `cudaHostRegister` splitting the mapping) is superseded by that one first and kept as the fallback if a private mapping also yields 0. The 0.272 to 0.354 ms band of the mmap arms is the rate of small pages of shared memory. The `cudaHostAlloc` results (all slow in a fresh process, the curve's slow shares, the slab, the curve) are unaffected. The retest is registered in `docs/CUDA-HUGEPAGE-RETEST-RECORD.md` (commit 4340c15, before any code).
