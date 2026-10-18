# CUDA lane session B1: experiment record (what sets the host-to-device rate of a pinned 9.95 MB buffer, and the S1 copy curve redone with qualified buffers; RTX 5090)

Written 2026-10-18 against runtime 0.66.23, **before any B1 instrument code and before any instance exists**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void (plan section 12). Labels follow plan section 0. This follows `docs/CUDA-N2S1-RESULTS.md` section 4 and plan section 17.5. Nothing here starts an instance: that needs Hamed's yes, and he creates it.

## 1. Baseline (recorded)

* On the N2 RTX 5090 instance (23 GiB host RAM, KVM guest, THP policy `madvise` with `defrag` set to `madvise`, 0 kB of anonymous huge pages in a process that held 1.2 GiB of pinned buffers), a pinned 9.95 MB host buffer copied to the device in **0.182 ms (54.7 GB/s, the link plateau) for most buffers and 0.31 to 0.36 ms (28 to 32 GB/s) for a minority**, stable on repeat per buffer (probe 4: 24 separately pinned buffers, 19 fast and 5 slow, the slow ones the later allocations; a second pass over eight of them reproduced each value). A 24-record slab sliced into 9.95 MB records gave a mix, 0.18 to 0.33 ms. In the replay, 3 or 4 of 64 records were slow in every cell (0.28 to 0.36 ms).
* Not the copy stream, the destination (a whole tensor, an arena slice at offset 0 or misaligned), event reuse, pre-synchronization, the host clock, warm-up or the iteration count (probes 1 to 3).
* S1 baseline, **now known to be confounded** by that effect: one pinned buffer per cell measured pinned H2D 27.1 GB/s at 3 MiB, 28.4 at 9.95 MB, 30.4 at 14.16 MB, 41.0 at 21.1 MiB, 52.4 at 64 MiB and 52.2 at 256 MiB; four baseline runs on one instance read the 9.95 MB cell within 1 % of each other, so the buffer that cell got stayed slow.
* **Working hypothesis H-backing:** the DMA rate of a pinned buffer is set by how its physical pages are laid out (many small, scattered pages against contiguous large ones). This is a hypothesis: no physical address is readable without privileges, and a KVM guest's pages may be backed by the hypervisor's own page size, which the guest cannot see or choose. The competing reading, **H-host-side**, is that the rate is set below the guest (host page backing, IOMMU or NUMA placement of the guest's memory), in which case nothing a guest process does with huge-page advice moves it.

## 2. Questions and cells (profile `backing`; `benchmarks/cuda/backing_probe.py`, to be built after this record is committed)

Pinned host memory comes from two routes: `cudaHostAlloc` (through `torch.empty(pin_memory=True)`, what S1, S2, V1 and N2 used) and `cudaHostRegister` on ordinary anonymous memory that the process maps itself, so that huge-page advice (`madvise`) can be applied **before** the pages are touched and the result read back from `/proc/self/smaps` (`AnonHugePages` of that mapping: an observation, not a physical address). All copies are one pinned buffer to a device buffer, CUDA events on one stream, 20 timed repetitions after 5 warm-up, p50 per buffer, size 9.95 MB unless stated.

| id | cell | what varies |
|---|---|---|
| B1 | `alloc_order`: 48 `cudaHostAlloc` buffers allocated one after another in a **fresh process**, each timed alone; run in two separate processes | allocation order, process-to-process spread |
| B2 | `thp`: 16 buffers = anonymous mmap, `MADV_HUGEPAGE`, touched, 2 MiB-aligned, `cudaHostRegister`; `AnonHugePages` fraction recorded per buffer | huge-page advice requested |
| B3 | `nothp`: 16 buffers as B2 with `MADV_NOHUGEPAGE`; and `default`: 16 with no advice | huge pages refused / not requested |
| B4 | `slab_chunks`: one 256 MiB `cudaHostAlloc` slab; every 4 MiB chunk copied alone (20 repeats), then the whole slab in one copy | structure inside a large buffer |
| B5 | `reroll`: 12 cycles of B3-`default` and of B2 (`mmap`, touch, register, time, unregister, unmap) | whether a buffer's class is sticky to the address or re-drawn per allocation |
| B6 | `curve`: pinned H2D at 1 MiB, 3 MiB, 9.95 MB, 14.16 MB, 21.1 MiB, 64 MiB and 256 MiB for K buffers each (K = 16, 16, 16, 12, 12, 8, 3; at most 1.2 GiB pinned at once), each buffer timed alone; pageable H2D (K = 4) and pinned D2H at the fastest buffer for the same sizes | buffer, size |

Gates: bytes survive a round trip (checked on every buffer, outside timing); no copy rate above the link's sanity bound (a cache hit or a timing bug); a cell whose registration or advice call fails is reported `unavailable` with the error, never as a number.

## 3. Predictions (registered before any code)

All **predicted**. "Slow" means p50 at or above 0.28 ms at 9.95 MB (the fast mode is 0.182 ms).

| id | quantity | prediction | falsifier |
|---|---|---|---|
| B-P1 | B1: shares and levels | 48 fresh-process buffers are bimodal: slow share 0.05 to 0.35, fast buffers 0.178 to 0.195 ms, slow buffers 0.28 to 0.40 ms (probability 0.8) | slow share outside, or more than 10 % of buffers between 0.20 and 0.28 ms (a third mode) |
| B-P2 | B1: order | the mean allocation index of slow buffers exceeds the fast buffers' by 8 or more (probability 0.65) | the difference is under 4 |
| B-P3 | B1: two fresh processes | slow shares within 0.15 of each other (probability 0.6) | differ by more than 0.15 |
| B-P4 | B2: huge pages are granted and visible | at least 14 of 16 buffers show `AnonHugePages` of 90 % or more of their size (probability 0.6; the guest's `defrag=madvise` allows direct compaction) | fewer than 14: B2 and B5-thp are then inconclusive about H-backing and said so |
| B-P5 (the causal test of H-backing) | B2: rate of huge-page-verified buffers | every huge-page-verified buffer is fast, 0.175 to 0.20 ms (probability 0.55) | at least one slow buffer among at least 8 huge-page-verified ones: H-backing in the guest is not supported and H-host-side is the survivor |
| B-P6 | B3: refused huge pages | `nothp` slow share at least 0.5 (probability 0.45); `default` (no advice) within 0.2 of `nothp` | `nothp` slow share under 0.2 (small pages alone do not make a buffer slow) |
| B-P7 | B5: stickiness | consecutive allocations' classes agree about as often as independent draws would: lag-1 agreement within 0.15 of p^2 + (1-p)^2, p the cell's slow share (probability 0.6) | agreement 0.15 or more above that: the class sticks to the address or the allocator, not to each draw |
| B-P8 | B4: inside a 256 MiB slab | slow 4 MiB chunks are 0.05 to 0.40 of the chunks, and they come in runs (mean run length of slow chunks 2 or more) (probability 0.55); fast chunks 45 to 56 GB/s, slow 25 to 35 | no slow chunks, or slow chunks isolated singly |
| B-P9 | B4: whole slab in one copy | 40 to 54 GB/s, between the two chunk classes' rates and weighted by the slow share within 10 % | outside |
| B-P10 | B6: fast-buffer curve (fastest of the K buffers at each size) | at least 50 GB/s at every size from 3 MiB up, 45 to 55 at 3 MiB, 52 to 56 at 21.1 MiB, 64 MiB and 256 MiB: no fixed-cost ramp (probability 0.7) | any size from 3 MiB up below 45 GB/s |
| B-P11 | B6: slow-buffer share against size | slow share (below 0.8 x the fastest buffer's rate at that size): 9.95 MB 0.1 to 0.35; 21.1 MiB 0.1 to 0.35; 64 MiB and 256 MiB under 0.15 (probability 0.5) | 64 MiB share above 0.30 |
| B-P12 | B6: S1's ramp explained | the single-buffer S1 curve (27 / 28 / 30 / 41 / 52 GB/s at 3 MiB / 9.95 MB / 14.16 MB / 21.1 MiB / 64 MiB) lies between the slowest and fastest buffer curves of this session at every size (probability 0.7) | any S1 point outside the range of this session's buffers at that size |
| B-P13 | pageable H2D | 4 to 26 GB/s at every size, within 15 % of its S1 value (24 to 25 at 9.95 MB), not dependent on the pinned buffer (probability 0.6) | outside |
| B-P14 | pinned D2H (fastest pinned buffer) | 50 to 56 GB/s from 3 MiB up | outside |
| B-P15 | completion | all cells inside 150 s of GPU run time, all gates true, no failed cell, registration and advice calls succeed (probability 0.75) | any gate false, a failed cell or truncation |

## 4. What each outcome means

* **B-P5 holds with B-P4 holding:** H-backing is supported in the guest, and a pinned expert store should be built from huge-page-backed registered memory; the production lever is an allocator, not a kernel. **B-P5 fails with huge pages verified:** H-host-side (nothing to do from a guest process; the instance's luck is the variable, and any copy cost must be qualified per buffer). **B-P4 fails:** the experiment cannot test H-backing here and says so.
* **B-P7 shows stickiness:** a slow buffer can be re-drawn by freeing and reallocating, a cheap production remedy (qualify and replace).
* **B-P10 holds:** there is no fixed cost per copy on this card at any size from 3 MiB; S1's "160 us" is entirely a slow-buffer artifact and the copy curve to cite is the fast-buffer curve with the slow-buffer curve as its pessimistic bound.
* Nothing here is a statement about the V100 (its link is the limit for every buffer, V1), about DeepSeek, MiniMax or GLM on NVIDIA hardware, or about another provider's hosts.

## 5. Stop rules and cost

Stop and bring the files home on a failed cell, a false gate, a truncated run, or 80 % of the cap before the first cell. A defect found on the instance is fixed on the Mac. Session cap **1 GPU-hour** from Ready (the instance takes about seven minutes to initialize and is not the same buffer population as the earlier instances, which the results state). Estimated GPU run time 60 to 120 s per process; the probe is run twice for B-P3.

## 6. Needs Hamed

A go and an RTX 5090 instance when the agent says it is ready (the same listing and key as S2 and N2); then the usual SSH arrangement for that one instance, no sudo. Housekeeping stands (log the N2 hours, delete the S2, V1 and N2 ssh configs, known_hosts files and key).
