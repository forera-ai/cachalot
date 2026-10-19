# CUDA lane session B2: results (huge-page retest with a private mapping, and allocation context; RTX 5090)

Run 2026-10-19 under `docs/CUDA-HUGEPAGE-RETEST-RECORD.md` (10 predictions committed in 4340c15 before any instrument code or instance). Instrument: `benchmarks/cuda/backing_probe.py` groups `h1` to `h7` at scaffold commit 225c663 (`REVISION` dirty=0), runtime 0.66.27. Instance: a fifth rented RTX 5090 virtual machine of the S2, N2 and B1 class (driver 580.159.03, torch 2.7.1+cu128, 7 vCPUs, 24 GiB RAM, PCIe 5.0 x16, kernel 6.8, THP `enabled=madvise`, `defrag=madvise`, `shmem_enabled=never`, no hugetlb pages). Raw rows, manifests and per-buffer samples: `docs/cuda-results/b2-rtx5090-2026-10-19/`. Labels follow plan section 0; all of it is measured on this one instance unless it says derived or hypothesis. Nothing here says anything about a V100, another provider's hosts, or the models on NVIDIA hardware.

## 1. What ran

Over SSH to the one instance Hamed named, no sudo: `check`, `setup`, `doctor`, a clean `smoke`, then `session.sh hugepage`: seven fresh processes (`h7`, `h6`, `h4`, `h1`, `h2`, `h3`, `h5`). Every process completed with `failed_cells` empty and the round-trip and sanity gates true; no instrument defect this time. GPU time 0.6 to 2.3 s a process (about 8 s in all against the 120 s bound). The Linux-only paths all worked on the first run: the private mapping, the three smaps reads with `AnonHugePages` and `ShmemPmdMapped`, `madvise(MADV_POPULATE_WRITE)` (accepted, kernel 6.8), the host policy read.

## 2. Results by field name (9.95 MB, 16 buffers a cell unless stated; p50 ms per buffer)

| cell (`arm`) | mapping and advice | huge-verified (make / register / copy) | `shmem_pmd_verified_count` | copy ms min / median / max | slow share (at or above 0.28 ms) |
|---|---|---|---:|---|---:|
| `private_thp` (H1) | private, `MADV_HUGEPAGE` | **16 / 16 / 16** (fraction 1.0, drop after registration 0.0) | 0 | 0.3562 / 0.3574 / 0.3583 | **1.0** |
| `private_populate` (H5) | private, `MADV_HUGEPAGE` then `MADV_POPULATE_WRITE` | **16 / 16 / 16** (drop 0.0) | 0 | 0.3580 / 0.3587 / 0.3620 | **1.0** |
| `private_nothp` (H2) | private, `MADV_NOHUGEPAGE` | 0 / 0 / 0 | 0 | 0.2557 / 0.3066 / 0.3192 | 0.75 |
| `private_default` (H3) | private, no advice | 0 / 0 / 0 | 0 | 0.2554 / 0.2864 / 0.3222 | 0.75 |
| `shared_thp` (H4) | shared (B1's), `MADV_HUGEPAGE` | 0 / 0 / 0 | **0** | 0.2510 / 0.2879 / 0.3228 | 0.8125 |
| `fresh48` (H7) | `cudaHostAlloc`, 48 buffers, fresh process | n/a | n/a | 0.3584 / 0.3588 / 0.3880 | **1.0** |
| `prefix` (H6) | 16 x 1 MiB and 16 x 3 MiB first, then 48 x 9.95 MB (`cudaHostAlloc`) | n/a | n/a | 0.3580 / 0.3588 / 0.3590 | **1.0** |

The smaller buffers allocated first copied at 26.4 GB/s (1 MiB) and 27.4 GB/s (3 MiB), the same ramp as B1. `huge_verified_slow_count` was 16 of 16 in both huge-page cells (`huge_verified_ms_max` 0.3583 and 0.362); `small_page_slow_share` 0.75 (H2, H3) and 0.8125 (H4). Host policy and counters before the cells: `enabled=[madvise]`, `defrag=[madvise]`, `shmem_enabled=[never]`, `AnonHugePages` 0, `HugePages_Total` 0.

## 3. Scoring the registered predictions

| id | outcome |
|---|---|
| R-P1 (at least 14 of 16 verified with a private mapping) | **held**: 16 of 16 in H1 and in H5 |
| R-P2 (shared control 0 of 16 by `AnonHugePages` and `ShmemPmdMapped`) | **held**: 0 and 0. B1's zero was the shared mapping, confirmed |
| R-P3 (`private_default` at most 3, `private_nothp` 0) | **held**: 0 and 0 (the host's `madvise` policy gives nothing without advice) |
| R-P4 (registration keeps the pages, within 0.05) | **held**: drop 0.0 for all 32 verified buffers |
| R-P5 (the causal test: every verified buffer fast, 0.175 to 0.20 ms, and at least 0.7 of the small-page buffers slow) | **falsified**: all 16 (and all 16 more in H5) verified buffers read 0.356 to 0.362 ms, the slow value; the registered falsifier (any slow buffer among at least 8 verified) fired 32 times. The small-page buffers were 0.75 slow, but at 0.25 to 0.32 ms, faster than the huge-page ones |
| R-P6 (populate within 2 of H1's verified count) | **held**: 16 against 16 |
| R-P7 (fresh-process control slow share at least 0.9) | **held**: 1.0 (this instance, like B1's, gave no fast `cudaHostAlloc` buffer in a fresh process) |
| R-P8 (smaller buffers first lower the slow share by at least 0.3) | **falsified**: 1.0 against 1.0, difference 0 |
| R-P9 (verified buffers at 0.175 to 0.20 ms, none above 0.25) | **falsified**: 0.356 to 0.362 ms |
| R-P10 (inside 120 s, all gates true, no failed cell, advice and registration succeed) | **held**: about 8 s, all gates true, 0 failed cells, no advice, populate or registration error |

Of 10: 7 held, 3 falsified (R-P5, R-P8, R-P9). The three falsified ones are the three that would have made page size or allocation order the explanation.

## 4. What the result is, and is not

- **Measured.** B1's zero huge pages was the shared mapping: a private mapping with `MADV_HUGEPAGE` before the touch was fully huge-page backed (16 of 16, fraction 1.0) on this guest, and registration kept it. The guest can back pinned memory with huge pages. The B1 erratum is confirmed.
- **Measured, the opposite of the hypothesis.** Huge-page-backed registered buffers were all **slow** (0.357 ms, 27.9 GB/s), the same value as this instance's fresh-process `cudaHostAlloc` buffers (0.359), and slower than the small-page registered buffers (0.255 to 0.32 ms). H-backing as registered (huge pages give the link plateau of 0.182 ms) is not supported in this guest: 32 huge-page-verified buffers, none fast. Neither page size nor allocation order (R-P8) produced a fast buffer here; no cell in this session produced a 0.182 ms buffer at all.
- **Hypothesis, not tested.** Because the `cudaHostAlloc` buffers read exactly the huge-page value, the CUDA allocator may be giving huge-page-backed memory on this instance; and the fast class (0.182 ms) seen in N2 and in B1's curve process would then be some third backing that neither a self-mapped buffer nor a fresh-process allocation reached. H-host-side (the rate depends on something below the guest, such as which host pages back the guest's memory) is the surviving explanation, still without a direct test; nothing a guest process did in B1 or B2 moved a buffer from slow to the 0.182 ms class, except that in B1's curve process some `cudaHostAlloc` buffers were fast (a context B2's prefix cell did not reproduce).
- **Not shown.** Why B1's curve process had fast 9.95 MB buffers (69 %) when the prefix cell here, with smaller buffers allocated first, had none: the curve process also allocated pageable buffers, device buffers and many sizes, and ran after other processes had run; the instance differs too.
- **Consequence for the lane (derived).** A pinned expert store cannot be assumed to reach 0.182 ms a copy by any allocation technique tried so far; its cost is a property of the instance. The copy cost to use in the N2 formula is still the per-buffer qualified one, with 0.185 ms (N2's fast class) and 0.359 ms (this and B1's fresh-process class) as the range, and the page-size route is closed as an explanation within the tested options.

## 5. Cost and housekeeping

GPU time on the instrument: about 8 s; the instance's console uptime read 2 minutes 28 seconds at the first screenshot, and the agent's commands ran for a few minutes after; the billed hours are Hamed's to enter from the console. The agent cannot terminate the instance. Housekeeping stands: fill the B1 and B2 hours in the private cost file; delete the S1, S2, V1, N2, B1 and B2 ssh configs, known_hosts files and the key (nothing was deleted here).
