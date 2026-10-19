# CUDA lane session B2: experiment record (the huge-page retest with a private mapping, and whether smaller buffers allocated first change the class of a pinned 9.95 MB buffer; RTX 5090)

Written 2026-10-19 against runtime 0.66.25, **before any B2 instrument code and before any instance exists**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void (plan section 12). Labels follow plan section 0. This follows `docs/CUDA-BACKING-RESULTS.md` (sections 4 and 6) and plan section 17.6. Nothing here starts an instance: that needs Hamed's yes, and he creates it.

## 1. Baseline (recorded)

* B1 (`docs/CUDA-BACKING-RESULTS.md`, RTX 5090, 24 GiB RAM, KVM guest, kernel 6.8, THP `enabled=madvise`, `defrag=madvise`, **`shmem_enabled=never`**): huge-page advice before first touch produced **0 of 16** huge-page-backed buffers in every arm; `AnonHugePages` read 0 kB in `/proc/meminfo` after the session. The first 48 pinned 9.95 MB `cudaHostAlloc` buffers of two fresh processes were all slow (0.359 ms, 27.7 GB/s); in a later process the same size was 31 % slow. Buffers I mapped and registered myself read 0.272 to 0.354 ms (a lower band than 0.359).
* **A defect in the B1 instrument, found on offline review after the run (derived from the code and the host's setting, not yet tested):** `MmapBuffer` maps with `mmap.mmap(-1, size)`, whose default flags are `MAP_SHARED`, so it is a *shared* anonymous mapping (shmem). Transparent huge pages for shared memory are governed by `shmem_enabled`, which was `never` on that host, and smaps reports such pages as `ShmemPmdMapped` and not `AnonHugePages`. B1's "0 of 16" is therefore the expected reading of that code on that host whatever the guest can do with private memory, and **B1 did not test whether this guest can back a pinned buffer with huge pages**. It also means B1's advice arms (`mmap_thp`, `mmap_nothp`, `mmap_default`) all had one backing, small pages of shared memory, and their rates (0.276 to 0.354 ms) are rates of that backing.
* S1, S2, V1, N2 and B1 together: the slow class of a 9.95 MB pinned buffer is 0 to 100 % of buffers depending on instance, process and allocation context (N2: 3 or 4 of 64 slow; B1 fresh process: 48 of 48; B1 curve process: 31 %).

**Working hypotheses.** **H-private:** with a private anonymous mapping, `MADV_HUGEPAGE` before first touch gives huge-page backing on this class of guest. **H-backing (carried from B1, still untested):** buffers backed by huge pages copy at the link plateau (about 0.182 ms for 9.95 MB), buffers on small pages are slow. **H-context:** the class of a `cudaHostAlloc` buffer depends on what the process allocated before it (here: smaller buffers first). **H-host-side (the rival):** the rate is set below the guest and none of this moves it.

## 2. Cells (profile `hugepage`; `benchmarks/cuda/backing_probe.py`, to be extended after this record is committed)

Pinned host memory from `cudaHostRegister` on a mapping the process makes itself, 9.95 MB rounded up to whole 2 MiB extents (10 MiB, 2 MiB aligned), advised **before** the first touch, populated by writing, then registered. smaps is read at three points per buffer: after the touch (before registration), after registration and after the timed copies; the fields recorded are `AnonHugePages`, `ShmemPmdMapped`, `Rss` and `THPeligible`. Timing, as in B1: one buffer to a device buffer, CUDA events on one stream, 20 repetitions after 5 warm-up, p50 per buffer. Each cell is its own fresh process unless stated.

| id | cell | what varies |
|---|---|---|
| H1 | `private_thp`: 16 buffers, `MAP_PRIVATE\|MAP_ANONYMOUS`, `MADV_HUGEPAGE` | the one registered fix |
| H2 | `private_nothp`: 16 buffers, `MADV_NOHUGEPAGE` | huge pages refused |
| H3 | `private_default`: 16 buffers, no advice (host policy `madvise`) | policy without advice |
| H4 | `shared_thp`: 16 buffers, `MAP_SHARED` as in B1, `MADV_HUGEPAGE` | the B1 control, replicated with the better smaps fields |
| H5 | `private_populate`: 16 buffers, `MADV_HUGEPAGE` then `MADV_POPULATE_WRITE` (kernel 5.14 and later; reported `unavailable` if the call fails) instead of a touch loop | how the pages are made |
| H6 | `prefix`: a fresh process allocates 16 x 1 MiB then 16 x 3 MiB pinned buffers with `cudaHostAlloc`, then 48 x 9.95 MB, each timed alone | allocation context (H-context) |
| H7 | `fresh48`: a fresh process allocates 48 x 9.95 MB with `cudaHostAlloc`, each timed alone (a replicate of B1's first cell, on this instance) | the control for H6 and a check on the instance's population |

Gates: bytes survive a round trip (every buffer, outside timing, **checked right after that buffer's own copy**); no copy rate above the link's sanity bound; a cell whose advice, populate or registration call fails reports `unavailable` with the error, never a number. The rehearsal tests on the Mac assert every gate a GPU run will check, with K above 1 (B1's lesson). Huge-page-verified means at least 90 % of the registered range reported as `AnonHugePages` at the post-touch read.

## 3. Predictions (registered before any code)

All **predicted**. "Slow" means p50 at or above 0.28 ms at 9.95 MB; "fast" 0.175 to 0.20 ms. Two caveats stated now: the instance may differ from B1's (N2's mostly fast, B1's fresh process all slow), and a prediction about fast and slow classes can only be scored on the classes the instance gives.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| R-P1 | H1: huge pages with a private mapping | at least 14 of 16 buffers are huge-page-verified (probability 0.7) | fewer than 14: this guest does not back private advised memory with huge pages either, and the B1 zero was not (only) the shared mapping |
| R-P2 | H4: the B1 control | 0 of 16 huge-page-verified by `AnonHugePages`, and `ShmemPmdMapped` also 0 (probability 0.85) | any buffer verified: the shared mapping was not the obstacle and the B1 reading needs another explanation |
| R-P3 | H3 and H2: no advice, advice refused | `private_default` has at most 3 of 16 huge-page-verified, `private_nothp` 0 of 16 (probability 0.8) | `private_default` has 8 or more: the host's policy grants huge pages without advice and `enabled=madvise` is not what governs this guest |
| R-P4 | H1: registration keeps the pages | for the verified buffers, the huge fraction after registration is within 0.05 of the post-touch value (probability 0.65) | a drop above 0.2: `cudaHostRegister` splits or migrates the mapping, which would have hidden any effect in B1 as well |
| R-P5 (the causal test of H-backing) | H1 against H2: rate | if H1 has at least 8 verified buffers and H2 at least 8 small-page ones: the verified buffers are all fast (0.175 to 0.20 ms) and at least 0.7 of the H2 buffers are slow (probability 0.5) | any slow buffer among at least 8 verified ones, or a slow share under 0.3 among the small-page ones: H-backing in the guest is not supported; H-host-side survives |
| R-P6 | H5: populate | the same verified count as H1 within 2 of 16 (probability 0.7), `unavailable` is a valid outcome | a difference above 4: the way pages are made matters and R-P1 holds only for a touch loop |
| R-P7 | H7: the fresh-process control on this instance | slow share at least 0.9 (probability 0.5; B1's instance gave 1.0, N2's gave 0.05 to 0.2) | slow share under 0.3: the instance's population differs from B1's, and R-P8 is scored against this instance only |
| R-P8 | H6 against H7: allocation context | the 48 buffers allocated after the 32 smaller ones have a slow share at least 0.3 lower than H7's (probability 0.4) | a difference under 0.1 (or H7 under 0.3, in which case R-P8 is not testable): small-first allocation is not what produced the curve process's fast buffers in B1 |
| R-P9 | huge pages and the plateau on the registered memory | huge-page-verified registered buffers copy 9.95 MB at 0.175 to 0.20 ms (probability 0.5; this is R-P5 restated for the absolute level) | above 0.25 ms for any verified buffer in a run with at least 8: the link rate is not set by page size alone |
| R-P10 | completion | all cells inside 120 s of GPU run time, all gates true, no failed cell, advice and registration succeed (probability 0.7) | any gate false, a failed cell or truncation |

## 4. What each outcome means

* **R-P1 holds with R-P5 holding:** H-backing is supported in this guest; the production lever is an allocator that registers privately mapped, huge-page-advised memory (which also avoids B1's slow `cudaHostAlloc` class on a fresh process). **R-P1 holds with R-P5 failing:** the page size is not the variable; H-host-side or the allocation context is; the rest of this record says which. **R-P1 fails:** the guest cannot give huge pages for pinned memory at all; the lever, if any, is the allocation context (R-P8) or nothing a guest process controls.
* **R-P2 holding** closes the B1 zero as a defect of the shared mapping and lets B1's mmap cells be quoted only as "small pages of shared memory". **R-P2 failing** means B1's zero needs another cause.
* Nothing here is a statement about the V100, about DeepSeek, MiniMax or GLM on NVIDIA hardware, or about another provider's hosts.

## 5. Stop rules and cost

Stop and bring the files home on a failed cell, a false gate, truncation, or 80 % of the cap before the first cell. A defect found on the instance is fixed on the Mac. Session cap **1 GPU-hour** from Ready (the instance takes about seven minutes to initialize and is not the same buffer population as the earlier ones, which the results state). Estimated GPU run time 40 to 100 s in all.

## 6. Needs Hamed

A go and an RTX 5090 instance when the agent says it is ready (same listing and key as S2, N2 and B1); then the usual SSH arrangement for that one instance, no sudo. Housekeeping stands (fill the B1 hours, delete the S1, S2, V1, N2 and B1 ssh configs, known_hosts files and the key).
