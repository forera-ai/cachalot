# Audit: which CUDA-lane results used the allocate-then-copy pattern, and what the follow-up L finding does and does not change

Written 2026-10-21 against runtime 0.66.59, Mac only, from the code of `benchmarks/cuda/` and the lane's results files (nothing was run). It answers open job 4 of `docs/CUDA-PIN-ROUTE-RESULTS.md` section 7 and corrects how far that file's conclusion reaches. Labels: **read** (a file and line), **recorded** (a number in a results file), **derived** (arithmetic or reasoning from those), **hypothesis**.

## 1. The finding and its reach

Follow-up L (`docs/CUDA-PIN-ROUTE-RESULTS.md` section 4, **measured**, one instance `hellish-cry-4960`): 24 pinned 9.95 MB `cudaHostAlloc` buffers each timed right after they were allocated copied at 0.362 ms (27.5 GB/s); the same 24 allocated all before the first copy copied at 0.192 ms (51.8 GB/s). The destination and the stream did not matter.

**What that does not show (recorded, from the lane's own files):**

* On the slow-population RTX 5090 of `docs/CUDA-LOW-C-RESULTS.md` the arena process, which allocates its 64 source records **all before** any copy (`replay_arena.py:365-368`, **read**), had a median source-record copy of **0.349 ms** (per-copy cost 0.355 ms). So batch allocation does not guarantee the fast rate; that instance was slow in the fast-order pattern.
* On B1's instance the curve process allocated all `K` buffers of a size before timing any (`backing_probe.py:481`, **read**) and still read 37.5 % slow at 9.95 MB in its first run and 17 to 31 % in the rerun (`docs/CUDA-BACKING-RESULTS.md`, **recorded**), against 48 of 48 slow in its interleaved cell on the same instance.
* On B1's instance a 256 MiB slab allocated once and timed in 4 MiB chunks read 14 % slow chunks (9 of 64) (**recorded**).
* The N2 and N3 stages' arena sources read 3 or 4 of 64 slow (4.7 to 6.25 %) on fast-population instances (**recorded**).

**So the supported statement is:** the allocation order (a buffer allocated and timed before the next is allocated, against all allocated first) **changes the slow share a great deal on the instances where both were measured** (B1: 100 % against 17 to 38 %; K: 100 % against 0 %), and a separate instance-level difference exists (the slow-population instance's batch-first arena was slow). "The slow class was largely the harness's allocation order" overstates it; **"the allocation order is a major, independent cause of the slow class, and an instance or population factor also exists"** is what the data support. The reason for either is untested.

## 2. Cell by cell

"Interleaved" is a pinned buffer allocated and then copied (possibly after earlier cells' copies in the same process) before the next is allocated. "Batch-first" is every pinned buffer of the cell allocated before the first timed copy.

| session and cell | pattern (code) | what was concluded | does the finding touch it |
|---|---|---|---|
| S1, S2, V1: `suite.py` `_copy`, `_overlap`, `_batchcopy` (`suite.py:448,484-520`) | one pinned buffer allocated per cell, validated, then timed, in a process that ran earlier cells' copies: interleaved | the pinned H2D curve (28 GB/s at 9.95 MB, 52 at 64 MiB), the "160 us fixed cost" of S1, batching conclusions of S2 | **yes, S1's and S2's RTX 5090 readings of the 9.95 MB rate** (the N2 correction of 0.66.23 already withdrew S1/S2/V1's 0.35 ms for the record size; this adds the pattern as a candidate cause). **V1 (V100, PCIe 3.0 x16, 11.8 GB/s flat) is link-limited at every size and is not touched**; the L40S's 26.3 GB/s (copy-cost sweep, arena sources) is the PCIe 4.0 x16 ceiling and is not touched |
| N2 stage 1 (`replay_arena.py`) | sources batch-first, arena, one stream | 0.183 to 0.185 ms a copy; 3 or 4 of 64 records slow | no (this is the fast-order pattern) |
| B1 `b1` (`alloc_order`, `backing_probe.py:359-369`) | interleaved | 48 of 48 slow, in two processes | **yes**: the "fresh process, every buffer slow" reading is the interleaved reading |
| B1 `rest`: `mmap_cell`, `reroll` (`:371-390,445-458`) | interleaved (self-made registered buffers) | 0.27 to 0.35 ms, "a lower band than 0.359"; huge-page 0 of 16 (a defect, B2 erratum) | yes (rates), and already voided for the huge-page question |
| B1 `rest`: `slab_chunks` (`:460-468`) | one slab batch-allocated, chunks timed | 14 % slow chunks, runs of 3 | not affected by the interleaving; shows slow chunks exist batch-first |
| B1 `curve` (`:470-499`) | batch-first per size, K buffers | 17 to 38 % slow at 9.95 MB on that instance | not affected; the sizes' slow shares are the batch-first reading on that instance |
| B2 `h1` to `h5` (`private_cell`, `:392-421`) | interleaved (each `MmapBuffer` made and timed before the next) | "every huge-page-backed buffer was slow (0.357 ms), H-backing falsified" | **yes: that falsification was made in the interleaved pattern only.** Huge-page backing in the batch-first pattern is untested |
| B2 `h6` (`prefix_cell`) and `h7` (`alloc_order`) | interleaved (h6: smaller buffers first, each timed; then 48 interleaved) | "allocation context not the cause" (R-P8 falsified), "fresh process all slow" | **yes** for the same reason; the "small-first context" test used the interleaved pattern throughout |
| N3 stages A to D, bridge, row timing, third C, low C, copy cost, L40S ends (`overlap_arena.py`, arena sources) | batch-first sources | all of the layer-ahead, slowdown and window results | **no**; their copy costs are the arena's. The slow-population instance (0.349 ms) is a separate, instance-level reading, recorded as such |
| Session K `k0` to `k5` | interleaved | class of the instance by the committed rule (slow), routes | **yes**: the committed DC-6 rule reads any instance as slow (K-run: 12 of 12) because of its pattern; the 64 MiB plateau of the K-run (1.6 to 2.1 ms) came from the same pattern |
| Follow-up L `l1` to `l4` | the pattern under test | the finding itself | n/a |

## 3. What changes in the lane's statements

1. **Withdrawn as stated:** "the slow class is a property of the buffer, stable per buffer, set below the guest" (N2 section 4, B1, B2 summaries) as a general statement about the host. It was measured in the interleaved pattern; the batch-first pattern gives 0 to 38 % slow depending on the instance. The notes added to `CUDA-BACKING-RESULTS.md` and `CUDA-HUGEPAGE-RETEST-RESULTS.md` (0.66.59) say it; this audit sharpens them (the pattern is *a* cause, not the whole story).
2. **Still standing:** every result that used the arena pattern (N2 onward), its copy costs, the window and slowdown models, the instance populations (0.194 or 0.349 ms) as recorded readings of the arena's sources. The lane's population labels (fast 0.194, slow 0.349) describe arena sources and stay valid as such.
3. **Open and now sharper:** (a) why interleaved is slow; (b) why the slow-population instance's batch-first arena was slow; (c) whether huge-page backing helps in the batch-first pattern; (d) whether a registered route helps in the batch-first pattern. Follow-up M (`docs/CUDA-PIN-ORDER-RECORD.md`) registers cells for (a), (c) and (d) and a re-specified classification.

## 4. What this audit did not do

It did not re-run anything, did not read every cell's raw data, and relies on the results files' recorded numbers. Whether a pinned allocation made after a copy is mapped differently (page pinning, IOMMU) is hypothesis.
