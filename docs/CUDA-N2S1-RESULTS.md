# CUDA lane N2 stage 1: results (RTX 5090, a real slot arena with real pinned host-to-device copies driven by the recorded DeepSeek trace)

Run 2026-10-18 under `docs/CUDA-N2S1-EXPERIMENT-RECORD.md` (12 predictions committed in b1974e2 before any harness code or instance). Instrument: `benchmarks/cuda/replay_arena.py` at scaffold commit 2224a7f (`REVISION` dirty=0), runtime 0.66.23. Instance: a third rented RTX 5090 virtual machine of the S1 and S2 class (driver 580.159.03, torch 2.7.1+cu128, 7 vCPUs, 23 GiB RAM, PCIe 5.0 x16). Every number is **measured** unless marked **derived**, and describes one VM on one day (plan section 9). Stage tag: **N2 stage 1** (a replay with real copies of dummy expert-sized records; no compute, prefetch or overlap, no real model routing beyond the recorded trace). Only the RTX 5090 arm of the record was run: the V100 column was not (it needs its own session). Sanitized manifests and rows: `docs/cuda-results/n2-rtx5090-2026-10-18/`; raw samples stay local. Billed time: to be logged by Hamed from the console's transactions page (GPU run time of the stage 1 cells themselves was 130 seconds); no hostname, address, key or amount appears here.

## 1. What ran

`doctor`, the S1 `smoke`, then `n2smoke` (two 40-token cells), then `n2full`: the recorded trace (`routing-trace-deepseek-long`, 1,972 decode tokens, 40 layers x 384 experts, top-6) at 8, 12 and 17 % resident (1,229 / 1,843 / 2,611 slots of 9,950,000 bytes) and the seeded generator at 17 % (200-token prefill, 1,972 decode tokens), LRU. Gates, all true in the final smoke runs and in the full run: event hash equal to the Mac simulator's for every cell (the host checks its own hash against `n2s1-expected.json` before timing), determinism (a second bookkeeping pass gives the same hash and the same rolling chain), arena read-back (a copied slot against its source after every 2,000th copy and a final seeded sample of 48 resident slots: 0 failures), copy-rate sanity, no failed cell, not truncated. Cell time 130 s for the four cells together.

Two defects were found by the first `n2smoke` and fixed on the Mac (commits fbebe58 and 2224a7f): the generator arrays imported `cachalot` (absent on the host), and the exploratory source-qualification copies ran outside the copy stream's context so their events read 6 us (voided; fixed, the qualification numbers below are from the fixed code). A third (the arena is not released between cells, so the allocator's reserved memory carries the previous cell's arena) is fixed in the code after the run and was not re-run; it affects only one row (section 3, N2-P9).

## 2. Results by field name (p50 over 1,972 tokens; ranges and p95 where useful)

| cell | misses a token | exposed copy ms a token (p50, p95) | event-time ms a token | ms per extra miss (fit R-squared) | per copy, layers with 1 miss / 5 or more (ms) | bookkeeping ms a token | bytes H2D a token |
|---|---:|---|---:|---|---|---:|---:|
| `trace.p8` | 90.84 | 16.74 (25.86) | 16.46 | 0.1835 (0.9988) | 0.1947 / 0.1843 | 0.156 | 903.8 MB |
| `trace.p12` | 69.64 | 12.83 (21.35) | 12.51 | 0.1841 (0.9994) | 0.1945 / 0.1841 | 0.138 | 692.9 MB |
| `trace.p17` | 51.85 | 9.43 (17.36) | 9.19 | 0.1853 (0.9994) | 0.1945 / 0.1843 | 0.119 | 515.9 MB |
| `gen.p17` | 72.26 | 13.53 (15.63) | 13.22 | 0.1833 (0.9969) | 0.1944 / 0.1840 | 0.138 | 719.0 MB |

Isolated-copy `control_drift` (start against end of a cell): 0.07 to 0.23 %. Read-back failures 0 in every cell. Reserved memory beyond the arena: 1.9 MiB (`trace.p8`), 0.07 MiB (`trace.p17` and `gen.p17`), 11,666 MiB (`trace.p12`, an artifact, see N2-P9).

**Source qualification (exploratory, not registered):** each of the 64 pinned source records copied alone into one slot, 15 repeats, event ms p50. In every cell the fastest record reads 0.1844 ms, the median 0.1847, and **3 or 4 records out of 64 are slow, 0.28 to 0.36 ms** (shares 0.047 to 0.063 of records at 1.5 or more times the fastest).

## 3. Scoring the registered predictions (RTX 5090 arm only)

| id | outcome |
|---|---|
| N2-P1 (event sequence equals the simulator's) | **held** for all six cells, on the Mac and again on the host (the host refused to time a cell whose hash differed; none did) |
| N2-P2 (exposed copy time a token at 8 / 12 / 17 %: 28 to 36 / 21 to 27 / 15.5 to 21 ms) | **missed low in all three**: 16.74 / 12.83 / 9.43 ms. The intervals assumed 0.33 to 0.36 ms a copy (S1 and S2); the replay's copies cost 0.18 to 0.19 ms (section 4) |
| N2-P3 (0.30 to 0.38 ms per extra miss, R-squared at least 0.95) | **missed low**: 0.183 to 0.185 ms; the fit held (R-squared 0.997 to 0.999) |
| N2-P4 (1-miss layers 0.34 to 0.38 ms, 5-or-more layers 0.25 to 0.34; falsifier: 5-or-more not cheaper) | both intervals **missed low** (0.1945 and 0.184); the falsifier did **not** fire: layers raising 5 or more misses are about 5 % cheaper a copy, in direction as predicted, far smaller than predicted |
| N2-P5 (CUDA-H9 replay test: 0.8 to 1.2 ms of exposed time per GiB between 8 and 17 %, falsifier: outside +-30 % of 3.0 x 0.35 = 1.05) | **missed low, falsifier fired as written**: (16.74 - 9.43) ms over 12.81 GiB = **0.571 ms per GiB**. With the copy cost the replay actually measured the registered formula holds: 3.04 misses a token a GiB (N0a) x 0.1847 ms = 0.562, within 2 % of 0.571, so the structure (slope times per-copy cost) is right and the assumed copy cost was wrong |
| N2-P6 (bytes H2D a token at 17 % within 1 % of 516 MB) | **held**: 515.9 MB |
| N2-P7 (V100 over 5090 ratio) | **not run** (no V100 arm) |
| N2-P8 (generator at default knobs misses more than the trace's 51.85 a token at 17 %) | **held**: 72.26 (a property of the replay, identical on any card) |
| N2-P9 (reserved memory beyond the arena under 300 MiB) | **held** in `trace.p8` (1.9 MiB), `trace.p17` and `gen.p17` (0.07 MiB); `trace.p12` reads 11,666 MiB because the previous cell's 11.4 GiB arena was still in the allocator's cache (the harness did not call `empty_cache` between cells): an instrument defect, fixed in the code, not re-measured |
| N2-P10 (bookkeeping under 1.5 ms a token) | **held**: 0.12 to 0.16 ms |
| N2-P11 (a cell inside 6 minutes on the 5090) | **held**: the four cells together took 130 s |
| N2-P12 (gates b, c, d hold) | **held** |

## 4. The finding the predictions did not cover: the H2D rate of a 9.95 MB pinned buffer is a property of the buffer (exploratory, post hoc)

The replay's copies cost 0.18 to 0.19 ms (about 54 GB/s, at the link's measured plateau), not the 0.33 to 0.36 ms of the S1 and S2 cells. The same instance, run with the S2 profile minutes later, reproduced the old number (`batchcopy.k1.single` 0.334 ms; control 27.8 and 30.0 GB/s). Four throwaway probes (`benchmarks/cuda/exploratory/`, output in `docs/cuda-results/n2-rtx5090-2026-10-18/exploratory-output.txt`) separated the causes:

* **Not** the copy stream (default or side stream: 0.185 and 0.191 ms), **not** the destination (a whole tensor, an arena slice at offset 0 and a misaligned slice at slot 5: 0.185, 0.192, 0.192), **not** event reuse, pre-synchronization or the host clock (six variants, all 0.357 ms for the slow buffer), **not** warm-up or iteration count.
* **It is the source buffer.** One pinned buffer measures 0.357 ms in the suite's cell and 0.185 ms in a second process's first buffer. Twenty-four separately pinned 9.95 MB buffers in one process: 19 read 0.182 ms and 5 read 0.31 to 0.36, **stable on repeat** (a second pass over the first eight gave 0.182 each), and the slow ones were the later allocations (the 24 medians rise with allocation order). Rotating among 8 sources gave a bimodal 0.19 to 0.36 ms. A single 24-record pinned slab sliced into records is also mixed (0.18 to 0.33).
* **Hypothesis, not tested here:** the physical page backing of the pinned allocation (fragmented small pages against contiguous large ones) sets the DMA rate; at read time transparent huge pages were `madvise` and the process held 0 kB of anonymous huge pages. A larger buffer (64 MiB and up) reached the plateau in every S1 and S2 run, consistent with that reading but not proof.

**Consequences for earlier results (this note corrects them in place; the earlier files keep their text, as the record rule says, and a correction is carried by the plan section 17.5):**

1. **S1 finding 1 ("0.35 ms at the DeepSeek record, a fixed cost of about 160 us") and the S1 CUDA-H1 falsification** (54 % of the plateau at the record size) are artifacts of which pinned buffer the cell happened to get. A fast-backed 9.95 MB buffer reaches the plateau (0.182 ms, 54.7 GB/s). H1 is restated: the rate depends on the buffer; a slow-backed buffer reaches about half. The pageable-against-pinned ratio of H2 is likewise buffer-dependent (pageable 24 GB/s is 0.44 of fast pinned and 0.85 of slow pinned).
2. **S2 finding 4 (batched or queued copies recover up to 40 %)** is confounded: the `k = 1` buffer was slow-backed and the larger buffers faster, so the apparent batching gain on the 5090 is at least partly a buffer effect. It is not established.
3. **V1 is unaffected in its numbers** (a PCIe 3.0 x16 link plateaus at 11.9 GB/s, far below even a slow-backed buffer's 28 GB/s, so every buffer is link-bound there). V1's reading that "H1 holds on the V100 and failed on the 5090, a card-specific ramp" is withdrawn: the 5090's ramp was the buffer. With the fast figure the cost ratio of the two cards is 0.84 / 0.185 = 4.5 (the PCIe 3 against PCIe 5 link ratio, about 4), not 2.4.
4. **CUDA-H9, derived, restated again:** the Mac's 2.0 ms a miss against 0.185 ms on a fast-backed 5090 is a cost ratio of 10.8 (against 5.7 and 2.4 before), and the measured exposed time per GiB is 0.571 ms against the Mac's 1.7 to 2.2: a ratio of 3.0 to 3.9, **still under the registered 5 times**, so H9 as stated is still not met on the ms-a-GiB definition, by a narrower margin than the V1 reading suggested for the V100.
5. **The N2 intervals themselves** (N2-P2 to P5) were derived from the S1 and S2 microbenchmark costs and are the casualties of finding 1; they are scored as written, not edited.

## 5. What this does not establish

One new VM, one day, 130 s of stage 1 GPU time, 64 distinct pinned source records of which 3 or 4 were slow. A host tier emulated by 64 records is not a 153 GB bank: a real expert store pinned in one large slab would see a mixture like probe 4's slab (0.18 to 0.33 ms), so the replay's 0.185 ms is a best-case, fast-backed figure that a production allocator would have to earn. No compute stand-in, no prefetch, no overlap (N3). The V100 arm of the record (N2-P7) was not run. No Cachalot CUDA claim follows.

## 6. Smallest next experiments

(1) Pin the cause: allocate pinned buffers with a deliberate backing (a large buffer first, then slice; or ask the kernel for huge pages with `madvise` where the host allows it) and read `/proc/<pid>/pagemap` or the smaps huge-page counters per buffer, then redo S1's copy curve with qualified (fast) buffers; predictions first. (2) The V100 arm of this record (link-bound, so the cost should be 0.84 ms a copy whatever the buffer) to close N2-P7 and test N2-P5's formula on a second card. Both need Hamed's go for GPU time.
