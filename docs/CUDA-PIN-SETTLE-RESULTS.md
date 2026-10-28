# CUDA lane follow-up N: results (a pinned buffer is slow after the CPU writes it, not after it is allocated)

Written 2026-10-28 against runtime 0.68.0. Record: `docs/CUDA-PIN-SETTLE-RECORD.md` (N-1 to N-8, committed first in 8f2077e before any code; an erratum on n2's buffer count committed with the build in 6afe0bb, before any run). Scored by `benchmarks/cuda/pin_route_settle_score.py --failed-attempts 1`; its output is `docs/cuda-results/pin-settle-2026-10-28.json`. Raw run directories are kept outside the repository under `~/cachalot-private/cuda-results/n1/` (archive sha256 `f9e0f45e…e03829`, the same on the instance and on the Mac). Labels follow plan section 0.

## 1. What ran

**Instance.** `calm-tongue-6106`, the RTX 5090 that follow-up M ran on minutes earlier (`docs/CUDA-PIN-ORDER-RESULTS.md` section 1), kept up at Hamed's word.

**Attempt 1 (failed, a harness defect).** Pack `6afe0bb` (dirty 0), 19:03:01 UTC: `k4` and `l1` complete; the `n1` process measured its six trains, wrote its manifest and raw samples, then stopped in the row writer (`ValueError: row: an unavailable row needs a reason in 'note'`: a train that never settles had an `unavailable` row without a note). The session stopped there by its rule. Its raw data is kept and agrees with attempt 2 (every train's last 100 copies 0.358 to 0.359 ms, no copy below 0.357 ms).

**Fix.** The row carries a note; a test that feeds a train that never settles through the row validator was added (f27844f).

**Attempt 2.** Pack `f27844f` (dirty 0), `session.sh pinroute4` from 19:04:12 to 19:05:03 UTC: seven fresh processes (`k4`, `l1`, `n1`, `n2`, `n3`, `n4`, `k4`), each status complete with every gate true, 16.6 s of run time in all.

## 2. Scores of N-1 to N-8

All numbers measured; the explanation is mine and a hypothesis unless stated. 9.95 MB buffers; "slow" at or above 0.28 ms, "fast" at or below 0.25 ms; per-buffer p50 of 20 copies after 5 warm-up unless stated.

| id | result | score | note |
|---|---|---|---|
| N-1 | both `k4` runs 24 of 24 slow (0.359 ms); `l1` 0 of 24 (0.182 ms) | in interval | |
| N-2 | n2, allocate then idle T ms: T = 0 and T = 5: 8 of 8 slow (0.359 ms); T = 20, 50, 100, 200, 500: 20 of 20 fast (0.182 to 0.183 ms) | **past falsifier** | time alone makes a buffer fast, between 5 and 20 ms (H-cache as stated is falsified) |
| N-3 | n3, allocate then unrelated CPU writes: none 8 of 8 slow (0.350 ms); 64 MiB (1.9 ms of writing) 7 of 8 slow (0.328 ms); 256 MiB (8.9 ms) 0 of 8 slow, median 0.210 ms (0.184 to 0.230) | in interval | not discriminating: the 256 MiB write also takes 8.9 ms, between N-2's 5 and 20 ms, and its buffers are only partly fast |
| N-4 | n4, buffers aged 500 ms: untouched 0 of 8 slow (0.182 ms); read by the CPU (`sum`, 8.9 ms) 0 of 8 (0.182 ms); rewritten by the CPU (24 ms) 8 of 8 slow (0.353 ms) | in interval | a CPU write makes an old buffer slow again; a CPU read does not (H-time as stated is falsified) |
| N-5 | n1, back-to-back copies for 1,000 ms: 6 of 6 buffers slow to the end (2,699 to 3,059 copies each; no copy at or below 0.25 ms, the fastest 0.294 ms) | in interval | copying keeps a buffer slow for at least 1,000 ms, while 20 ms of idle makes it fast |
| N-6 | slow median over the session's fast median: `k4` 1.97, n1 1.89, n2 1.97, n3 1.86, n4 1.94 | in interval | the slow mode is about half rate throughout |
| N-7 | N-2 reads on the H-time side, N-3 and N-4 on the H-cache side | **past falsifier** | neither hypothesis as registered |
| N-8 | attempt 2 clean, 16.6 s; attempt 1 stopped on the row writer | **past falsifier** | the falsifier counts a failed cell of any attempt |

**Four held (N-1, N-3, N-4, N-6), one held as registered but on the wrong premise (N-5: it was written for H-cache), three falsified (N-2, N-7, N-8).**

## 3. What it shows

Measured on this instance:

1. **Allocation is not the variable; the last CPU write is.** A buffer that has just been allocated is slow because allocation writes it (`pinned()` fills it with random bytes); a buffer aged 500 ms and then rewritten by the CPU is slow again (n4), while one read by the CPU stays fast.
2. **Without device reads, a written buffer becomes fast within 5 to 20 ms** (n2), with the same 0.182 ms as a buffer that was never slow.
3. **With device reads back to back, it stays slow** for at least 1,000 ms and 2,700 to 3,060 copies (n1). Copying the buffer prevents the recovery that idling allows.
4. CPU writes elsewhere (n3) take as long as the idle window, so they do not separate eviction from time; the 256 MiB buffers sit between the modes (0.184 to 0.230 ms).

**Hypothesis (H-dirty-resident, not tested here).** The device reads a buffer at half rate while the CPU's recent writes to it are still held dirty in the host's caches, served through the coherence path. With the buffer idle, other traffic on the host (other tenants of a shared last-level cache, the hypervisor) evicts those lines within 5 to 20 ms. Back-to-back device reads keep the lines resident (an I/O read that hits refreshes them). A CPU read leaves lines clean, so they do not slow the copy. This explains every cell of K, L, M and N; it is the reading, not a measurement. A direct test needs a cache-line flush (`clflushopt` over the buffer) or non-temporal stores in the fill, and a second instance or card.

**For a CUDA expert-streaming design (derived from the above; not measured in a streaming loop).** A host tier filled by a CPU copy (a page-cache read, a decompressor) and sent to the device at once moves at half rate. Filling by DMA (an `O_DIRECT` read into the pinned buffer, which the CPU never writes), or leaving 20 ms or more between the fill and the copy (a staging ring), should avoid it. The N2 and N3 arenas wrote their sources once and timed them later, so they measured the fast mode. Every "slow class" of S1, B1, B2, K and M was a buffer copied right after the CPU wrote it.

## 4. Corrections to earlier sections

* 0.66.59 (§18.214 and §18.215) and the 0.66.60 audit read the slow rate as the harness's "allocate-then-copy order". More precisely, it is "copied within about 5 ms of a CPU write, or kept busy since". Allocation matters only because allocation writes the buffer.
* M's H-after-copy (M-2, M-3) is falsified there. N's H-time (an allocation-age effect) and H-cache as stated (time alone does not help) are each falsified by one cell here.

## 5. Limits

One instance, one card, the fast population, 9.95 MB buffers only, one host CPU of unknown model (7 vCPUs of a shared host). The settle window is bracketed between 5 and 20 ms, not measured finer. The slow-population instances (the 0.349 ms arena of 0.66.52) are not explained by this and were not re-measured: an arena whose sources were written long before still read slow there. Nothing here is about a model, a default or the Mac runtime.

## 6. Next cells (not registered)

A finer idle sweep between 5 and 20 ms; a train with idle gaps (does the recovery accumulate?); a fill with non-temporal stores and a `clflushopt` cell (a small C helper); an `O_DIRECT` read straight into a pinned buffer and copied at once; the same cells on a slow-population instance and on an L40S.
