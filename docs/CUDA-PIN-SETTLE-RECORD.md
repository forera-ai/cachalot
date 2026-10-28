# CUDA lane follow-up N: is a just-allocated pinned buffer slow for a time after allocation, or while its bytes are still in the CPU's caches? (experiment record; predictions before any build)

Written 2026-10-28 against runtime 0.68.0, **after follow-up M ran on `calm-tongue-6106` and before any code for this session is written or run**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in `docs/CUDA-PIN-SETTLE-RESULTS.md` afterwards. A prediction edited after its run is void. Hamed asked for it on 2026-10-28 ("Keep calm-tongue-6106 up. Register follow-up N (allocate, wait T, then time; T swept), predictions committed first, build it, run it, then write everything up and release."). The instance is the one follow-up M ran on, so the population is known (M-1 held there).

## 1. Baseline (follow-up M, measured on `calm-tongue-6106`, 2026-10-28)

* `k4` (each 9.95 MB `cudaHostAlloc` buffer timed right after it is made): every one of 24 buffers 0.359 ms (27.7 GB/s); `l1` (all 24 made first, then timed): every one 0.182 ms (54.6 GB/s). The two modes are 1.97 times apart and there is nothing between them.
* `m1` (12 made, one copy, 12 more, then all timed) and `m4` (one copy of another buffer, then 24 made, then timed): all 48 buffers fast. **H-after-copy is falsified (M-2, M-3):** a copy run earlier in the process does not make later allocations slow.
* So what separates `k4` from `l1` is what happens between a buffer's allocation and its timing. In `l1` the first buffer is timed after 23 further allocations and no copy; the last after 23 other buffers' timings and their byte round trips, and no allocation. Both were fast.

**What the instrument does between allocation and timing (read from `backing_probe.py`):** `pinned()` writes 9.95 MB of random bytes into a pageable tensor and `pin_memory()` copies them into the pinned buffer, so the pinned buffer's bytes have just been written by the CPU; `roundtrip_ok()` copies the device bytes into a new 9.95 MB pageable tensor and compares it with the host buffer (about 30 MB of CPU memory traffic).

**Two working hypotheses, each consistent with all of M:**

* **H-time:** a newly pinned buffer copies at half rate for some time after allocation (for example, a driver or hypervisor mapping that completes asynchronously), whatever the CPU does meanwhile. From M's timings: slow for at least the 25 copies of one timing (about 10 ms) and settled within the time `l1`'s last buffer waited (23 timings and round trips, of the order of 100 ms).
* **H-cache:** the buffer is slow while its freshly written bytes are still held (dirty) in the CPU's caches, so the device's reads are served through the coherence path at half rate; it becomes fast once other CPU memory traffic (further allocations, round trips) evicts those lines, and only then. Time alone does not settle it; rewriting an old buffer makes it slow again.

## 2. Cells (each its own fresh process; `benchmarks/cuda/pin_route.py`, groups `n1` to `n4` new, `k4` and `l1` reused as controls)

All buffers 9.95 MB through `cudaHostAlloc` (torch `pin_memory`), kept alive to the end of the cell, a 9.95 MB device destination, default stream. "Timed" means the standard measurement (CUDA events, 20 repetitions after 5 warm-up, p50, GPU synchronised between repetitions), and the byte round trip is checked **after** the timing. The host time from the end of `pin_memory()` to the first timed copy is recorded for every buffer.

| id | cell | what it does |
|---|---|---|
| k4 | control, interleaved | as in M (start and end of the session) |
| l1 | control, batch-first | as in M |
| n1 | `settle_train` | 6 buffers in turn: allocate, then a back-to-back train of single timed copies (GPU synchronised between copies, no CPU work but the copies) for 1,000 ms, every copy's ms and its host time since allocation recorded; round trip after the train |
| n2 | `wait_sweep` | 27 buffers: allocate, `time.sleep(T)` with the GPU idle and no other work, then time; T in {0, 5, 20, 50, 100, 200, 500} ms plus 0 and 500 again (three each of 0 and 500, three of every other value), in a fixed shuffled order |
| n3 | `evict` | 24 buffers: allocate, then write E bytes into one unrelated pageable buffer allocated once at the start (`fill_`), then time at once; E in {0, 64 MiB, 256 MiB}, eight each, interleaved; the fill's host duration recorded |
| n4 | `retouch` | 24 buffers all allocated first and left 500 ms, then for each in turn: one of {nothing, a CPU read of the whole buffer (`sum`), a CPU rewrite of the whole buffer with new bytes}, cycled, then time at once |

Session order (`session.sh pinroute4`, output directory `benchmarks/results/cuda-settle`): `k4`, `l1`, `n1`, `n2`, `n3`, `n4`, `k4`, seven fresh processes. Gates: the byte round trip of every buffer (n4's rewritten buffers are checked against their new bytes); no copy rate above the PCIe sanity bound; no failed cell; not truncated.

## 3. Predictions (registered before any code)

All **predicted**; probabilities mine. "Slow" is the absolute rule of M (p50 at or above 0.28 ms), "fast" at or below 0.25 ms; a copy in n1's train is classed the same way by its own time. The predictions are written so that H-time and H-cache predict opposite outcomes in N-2, N-3 and N-4.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| N-1 | the controls separate again | both `k4` runs slow share at least 0.9 and `l1` at most 0.1 (probability 0.90) | either fails: nothing below is scored |
| N-2 | time alone (n2) | the T = 500 ms buffers have a slow share at least 0.8 (H-cache; probability 0.55); the T = 0 buffers at least 0.8 (0.90) | the T = 500 ms buffers have a slow share at most 0.2 (time alone settles a buffer: H-time) |
| N-3 | eviction without waiting (n3) | the E = 256 MiB buffers have a slow share at most 0.2 (0.55) and the E = 0 buffers at least 0.8 (0.90); the 64 MiB buffers are not scored as a prediction (reported) | the E = 256 MiB buffers have a slow share at least 0.8 and the fill took under 20 ms |
| N-4 | re-dirtying a settled buffer (n4) | the "nothing" buffers slow share at most 0.1 (0.85); the rewritten buffers at least 0.8 (0.55); the read-only buffers are reported, not predicted | the rewritten buffers at most 0.2 (rewriting does not make an old buffer slow again) |
| N-5 | the train (n1) | at least 5 of 6 buffers stay slow for the whole 1,000 ms (the last 100 copies slow; probability 0.50); if they settle, the median settle time (first copy of 10 fast in a row) is between 10 and 150 ms (conditional, 0.50) | every buffer settles within 150 ms of copies while N-2 reads slow at 500 ms: settling needs copies, neither hypothesis as stated |
| N-6 | the slow rate is half the fast rate | in every cell with slow buffers, the slow median over the fast median of the same session is between 1.8 and 2.2 (0.75) | outside 1.6 to 2.4 in any cell |
| N-7 | the hypotheses are separated | N-2, N-3 and N-4 all read on the H-cache side, or all on the H-time side (0.70) | they split (some each way): the mechanism is neither as stated |
| N-8 | completion | all gates true, no failed cell, seven processes under 60 s of GPU time together (0.85) | any gate false, a failed cell or truncation |

## 4. What each outcome means

* **H-cache side (N-2 slow at 500 ms, N-3 fast after 256 MiB, N-4 rewritten slow):** a pinned buffer is slow to copy while the CPU still holds its freshly written lines. For a CUDA expert-streaming design this is a rule about the host tier: a buffer filled by the CPU (a decompressor, a file read through the cache) and copied to the device at once is copied at half rate until its lines leave the CPU's caches; reading from disk with direct I/O (DMA into the pinned buffer) or letting a buffer age before its copy avoids it. Every earlier "slow class" of S1, B1, B2, K and M is then a measurement of a cache-hot source, and the arena's fast sources were cold.
* **H-time side (N-2 fast at 500 ms, N-3 slow, N-4 rewritten fast):** the slowness is a property of a new pinned allocation that wears off with time, and n2 gives its length; the cure is to allocate the host tier ahead of use.
* **A split (N-7 falsified):** neither story; the cells say which factor (time, CPU traffic, rewriting) moves the rate and the next record names a mechanism from them.
* Nothing here is a statement about another card or instance population, a model or a default; the instance's population (fast, per M) is fixed.

## 5. Stop rules and cost

Stop and bring the files home on a failed cell, a false gate, or truncation. A defect found on the instance is fixed on the Mac. The instance has been up since about 18:49 UTC for follow-up M; this session's cap is the same **1 GPU-hour from Ready** (terminate by about 19:50 UTC), so the build is kept to the four cells above. Estimated GPU run time: `n1` 6 s, `n2` about 4 s, `n3` and `n4` about 2 s each, the controls 1 s each: under 30 s.

## 6. Needs Hamed

Nothing new: the instance exists and he said go. He terminates it after the files are home.
