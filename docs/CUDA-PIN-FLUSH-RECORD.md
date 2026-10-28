# CUDA lane follow-up O: is the half-rate pinned copy the CPU's dirty cache lines, how long do they take to clear, and does a DMA fill avoid it? (experiment record; predictions before any build)

Written 2026-10-28 against runtime 0.68.1, **after follow-up N and before any code for this session is written or run**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in `docs/CUDA-PIN-FLUSH-RESULTS.md` afterwards. A prediction edited after its run is void. Hamed chose the cells on 2026-10-28 (all four of `docs/CUDA-PIN-SETTLE-RESULTS.md` section 6: a `clflushopt` or non-temporal fill, an `O_DIRECT` read into a pinned buffer, a finer idle sweep, a copy train with idle gaps) and the card: an RTX 5090 of the fast population, the population of M and N.

## 1. Baseline (follow-up N, measured on `calm-tongue-6106`, 2026-10-28)

The instance: one RTX 5090, 7 vCPUs of an **AMD Ryzen 7 7800X3D** (one CCD, 96 MB of L3 with 3D V-Cache, shared by every tenant on that CCD), PCIe Gen 5 x16. 9.95 MB `cudaHostAlloc` buffers; "slow" p50 at or above 0.28 ms, "fast" at or below 0.25 ms.

* A buffer copied right after the CPU wrote it is slow (0.359 ms against 0.182 ms, two modes 1.97 times apart). `pinned()` writes the buffer (`pin_memory()` copies random bytes into it), so every freshly allocated buffer is in this state.
* Idle T ms after the write (n2): T = 0 and 5 ms slow (8 of 8), T = 20, 50, 100, 200, 500 ms fast (20 of 20).
* A back-to-back train of copies (n1) keeps it slow for at least 1,000 ms (2,700 to 3,060 copies, none at or below 0.25 ms).
* A buffer aged 500 ms is fast; rewritten by the CPU it is slow again; read by the CPU (`sum`) it stays fast (n4).
* 256 MiB of unrelated CPU writes (8.9 ms of writing) leave the buffers between the modes (0.184 to 0.230 ms, median 0.210; n3).

**Working hypothesis from N, H-dirty (never tested directly):** the device reads a buffer at half rate while the CPU's recent writes to it are held dirty in the host's caches, served through the coherence path; with the buffer idle, other traffic evicts the lines within 5 to 20 ms; back-to-back device reads keep them resident.

**Its competitors, each consistent with all of N:**

* **H-page:** the slowness is keyed on the page having been written recently, outside the CPU caches (for example the hypervisor's or the IOMMU's tracking of written guest pages), and wears off with time. A cache flush does not help; a non-temporal rewrite (a write that bypasses the caches) is still a write and is slow.
* **H-accumulate** (about the train only): idle time clears the state whatever the device does in between, so idle gaps between copies add up. N's train argues against it only weakly: its gaps between copies (about 0.01 ms each, about 27 ms over 1,000 ms) were short.

## 2. Cells (each its own fresh process; `benchmarks/cuda/pin_route.py`, groups `o1` to `o4` new, `k4` and `l1` reused as controls)

All buffers through `cudaHostAlloc` (torch `pin_memory`), kept alive to the end of the cell, a 9.95 MB device destination, default stream. "Timed" means the standard measurement (CUDA events, 20 repetitions after 5 warm-up, p50, GPU synchronised between repetitions); the byte round trip and any content check are done **after** the timing, never between an operation and its timing.

| id | cell | what it does |
|---|---|---|
| k4 | control, interleaved | as in M and N (start and end of the session) |
| l1 | control, batch-first | as in M and N |
| o1 | `fine_wait` | 36 buffers: allocate, idle T ms (GPU idle, `time.sleep`), then time; T in {0, 5, 7.5, 10, 12.5, 15, 17.5, 20, 50} ms, four of each, in a fixed shuffled order |
| o2 | `gap_train` | 10 buffers, two for each gap G in {1, 2, 5, 10, 20} ms in a fixed interleaved order: allocate, then for 400 ms single timed copies each followed by G ms of idle; every copy's ms and its host ms since allocation kept; round trip after the train |
| o3 | `flush` | 25 buffers, five for each mode, cycled: allocate, then one of {`none`; `memcpy`: rewrite the buffer with new bytes by an ordinary `memcpy`; `nt`: rewrite it with new bytes by non-temporal 16-byte stores then `sfence`; `clflushopt`: flush every line of the buffer then `sfence`; `clwb`: write every line back, keeping it, then `sfence`}, then time at once. The four operations run in a small C helper (`benchmarks/cuda/cache_touch.c`, compiled on the instance with the system `cc`; `clflushopt` and `clwb` used only if CPUID reports them). The operation's host ms is recorded |
| o4 | `direct_read` | 24 buffers of 9,953,280 bytes (the next multiple of 4,096 above 9.95 MB, so `O_DIRECT` can read whole blocks; only the first 9.95 MB are timed) all allocated first and left 500 ms; a file of 18 records written, `fsync`ed and dropped from the page cache before the allocation; then each buffer in turn, cycled: one of {`none`; `direct`: an `O_DIRECT` read of a record into the buffer; `buffered`: an ordinary read of a record into the buffer; `direct_wait`: an `O_DIRECT` read, then 50 ms idle}, then time at once. The read's host ms is recorded; after the timing the buffer is checked against the record's bytes |

Session order (`session.sh pinroute5`, output directory `benchmarks/results/cuda-flush`): `k4`, `l1`, `o1`, `o2`, `o3`, `o4`, `k4`, seven fresh processes. Gates: the byte round trip of every buffer; o3's rewritten and o4's read buffers checked against their new bytes; no copy rate above the PCIe sanity bound; no failed cell; not truncated. A mode the instance cannot run (no C compiler, an instruction CPUID does not report, a file system that refuses `O_DIRECT`) is an `unavailable` row with the reason, not a failed cell; its prediction is then not testable.

## 3. Predictions (registered before any code)

All **predicted**; probabilities mine. "Slow" share is the share of buffers whose p50 is at or above 0.28 ms; "fast" at or below 0.25 ms; a copy in a train is classed by its own time.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| O-1 | the controls separate again | both `k4` runs slow share at least 0.9 and `l1` at most 0.1 (0.90) | either fails: nothing below is scored |
| O-2 | o1 reproduces N's ends | T = 0 and T = 5 ms slow share at least 0.75; T = 20 and T = 50 ms at most 0.25 (0.80) | T = 5 ms at most 0.25 or T = 20 ms at least 0.75 |
| O-3 | o1 transition | the settle time T50 (the T at which the slow share, interpolated linearly between the swept values, first falls to 0.5 or below) is between 7.5 and 15 ms (0.60) | T50 at or below 5 ms or above 20 ms |
| O-4 | o1 shape | the transition is gradual, as line-by-line eviction would be: at least 3 of the 20 buffers with T in {7.5 ... 17.5} have a p50 strictly between 0.20 and 0.28 ms (0.50) | every one of the 36 buffers is at or below 0.20 ms or at or above 0.28 ms (a switch, not an eviction) |
| O-5 | o2 short gaps | the G = 1 and G = 2 ms trains stay slow: slow share of each train's second half at least 0.9 (H-dirty with reads refreshing the lines; 0.70) | any G = 1 or G = 2 ms train has 10 fast copies in a row (idle time accumulates across the gaps: H-accumulate) |
| O-6 | o2 long gaps | the G = 20 ms trains are fast from the second copy on: fast share of copies 2 to the end at least 0.9 (0.75); G = 5 and G = 10 ms reported, not predicted | a G = 20 ms train's copies 2 to the end have a fast share at most 0.5 |
| O-7 | o3 controls | `none` and `memcpy` slow share at least 0.8 each (0.85) | either at most 0.2 |
| O-8 | o3 flush | `clflushopt` slow share at most 0.2 (H-dirty; 0.70) | `clflushopt` slow share at least 0.8 (the cache lines are not the cause: H-page) |
| O-9 | o3 non-temporal rewrite | `nt` slow share at most 0.2 (H-dirty; 0.60) | `nt` slow share at least 0.8 (a write is slow even if it bypasses the caches: H-page) |
| O-10 | o3 clean and resident | `clwb` slow share at most 0.2 (dirtiness, not residency, is the variable; 0.55) | `clwb` slow share at least 0.8 (resident lines are slow even when clean) |
| O-11 | o4 controls | `none` slow share at most 0.1 and `direct_wait` at most 0.2 (0.80); `buffered` at least 0.8 (the kernel's copy writes the buffer through the CPU caches; 0.75) | `none` at least 0.5, or `buffered` at most 0.2 |
| O-12 | o4 DMA fill | `direct` slow share at most 0.2 (0.55; low because how this VM's disk reads reach guest memory is not known: a host CPU copy into guest pages would leave the lines dirty in the shared L3) | `direct` slow share at least 0.8 (N's design rule "fill by `O_DIRECT`" fails on this instance) |
| O-13 | slow is half rate | in every cell with slow buffers, the slow median over the session's fast median (the `l1` median) is between 1.8 and 2.2 (0.75) | outside 1.6 to 2.4 in any cell |
| O-14 | the hypotheses are separated | O-8 and O-9 read the same way: both on the H-dirty side or both on the H-page side (0.75) | they split (`clflushopt` fast and `nt` slow, or the reverse) |
| O-15 | completion | all gates true, no failed cell in any attempt, every mode available (helper compiled, `clflushopt` and `clwb` reported, `O_DIRECT` accepted), seven processes under 60 s of run time together (0.65) | any gate false, a failed cell or truncation |

## 4. What each outcome means

* **H-dirty side (O-8, O-9 and O-10 fast, O-5 slow):** the half rate is the device reading lines the CPU holds dirty. For a CUDA expert-streaming host tier: write the tier with non-temporal stores, flush it, or fill it by DMA (O-12 says whether `O_DIRECT` counts as DMA on such a VM), or let it age past T50 (O-3) before its copy; never re-copy a just-filled buffer in a tight loop expecting it to warm up (O-5).
* **H-page side (O-8 and O-9 slow):** the cause is outside the CPU caches and keyed on recent writes; no CPU instruction avoids it, only time (O-3 gives the wait) or a fill the guest CPU does not perform (O-12).
* **A split (O-14 falsified):** for example `clflushopt` fast and `nt` slow would mean the non-temporal stores still left lines behind (write-combining buffers, or an implementation that keeps the line), and the next record would test that alone.
* **O-5 falsified (short-gap trains settle):** idle time accumulates and device reads do not refresh the state; N's train stayed slow only because its gaps were too short in total.
* Nothing here is a statement about another card, another host CPU or instance population, a model or a default.

## 5. Stop rules and cost

Stop and bring the files home on a failed cell, a false gate, or truncation. A defect found on the instance is fixed on the Mac and the scorer counts the failed attempt (O-15). Cap **1 GPU-hour from Ready**. Estimated run time: `o1` about 2 s (36 buffers, mean wait 15 ms, plus timing and round trip), `o2` about 4.5 s (10 trains of 400 ms), `o3` and `o4` about 2 s each (o4 writes 180 MB first), the controls about 1 s each: under 15 s. The setup (unpack, `setup`, `doctor`, `smoke`) is about 10 minutes as in M and N.

## 6. Needs Hamed

One RTX 5090 of the fast population (if `calm-tongue-6106` is still up it can be reused; otherwise a new one, and O-1 checks its population), and his go to run `session.sh pinroute5` on it. He terminates it after the files are home.
