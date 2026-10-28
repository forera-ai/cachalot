# CUDA lane follow-up P: the copy period that holds a pinned buffer slow, and a fill-then-flush host pipeline timed end to end (experiment record; predictions before any build)

Written 2026-10-28 against runtime 0.68.2, **after follow-up O and before any code for this session is written or run**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in `docs/CUDA-PIN-PIPELINE-RESULTS.md` afterwards. A prediction edited after its run is void. Hamed asked for it on 2026-10-28 ("Register follow-up P for the CUDA lane: the 1-2 ms copy-frequency threshold and a fill-then-flush pipeline timed end to end. Predictions first, build and rehearse on the Mac, no instance yet."). The card is an RTX 5090 of the fast population; the two instances so far (`calm-tongue-6106`, `pricey-voice-4663`) both ran on AMD Ryzen 7 7800X3D hosts.

## 1. Baseline (follow-up O, measured on `pricey-voice-4663`, 2026-10-28)

* A just-written pinned 9.95 MB buffer copies at 0.359 ms; flushed (`clflushopt`, 0.38 ms), written back (`clwb`) or filled with non-temporal stores it copies at 0.182 ms. The CPU's dirty lines are the cause.
* Left alone, the lines clear in one step 10 to 22 ms after the write (o1 T50 15.0 ms; o2 first fast copies at 10.8 to 42.7 ms).
* Copy trains: a single timed copy, then `time.sleep(G)`. **G = 1 ms gave a measured period of 1.44 ms and held the buffer slow for 400 ms (2 of 2 trains); G = 2 ms gave 2.34 ms and settled (2 of 2, at 20.7 and 42.7 ms).** The threshold period is somewhere in between.
* A lone copy after an idle gap costs about 0.225 ms when fast (0.04 ms of wake-up above `time_copy`'s warm 0.182); in a dense train a fast copy is 0.193 to 0.197 ms.
* Host costs on that CPU (o3, cache-hot source): a 9.95 MB `memcpy` rewrite 0.18 to 0.21 ms, a non-temporal rewrite 0.37 to 0.44 ms, a `clflushopt` pass 0.38 ms. None of these had a cold source, which a real host tier has (decompressed or read data that is not already in the L3).

**The two questions.**

1. **Where between 1.44 and 2.34 ms is the threshold, and is it sharp?** A sharp threshold (every train at one period holds, every train one step longer settles) points at a deterministic cadence on the host; a wide band of mixed outcomes points at a probabilistic eviction that frequent device reads make less likely.
2. **What does the dirty-line effect cost a real host pipeline, and which fill wins?** An expert-streaming host tier fills a pinned staging buffer from a cold source and copies it to the device while the next buffer is filled. Per O, a `memcpy` fill makes the copy slow (0.359 ms), a `memcpy` plus `clflushopt` makes it fast but adds a 0.38 ms host pass, and a non-temporal fill makes it fast with no extra pass. With fill and copy overlapped, the token-relevant number is the per-record wall time, `max(host work, copy)` plus overheads, and that can rank the arms differently from the copy time alone.

## 2. Cells (each its own fresh process; `benchmarks/cuda/pin_route.py`, groups `p1` and `p2` new, `k4` and `l1` reused as controls)

| id | cell | what it does |
|---|---|---|
| k4 | control, interleaved | as in M, N and O (start and end of the session) |
| l1 | control, batch-first | as in M, N and O |
| p1 | `period_train` | 18 trains of 300 ms, three for each target period P in {1.4, 1.6, 1.8, 2.0, 2.2, 2.4} ms, in a fixed shuffled order: allocate a 9.95 MB buffer, then single timed copies (CUDA events, GPU synchronised after each) started on a fixed schedule every P ms (the wait is `time.sleep` for all but its last 0.2 ms, then a spin, so the host idles as in O); every copy's ms and its host ms since allocation kept, the realised periods recorded; round trip after the train |
| p2 | `fill_pipeline` | 16 pageable source records of 9.95 MB (159 MB, more than the 96 MB L3, so a source read cycling through them is cold), written once and left 200 ms; 2 pinned staging buffers; 4 device slots. Four fill modes: `aged` (the staging buffers filled once at the start and left; no fill per record: the copy-only bound), `memcpy` (an ordinary copy of the next source into the staging buffer), `flush` (the same `memcpy`, then `clflushopt` over the staging buffer), `nt` (non-temporal stores of the next source into the staging buffer). Each pass is 64 records, run twice: **serial** (fill, then copy, then wait for the copy, record by record) and **pipelined** (the copy of record r runs on a copy stream while the host fills record r + 1 into the other staging buffer; a staging buffer is refilled only after its previous copy's event completes). Per record: the host fill ms (and the flush ms), the device copy ms (CUDA events around the copy on the stream); per pass: the wall ms from the first fill to the last copy's completion. Twelve passes in the fixed order aged, memcpy, flush, nt, nt, flush, memcpy, aged, aged, memcpy, flush, nt. After every pass the last four device slots are compared with their sources |

The helper is `benchmarks/cuda/cache_touch.c` from O (compiled on the instance with `cc`); its `ct_copy` is the `memcpy` and `ct_stream_copy` the non-temporal fill, so both fills go through the same code path. A mode the instance cannot run is an `unavailable` row with the reason, not a failed cell.

Session order (`session.sh pinroute6`, output directory `benchmarks/results/cuda-pipeline`): `k4`, `l1`, `p1`, `p2`, `k4`, five fresh processes. Gates: the byte round trip of every p1 buffer; every p2 pass's device slots equal their sources; no copy rate above the PCIe sanity bound; no failed cell; not truncated.

## 3. Predictions (registered before any code)

All **predicted**; probabilities mine. A copy is "fast" at or below 0.25 ms and "slow" at or above 0.28 ms by its own CUDA-event time. A p1 train **holds** if none of its copies is fast; otherwise it **settles**, at the host time of its first fast copy. "Per-record wall" is a pipelined pass's wall ms over 64; medians are over a mode's three passes.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| P-1 | the controls separate | both `k4` runs slow share at least 0.9 and `l1` at most 0.1 (0.90) | either fails: nothing below is scored |
| P-2 | p1 anchors reproduce O | the 1.4 ms trains hold (3 of 3) and the 2.4 ms trains settle (3 of 3) (0.75) | a 1.4 ms train settles or a 2.4 ms train holds, and the realised median period is within 0.05 ms of its target |
| P-3 | the threshold | P50, the period at which the hold share first falls to 0.5 or below (linear between the swept periods), is between 1.6 and 2.2 ms (0.60) | P50 at or below 1.4 ms or above 2.4 ms |
| P-4 | its sharpness | at most one swept period has mixed outcomes (some trains hold, some settle) (0.50) | three or more periods have mixed outcomes (a probabilistic band, not a cadence) |
| P-5 | p2 serial copy class | in the serial passes: `memcpy` copies slow share at least 0.9; `flush`, `nt` and `aged` copies fast share at least 0.9 each (0.80) | `memcpy` fast share at least 0.5, or `nt` or `flush` slow share at least 0.5 (the cold-source fill behaves unlike O's hot-source rewrite) |
| P-6 | p2 serial host costs | `memcpy` fill median 0.25 to 0.60 ms (cold source; 0.60); the `clflushopt` pass median 0.30 to 0.50 ms (0.70); the `nt` fill median at or below the `memcpy` fill median (non-temporal stores skip the read-for-ownership; 0.60) | `memcpy` fill median below 0.15 ms or above 1.0 ms; the flush median above 0.8 ms |
| P-7 | the copy-only bound | `aged` per-record wall 0.18 to 0.23 ms (0.70) | above 0.30 ms (the pipeline machinery itself costs more than a copy) |
| P-8 | which fill wins pipelined | `nt` has the lowest per-record wall of `memcpy`, `flush` and `nt` (0.55), and `flush` is slower than `memcpy` (its extra host pass outweighs the faster copy; 0.60) | `nt` per-record wall at least 1.1 times `memcpy`'s |
| P-9 | the pipeline model | for `memcpy`, `flush` and `nt`, the per-record wall is within 25 % of max(serial host ms, serial copy ms) of the same mode (0.50) | any of the three off by more than 50 % |
| P-10 | the dirty-line penalty in a pipeline | `memcpy`'s per-record wall over `aged`'s is between 1.5 and 2.5 (0.55) | below 1.2 (the slow copy hides behind the fill) |
| P-11 | completion | every gate true, no failed cell, every mode available, five processes under 60 s of run time together (0.70) | any gate false, a failed cell or truncation |

## 4. What each outcome means

* **P-3 and P-4 held (a sharp threshold between 1.6 and 2.2 ms):** a cadence on the host clears dirty lines unless the device has touched them within about that period; the trigger is then a periodic event to look for (a timer, a scrubber, another tenant's periodic stream), and an expert tier that re-reads the same buffer faster than that never warms up.
* **P-4 falsified (a band):** each read makes eviction less likely rather than impossible; the trigger is probabilistic traffic, and the number to use is a hold probability against period, not a threshold.
* **P-8 held:** an expert host tier on such a VM should fill with non-temporal stores; a flush pass is a fallback when the filler cannot use them (a library decompressor). **P-8 falsified with `memcpy` best:** the fill dominates and the half-rate copy hides behind it, so the dirty-line effect does not matter at this pipeline depth (P-10 says how much it does).
* **P-9 falsified:** the overlap is not free (host fills and device reads of host memory compete for DRAM bandwidth, hypothesis), and the row models of the N3 lane need a host-contention term.
* Nothing here is a statement about another card, host CPU, instance population, a real model or a default.

## 5. Stop rules and cost

Stop and bring the files home on a failed cell, a false gate, or truncation. A defect found on the instance is fixed on the Mac and the scorer counts the failed attempt (P-11). Cap **1 GPU-hour from Ready**. Estimated run time: `p1` 18 trains of 300 ms plus allocation and round trips, about 6 s; `p2` writes 159 MB of sources, then 12 passes of 128 records at about 0.2 to 1.2 ms each, about 2 s; controls about 1 s each; under 15 s. The setup (unpack, `setup`, `doctor`, `smoke`) is about 5 minutes.

## 6. Needs Hamed

One RTX 5090 of the fast population and his go to run `session.sh pinroute6` on it. He terminates it after the files are home.
