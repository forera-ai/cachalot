# CUDA lane follow-up M: results (why an allocate-then-copy pinned buffer is slow, and the huge-page, route and classification answers in the fast pattern)

Written 2026-10-28 against runtime 0.68.0. Record: `docs/CUDA-PIN-ORDER-RECORD.md` (M-1 to M-9, committed first in debaddb, 2026-10-21, before any code or run). Scored by `benchmarks/cuda/pin_route_order_score.py`; the scorer's output is `docs/cuda-results/pin-order-2026-10-28.json`. Raw run directories are kept outside the repository under `~/cachalot-private/cuda-results/m1/` (archive sha256 `c30a402f…5344e`, the same on the instance and on the Mac). Labels follow plan section 0. Follow-up N, which ran on the same instance afterwards, is in `docs/CUDA-PIN-SETTLE-RESULTS.md`.

## 1. What ran

**Instance.** `calm-tongue-6106`, one RTX 5090 32 GB (the Korean listing: 7 CPUs, 23 GB RAM), driver 580.159.03, the NVIDIA module loaded (`lsmod` showed 3 modules), PCIe Gen 5 x16 at rest, Ready at about 18:49 UTC on 2026-10-28. The agent ran over SSH at Hamed's instruction, with a private ssh config outside the repository, and used no sudo.

**Code.** Pack `cuda-scaffold-dcadfa1.tar.gz` (REVISION `sha=dcadfa1 dirty=0`, sha256 `9a03e846…aa600b`); the multi-lead trace `n3d-trace.npz` hashed to `b68aefa4…e3f488`, the value in `n3d-expected.json`. `check`, `setup` (torch 2.7.1+cu128, arch gate true for compute capability 12.0), `doctor` and `smoke` (106 rows, 2.0 s) were clean.

**Runs.** `session.sh pinroute3` from 18:55:09 to 18:56:03 UTC: nine fresh processes (`m0`, `k4`, `l1`, `m1`, `m4`, `m6`, `m7`, `m8`, `m9`), each status complete with every gate true (round trip, sanity, no failed cell), 0.8 to 1.3 s each, 9.9 s in all. Then `session.sh n3ksmoke` (22.2 s): status complete, every gate true, including `guard_sees_planted`.

## 2. Scores of M-1 to M-9

"In interval", "outside interval" and "past falsifier" are the scorer's words; the explanation column is mine and a hypothesis unless stated. Every buffer is 9.95 MB; medians are of the buffers' p50 (20 copies after 5 warm-up).

| id | result (measured) | score | note |
|---|---|---|---|
| M-1 | `k4` (interleaved) 24 of 24 slow, 0.359 ms, 27.7 GB/s; `l1` (batch-first) 0 of 24 slow, 0.182 ms, 54.6 GB/s | in interval | the controls separate, so M-2 to M-6 are testable |
| M-2 | `m1`: the 12 buffers made before the one copy and the 12 made after it are all fast (slow share 0 and 0; 0.182 ms) | **past falsifier** | H-after-copy is falsified: a copy that ran before an allocation does not make it slow |
| M-3 | `m4`: one copy of a separate 1 MiB pinned buffer first, then 24 buffers made and timed: slow share 0, 0.182 ms | **past falsifier** | the same, for a copy of another buffer |
| M-4 | `m6`: 24 private mappings with `MADV_HUGEPAGE`, all made first: 24 of 24 huge-page verified, slow share 0, 54.6 GB/s (within 0.1 % of `l1`) | in interval | huge pages do not hurt in the fast pattern |
| M-5 | `m7` (`MADV_NOHUGEPAGE`): slow share 0, 54.5 GB/s (0.2 % below `m6`) | in interval | page size does not matter in the fast pattern; B2's falsification stands and its pattern caveat is closed |
| M-6 | `m8` (ordinary allocator memory, registered with flag 0, made first): slow share 0, 0.209 ms, 47.7 GB/s, 12.7 % below `l1` | in interval (the interval was 15 %) | registered ordinary memory is fast in this pattern but not quite `cudaHostAlloc`'s rate |
| M-7 | `m0` and `m9`: class `fast` (0 of 12 slow), plateau 55.16 and 55.17 GB/s | in interval | the committed rule reads the instance correctly on the batch-first pattern |
| M-8 | arena source records 0.1845 ms (`n3ksmoke`) against `m0`'s 0.1821: 1.3 % apart | in interval | the rule agrees with the arena |
| M-9 | every gate true, no failed cell, the nine processes 9.9 s, the smoke clean | in interval | |

**Seven held, two falsified (M-2, M-3).**

## 3. What it means

* **The slow rate is not caused by an earlier copy.** In every cell where buffers were all made before any was timed, all were fast, whether or not a copy ran in between (`m1`, `m4`). What separates `k4` from `l1` is what happens between one buffer's allocation and its own timing.
* **The rates are two modes, not a spread.** Within a cell the buffers agree to 0.001 ms: `k4`'s 24 buffers are all 0.358 to 0.359 ms and `l1`'s all 0.182 to 0.183; the slow mode is 1.97 times the fast one.
* **Huge pages, page size and route in the fast pattern:** no effect from huge pages or page size; registered ordinary memory is about 13 % slower than `cudaHostAlloc`. The 14 to 26 % difference between the routes measured by session K (DC-3) was a property of the slow pattern.
* **Classification (DC-6):** on the batch-first pattern the committed rule gives `fast` at both ends of the session with a 55.2 GB/s plateau, and agrees with the arena. It is adopted in that pattern for this card.
* The per-buffer data suggested the question follow-up N then asked: in `k4` the buffer has just been written by the CPU (`pinned()` fills a pageable tensor and `pin_memory()` copies it into the pinned buffer) when it is timed; in `l1` it has not.

## 4. Limits

One instance and one card. The population is the fast one by the committed rule; the slow-population instances of 0.66.52 and of the 0.66.60 audit were not re-measured. 24 buffers a cell. Nothing here is about a model, a default or the runtime on the Mac.
