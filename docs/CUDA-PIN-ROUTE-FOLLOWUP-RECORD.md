# CUDA lane session K, follow-up L: what makes the arena process's pinned copies fast when a fresh process's are slow? (experiment record; predictions before any build)

Written 2026-10-21 against runtime 0.66.58, **after** the first run of session K (`docs/CUDA-PIN-ROUTE-RECORD.md`, run on the RTX 5090 instance `hellish-cry-4960`, results home under `~/cachalot-private/cuda-results/k1/`, not yet written up) and **before any code for this follow-up is written or run**. Charter section 7.2 fields 1 to 3 are fixed here; this file is committed first and results go in the session's results file. A prediction edited after its run is void. Hamed asked for it on 2026-10-21 ("Register a short follow-up first (reversed order, arena-first context, predictions committed before the run), keep the instance up, then write results"), and the instance stays up for it.

## 1. What the first run left open (recorded, from the run's rows, not yet scored in a results file)

* In fresh processes, the 9.95 MB `cudaHostAlloc` (torch `pin_memory`) buffers of cells `k0`, `k4` and `k5` all copied at 0.362 ms (27.5 GB/s; 48 of 48 buffers within 0.001 ms of each other), and the 64 MiB buffers at 1.6 to 2.1 ms (32 to 41 GB/s, below the 52 GB/s plateau of the earlier instances).
* In the arena process of the same session (`n3ksmoke`), the 64 pinned source records of 9.95 MB, qualified by the harness itself, copied at **0.194 ms** (the median source copy), 0 of them slow. The arena process differs from `alloc_order` in four ways at once: a 24 GiB device buffer is allocated **first** and the copy destination is a slice of it; the 64 records are all pinned **before** any is timed; the copies are timed on a **non-default stream**; and the content comes from a seeded generator.
* The three registered routes (`k1`, `k2`, `k3`) ran at 0.27 to 0.31 ms: faster than `cudaHostAlloc` (0.362) in the same context, slower than the arena's 0.194.
* Older sessions: B1 (fresh process, interleaved allocation and timing): 48 of 48 slow; B1's curve process (all `K` buffers allocated, then each timed): 17 to 31 % slow; N2 (arena): 3 or 4 of 64 slow.
* The cells of the first run ran in a fixed order (`k0`, `k1`, `k2`, `k3`, `k4`, `k5`).

**Working hypotheses** (not exclusive): **H-batch**, allocating all the pinned buffers before timing any makes them fast; **H-dest**, a destination inside a large device allocation (or its alignment) makes them fast; **H-stream**, timing on a non-default stream makes them fast; **H-order**, the first run's route differences are an order or time effect. The follow-up separates them with one cell each and a cell that has all three, plus a reversed-order repeat of the routes.

## 2. Cells (each its own fresh process; `benchmarks/cuda/pin_route.py`, groups `l1` to `l4` new, `k1` to `k4` reused)

All buffers 9.95 MB, 24 of them, `cudaHostAlloc` through torch `pin_memory` (as `k4`), timed alone (CUDA events, 20 repetitions after 5 warm-up, p50 per buffer), the byte round trip checked after each buffer's own copy.

| id | cell | allocation | destination | stream |
|---|---|---|---|---|
| l1 | `batch_first` | all 24 first, then timed | a 9.95 MB device buffer | default |
| l2 | `arena_dest` | interleaved (allocate one, time it) | slot 0 of a 24 GiB device buffer allocated first | default |
| l3 | `side_stream` | interleaved | a 9.95 MB device buffer | a non-default `torch.cuda.Stream` |
| l4 | `arena_full` | all 24 first, then timed | slot 0 of a 24 GiB device buffer allocated first | a non-default stream |

(`k4`, the control, is interleaved, a 9.95 MB destination, the default stream.) The session order is **reversed and bracketed**: `k4`, `k3`, `k2`, `k1`, `l1`, `l2`, `l3`, `l4`, `k4`: the routes in the reverse of the first run's order, the four new cells, and the control again at the end. Output goes to its own directory (`benchmarks/results/cuda-followup` on the host) so the first run's directories are untouched.

## 3. Predictions (registered before any code)

All **predicted**; probabilities mine. "Slow" is the absolute rule (p50 at or above 0.28 ms; this instance's slow class sits at 0.362 ms); "fast" is at or below 0.25 ms. The slow share is over the 24 buffers of a cell.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| L-1 | the arena-like cell reproduces the arena (the positive control) | `l4` has a slow share at most 0.2 and a median under 0.25 ms (probability 0.75) | slow share at least 0.9: the arena's fast copies are not reproduced by these four factors, and a fifth (the seeded content, the number of records, the allocator state) remains |
| L-2 | batch allocation alone (H-batch) | `l1` has a slow share at most 0.5 (probability 0.45) | slow share at least 0.9 (allocation order does not matter) |
| L-3 | destination alone (H-dest) | `l2` has a slow share at most 0.5 (probability 0.35) | slow share at least 0.9 |
| L-4 | stream alone (H-stream) | `l3` has a slow share at least 0.9 (probability 0.60, a non-default stream does not matter) | slow share at most 0.5 |
| L-5 | the cause is one factor | if `l4` is fast (L-1), at most one of `l1`, `l2`, `l3` is fast on its own (probability 0.55: the factors interact, or one dominates) | two or more of the three fast alone: more than one factor suffices |
| L-6 | order | the reversed-order routes reproduce the first run: the median rate of each of `k1`, `k2`, `k3` within 8 % of the first run's (33.8, 34.5, 31.3 GB/s) (probability 0.75) | any route more than 20 % off its first-run rate: the first run's route differences were an order or time effect |
| L-7 | the control at both ends | the first and the last `k4` have the same slow share (1.0 each) and medians within 1 % of 0.362 ms (probability 0.85) | either `k4` below 0.30 ms or the two medians more than 5 % apart: drift inside the session |
| L-8 | route ranking survives | in the reversed order the median rate ranks `k1` and `k2` above `k3` above `k4` as in the first run (probability 0.70) | `k4` not the slowest of the four |
| L-9 | completion | all gates true, no failed cell, GPU time of the nine processes under 300 s (probability 0.80) | any gate false, a failed cell or truncation |

## 4. What each outcome means

* **L-1 holds and L-5 holds:** the fast class in this harness is a property of a pair or a combination of the three factors; the production statement is "the context decides, not the instance": a pinned host tier should be allocated in the arena's pattern, and the finding explains the earlier fresh-process slow classes (B1, B2, the first run's `k0`).
* **L-1 holds with one of `l1` to `l3` fast alone:** that single factor is the lever (and the paper's "pinning route" question is secondary to it).
* **L-1 falsified:** none of these four factors explains the arena's 0.194 ms; the remaining differences (the seeded content, 64 records against 24, the allocator's state after the 24 GiB allocation) become the next record.
* **L-6 falsified:** the route result of the first run is an order effect and K-5 cannot be read as a route effect.
* Nothing here is a statement about another card, a model or a default; it is one instance on one day.

## 5. Stop rules and cost

Stop on a failed cell, a false gate or truncation. A defect found on the instance is fixed on the Mac and the instance stays up for the rerun. Estimated GPU run time 3 to 5 minutes for nine processes (the 24 GiB allocation adds a second or two to `l2` and `l4`). The instance of the first run is the one that stays up; the session cap of one GPU-hour from its Ready time is shared with the first run (about 25 minutes used when this record is committed).
