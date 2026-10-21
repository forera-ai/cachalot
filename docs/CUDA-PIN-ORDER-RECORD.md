# CUDA lane follow-up M: why is a pinned buffer allocated and then copied slow, and does the fast pattern change the huge-page, route and classification answers? (experiment record; predictions before any build)

Written 2026-10-21 against runtime 0.66.59, **before any code for this session is written, before any instance exists, and without a run**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void. Labels follow plan section 0. Hamed asked for it on 2026-10-21 ("go for all 4 in order"; item 2 of the proposal: register the next cell and build it on the Mac). Nothing here starts an instance: that needs his yes and a new instance (an RTX 5090, as before; the list of 2026-10-21 has no other card the questions fit).

## 1. Baseline (recorded)

* Follow-up L (`docs/CUDA-PIN-ROUTE-RESULTS.md` section 4, one instance `hellish-cry-4960`): 24 pinned 9.95 MB buffers, each timed right after allocation: 0.362 ms (27.5 GB/s), slow share 1.0; all 24 allocated first, then timed: 0.192 ms (51.8 GB/s), slow share 0; the destination (a slot of a 24 GiB device buffer) and a non-default stream did nothing.
* The audit of 0.66.60 (`docs/CUDA-PIN-PATTERN-AUDIT.md`): batch allocation is not sufficient everywhere: the slow-population instance's batch-allocated arena read 0.349 ms, and B1's batch-first curve read 17 to 38 % slow at 9.95 MB.
* Interleaved results still to be re-read in the batch-first pattern: B2's huge-page falsification (16 of 16 huge-page-backed registered buffers slow, interleaved), the registered routes of session K (0.27 to 0.31 ms against 0.362, interleaved), and the DC-6 classification rule, which reads any instance slow in the interleaved pattern (12 of 12) while the arena in the same session read it fast (0.194 ms).

**Working hypotheses.** **H-after-copy:** a pinned buffer allocated after any host-to-device copy has run in the process is slow; one allocated before is fast. **H-while-timing** (the rival): the slowness comes from allocating while earlier buffers are being timed (the pattern itself), not from the existence of an earlier copy. **H-hugepage-fast:** in the fast pattern, huge-page backing makes no difference (the rate is set by the pattern, as in L). **H-route-fast:** registered and ordinary memory match `cudaHostAlloc` in the fast pattern.

## 2. Cells (each its own fresh process; `benchmarks/cuda/pin_route.py`, groups `m0` to `m9` new, `k4` and `l1` reused as the two controls)

All buffers 9.95 MB, 24 a cell (the classification cells 12 plus three of 64 MiB), each timed alone (CUDA events, 20 repetitions after 5 warm-up, p50), the byte round trip checked after each buffer's own copy.

| id | cell | what it does |
|---|---|---|
| m0, m9 | `classify_batch_start`, `classify_batch_end` | the committed classification rule (section 3.1 of `docs/CUDA-PIN-ROUTE-RECORD.md`, unchanged: slow below 0.75 x the 64 MiB plateau; fast with at most 1 slow of 12) on the **batch-first** pattern: the 12 record buffers and the 3 plateau buffers all allocated before any is timed; the device copy and the link sampler as in `k0` |
| k4 | control, interleaved | as in the first run |
| l1 | control, batch-first | as in follow-up L |
| m1 | `half_copy_half` | allocate 12 buffers; run **one** host-to-device copy of the first of them; allocate 12 more; then time all 24 in allocation order |
| m4 | `copy_first` | run one host-to-device copy of a separate 1 MiB pinned buffer; then allocate all 24 buffers; then time them |
| m6 | `private_thp_batch` | 24 private 2 MiB-aligned mappings, `MADV_HUGEPAGE`, registered (flag 1, as B2), **all made before any is timed**; smaps at three points |
| m7 | `private_nothp_batch` | the same with `MADV_NOHUGEPAGE` (the control for m6) |
| m8 | `ordinary_batch` | 24 ordinary allocator buffers (`torch.empty`), registered with flag 0, all made before timing |

Session order (`session.sh pinroute3`, output directory `benchmarks/results/cuda-order`): `m0`, `k4`, `l1`, `m1`, `m4`, `m6`, `m7`, `m8`, `m9`, nine fresh processes; then `n3ksmoke` (the arena, for the cross-check of M-8). Gates: the byte round trip after every buffer; no copy rate above the PCIe sanity bound; a failed registration is `unavailable`, never a number.

## 3. Predictions (registered before any code)

All **predicted**; probabilities mine. "Slow" is the absolute rule (p50 at or above 0.28 ms); "fast" at or below 0.25 ms. The instance is a new draw and may not be the same population as `hellish-cry-4960`; M-1 says so first and the others are scored **only if M-1 holds** (otherwise they are reported as not testable on that instance).

| id | quantity | prediction | falsifier |
|---|---|---|---|
| M-1 | the controls separate on the new instance | `l1` slow share at most 0.2 and `k4` slow share at least 0.9 (probability 0.55) | either fails: the instance is another population, M-2 to M-6 are not testable |
| M-2 | H-after-copy, the half-then-copy test | in `m1` the first 12 buffers have a slow share at most 0.2 and the last 12 at least 0.8 (probability 0.35); the rivals (all 24 fast, 0.35; all 24 slow, 0.15; other, 0.15) | the last 12 have a slow share at most 0.2 (a copy running before an allocation does not slow it) |
| M-3 | H-after-copy, any earlier copy | in `m4` the slow share is at least 0.8 (probability 0.30) | at most 0.2 (an earlier copy of another buffer does not slow later allocations) |
| M-4 | huge pages in the fast pattern | `m6` has at least 14 of 24 buffers huge-page-verified (probability 0.60) and its median rate is within 10 % of `l1`'s (probability 0.55) | a verified buffer share with slow share at least 0.5: huge pages hurt in the fast pattern |
| M-5 | page size does not matter in the fast pattern | `m7` has a slow share at most 0.2 and a median rate within 10 % of `m6`'s (probability 0.55) | `m7` slow share at least 0.5 or more than 30 % below `m6` |
| M-6 | the route in the fast pattern | `m8`'s median rate is within 15 % of `l1`'s (probability 0.50) | more than 30 % below `l1` |
| M-7 | the re-specified classification | `m0` and `m9` both read class `fast` (code 0, probability 0.60 given M-1 holds) and have the same class (0.85); the 64 MiB plateau of `m0` is between 48 and 57 GB/s (0.60) | `m0` reads `slow` while the arena (M-8) reads fast: the rule is wrong even in the fast pattern; plateau below 40 GB/s |
| M-8 | the rule agrees with the arena | the median source-record copy of `n3ksmoke` is within 10 % of `m0`'s 9.95 MB median (probability 0.65) | more than 25 % apart |
| M-9 | completion | all gates true, no failed cell, the nine processes under 300 s of GPU time, the smoke clean (probability 0.80) | any gate false, a failed cell or truncation |

## 4. What each outcome means

* **M-2 holds (and M-3 holds):** a pinned allocation made after a host-to-device copy has run is slow; the cure is to allocate the whole pinned host tier before the first copy (a one-line rule for any CUDA expert-streaming design). **M-2 falsified with all 24 fast:** a copy before an allocation does not matter; interleaving with *timing* (or the allocator's behaviour while earlier buffers are in use) is the variable, and a further cell (allocate while a copy is in flight) is the next record.
* **M-4 and M-5 hold:** huge-page backing does not matter in either pattern: B2's falsification stands and its pattern caveat is closed. **M-4 or M-5 falsified:** huge pages matter in the fast pattern and the earlier falsification was an artifact.
* **M-6 holds:** registration route is irrelevant in the fast pattern, so DC-3's interleaved difference (14 to 26 %) is a property of the slow pattern only.
* **M-7 and M-8 hold:** the classification rule is adopted on the batch-first pattern, and the lane's fast and slow populations are both readable before a run.
* Nothing here is a statement about another card, a model or a default.

## 5. Stop rules and cost

Stop and bring the files home on a failed cell, a false gate, truncation, or 80 % of the cap before the first cell. A defect found on the instance is fixed on the Mac. Session cap **1 GPU-hour** from Ready (setup about a quarter of it). Estimated GPU run time: nine processes of 1 to 2 s each and the smoke 25 s: under 2 minutes; the rest is setup, the trace copy and results.

## 6. Needs Hamed

A go and an RTX 5090 instance (the Korean listing); `nvidia-smi` and `lsmod | grep nvidia` first, and a replacement created before a faulty instance is terminated. The agent runs over SSH only for that one instance and never runs sudo.
