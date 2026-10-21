# CUDA lane session K: does the pinning route set the host-to-device copy class (DC-3), is the issue loop free of hidden synchronisation (DC-5), and can the instance's copy population be classified before a run (DC-6)? (experiment record; predictions before any build)

Written 2026-10-21 against runtime 0.66.57, **before any code for this session is written or run and before any instance exists**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file (`docs/CUDA-PIN-ROUTE-RESULTS.md`). A prediction edited after its run is void (plan section 12). Labels follow plan section 0. It takes the three cheapest items of `docs/DEEPINFER-LEVER-LANE.md` section 5 (DC-3, DC-5, DC-6, ranked there as "minutes, no new harness") and Hamed's 2026-10-21 request to prepare them for a further GPU type, Mac only.

## 0. Which card (decided from the provider's list of 2026-10-21, not from this record's wishes)

Hamed's screenshot of the provider's list shows, with availability "few" or "some": L40S 48 GB (Taiwan), V100 SXM3 32 GB (two listings, Texas), V100 SXM2 16 GB (New York) and RTX 5090 32 GB (Korea). **No A100 or RTX 4090 is listed**, so the "third type" the previous prompt named does not exist this week; none of the listed cards is new silicon (the V100 SXM2 is the V100 of session V1 with half the memory and another host). The three items below are about the host-to-device copy rate, and only the RTX 5090 has shown the slow and fast copy classes they address (the L40S gave one class, the V100 a flat curve), and it is the only listed card with a PCIe 5.0 x16 link. The session is therefore prepared for **a new RTX 5090 instance** (the listed Korean one: 24 GB RAM, 800 GB disk, as B1, B2, N2, N3 and the later sessions used). The fallback if Hamed prefers a different card is the V100 SXM2 16 GB; DC-5 would then need a smaller arena (the `p8` group), DC-3 and DC-6 need none. Nothing else in this record depends on the card except the numbers of section 4.

## 1. What this asks, and what it cannot claim

**DC-3, the pinning route.** B1 and B2 found that the rate of a pinned 9.95 MB buffer is a property of the buffer, 0.18 ms for most buffers on some instances and 0.31 to 0.36 ms on others, that every buffer of a fresh process was slow on the B1 and B2 instances, and falsified huge-page backing as the cause (private mapping, 16 of 16 huge-page-backed, still 0.357 ms). All those sessions pinned with `cudaHostAlloc` (through torch's `pin_memory`) or with `cudaHostRegister` on a self-made 2 MiB-aligned mapping **with the portable flag**. DeepInfer pins memory that already exists with `cudaHostRegister(ptr, nbytes, 0)` (flag 0, no portable) on ordinary buffers. **H-route:** the pinning route (allocated, registered with the portable flag, registered with flag 0, registered ordinary allocator memory) does not move the class of the copy; the rate is set below the process. **H-route-moves** (the rival): at least one route yields the fast class on an instance whose `cudaHostAlloc` buffers are slow.

**DC-5, the issue loop's synchronisation.** The N3 stage B result "the host issue loop is about 1.35 ms a token, about 19 microseconds a copy of torch dispatch" is a timing. It does not say the loop contains no blocking call. DeepInfer's guard (`set_sync_debug_mode("error")` around launch and sampling) turns it into a statement. **H-clean:** the part of the row loop between the start of a row's bookkeeping and the router synchronisation (copy issue, event records, waits, stand-in launches) contains no call that torch classifies as a synchronising CUDA operation, at levels 3 and 4. **H-dirty** (the rival): it contains at least one (for instance an `.item()`, a host read or an implicit copy from pageable memory).

**DC-6, classifying the population before the run.** The lane read the instance's class after the fact from the median source copy (0.194 against 0.349 ms). This session commits the rule beforehand (section 3) and applies it at the start and at the end, and adds one observation the earlier sessions never recorded: the PCIe link generation and width reported by `nvidia-smi` during the copies. The slow class's rate (27.7 GB/s) equals a PCIe 4.0 x16 rate and the fast class's (54.6 GB/s) a PCIe 5.0 x16 rate on a PCIe 5 card; if the link is retrained to a lower generation under some conditions, that is a host-side cause of exactly the observed kind. **H-link** (hypothesis, never tested): the slow class coincides with a lower reported link generation during the copies. The sampler is read-only.

**It cannot claim** anything about real expert kernels, about DeepSeek, MiniMax or GLM on NVIDIA hardware, another card, or why a buffer is slow beyond the three hypotheses above; one instance of one card. DC-5's guard sees the calls torch routes through its synchronisation hook; a driver-level block inside an asynchronous call (a full launch queue, for example) is not visible to it.

## 2. Baseline (recorded)

* RTX 5090 pinned 9.95 MB H2D: fast class 0.182 to 0.194 ms (52 to 54.6 GB/s), slow class 0.349 to 0.362 ms (27.4 to 28.5 GB/s); plateau at 64 MiB and above 52 GB/s (S1, N2, B1, B2, instance sections of the later results). The slow share of a set of buffers is 0 to 100 % depending on instance, process and allocation context; fresh-process `cudaHostAlloc` was 48 of 48 slow on the B1 and B2 instances.
* B2: private mapping, `MADV_HUGEPAGE`, registered with the portable flag: 16 of 16 huge-page-verified and every buffer slow (0.356 to 0.362 ms); a shared mapping, no huge pages, was slow as well (0.255 to 0.32 ms in B1's small-page cells).
* L40S (PCIe 4.0 x16): source copy 0.378 ms (26.3 GB/s), one class; V100: 0.84 ms (11.8 GB/s), flat from 3 MiB.
* N3 stage B (RTX 5090, fast population): the unguarded issue loop is about 1.35 ms a token at level 4 with the oracle arm and 19 microseconds a copy; the closure and `rowsum` gates hold at levels 3 and 4.
* The source-copy qualification inside `overlap_arena.py` already reports a within-run slow share (records at or above 1.5 times the fastest record's copy time); no earlier rule was committed before a run.

## 3. Design (fixed here)

### 3.1 The committed classification rule (DC-6)

Canonical copy: a **fresh process** allocates twelve 9.95 MB pinned buffers with `cudaHostAlloc` (torch `pin_memory`), each timed alone (CUDA events, one stream, 20 repetitions after 5 warm-up, p50 per buffer, the GPU idle between), and three 64 MiB pinned buffers timed the same way. `plateau_gbps` = the highest p50 rate of the three 64 MiB buffers. A 9.95 MB buffer is **slow** if its rate is below **0.75 x plateau_gbps** (the time above 1.333 x the plateau-scaled time). The **instance class** is `fast` if at most 1 of 12 buffers is slow, `slow` if at least 11 of 12, `mixed` otherwise. The earlier absolute rule (slow = p50 at or above 0.28 ms) is reported beside it, never instead. Canonical stream kernel: one device-to-device copy of 256 MiB, p50 of 20 after 5 warm-up, in GB/s; a later reading differs from an earlier one significantly if it differs by more than 15 % **and** by more than 3 microseconds (DeepInfer's kernel-regression rule, DM-4, applied to a card with no committed baseline: the start reading is the baseline). The classification is run **twice**, cell `k0` at the start of the session and cell `k5` at its end, each in its own fresh process.

### 3.2 DC-3 cells (each its own fresh process; `benchmarks/cuda/pin_route.py`)

All buffers 9.95 MB, 24 per cell, each timed alone as above, the byte round trip checked after each buffer's own copy.

| id | cell | route |
|---|---|---|
| k1 | `register_portable` | self-made 2 MiB-aligned private anonymous mapping, no advice, registered with `cudaHostRegisterPortable` (flag 1: what B1 and B2 used) |
| k2 | `register_flag0` | the same mapping, registered with flag 0 (DeepInfer's call) |
| k3 | `ordinary_flag0` | memory from the ordinary allocator (`torch.empty` on the CPU, 9.95 MB, not 2 MiB aligned), registered with flag 0: "memory that already exists" |
| k4 | `alloc_control` | `cudaHostAlloc` through torch `pin_memory`, the route of S1 to N3 (the control) |

`k0` and `k5` are the classification cells of section 3.1 (their twelve `cudaHostAlloc` buffers are also a control). smaps is read after the content is made, after registration and after the timed copies, as in B2, so the page backing of each registered route is on record. A registration that fails is reported `unavailable` with the error, never as a number.

### 3.3 DC-6 link observation

In cells `k0`, `k4` and `k5` a read-only sampler (`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader -lms 100`) runs in a separate process for the cell's duration and its samples are kept (generation, width, count). If `nvidia-smi` cannot report them the rows are `unavailable`. The sampler's own cost is part of the three cells and is not removed; the other cells run without it, so the comparison of `k4` against `k1` to `k3` includes it as a difference of design (stated, and not a hidden confound: `k0` and `k5` run both with it).

### 3.4 DC-5 cells (profile `k-full`, `overlap_arena.py`)

The N3 stage cells on the multi-lead trace (`n3d-trace.npz`, `n3d-expected.json`, unchanged), 17 % resident, full copy, 400 decode tokens, arms `N`, `D`, `P0L`, `N` and the arm hash and count gates as before. A **guarded** cell wraps, in every row, the part of the row loop from the start of the row's bookkeeping to just before the router synchronisation (`sync_compute`) in `torch.cuda.set_sync_debug_mode("warn")`, with the warnings captured; the router synchronisation, the timing reads and the token flush stay outside (they synchronise by design). "warn" is used instead of "error" so that a violation cannot leave a half-issued row and fail the arm's counts. Entries (one group, one calibration per C):

| entry | level | C (ms) | guard | arms |
|---|---|---|---|---|
| 1 | 3 | 1.0 | no (rows recorded) | N, D, P0L, N |
| 2 | 3 | 1.0 | yes | N, D, P0L, N |
| 3 | 4 | 1.0 | no | N, D, P0L, N |
| 4 | 4 | 1.0 | yes | N, D, P0L, N |
| 5 | 4 | 0.3 | no | N, D, P0L, N |
| 6 | 4 | 0.3 | yes | N, D, P0L, N |
| 7 | 4 | 0.3 | yes + a **planted** synchronisation | N, P0L, N |

Entry 7 calls `.item()` on a device scalar once per token inside the guarded window: it must be counted (the guard's own test). 27 cells a run, one run (no replicate: the question is a count, and the unguarded entries give the timing). Smoke profile `k-smoke`: 40 tokens, entries 2 and 7 only (7 cells).

### 3.5 Gates and order

DC-3 and DC-6: the byte round trip after every buffer; no rate above the PCIe sanity bound; the fresh process per cell; an unavailable registration is a result. DC-5: all N3 stage gates (event hash, determinism, counts, chain2, readback, calibration, drift, closure, sanity, floor, `rowsum` at level 3), and the guard's own gate: entry 7 must count at least one violation per guarded token (otherwise the guard is blind and K-8 is void). Order on the instance: `check`, `setup`, `doctor`, `smoke` (all clean), **`session.sh pinroute`** (k0, k1, k2, k3, k4, k5 in that order, stopping at the first failure), then `n3ksmoke`, then `n3kfull`. The classification cells bracket the DC-3 cells so that drift inside the session shows.

### 3.6 Offline analysis

`benchmarks/cuda/pin_route_score.py` (new) scores K-1 to K-11 from the run directories and prints them beside the registered intervals; rehearsed on planted runs before any instance run.

## 4. Predictions (registered before any build)

All **predicted**; probabilities are my own. The instance's population is unknown, so predictions that depend on it are conditional and say so.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| K-1 | classification stability | the instance class of `k0` equals that of `k5` (probability 0.80) | different classes |
| K-2 | plateau | `plateau_gbps` of `k0` between 48 and 57 GB/s (probability 0.70; recorded 52 to 54.6) | below 40 or above 62 |
| K-3 | the committed rule against the absolute rule | the two rules give the same slow or fast label for at least 90 % of the 12 + 12 buffers of `k0` and `k5` (probability 0.80) | below 75 %: the relative rule is not a replacement for the 0.28 ms rule |
| K-4 | stream baseline | the 256 MiB device copy of `k0` lies between 1,300 and 1,700 GB/s (probability 0.70; recorded 1,517) and `k5` differs by less than 15 % or less than 3 microseconds (probability 0.90) | outside either |
| K-5 | route does not move the class (H-route), the headline | the slow share of each of k1, k2 and k3 is within 0.25 of k4's (probability 0.65) | any route more than 0.45 away from k4 (the route moves the class: H-route-moves, and it becomes the lever) |
| K-6 | flag 0 against portable | the median rates of k1 and k2 differ by less than 5 % (probability 0.80) | more than 15 % |
| K-7 | ordinary memory | k3's median rate is within 10 % of k4's, or k3 is `unavailable` (probability 0.70) | k3's median rate more than 30 % above k4's |
| K-8 | DC-5, H-clean | the guarded windows count **zero** synchronising calls at levels 3 and 4 in every guarded cell (probability 0.70) | at least one counted in a guarded cell of entries 2, 4 or 6: H-dirty, and the warnings' source lines are the finding |
| K-9 | guard cost | the median `host_ms` per token of `P0L` guarded and unguarded differ by less than 10 % at level 4, C = 1.0 and 0.3 (probability 0.60) | more than 25 %: guarded cells are never used for timing, already the design |
| K-10 | the guard sees a sync (control) | entry 7 counts at least one violation in at least 95 % of its guarded tokens, and the counted calls are the planted ones (probability 0.90) | fewer than 50 %: the guard is blind on this torch build and K-8 is void |
| K-11 | link generation (H-link) | the sampled `pcie.link.gen.current` equals its maximum in every sample of `k0`, `k4` and `k5` (probability 0.55; the 5090 is PCIe 5.0 and the link may idle lower between copies, hence not higher) | a sample below the maximum **in a cell whose slow share is at least 0.5 while a cell with slow share under 0.5 has none**: H-link supported (reported as such, not as a failure of the run) |
| K-12 | completion | all gates true, no failed cell, GPU time of `pinroute` under 600 s and of `n3ksmoke` + `n3kfull` under 900 s (probability 0.65) | any gate false, a failed cell, or truncation |

## 5. Decision rules (fixed here)

* **H-route is reported supported** if K-5 holds (no route more than 0.25 from the control) with k4 itself not `unavailable`; if k4's slow share is under 0.2 the instance has no slow class to move and K-5 is **not testable** (reported so; the session's DC-3 answer is then "no slow class on this instance", and a second instance is the open item). **H-route-moves** is reported if any route is more than 0.45 from k4 in slow share with the other cells' variation smaller than that difference.
* **The guard result** (K-8) is reported only if K-10 holds. A counted violation is reported with its warning text and source line; no claim is made that it costs time.
* **H-link** is reported supported only under K-11's falsifier wording; a link that reads at its maximum in every sample does not support it and does not exclude it (samples at 10 Hz may miss a transition).
* **The committed rule is adopted** for the lane's sessions if K-1 and K-3 hold; if K-3 is falsified the absolute rule stays and the relative rule is reported as a second label only.
* Nothing here predicts a change in `src/` or a default. These are measurements of a host-side path and of a harness.

## 6. Stop rules and cost

Stop and bring the files home on a failed cell, a false gate, truncation, or 80 % of the cap before the first cell. A defect found on the instance is fixed on the Mac. Session cap **1 GPU-hour** from Ready (setup, wheel download and the arena's 24 GiB allocation take most of the first quarter hour). Estimated GPU run time: `pinroute` 3 to 6 minutes (six fresh processes, each with a CUDA context start), `n3ksmoke` 1 to 2, `n3kfull` 6 to 9 (27 cells), plus the trace copy (39.8 MB over a slow link). Every run passes an explicit `--max-seconds`.

## 7. Needs Hamed

A go and one **RTX 5090** instance (the Korean listing, as before; the runbook says when to create it and the instance takes about seven minutes to initialise), then the usual SSH arrangement for that one instance, no sudo. If an instance has a GPU that does not initialise, a replacement is created before the faulty one is terminated. Housekeeping from earlier sessions stands: terminate `aquatic-existence-2628`, `flat-battle-1078`, `imperfect-duck-7094`, `elegant-things-5691` if still up, enter the billed hours, delete the ssh configs and known_hosts of `j1`, `i1`, `h2`, `h1`, `g1`, `f1`, `f2`, `f3`.

## 8. Clarifications written after the instruments were built and before any instance run

Three terms of sections 3 and 4 were left undefined and are fixed here, with no prediction, interval or falsifier changed (no run exists).

* **K-4, the unit.** The stream baseline's GB/s counts read plus write, `2 x bytes / p50`, as session S1's 1,517 GB/s did (its 512 MiB copy); the interval 1,300 to 1,700 GB/s and the 15 % and 3 microsecond rule apply to that number. The copy is 256 MiB.
* **K-5 and K-7, "slow share" of a route cell.** The route cells have no plateau of their own, so a route buffer is slow by the committed rule against the **higher of the two plateau rates of `k0` and `k5`** (the session's measured link plateau). The absolute 0.28 ms rule's share is reported beside it and is not scored.
* **K-3, the buffers pooled.** The agreement is over the 24 buffers of `k0` and `k5` together (12 each).
* **Field names.** The instance class is the row `instance_class_code` (0 fast, 1 mixed, 2 slow). Scoring is by field name, never by column position. The scorer is `benchmarks/cuda/pin_route_score.py`.
