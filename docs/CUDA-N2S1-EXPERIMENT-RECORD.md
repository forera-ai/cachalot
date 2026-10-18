# CUDA lane N2 stage 1: design and experiment record (a real slot arena and real host-to-device copies driven by a routing trace)

Written 2026-10-18 against runtime 0.66.21, **before any stage 1 harness code and before any stage 1 GPU measurement**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file afterwards. A prediction edited after its run is void (plan section 12). Labels follow plan section 0. This is plan phase N2, **stage 1** of its three stages (plan section 6, N2): a replay with real copies, nothing else. It follows `docs/CUDA-S1-RESULTS.md`, `docs/CUDA-S2-RESULTS.md` and `docs/CUDA-V1-RESULTS.md` (measured copy costs on two cards) and the N0a policy records. Nothing here starts an instance; running it needs Hamed's go and a card choice.

## 1. What stage 1 can and cannot claim (plan section 6, N2 table)

Stage 1 drives a real VRAM slot arena with real pinned host-to-device copies of **dummy expert-sized buffers** from a routing trace (the recorded DeepSeek trace and, as a controlled contrast, the seeded synthetic generator). It can claim: the cache behaviour (hits, misses, evictions) and the **exposed demand-fetch copy time per decode token** of that pattern on that card, the resident-fraction curve of that time, and whether the CUDA replay's event sequence equals the Mac simulator's (CUDA-H5). It cannot claim: anything about real model routing beyond the recorded trace, real compute, hidden copy time (no compute stand-in runs; that is N3), prefetch, a token time, quality, or anything about DeepSeek V4.1 Flash on NVIDIA hardware. Every number carries the stage tag "N2 stage 1".

## 2. Baseline (recorded)

* **Copy costs** (pinned H2D, one expert record, 9.95 MB): RTX 5090 0.34 to 0.36 ms isolated, about 0.33 ms per copy when 4 are queued and 0.22 to 0.29 when 8 are queued (S2, two instances); V100 SXM3 over PCIe 3.0 x16 0.84 to 0.86 ms, flat for every queue length (V1). Event timer overhead 5 us (5090) and 16 us (V100). Router-style per-layer sync 7.5 us (5090) and 20 us (V100).
* **Mac** (recorded, not reproduced): a decode miss costs 2.0 ms at 48 GiB on the internal SSD (LEDGER DS-MISS-48), a token is about 80 ms plus 2.0 ms a miss.
* **Offline layer-miss profile** (new instrument `benchmarks/cuda/layer_miss_profile.py`, Mac only, `routing-trace-deepseek-long`, 1,972 decode tokens, 40 layers x 384 experts, LRU, tau 0, no prefetch; totals equal `cuda_offline_policy_table.replay`, so the replay is consistent with N0a):

| resident (slots, arena at 9.95 MB) | misses a token | layers with a miss a token | mean misses in a missing layer | share of misses in layers raising 3 or more |
|---|---:|---:|---:|---:|
| 8 % (1,229; 11.4 GiB) | 90.84 | 35.94 | 2.53 | 0.67 |
| 17 % (2,611; 24.2 GiB) | 51.85 | 28.35 | 1.83 | 0.41 |
| 34 % (5,222; 48.4 GiB) | 24.78 | 17.45 | 1.42 | 0.19 |

  Layers raising 1, 2, 3, 4, 5 or more misses at 17 %: 27,589 / 17,506 / 8,293 / 2,989 / 1,107 of 80,000 decode layer events (maximum 6, the top-k).
* **VRAM limit (derived):** 31.4 GiB is free on both cards after the framework context; a 34 % arena (48.4 GiB) and a 25 % arena (35.6 GiB) do not fit either card. Stage 1 therefore covers **8, 12 and 17 %** (11.4, 17.1 and 24.2 GiB), plus the generator contrast at 17 %; 25 and 34 % are not measurable here and are stated as such.
* **Host tier (derived):** 15,360 experts x 9.95 MB is 153 GB; the 5090 hosts had 24 GiB of RAM and the V100 host 83 GiB. The host tier is emulated by 64 distinct pinned source records (637 MB, above any CPU cache) indexed by expert id modulo 64: the copy cost does not depend on content, and the emulation is stated beside every result.

## 3. Design of the harness (to be built after this record is committed)

`benchmarks/cuda/replay_arena.py`, standalone, no import from `src/cachalot`, in the layering of `suite.py` (pure bookkeeping testable on the Mac; torch cells behind a bench object; a CPU rehearsal; a GPU smoke before any number counts, because the S2 d2d race showed the rehearsal cannot test stream ordering).

1. **Arena.** One uint8 device buffer of `slots x 9,950,000` bytes allocated once; slot `s` is the slice `[s x n, (s+1) x n)`. Fragmentation is zero by construction; `torch.cuda.memory_reserved` and `memory_allocated` are recorded.
2. **Bookkeeping.** An LRU map from (layer, expert) to slot, written independently of `simulate_policies.Store` (so equality is a test, not a tautology). The initial resident set is the recorded prefill's (the `cache_sim` prefill walk, populating the arena with real copies **before** timing starts).
3. **Decode walk, per token, per layer in trace order:** the misses of that layer are issued as queued `copy_` calls of one source record into the victim slot on one copy stream, one event recorded after them, and the host waits on the event (the router-style sync). The **exposed copy time of the layer** is the host monotonic time from the first issue to the event's completion, and the CUDA-event time over the same span is recorded beside it; the token's exposed copy time is the sum over its layers. Hits cost no copy; the host bookkeeping time is measured separately.
4. **Gates (a cell is void if any fails):** (a) the event sequence (hit or miss, victim and slot per request) hashes equal to the Mac simulator's for the same trace, slots and policy, **checked on the Mac before any GPU time** for every cell; (b) a sampled read-back of arena slots against their source records after every 2,000th copy; (c) two runs of one cell give the same event hash; (d) no copy rate above the link's sanity bound (a cache hit or a timing bug); (e) no failed cell, not truncated.
5. **Cells.** Trace `routing-trace-deepseek-long` (all 1,972 decode tokens) at 8, 12 and 17 % with LRU; the seeded generator at its default knobs (40 layers, 384 experts, top-6, skew 1.0, locality 0.3, 1,972 tokens) at 17 %; the pinned control cell of the S1 profile at the start and at the end.
6. **Estimate (not a measurement).** Sum of misses x copy cost: the three trace cells and the generator cell are about 210 copies-per-token-equivalents x 1,972 tokens, about 150 s on a 5090 and 350 s on a V100 of copy time, so a card session is about 40 minutes with setup, cap 1 GPU-hour from Ready.

## 4. Predictions (registered before any harness code)

All **predicted**. "Exposed copy time a token" is the sum over the token's layers of the host-measured time from the first copy issue to the layer's event, p50 over tokens, n = 1,972. Per-copy cost c below is the S2 and V1 measured cost, so a number outside its interval is a statement about how the real trace pattern differs from the S2 and V1 microbenchmarks.

| id | quantity | RTX 5090 | V100 (PCIe 3.0 x16) | falsifier |
|---|---|---|---|---|
| N2-P1 | the CUDA replay's event sequence equals the Mac simulator's (CUDA-H5), every cell | holds, probability 0.95 | holds | any differing event: a harness bug until proven otherwise |
| N2-P2 | exposed copy time a token, trace, 8 % / 12 % / 17 % | 28 to 36 / 21 to 27 / 15.5 to 21 ms | 70 to 84 / 53 to 65 / 39 to 49 ms | a value outside its interval |
| N2-P3 | per-token cost of one extra miss (least squares of token time on misses a token, within one cell) | 0.30 to 0.38 ms | 0.82 to 0.92 ms | outside, or a fit R-squared below 0.95 |
| N2-P4 | per-copy cost by layer-event size (mean copy time over events with 1 miss, and over events with 5 or more) | 1 miss: 0.34 to 0.38 ms; 5 or more: 0.25 to 0.34 (queued copies overlap their launch cost, S2) | both 0.82 to 0.92 (flat, V1) | 5090: the 5 or more figure at or above the 1-miss figure; V100: they differ by more than 8 % |
| N2-P5 (CUDA-H9 test, derived from measured curves) | extra exposed ms a token per GiB of resident experts between 8 % and 17 % | 0.8 to 1.2 ms per GiB | 2.0 to 2.9 ms per GiB | outside +-30 % of misses-slope x copy-cost (3.0 x 0.35 = 1.05 and 3.0 x 0.84 = 2.5 ms per GiB): the copy cost then depends on the pattern |
| N2-P6 | bytes H2D a token at 17 % | 516 MB (51.85 misses x 9.95 MB, identical on both cards) | the same | differs by more than 1 % (a replay bug) |
| N2-P7 | ratio of the two cards' exposed copy time at 17 % (V100 over 5090) | 2.3 to 2.7 | (same cell) | outside |
| N2-P8 | the generator at default knobs, 17 %, misses a token | more than the recorded trace's 51.85 (the generator is less concentrated), probability 0.7 | the same count (a property of the trace, not of the card) | at or below 51.85 |
| N2-P9 | allocator: reserved memory beyond the arena | under 300 MiB (zero fragmentation by construction), probability 0.85 | under 300 MiB | 300 MiB or more |
| N2-P10 | host bookkeeping time a token (miss decisions and issue, not the copy) | under 1.5 ms (5090 host, 7 vCPUs) | under 2.5 ms (V100 host, 4 vCPUs) | above those |
| N2-P11 | a whole cell finishes inside 6 minutes on the 5090, 12 on the V100 | probability 0.8 | probability 0.8 | truncated |
| N2-P12 | gates (b), (c), (d) hold | all hold, probability 0.85 | all hold | any false |

## 5. Stop rules and what each outcome means

* Stop and bring the files home on a failed cell, a false gate, a truncated run, or 80 % of the cap before the first trace cell. A defect found on the instance is fixed on the Mac. A stream-ordering defect found by the GPU smoke voids the cells before it.
* If N2-P1 fails, nothing else in this record counts until the harness bookkeeping is fixed: a replay that does not equal the simulator measures something else.
* If N2-P5 holds on both cards, CUDA-H9's restated reading ("the value of a resident GiB scales with the measured miss cost times the steeper small-fraction slope") is supported by a replay and not only derived; the plan's N2 budget advice then stands. If it fails, the first suspect is the layer-grouped issue pattern, not the cache.
* The result is a statement about pinned host-resident source records and a recorded DeepSeek routing trace: no network volume, no NVMe tier, no prefetch, no compute overlap. The hidden-copy question is N3.

## 6. Needs Hamed

(1) A go to build the harness (Mac only, with predictions already committed here): arena, bookkeeping, equality tests against the simulator, CPU rehearsal. (2) Later, a go and a card choice (5090, V100 or both) for the GPU smoke and the session; the agent does not provision. The housekeeping from S2 and V1 stands (log hours, delete the ssh configs, known_hosts and key).
