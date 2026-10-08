# Architecture comparison (track L6)

Written 2026-10-07 against runtime 0.61.10, as track L6 of `docs/RESEARCH-DIRECTION.md`. It adds no measurement. Every number is read from `docs/LEDGER.md` (row ids in brackets), from the model configs in the code, or derived from those by arithmetic that is shown. Tags: **measured** (a ledger row), **derived** (arithmetic from measured rows), **hypothesis** (an explanation nothing here tests). Regimes differ between columns (drive, budget, version), and the table says so; where a cell would need a measurement that does not exist it says "unmeasured" rather than borrowing a number.

The charter's L6 stop rule: conclusions are phrased in architectural terms (expert size, access pattern, working set), not as a per-model percentage. Section 3 does that; a per-model percentage belongs in the CHANGELOG.

## 1. The three architectures as the runtime sees them

| property | DeepSeek V4.1 Flash | MiniMax-M3 | GLM-5.3-Flash |
|---|---|---|---|
| routed experts: layers x per layer | 40 x 384 = 15,360 | 60 x 128 = 7,680 | 42 x 288 = 12,096 |
| top-k and expert uses a token | 6, 240 | 4, 240 | 8, 336 |
| share of experts touched a token (derived) | 1.56 % | 3.13 % | 2.78 % |
| expert record in the bank | 9.49 MiB (9.95 MB), 2-bit affine g128 | 21.1 MiB slot image, 3-bit | 13.5 MiB (14.16 MB), 4-bit |
| expert bank on disk | 142 GiB | 158 GiB | 159 GiB (163 GB contiguous bank) |
| non-expert weights read a token | ~5.14 GB FP8 trunk [DS-BYTES-TOKEN, derived] | ~6 GiB resident non-expert model (not read a token from disk) | not recorded |
| attention | sliding window + compressed sparse, Engram n-gram tables (189 GiB on the X10Pro) | GQA, KV grows with context | 34 linear-attention layers of 42 |
| expert budget in the shipped config | 48 GiB (5,179 slots at 9.95 MB, `cache_sim` output) | 68 GiB (about 3,300 slots at 21.1 MiB, derived) | 46 GiB in the 0.61.9 A/B (about 3,490 slots at 13.5 MiB, derived), 52 shipped |
| resident share of all experts (derived) | 33.7 % | about 43 % | about 29 % |
| decode hit rate | 86-93 % live [DS-HIT-48] | 89.9 % at 68 GiB [MM-TOKEN-SPLIT] | 60-70 % at 52 GiB, internal drive [GLM-HIT] |
| expert drive | internal SSD, 6.8 GB/s | internal + X10Pro mirror, ~6.5-7 GiB/s [ST-MIRROR] | X10Pro, 1.0 GB/s wall [ST-X10] |

Cross-check of the derived slot counts: GLM's 0.61.9 arms read 1.932 GB a token from 136.5 reads [GLM-BANK-AB], and 136.5 x 14.16 MB = 1.933 GB, so the 13.5 MiB record size and the read counts agree.

## 2. What a token costs, by term

Same decomposition as the charter (floor plus exposed expert reads), each cell with its row and regime. The three floors are not the same quantity and are not comparable without that caveat.

| term | DeepSeek | MiniMax | GLM |
|---|---|---|---|
| all-resident token, server path | 79-80 ms [DS-FLOOR-SRV] (84.8 at 25k context, DS-FLOOR-CTX) | 44-49 ms [MM-FLOOR] | **unmeasured** (LEDGER open question 3) |
| misses a token | 23.5-28 [DS-TOKEN-48, DS-HIT-48] | 23.0 at 68 GiB [MM-TOKEN-SPLIT, 0.38.0, stale in time] | 86-123 [GLM-TOKEN-USB-M at 52 GiB: 86.0; GLM-BANK-AB at 46 GiB: 122.7] |
| marginal cost of a miss | 2.0 ms [DS-MISS-48] | ~3.6 ms [MM-MISS] | about 12.5 ms an average miss, derived as 1,531 ms / 122.7 (an upper bound: includes the floor, and reads queue on one pipe [GLM-TOKEN-USB-M mean read 52.9 ms]) |
| bytes read for a miss | 9.95 MB | 22.1 MB | 14.16 MB |
| bytes read a token, all reads (derived) | ~454 MB (45.6 reads) [DS-PRED-USB] | ~508 MB (23 misses, plus speculative loads not counted) | 1.93 GB [GLM-BANK-AB] |
| decode token | 120-126 ms internal (budget 0: ~109) | 93.0 ms agent decode at 68 GiB [0.37.0, NEXT-SESSION v66 row] | 1,531 ms on the X10Pro (bank) |
| share of the token waiting on reads | ~0.2 (misses x 2 ms of 120) | ~0.63 [MM-TOKEN-SPLIT] | 0.91-0.93 [GLM-BANK-AB] |
| what binds now | the floor: GPU 70 % busy, 27 % idle between 44 host syncs [DS-FLOOR-SPLIT-0609] | the drives' wall [bottleneck table in LEDGER section 6] | the drive: 0.95-1.01 GB/s while reading [GLM-TOKEN-USB-M, GLM-BANK-AB] |

## 3. Findings stated in architectural terms

Each is tagged. None is a new measurement.

1. **The read share of a token is set by bytes a miss against drive rate, not by the model family** (derived). A DeepSeek miss moves 9.95 MB at the internal drive's ~5-7 GB/s and costs 2.0 ms; a MiniMax miss moves 22.1 MB over two drives for ~3.6 ms; a GLM miss moves 14.16 MB at ~1 GB/s and its token is 91-93 % reads. The ordering of read share (0.2, 0.63, 0.92) follows bytes divided by drive rate times misses a token. Storage bandwidth buys a lot only below the knee (DeepSeek: ~2.6 GB/s [ST-BW-CURVE]); above it the floor binds.
2. **Hit rate at equal resident share depends on routing skew, and the three models differ** (derived, hypothesis for the cause). At a resident share of 29-43 % of all experts, DeepSeek hits 86-93 %, MiniMax about 90 %, GLM 60-70 % (GLM's budget was 52 GiB, internal drive, an older version; its share is the smallest). Part of the gap is the smaller share; whether GLM's routing is also flatter (more uniform use of experts) is not tested: the repository has a routing trace for DeepSeek only (see `docs/SEAMS.md`, tracer). L6's next measurement is a GLM and a MiniMax routing trace through `cache_sim.py`.
3. **Expert-record size decides which I/O levers pay** (measured, one result per model, generalization is a hypothesis). The mirror-striping result is the clearest: an 18 % loss at 9.49 MiB, a gain at 17.93 MiB [ST-MIRROR-SIZE, stale for DeepSeek]. Contiguous layout: a nine-range 9.95 MB record and a 5-9-range GLM record both plateau ~0.94-0.97 GB/s on the X10Pro; making the GLM record one pread gained +6.6 % rate and -6.1 % of a token with identical output [GLM-LAYOUT-PAIRED, GLM-BANK-AB]; the DeepSeek bank has not been rewritten that way, and its X10Pro case is moot while its bank is internal.
4. **Prefetch pays by precision times the drive's idle share, not by model** (measured on two models, derived for the rule). DeepSeek's predictions are 35-40 % used: a ~1 ms win on the internal SSD and an 11-18 % loss at 1 GB/s [DS-PRED-AB, DS-PRED-USB]. GLM's are 72 % used and roughly break even at 1 GB/s (off -4.4 %, inside drift) [GLM-PRED-USB]. MiniMax's decode prefetch reads from routing the GPU-select loop already ran: ~88-91 % used, -15 % [0.33.0 row; MM-SPEC-PREC 90 %]. A bandwidth-aware or precision-gated prefetch is the architecture-level lever; the gate was priced and not built [DS-PRED-GATE].
5. **Dropping against substituting a missing expert depends on how much routing weight the cheap misses carry** (measured). MiniMax's substitution (miss under 20 % of the layer's weight) is on by default; DeepSeek's lost to the plain drop at equal speed (ten times the dNLL) [Q-SUBST], and DeepSeek's budget-0 drop is the only approximate lever, off in the exact path. One of three C# results showed a budget-0 code cost [Q-MB0-CSHARP, Q-MB-CSHARP-2, Q-MB0-CSHARP-3].
6. **Speculative decoding is closed for all three for one reason** (derived): a K-position verify block reads K tokens' worth of misses, so only the floor amortizes [DS-D2-VERIFY, DS-DSPARK-ACCEPT; MTP for GLM in the prompt's v83 row].
7. **GPU-side expert selection pays in proportion to how many layers are all-hit** (measured, one model each). MiniMax: -9.5 % [MM-GPUSEL, 0.31.0]. DeepSeek: 52.8 % of decode layers are all-hit at 48 GiB, and a layer with a miss rewinds, so the net is +0.1 ms and the lever is held [DS-ALLHIT-LAYERS, DS-D3-SAVE]. GLM: 16.3 % all-hit, closed.
8. **The cliffs are the same mechanism with different constants** (measured): a model's GPU allocation plus every other process's crosses the driver's working set or the system's wired ceiling and the token doubles. DeepSeek 52 GiB (wired ~74.5 GiB, DS-CLIFF-WIRED), MiniMax 63 GiB at the default sysctl and 70 GiB at 88064 [MM-CLIFF-62, MM-BUDGET-68], GLM 64 GiB [GLM-CLIFF-64]. GLM's wired set is its budget plus ~24 GiB, so its usable budget is the smallest.
9. **Context length moves each model differently** (measured for two). DeepSeek's floor adds ~4 ms over the first ~2k tokens and 0.09 ms per 1k after [DS-FLOOR-CTX]; MiniMax's GQA decode attention is 4.3 / 10.5 / 17.5 ms at 2k / 24k / 45k [MM-GQA-KERNEL] and the model decodes at 2.65 tok/s at 64k [MM-64K, stale]. GLM's linear-attention layers should flatten the growth: unmeasured.

## 4. What this table cannot say, and the measurements that would close it

| gap | what is missing | cheapest instrument that exists |
|---|---|---|
| GLM all-resident floor | no row; every GLM token is read-bound | `serve-glm.sh` with a budget that holds a replayed prompt, repeated; estimate and confirm with Hamed first (machine) |
| MiniMax constants since substitution became the default | MM rows pre-date 0.43.0 (LEDGER open question 3) | the M0 harness of 0.55.0 (`serve-minimax.sh`, machine) |
| routing skew for GLM (MiniMax measured in 0.62.1: LEDGER MM-ROUTING-SKEW, 35 % against DeepSeek's 50-53 % in the busiest 10 % of experts) | no GLM trace (finding 2) | a tracer for both exists since 0.62.0 (`CACHALOT_ROUTING_TRACE`, unweighted); the trace itself still has to be recorded on the machine |
| bytes a token, useful against wasted, per model on one scale | counters exist per model but only DeepSeek and GLM report predicted loads used [DS-PRED-PREC, GLM-PRED-USB]; MiniMax's spec precision is from 0.38.0 | `/v1/stats` store counters (GLM and MiniMax carry them since 0.61.1) |
| energy | none | L5, Hamed runs `sudo powermetrics` per arm |

## 5. Honest limits

- The columns are not one regime: DeepSeek is internal-drive at 48 GiB, GLM is X10Pro at 46 GiB, MiniMax's decode numbers are from 0.37.0-0.38.0 at 68 GiB. A difference between columns mixes the architecture with the drive and the version.
- The derived slot counts for MiniMax and GLM use the budget over the record size and ignore transient slots and the partial last slab; they are within a few percent, not exact.
- Findings 2, 7 and 9 each name an architecture-level cause that was not isolated by a controlled comparison. They are the lab's hypotheses to test, not results.
