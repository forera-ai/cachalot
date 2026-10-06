# Constants and bottleneck ledger (track L1)

Written 2026-10-06 against runtime 0.60.4, as track L1 of `docs/RESEARCH-DIRECTION.md`. It collects, in one place, every constant the project's decisions rest on, with the regime it was measured in and whether anything has since reproduced or contradicted it. It adds no new measurement except where a row says so.

**How to use it.** A predictive model, a pricing note or a "lever closed" claim cites a row id (for example `DS-FLOOR-48`). A row is only as good as its regime: when the drive, the budget, the `iogpu.wired_limit_mb` sysctl, the runtime path or the context changes, the row is suspect until it is reproduced. The charter's stop rule for L1 is that every constant used by `benchmarks/cache_sim.py` and by any what-if cites a row here.

**Tags.** *measured*: a recorded measurement (with its HANDOFF section). *derived*: arithmetic from measured rows. *estimate*: arithmetic from assumptions or from an older regime. **Status**: *current* (nothing since contradicts it and its regime still holds), *stale* (its regime has changed; not contradicted, not reproduced), *contradicted* (a later measurement disagrees; the row that does is named), *superseded* (replaced by a better measurement of the same thing).

**Regime fields used below.** drive: `int` = internal SSD (~6.8 GB/s), `X10` = X10Pro over USB (1.0 GB/s, queue depth 1), `int+X10` = MiniMax's mirror over both. sysctl: `88064` = `iogpu.wired_limit_mb=88064` (an 86 GiB GPU working set; Hamed sets it, it resets at reboot), `default` = unset (Metal recommends 77.76 GiB on this machine, read 2026-10-06). ctx = context length at decode. W-class = the charter's workload classes (section 7.1).

**Regime warning, 2026-10-06.** The machine rebooted on 2026-10-06 and the sysctl read 0 (default) until Hamed re-applied 88064 the same morning (it read 88064 again at 0.60.7); check it before every speed arm. Every DeepSeek speed row from 0.53.0 to 0.60.2 was measured at `88064`. No DeepSeek row has been measured at the default sysctl; the wired governor (`DS-CLIFF-WIRED`) acts on system wired memory, not on the GPU working set, so the 48 GiB rows probably hold, but that is a hypothesis until one arm reproduces `DS-FLOOR-48`.

## 1. DeepSeek V4.1 Flash (priority 1)

Architecture: 40 MoE layers x 384 routed experts = 15,360, top-6 (240 expert uses a token), 2-bit affine g128 bank at 9.49 MiB (9.95 MB) a record, 142 GiB, FP8 trunk, Engram tables (189 GiB on X10Pro).

| id | constant | value | tag | version, date | regime | method, W-class | source | status | reproduced by |
|---|---|---|---|---|---|---|---|---|---|
| DS-FLOOR-48 | all-resident token, server path, short context | 70 ms (fit intercept 69-71 at 36-48 GiB) | measured | 0.53.0, 2026-10-03 | int bank, 48 GiB, wired limit 80, sysctl 88064, ctx < 1k | `mix.sh` 11 requests x 120 tokens, fit of ms on misses, W1 | §18.52 | **contradicted** by DS-FLOOR-SRV (0.60.10): a fit intercept, not a floor; the same workload fit 93.4 + 1.40 m and 73.4 + 2.30 m in two arms on one day | §18.53 governor-on arm: 70 + 2.15 m |
| DS-FLOOR-SRV | all-resident token, server path, short context, exact | **79.2-80.3 ms** (repeats 1-2 of one greedy prompt, 0 misses); 87.8 after 12 mixed prompts in the same process; budget 0 (shipped): 81.3-81.9 | measured | 0.60.10, 2026-10-06 | int bank, 48 GiB, wired limit 80, sysctl 88064, no snapshots preloaded, ctx 27 | `serve.sh` + `floor-server-0.60.10/driver.py`, W1 | §18.76 | current | §18.53 `floor.sh` 78.6 ms (12.73 tok/s) at 48 GiB; in-process DS-FLOOR-INPROC 75-77 the same day |
| DS-PRED-FLOOR | cost of decode prediction on an all-resident server token | ~5 ms (A 79.2-80.3 against `CACHALOT_PREDICT_TOPK=0` 72.8-76.8); on 23.5-miss tokens prediction nets ~1 ms (126.2 against 127.4 ms) | measured | 0.60.10 | as DS-FLOOR-SRV | two fresh servers, same driver, n = 2-3 floor requests and 12 mixed each | §18.76 | current, small n | §9.15 (2026-09-21, in-process): ~6 ms (3.9 CPU + 2.5 eval) |
| DS-MB0-HOSTLAYERS | budget 0: decode layer-calls that route to a predicted load still in flight | **26.4 %** (12.6 experts a token; 73.6 % of layers could skip the host); server arm 11.5 awaited a token, 16.5 dropped | measured | 0.60.10 | int bank, 48 GiB, budget 0, in-process, 12 mixed prompts x 120 greedy tokens | `floor-server-0.60.10/layer_share.py` (a monkeypatched counter), W1 | §18.76 | current | none |
| DS-MISS-48 | cost of one decode miss | 2.0 ms (fits 1.83-2.41) | measured | 0.53.0 | as DS-FLOOR-48 | slope of the same fits | §18.52, §18.53 | current | §18.53 (2.15, 2.27); 0.60.10: 1.97 from the mean token over the measured floor (the fits' slopes, 1.40 and 2.30, are not stable) |
| DS-TOKEN-48 | decode token at 48 GiB | 120-125 ms (7.9-8.3 tok/s) at 25 misses | measured | 0.53.0-0.54.0 | as DS-FLOOR-48 | `mix.sh`, `serve.sh` twice | §18.52, §18.53 | current | §18.53 (125.2 ms); 0.60.10: 126.2 ms at 23.5 misses (= 79.8 + 1.97 m against DS-FLOOR-SRV) |
| DS-FLOOR-INPROC | all-resident token, in-process | 75.8-77.5 ms (README 76.4) | measured | 0.53.0 | int bank, budget 36, `MLX_METAL_FAST_SYNCH=1`, short ctx | `decode_resident.py` | §18.52 | current | 0.60.9 (§18.75): 77.4 mean of five all-resident passes (75.7-79.8), 75.4 median in the sync profiler, 75.7-76.9 in the GPU profiler; 48 GiB, sysctl 88064 |
| DS-FLOOR-22K | all-resident token, server, 22k Hermes context | 95.6 ms | measured | 0.57.0, 2026-10-04 | int bank, 48 GiB, sysctl 88064, ctx 22k | one repeated warm-up turn in `server_miss_budget_ab.py`, W6 | §18.59 | current, n = 1 | none |
| DS-TOKEN-22K-EXACT | decode token, server, 22k context, exact | 140.3 ms (7.1 tok/s) | measured | 0.57.0 | as DS-FLOOR-22K | per-turn alternation, 11 prompts x 220 greedy tokens, W2/W6 | §18.59 | current | §18.68: 138.3 ms on 1,500-token C# replies |
| DS-HIT-48 | decode hit rate, agent context | 86-93 % live, 88.3 % simulated; ~24-28 misses a token | measured | 0.55.0 | int bank, 48 GiB, ctx 22k | weighted routing trace (1,958 decode tokens) + `cache_sim.py`, W6 | §18.54 | current | simulator matches live within spread |
| DS-BUDGET-CURVE | misses a token against expert budget (simulated) | 36: 34.1, 44: 29.4, 48: 28.0, 52: 27.0 | derived | 0.55.0 | trace of DS-HIT-48 | `cache_sim.py`, LRU | §18.54 | current | live 36/44/48 in §18.52 (31.8 / 27.1 / 25.3) |
| DS-ALLHIT-LAYERS | share of decode layer-calls with every expert resident | 36: 45.1 %, 48: 52.8 %, 52: 54.2 % | derived | 0.55.1 | trace of DS-HIT-48 | `d3_layer_hits.py` | §18.55 | current | none |
| DS-CLIFF-GPU52 | GPU allocation cliff at a 52 GiB budget | floor 157-182 ms (token 210 ms, 4.8 tok/s) at 80.5 GiB "Alloc system memory" | measured | 0.53.0 | int bank, wired limit 80 and 84 | `mix.sh`, floor runs | §18.52, §18.53 | current (mechanism corrected by DS-CLIFF-WIRED) | §18.53 at wired limit 84 |
| DS-CLIFF-WIRED | the edge as system wired memory | floor flat to ~72 GiB wired, 85 ms at 74.2, 91 at 74.6, 158-182 at 75.1+ | measured | 0.54.0 | sysctl 88064 | `floor.sh`, `vm_stat` wired pages during decode | §18.53 | current | governor arms of §18.53 |
| DS-GOV-CEIL | governor ceiling | 76 % of `hw.memsize` = 72.96 GiB; 77 % held 74.1 GiB and gave a 92 ms floor | measured | 0.54.0 | sysctl 88064 | holder arms (4 GiB mlock) | §18.53 | current | none at default sysctl |
| DS-PREFILL-COLD | cold prefill, short-mid prompt | 86.5 tok/s (4,501 tokens in 52 s) at 48 GiB; 64.9 at 52 | measured | 0.53.0 | int bank, cold text | `pf.sh`, n = 1 per arm, W3 | §18.52 | current, n = 1 | none |
| DS-PREFILL-BLOCK | Hermes system block, cold | 205-227 s for ~22.3-22.5k tokens (~100 tok/s) | measured | 0.55.0-0.58.2 | int bank, 48 GiB | live sessions and the trace run, W6 | §18.54, §18.63, §18.64 | current | three sessions |
| DS-PREFILL-SHORT | short side requests (200-400 tokens) | 25-45 tok/s (~165 s a session) | measured | 0.58.1 | int bank, live Hermes | server `[request]` lines, W6 | §18.63 | current, unexplained | none |
| DS-PIN-4096 | first in-block chunk pin | 4,096 tokens (~41 s) reused after a restart or a changed block | estimate | 0.59.0 | as DS-PREFILL-BLOCK | priced from dumps, built, not measured live | §18.65 | open: live check pending | none |
| DS-ENGRAM-USB | Engram row read, decode shape, X10Pro | 3.90 ms mean (p90 4.19, max 7.1) per two-layer read of 24 rows | measured | 0.53.0 | X10 | 300 random reads, cold | §18.52 | current; hidden behind the layers | none |
| DS-PRED-PREC | predicted expert loads that are used | 34 % (prefill + decode), 39-41 % in later runs; 36 % in the 2026-10-03 smoke run | measured | 0.53.0 | int bank | `/v1/stats` counters | §18.52, SPEED-RESEARCH §0 | current | three runs |
| DS-PRED-AB | value of prediction at 48 GiB | +3 % (inside noise); kept | measured | 0.53.0 | int bank, 48 GiB | `CACHALOT_PREDICT_TOPK=0` arm | §18.52 | current | none |
| DS-PRED-USB | value of prediction on a 1 GB/s expert drive | **prediction off is faster**: -17.5 % emulated (445.7 to 367.7 ms), -11.4 % on the X10Pro (390.1 to 345.6); reads a token 45.6 to 26.9 (the ~18.6 unused speculative loads, 185 MB a token, queue ahead of demand misses on one pipe) | measured | 0.61.0, 2026-10-06 | 48 GiB, sysctl 88064, exact path | two pairs of fresh `serve.sh` arms, W1 | §18.77 | current, n = 1 a pair | the emulated and the real pair agree in sign |
| DS-MB0-USB | decode token on a 1 GB/s expert drive under budget 0 (shipped) | **281.2 ms** (exact 445.7, -37 %): 12.8 predicted loads awaited and 17.0 experts dropped a token, 32.1 predicted loads a token (40 % used); with prediction off as well **76.1 ms** but 78.0 of 240 routed uses dropped a token (frozen resident set; quality unmeasured, a bound only) | measured | 0.61.1, 2026-10-06 | int bank throttled to 1 GB/s, 48 GiB, sysctl 88064 | two fresh `serve.sh` arms, W1 | §18.78 | current, n = 1 an arm | none |
| DS-MB0-STEP | decode step with miss budget 0 | -27 % harness (131 to 95 ms), -22.6 % server at 22k (140.3 to 108.7), -18 % on mixed replies, -25 % on long C# replies | measured | 0.56.1-0.60.2 | int bank, 48 GiB, sysctl 88064 | Pareto harness; per-turn alternation; blind runs | §18.58, §18.59, §18.67, §18.68 | current | four runs |
| DS-MB1-STEP | decode step with miss budget 1 | -4 % harness (131 to 126 ms); about -6 % topic-shift; **-2.7 % paired [0.967, 0.978] on 1,000-token C# replies at 26k context** (2,112 experts dropped a reply, ~10 % of budget 0's) | measured | 0.56.1-0.60.6 | int bank, 48 GiB, sysctl 88064 | harness; `topic_shift.py`; three-arm blind run, W4 | §18.58, §18.61, §18.72 | current | three runs |
| DS-FLOOR-SPLIT | the floor's parts (internal-drive era) | trunk GEMV 14.0 ms (5.14 GB at ~367 GB/s), routing prediction 11, 40 router `mx.eval` round trips 9-12, routed experts 6.9, compressor/indexer 5.5, `wo_a` 4.3, hyper-connection glue 4.2, head/Engram/router 3.2, sparse attention 2.6, Engram reads 1.9; 56 ms inside `mx.eval`, 21.5 ms CPU outside | measured | 0.9.x, 2026-09 | int bank, 52 GiB (pre-cliff), floor 77-80 | profiling (§9.15) | SPEED-RESEARCH §1.1, README | superseded by DS-FLOOR-SPLIT-0609 | none |
| DS-BYTES-TOKEN | bytes a token | ~7.5 GB (trunk 5.14 + 240 x 9.95 MB); ~98 GB/s over the floor, 12 % of 819 GB/s | derived | 2026-10-03 | as DS-FLOOR-SPLIT | arithmetic | SPEED-RESEARCH §1.1 | estimate; the floor used (77 ms) is stale, at 70 ms it is ~107 GB/s | none |
| DS-FLOOR-SPLIT-0609 | the floor's parts, re-profiled | token 75.4 ms (sync arm median) = inside `mx.eval` 55.5 + CPU graph building 17.9 + store-blocked 1.0 + Engram reads 1.2; 44 evals. Chained GPU rows on the shipped path sum to 52.2 ms: compressed reuse attention 15.9 (30 layers), compressed source attention 8.7, sliding-window attention 1.0, routed experts traced 10.1, shared expert 6.0, hyper-connection glue 4.9, source compressor/indexer 2.3, head 1.6, router (fused) 0.9, Engram forwards 0.8 | measured (parts), derived (the sum) | 0.60.9, 2026-10-06 | int bank, 48 GiB, wired limit 80, sysctl 88064, `MLX_METAL_FAST_SYNCH=1`, ctx 17 (GPU arm) / 512 (sync arm) | `profile_decode_sync.py --mode resident`, `profile_decode_gpu.py`, W1 | §18.75 | current | none |
| DS-GPU-IDLE | GPU idle inside an all-resident token | ~20 ms of 75 (27 %): the CPU builds the next graph between the 44 syncs (17.9), plus store and Engram calls (2.2); inside-eval time exceeds the chained GPU sum by ~3.3 ms | derived | 0.60.9 | as DS-FLOOR-SPLIT-0609 | arithmetic on the two arms | §18.75 | current | none |
| DS-EVAL-DRAIN | one router `mx.eval` round trip, isolated | 0.272 ms over a 0.019 ms kernel (x 44 = 11.9 ms a token if none overlapped) | measured | 0.60.9 | as above | `profile_decode_gpu.py` eval-drain rows | §18.75 | current; in the token most of it overlaps (see DS-GPU-IDLE) | §9.x micro_eval_floor 0.31 ms |
| DS-SPEC-RESIDENT | speculative expert loads while every needed expert is resident | 23.0 loads a token, 23.0 expired unused, 218 MiB a token read (~2.8 GB/s at the floor) | measured | 0.60.9 | as DS-FLOOR-SPLIT-0609, the same token replayed 12 times | `profile_decode_sync.py` counters | §18.75 | current; off the critical path (store-blocked 1.0 ms), cost is drive time and energy | none |
| DS-D2-VERIFY | marginal cost of a verified draft position | 26.9 ms (multi-position path = prefill) | measured | 0.52.x | int bank | §18.50 | §18.56 | current; reopen under ~8 ms | none |
| DS-DSPARK-ACCEPT | DSpark acceptance | 2.85 tokens a forward at width 5 (p ~0.72), greedy, four short prompts | measured | 2026-09 | | §15 | §18.56 | current, small sample | none |
| DS-D3-SAVE | host round trip saved per all-hit layer by GPU-side selection | 0.63 ms (MiniMax's constant, not measured on DeepSeek); rewind ~0.7 ms (assumed) | estimate | 0.55.1 | | borrowed from MM-GPUSEL | §18.55 | estimate; DeepSeek's own value unmeasured | none |
| DS-BATCH | batched decode, offline (13 streams of the 22k trace, 48 GiB) | misses a token 24.9 / 28.2 / 30.5 / 34.9 at B = 1 / 2 / 4 / 8; union share of routed uses 96.5 % at B = 2, 89.4 % at B = 4; aggregate ms a token 84.7-94.6 at B = 4 (1.27-1.42x) | derived | 0.60.6 | trace of DS-HIT-48 | `batch_union.py`; the floor-per-lane models are hypotheses | §18.72 | estimate; not built | none |

## 2. MiniMax-M3 (priority 2)

Architecture: 60 MoE layers x 128 experts = 7,680, top-4, 21.1 MiB slot images, 158 GiB bank, ~6 GiB non-expert model.

| id | constant | value | tag | version, date | regime | method, W-class | source | status | reproduced by |
|---|---|---|---|---|---|---|---|---|---|
| MM-FLOOR | all-hit token, server path | 44-49 ms (repeats at 0.4-1.1 misses: 49.3 / 46.9 / 44.4) | measured | 0.55.0, 2026-10-03 | int+X10, 68 GiB, sysctl 88064 | M0: 12 prompts + one repeated, `serve-minimax.sh`, W1 | §18.54 | current | §18.19's 35-43 ms in-process |
| MM-MISS | cost of one decode miss | ~3.6 ms average (read 4.4-4.6 ms); fit slope 5.2 on mixed requests | measured | 0.55.0 | as MM-FLOOR | M0 fit `27.1 + 5.20 x misses` (r 0.95, n 12) | §18.54 | current | holder arm 3.7 ms |
| MM-FLOOR-77 | the research's floor | 77 ms intercept of `77 + 3.42 x misses` | estimate | 0.52.2 | 14 replies, narrow miss range | fit | SPEED-RESEARCH §1.2 | **contradicted** by MM-FLOOR (an intercept, not a floor) | |
| MM-TOKEN-SPLIT | where a token went at 68 GiB | 115.6 ms = 72.8 store wait + 42.8 other, 23.0 misses, hit 0.899 | measured | 0.38.0, 2026-09-28 | int+X10, 68 GiB, sysctl 88064, display asleep | `glm_prefill_timeline.py 2048`, `TF_DECODE=200` | §18.19 | stale in time (0.39-0.45 levers since), not contradicted | none |
| MM-READ-BW | effective read rate per miss | 21.1 MiB at ~6.5-7 GiB/s over both drives (2.9-3.2 ms of wait a miss) | measured | 0.36.0-0.38.0 | int+X10 | wait / misses | §18.17, §18.19 | current | two sections |
| MM-WAIT-METRIC | `decode_wait_ms_per_miss` in stats | 1.2 ms shown against ~3.4 ms fitted | measured | 0.52.2 | | comparison | SPEED-RESEARCH §2 | current: the stat understates a miss ~3x | |
| MM-CLIFF-62 | GPU working-set cliff, default sysctl | 62 GiB fine (110-115 ms), 63: 120-139 ms, 64: 192-193 ms; 66 cannot build the slab kernel | measured | 0.36.0 | int+X10, sysctl default (77.76 GiB) | `stream_agent.py`, display asleep | §18.17 | current at default sysctl | |
| MM-BUDGET-68 | budget at the raised working set | 68 GiB: -11 % decode, -12 % turns, -13 % short prefills against 62; 70 and 72 over-commit at load (+0.43 / +0.96 GiB swap) | measured | 0.37.0 | int+X10, sysctl 88064 | `stream_agent.py`, `memlog.py` | §18.18 | current; **needs the sysctl** (today 0) | §18.54 ran at 68 |
| MM-HOLDER | another process wiring 4 GiB | no cliff: 86.2 ms at 11.6 misses against 82.9 at 10.7 | measured | 0.55.0 | as MM-FLOOR | holder arm, n = 1, no alternation | §18.54 | current | |
| MM-GQA-KERNEL | decode attention, 60 layers | 4.3 / 10.5 / 17.5 ms at 2k / 24k / 45k; matmul-bound | measured | 0.38.0 | | `minimax_gqa_decode.py` | §18.19 | current | |
| MM-PREFILL-ATTN | prefill attention at long context | 2,048 queries over 45k keys: 177.6 ms a layer (17 TFLOPS); prefill at the GPU's matmul peak | measured | 0.38.0-0.38.1 | | micro-benchmark | §18.19, §18.20 | current | |
| MM-64K | decode at 64k context | 2.65 tok/s | measured | 0.21.0 | | | §18.2 | stale (many levers since) | |
| MM-GPUSEL | host round trip saved per all-hit layer by GPU-side selection | 0.63 ms; all-hit floor ~72 to 34 ms | measured | 0.31.0 | | `TF_ALTERNATE` | §18.12, SPEED-RESEARCH §2 | current | |
| MM-SPEC-PREC | speculative loads used | 12.6 of 14 a token (90 %) | measured | 0.38.0 | int+X10, 68 GiB | timeline | §18.19 | current | |

## 3. GLM-5.3-Flash (priority 3)

Architecture: 42 MoE layers x 288 experts = 12,096, top-8 (336 uses a token), 13.5 MiB an expert, 159 GiB; 34 linear-attention layers.

| id | constant | value | tag | version, date | regime | method, W-class | source | status | reproduced by |
|---|---|---|---|---|---|---|---|---|---|
| GLM-TOKEN-INT | decode token, internal drive | ~365 ms at ~100 misses, 70-75 % store wait; `105 + 2.6 x misses` | measured | 0.48.1-0.50.0 | int, 52 GiB | timeline; method noise ~25 % | §18.33, SPEED-RESEARCH §1.3 | **stale**: GLM moved to X10Pro in 0.52.3 | |
| GLM-READ-INT | drive rate during decode | 5.2-5.8 GiB/s, 84 % busy | measured | 0.48.1 | int | | §18.33 | stale (drive moved) | |
| GLM-HIT | decode hit rate | 60-70 % at 52 GiB | measured | 0.48.1 | int, 52 GiB | | §18.33 | current in hit terms (drive-independent) | |
| GLM-BUDGET-CURVE | decode against budget, live Hermes | 44 / 48 / 50 GiB: 2.79 / 3.00 / 3.10 tok/s (~1.7 % a GiB) | measured | 0.51.5 | int, n = 1 per arm | Hermes sessions, W6 | §18.42 | stale (drive moved) | |
| GLM-CLIFF-64 | 64 GiB budget | tokens 1.5-2.3x slower, wired 87-90 GiB, swap rising; GLM's wired set is budget + ~24 GiB | measured | 0.48.1 | int, wired limit 84 | | §18.33 | current | |
| GLM-PRED | next-layer prediction | top 8 holds 67 % of the next layer's experts and 59 % of its misses; K = 5 shipped, -8 to -14 % | measured | 0.50.0 | int, 52 GiB | `TF_ALTERNATE` swapped pairs | §18.35 | current in hit terms | |
| GLM-TOKEN-USB | decode token, X10Pro | ~14 ms a miss x ~100 misses = ~1.4 s a token (~0.7 tok/s) | estimate | 0.52.3 | X10 | arithmetic | SPEED-RESEARCH §0 | **superseded** by GLM-TOKEN-USB-M | |
| GLM-TOKEN-USB-M | decode token, X10Pro, K = 5 (shipped) | **1,518 ms** (0.66 tok/s) on 12 mixed prompts, 86.0 misses a token, 120.9 reads (1.71 GB) a token, 91 % of decode waiting on reads, drive 0.95 GB/s while reading, mean read 52.9 ms (queueing); repeats of one prompt 1,362-1,431 ms (not all-resident at 52 GiB) | measured | 0.61.1, 2026-10-06 | X10, 52 GiB, sysctl 88064 | fresh `serve-glm.sh`, W1 | §18.78 | current, n = 1 | none |
| GLM-PRED-USB | value of K = 5 prediction on the X10Pro | off is -4.4 % on mixed tokens (1,451 ms), -3.3 % on repeats: inside drift; 48.4 predicted loads a token, **72 % used**, 13.4 unused (190 MB); bytes a token -11 % without it | measured | 0.61.1 | as GLM-TOKEN-USB-M | one pair of fresh arms | §18.78 | current, one pair | none |
| GLM-PREFILL-USB | cold prefill of short prompts, X10Pro | ~0.7 tok/s (365 tokens of 12 prompts in 517-541 s; 25-62-token prompts 36-80 s) | measured | 0.61.1 | as GLM-TOKEN-USB-M | `[request]` lines of the two arms | §18.78 | current | two arms |
| GLM-CODE | garbled C# in replays | 12/21 (MiniMax 1/19); 8-bit non-expert hybrid 12/12 | measured | 0.51.7-0.52.1 | | C# replay, W4 | §18.44, §18.49 | current | two runs |

## 4. Storage and machine

| id | constant | value | tag | source | status |
|---|---|---|---|---|---|
| ST-INT | internal SSD sequential read | ~6.8 GB/s; 1.46 ms raw for a 9.95 MB DeepSeek record | measured | SPEED-RESEARCH §1.1 | current |
| ST-X10 | X10Pro over USB | 1.0 GB/s wall, queue depth 1 | measured | SPEED-RESEARCH §1.1, HANDOFF 0.48.x | current; queue depth recorded, never swept (L3 item 6) |
| ST-X10-EFF | X10Pro effective rate on DeepSeek decode (mixed W1, exact, 48 GiB) | ~1.15 GB/s (453 MB a token in 390 ms); mean read 17.3 ms (queueing) | derived | §18.77 | current, one arm |
| ST-THROTTLE-VALID | `CACHALOT_READ_THROTTLE_GBPS` against the drive it imitates | emulated 1 GB/s 445.7 ms against real X10Pro 390.1 ms (-12.5 %, inside the +-15 % written before); prediction-off arms 367.7 against 345.6 (-6 %) | measured | §18.77 | current; the emulation is a serial pipe, slightly pessimistic for the X10Pro |
| ST-BW-CURVE | DeepSeek token against expert-drive bandwidth (mixed W1, exact, 48 GiB, prediction on) | internal 118.6 ms, 4 GB/s 139.3, 2 GB/s 227.4, 1 GB/s 445.7 (all-resident 77-88 in every arm; 23.5 misses everywhere). Model `token = max(78 + 23.5 x (1.73 + 9.95 x (1/B - 1/6.8)), 453 MB / B)`: within 1 % at 1-2 GB/s, -3 % at 4; knee ~2.6 GB/s. Marginal value of storage bandwidth: ~219 ms a token per GB/s at 1-2 GB/s, ~44 at 2-4, ~7 at 4-6.8 | measured (points), derived (model) | §18.77 | current; W1 only |
| ST-MIRROR | MiniMax over both drives | ~6.5-7 GiB/s | measured | §18.17, §18.19 | current |
| ST-MIRROR-SIZE | mirror striping against expert size | an 18 % loss at 9.49 MiB, a gain at 17.93 MiB | measured | README roadmap item 6 | stale for DeepSeek (priced at old constants) |
| MC-WORKSET | Metal recommended working set | 77.76 GiB at default sysctl; 86 GiB at 88064 | measured | §18.18; read again 2026-10-06 | current |
| MC-SCREENSAVER | Flurry screensaver | ~13 % slower decode | measured | memory "slow decode window" | current |
| MC-HERMES-WINDOW | visible Hermes Desktop window | up to 30 % slower | measured | §18.19 (M28) | current |
| MC-READ-SIZE | GPU streaming read rate against bytes a launch reads | dense bf16 GEMV: 47 % of 819 GB/s at 2 MiB, 71 % at 4, 83 % at 8, 87-89 % at 16 MiB and above; `mx.sum` 31 / 50 / 63 / 76 % at 2 / 4 / 8 / 16 MiB, 93 % at 128 MiB+; ~5 us a launch | measured | §18.76, `benchmarks/micro_read_size_roofline.py` | current |
| MC-READ-SIZE-Q | quantized kernels at the same bytes a launch | 2-bit `quantized_matmul` (g128, scales and biases counted): 31 / 43 / 56 / 61 / 68 / 70 / 72 % of 819 GB/s at 1 / 2 / 3 / 4 / 8 / 11-32 / 128 MiB; FP8 GEMV (`fp8_linear_quantized`): 17 / 33 / 37 / 46 / 56 / 58-62 / 65 %; the bf16 GEMV beside them 32 / 64 / 77 / 77 / 87 / 86-88 / 90 % | measured | §18.78, `benchmarks/micro_read_size_roofline.py --arms` | current |
| MC-DRIFT | process-to-process drift | +-10 %; levers under ~15 % need per-token alternation | measured | SPEED-RESEARCH §2 | current |

## 5. Quality constants

| id | constant | value | tag | version | method, W-class | source | status |
|---|---|---|---|---|---|---|---|
| Q-NOISE | rounding band (prefill in 96-token chunks) | dNLL +0.0081 [-0.0071, +0.0243], KL mean 0.0144 / max 0.230, top-1 92.4 % | measured | 0.56.1 | Pareto harness, 576 positions | §18.58 | current |
| Q-MB0-NLL | miss budget 0, harness | dNLL +0.0073 [-0.0148, +0.0281], KL 0.0271 / 1.305, one more failed task | measured | 0.56.1 | Pareto harness | §18.58 | current |
| Q-MB0-AGENT | miss budget 0, agent bodies | 6/48 vs 14/48 exact (p 0.077, one grader); 28/96 vs 26/96 (p 0.87, model graders) | measured | 0.57.1, 0.60.1 | blind sheet, W6 | §18.60, §18.67 | current |
| Q-MB0-CSHARP | miss budget 0, C# | **26/48 flawed vs 11/48 exact (p 0.003); 15 invented API members vs 1**; TypeScript 8/48 vs 9/48 | measured | 0.60.2 | blind sheet, four model graders, W4 | §18.68 | current; one prompt; the dump it used is lost (see note) |
| Q-MB-CSHARP-2 | three arms on a rebuilt C# body (Hermes store, 25,855 prompt tokens) | exact 10/48, **budget 1 10/48 (p 1.00), budget 0 13/48 (p 0.63)**; invented 6 / 2 / 7 | measured | 0.60.6 | blind sheet, three model graders, W4 | §18.72 | current; **the positive control did not reproduce Q-MB0-CSHARP**, so the budget-1 equality is inconclusive and the budget-0 code cost is now split across two bodies |
| Q-MB0-CSHARP-3 | budget 0 on a second, independent C# task (CsvHelper + System.Text.Json configuration, ~275-token replies, short context) | **exact 2/48, budget 0 2/48 (p 1.00)**; decode ratio 0.776 [0.769, 0.782] | measured | 0.60.8 | blind sheet, two model graders, W4 | §18.74 | current; low base rate (4 %), rules out a large effect only |
| Q-MB0-SHIFT | miss budget 0 after a topic shift | +0.11 nats a token for the first 100 tokens in one run, +0.004 in the second | measured | 0.57.2-0.58.0 | `topic_shift.py`, W5 | §18.61, §18.62 | current; within run-to-run noise at this n |
| Q-SUBST | router-share substitution (DeepSeek) | dNLL +0.134, KL max 3.6-3.9 at 92.5 ms against budget 0's +0.013 at 98.3 ms | measured | 0.58.0 | `topic_shift.py` | §18.62 | current; closed |

**Note on Q-MB0-CSHARP's workload.** The dump the 0.60.1-0.60.2 runs used (`/tmp/cachalot-requests-0.58.1.jsonl`) was deleted by the reboot of 2026-10-06; `/tmp` is not a place to keep a workload. The 0.60.6 run rebuilt an approximate C# body from Hermes's own store into `benchmarks/results/quality-blind-0.60.5-budget1/csharp-body.jsonl`; dumps that a result depends on now belong under `benchmarks/results/`.

## 6. Bottleneck migration

What bound each model, what removed it, and what bound next. "Bound" means the term that dominated the token or turn in the regime of the time.

### DeepSeek V4.1 Flash

| version | bound by | evidence | removed or moved by | next bound |
|---|---|---|---|---|
| 0.1-0.8 (September) | SSD bandwidth (FP4 experts, 17.93 MiB) | charter §3.2; early HANDOFF sessions | the 2-bit bank (9.49 MiB) | hit rate |
| 0.9.x | hit rate and the 77-80 ms floor (9.4-9.6 tok/s at 52 GiB, internal) | README "Performance" | | (regime change) |
| 0.48.1-0.52.2 | the USB drive (bank on X10Pro, ~10 ms a miss; estimated 3.8 tok/s) | SPEED-RESEARCH §1.1 (estimate) | bank back on the internal SSD (0.52.3) | the GPU allocation cliff |
| 0.52.3 | GPU allocation cliff at 52 GiB (floor doubled to 157 ms) | DS-CLIFF-GPU52 | budget 48 (0.53.0), wired governor (0.54.0) | the floor |
| 0.53.0-now | the floor (80 ms through the server, DS-FLOOR-SRV; 70 was a fit intercept) plus 25 misses x 2 ms | DS-FLOOR-48, DS-MISS-48 | every bit-identical lever priced below its stop rule (D1 moot, D2 held, D3 held); budget 0 drops the misses (default since 0.60.0) | the floor: GPU busy ~70 % (attention 25.6 ms, routed experts 10.1, shared expert 6.0) and GPU idle ~27 % while the CPU builds graphs between 44 host syncs (DS-FLOOR-SPLIT-0609, DS-GPU-IDLE) |
| agent turns | the 22k system block's cold prefill (205-227 s) | DS-PREFILL-BLOCK | date reuse (0.60.0), in-block pin (0.59.0), not measured live | short side requests (DS-PREFILL-SHORT) |

### MiniMax-M3

| version | bound by | removed or moved by | next bound |
|---|---|---|---|
| 0.20.0 | per-layer GPU round trips | host-side work cut (0.21.0) | reads |
| 0.23.0-0.35.0 | reads (bandwidth-bound, MM-READ-BW) | bias-free bank, slot images, pair slots, GPU-side selection (0.31.0, MM-GPUSEL), prediction and prefetch | the GPU memory ceiling |
| 0.36.0-0.37.0 | the GPU working set (MM-CLIFF-62) | sysctl 88064 and budget 68 (MM-BUDGET-68), embedding on the host | the drives' wall |
| 0.38.0-now | the drives' wall: reads ~63 % of a token (MM-TOKEN-SPLIT) | miss substitution (output-changing, on by default since 0.43.0); M19 (a Thunderbolt drive) halted | reads; prefill at the GPU's matmul peak (MM-PREFILL-ATTN) |

### GLM-5.3-Flash

| version | bound by | removed or moved by | next bound |
|---|---|---|---|
| 0.48.1-0.51.x | store wait, 70-75 % of a token (GLM-TOKEN-INT) | contiguous bank (0.49.0), next-layer prediction (0.50.0, -8 to -14 %) | store wait |
| 0.52.3-now | the USB drive (moved there by priority): 1.52 s a token, 91 % of decode waiting on reads (GLM-TOKEN-USB-M, measured 0.61.1) | none planned; prefetch off -4 % (inside drift) | the drive |

## 7. Stale constants in code

| where | constant | ledger row it should cite | action |
|---|---|---|---|
| `benchmarks/cache_sim.py` `--floor-ms` default | 77.0 | DS-FLOOR-SRV (80) | changed to 70 in 0.60.6 (DS-FLOOR-48), to 80 in 0.60.10 (DS-FLOOR-SRV) |
| `benchmarks/cache_sim.py` `--miss-ms` default | `internal=1.7,usb=10.2` | DS-MISS-48 (2.0, internal bank, 48 GiB) | changed to `internal=2.0,usb=10.2` in 0.60.6; the USB figure stays an estimate (ST-X10) |
| `benchmarks/d2_verify_union.py` defaults | floor 70, miss 2.0, marginal 8, draft 15 | DS-FLOOR-SRV (80, was DS-FLOOR-48), DS-MISS-48; marginal 8 is Rapid-MLX's, DS-D2-VERIFY is ours (26.9) | unchanged; the section-18.56 table prints both |
| `benchmarks/d3_layer_hits.py` | saving 0.63, rewind 0.7 | DS-D3-SAVE (borrowed) | unchanged; flagged. Under budget 0 the all-hit share is DS-MB0-HOSTLAYERS's 73.6 %, not the trace's 52.8 % |

## 8. Open questions this ledger exposes

1. **No DeepSeek constant has been measured at the default sysctl.** One floor arm (`mix.sh`-shaped, 48 GiB) would say whether DS-FLOOR-48 and DS-TOKEN-48 hold today.
2. ~~DS-FLOOR-SPLIT is stale~~ re-profiled in 0.60.9 (DS-FLOOR-SPLIT-0609). ~~The 70 ms against 75-77 gap~~ settled in 0.60.10: the server's all-resident token is 79-80 ms (DS-FLOOR-SRV), 70 was a fit intercept. ~~What limits the weight-streaming kernels~~ partly answered (MC-READ-SIZE): at the bytes a decode launch reads (~3 MiB an expert projection, ~11 MiB a shared-expert projection) a dense bf16 GEMV reaches 60-85 % of 819 GB/s, so launch size explains part of the gap; the rest (routed experts at ~29 %) is the 2-bit dequantizing kernel itself, not the memory system. Doubling memory bandwidth would therefore move those kernels far less than 2x.
3. **MiniMax has no constants measured since its substitution defaults.** GLM's X10Pro token is measured since 0.61.1 (GLM-TOKEN-USB-M).
4. **DS-PREFILL-SHORT (25-45 tok/s on 200-400 tokens)** is a measured cost with no explanation.
5. **Queue depth (ST-X10) and read latency distributions** are recorded as part of a wall, never swept.
6. **Prefetch depth against storage bandwidth** (DS-PRED-USB, DS-PRED-AB, GLM-PRED-USB): prediction is a ~1 ms win on the internal SSD and an 11-18 % loss at 1 GB/s for DeepSeek (precision 35-40 %); GLM's K = 5 at 72 % precision roughly breaks even at 1 GB/s (-4.4 % off, inside drift; answered in 0.61.1). Where between the drives DeepSeek's turns is still unmeasured; a precision-gated prefetch needs a trace that records the predicted sets.
