# Cachalot research direction: an inference systems laboratory

Written 2026-10-05 against runtime 0.60.1, from Hamed's long-term direction prompt (twenty sections, reproduced in condensed form in section 2). This document is the charter that prompt asks for. It does not change any runtime default, any model priority, or any instruction that Hamed has already given. Where the new direction and an existing rule meet, section 8 says which wins and why, and section 9 lists what only Hamed can decide.

Revised the same day in a second pass against the full prompt. The second pass added what the first had left out: representative workload classes (prompt section 14), concurrency and batching (the mission list and `tokens/s = f(batch size)`), a written experiment record (prompt section 3), the GPU floor as a roofline so that "compute 2x" and "memory bandwidth 2x" can be asked (prompt section 7), cost efficiency, the request-level storage questions (worker count, queue depth, latency distribution), and the rule that the Mac is never presented as a datacenter equivalent (prompt section 4). Section numbers from the first pass are unchanged.

How to read the evidence in this file. Every number cited from the repository's history is labelled **recorded** and carries its HANDOFF section: it was measured by an earlier session, and this session did not reproduce it. Numbers computed here from recorded constants are labelled **derived**. Anything about what a change would do is a **hypothesis** or a **prediction**, and says what measurement would settle it. Nothing in this document is a new measurement.

## 1. The amended mission

Cachalot began as a runtime that makes very large open-weight models usable on one Mac. That work succeeded: three model families run (DeepSeek V4.1 Flash, MiniMax-M3, GLM-5.3-Flash), with a prefix cache, vision, an agent-grade server and a long record of levers that moved decode from roughly 3 to 9 tokens a second on the machine they run on.

The mission now becomes:

> Cachalot is a measurement-driven AI inference systems laboratory. Its purpose is to understand, measure, model, explain and improve the complete path by which an autoregressive language model produces a token, and the relationship among model architecture, compute, memory hierarchy, storage, scheduling, data movement, energy and end-to-end performance.

Three consequences follow, and they are the amendments that matter.

1. **The unit of result changes.** A result is no longer "tokens per second went from A to B". A result is a prediction that was made before the change, a measurement that was made after it, and an explanation of any gap between them. A speedup without a prediction is still shipped if it is correct, but it is recorded as engineering, not as research.
2. **The central question changes.** From "how fast can this model go" to "why does this token take N milliseconds, which dependencies are on its critical path, what is hidden by concurrency, what is exposed as latency, and what change to the whole system would help".
3. **Optimization stays.** Performance work is the experiment that tests whether the explanation is right. The direction is not permission to stop shipping speed, and it is not permission to rewrite the project.

The research cycle every non-trivial piece of work follows: **measure, explain, hypothesize, change, benchmark, validate, generalize.**

## 2. The direction in one page

The twenty sections of the prompt reduce to eleven commitments. Each is stated here as the rule Cachalot will follow, with the prompt section numbers in brackets.

| # | Commitment | Prompt sections |
|---|---|---|
| C1 | Preserve what exists. No rewrite. Read the repository and its negative results before proposing structure. | 1, 17 |
| C2 | Model the anatomy of one token as a dependency graph with overlap, not as a sum. Keep resource time and critical-path time apart. Make exposed latency, hidden latency, overlap, queueing, dependency delay and useful work against waiting first-class quantities. | 2, 8, 9 |
| C3 | Instrument before optimizing. Every meaningful change records baseline, hypothesis, intervention, result, explanation and generalization. | 3, 13 |
| C4 | Prefer curves to points. Sweep memory budget, storage bandwidth, prefetch depth, concurrency, context, quantization and architecture, and look for knees, saturation, thresholds and regressions. Record where the bottleneck moves after each win. | 4, 5, 6 |
| C5 | Build predictive models validated against measurements, then use them to answer hardware what-ifs. | 7 |
| C6 | Treat data movement and energy as costs equal to arithmetic. Report bytes moved per token, useful against wasted bytes, and joules per token where the platform can measure them honestly. | 8, 9, 11 |
| C7 | Treat model architecture as an experimental variable. State findings in architectural terms (expert size, access pattern, working set), not as a per-model percentage. | 10 |
| C8 | Be reproducible and honest. Record the configuration of every serious claim. Keep measured fact, derived result, hypothesis, prediction and speculation apart. Keep measurement separate from policy. | 12, 15, 16, 18 |
| C9 | Measure on representative workloads, not on one benchmark. A lever is judged on a fixed set of workload classes (section 7.1), and a win on one class that harms another is not a default. | 14 |
| C10 | Price resources, not only components. Express results as the value of a resource unit (milliseconds a token per GiB of fast memory, per GB/s of storage, per joule) so that the question "which improvement reduces total system cost" can eventually be answered. | mission, 7, 19 |
| C11 | The Mac is a laboratory for transferable concepts, never a stand-in for a datacenter GPU. Keep platform-neutral what is already neutral (manifest, scorecard, traces, the cost model) and build no speculative portability layers. | 4, 17 |

Sections 19 and 20 (the long-term questions and the vision) become the lab's standing question list, in section 6 of this file, and its definition of success: Cachalot can say why it is this fast, what prevents it from being faster, what resource would improve it, by about how much, at what cost, what bottleneck appears next, and that a predicted improvement, once built, measured what the model said.

## 3. What Cachalot already is: the baseline

The prompt's first rule is that the existing system is the baseline and its negative results are evidence. The repository is further along this path than its release notes suggest, because the work was done as engineering and never named as a method. This section is the inventory. It was read from the repository on 2026-10-05; the numbers are recorded, not reproduced.

### 3.1 Instruments that already exist

| Capability the direction asks for | What exists | Where |
|---|---|---|
| Cache hit and miss counts, bytes read, read seconds, reads in flight, decode wait | `ResidentStoreStats`: hits, misses, `ssd_bytes_read`, `ssd_read_seconds`, `reads`, `fast_reads` (page-cache share), `read_wall_seconds`, `read_busy_seconds`, `decode_wait_seconds`, `decode_waited_misses` | `src/cachalot/cache/resident_store.py`, served at `/v1/stats` |
| Per-request anatomy line | `[request] prompt= reused= prefill= completion= decode= (tok/s) miss/tok= hit= read= fast= mlx=active/peak/cache` | `src/cachalot/server/engine.py` |
| Decode split into expert wait and everything else, plus overlap | `decode_anatomy.py`: wall clock split into expert wait and rest; reads bucketed by worker pool (demand against speculative); the union of read intervals gives the share of decode with a read outstanding, and the ratio of summed read time to that union gives mean reads in flight | `benchmarks/decode_anatomy.py` |
| Per-layer main-thread timeline | `profile_decode_timeline.py` and the `profile_*` family (router wait, `get_many` wall, expert-op construction, inter-layer gap) | `benchmarks/` |
| Routing trace with router weights | `CACHALOT_ROUTING_TRACE`, a columnar `RoutingTracer` (phase, layer, position, experts, weights) | `src/cachalot/metrics/routing_trace.py` |
| A trace-driven predictive model | `cache_sim.py`: routing trace to misses per token to milliseconds per token, per memory budget and drop threshold, with `token = floor + misses x cost per miss`. Recorded: it reproduced the live 88 % hit and 7.9 tok/s at 48 GiB (HANDOFF 18.54) | `benchmarks/cache_sim.py` |
| Quality as a measured axis | `pareto.py`: arms over env, budget, miss budget and prefill chunk, paired NLL with a block-bootstrap interval, KL, top-1, a 21-task battery, a noise arm that sets the band | `benchmarks/pareto.py` |
| Controlled measurement hygiene | `guarded_run.sh`, `settle.sh`, `TF_ALTERNATE` token-level alternation with swapped pairs, `slow_window_sampler.py`, `memwatch.sh`, the cold-text rule for prefill A/B | `benchmarks/`, `MEASUREMENT.md` |
| Memory ceiling awareness | `wired_governor.py` and the MiniMax governor read the driver's "Alloc system memory" and the system's wired memory | `src/cachalot/cache/`, `src/cachalot/glm/` |
| Reproducing a live session offline | `CACHALOT_SERVER_DUMP` request bodies plus replay instruments | HANDOFF 18.24, 18.28 |

### 3.2 Results that already have the shape the direction wants

These are the existing evidence that the research cycle works here. Each is recorded, with its section.

- **A curve with a knee, found.** DeepSeek at a 52 GiB expert budget decoded 4.8 tok/s because the GPU's system allocation crossed the wired limit; at 48 GiB it is 7.9 to 8.3 (+66 %). A constant model came out of it: `token = 70 ms + 2.0 ms x misses` at 48 GiB or less (HANDOFF 18.52). The edge was then traced to the system's wired memory, about 74.5 GiB, and a governor was built from that explanation (18.53). That is a measured threshold, a mechanism, and an intervention validated against it.
- **A decomposition of a token.** MiniMax at 68 GiB: 115.6 ms = 72.8 ms of read wait (23 misses of 21.1 MiB at about 6.5 GiB/s) plus 42.8 ms of everything else (18.19). A later correction showed the all-hit token is 44 to 49 ms through the server, not 77 (18.54, 0.55.0): an earlier fitted intercept had been mistaken for a floor. That correction is itself a lesson for the lab (section 7).
- **Bottleneck migration, observed.** MiniMax decode went from per-layer GPU round trips (0.20.0), to reads (0.23.0 to 0.35.0), to the memory ceiling (0.36.0, 0.37.0), to the drive wall (0.38.0). DeepSeek went from SSD bandwidth to hit rate to the GPU allocation cliff to the 70 ms floor. The record narrates this; it has not been written as a migration table (section 5, track L1).
- **Wasted data movement, measured but not first-class.** Speculative expert loads: about two thirds of predicted loads are never used (recorded precision 32.6 % to 48 %, 26 to 28 wasted loads per token in several sessions, HANDOFF sections near lines 2010 to 2632 and 18.35). The prefetch is still a net win because the drive would otherwise idle. That is exactly the useful-against-wasted-bytes trade the direction asks Cachalot to price, and the repository has the counters but no standing report.
- **Hidden latency, measured.** `decode_anatomy.py`'s read-interval union and reads-in-flight, and the MiniMax finding that hit experts run on the GPU while misses are read, are overlap measurements. They live in benchmarks, not in the server.
- **Negative results with mechanisms.** The "What is closed" list in `docs/NEXT-SESSION-PROMPT.md` is a research record: entropy-coded weights lose (GPU decode 79 against 14 microseconds a matmul), a wider mirror hits the USB wall, speculative decoding is arithmetic-closed (a verify block reads K tokens of misses, so only the floor amortizes), miss substitution lost to the plain drop at equal speed (ten times the dNLL), GLM prefetch beyond K=5 gains nothing. They are the C1 evidence and must be kept in the new ledger, not discarded.
- **The floor, already partly a roofline.** DeepSeek's all-resident floor in the internal-drive era, 77 to 80 ms, was decomposed: the FP8 trunk GEMV family 14.0 ms moving 5.14 GB a token at about 367 GB/s, routing prediction 11 ms, 40 per-layer `mx.eval` round trips 9 to 12 ms, routed experts 6.9 ms, and smaller terms; 56 ms inside `mx.eval` and 21.5 ms of CPU outside it. Estimated whole-token traffic, about 7.5 GB, streams at about 98 GB/s, 12 % of the M3 Ultra's 819 GB/s (recorded and estimated, `SPEED-RESEARCH-2026-10-03.md` section 1, README "Where a token's time goes"). This is the evidence base for the "memory bandwidth 2x" and "compute 2x" what-ifs: it already says the floor is neither bandwidth-bound nor compute-bound at the chip level but dependency- and synchronization-bound, which is a hypothesis L4(f) must test.
- **Architecture as a variable, accidentally.** The same ideas were run on three expert sizes and counts (DeepSeek 9.49 MiB, 384 experts a layer, top-6; GLM 288 experts, top-8; MiniMax 21.1 MiB slot images). The mirror drive result "a property of the expert size, an 18 % loss at 9.49 MiB where it was a gain at 17.93" (README roadmap item 6) is already a statement in architectural terms.

### 3.3 Gaps against the direction

Honest list. These are what the direction asks for and the repository does not have.

| Gap | Today | Consequence |
|---|---|---|
| **G1 No per-token critical-path trace in the server.** | Aggregates (`decode_wait_seconds`, hit rate, `read=`) and benchmark-only timelines. No per-token record of what ran when, on which resource, and what it waited for. | Exposed against hidden latency exists only inside `decode_anatomy.py`, only for DeepSeek's reader, only when a benchmark is run. Nothing can yet explain one slow token in a live session. |
| **G2 No energy or power measurement of any kind.** | A search finds only two passing mentions of `sudo powermetrics` in HANDOFF, as a thing to try. | Joules per token, tokens per joule and energy-aware verdicts on levers are impossible today. `powermetrics` needs root; Claude never runs sudo (section 8). |
| **G3 No automatic run manifest.** | Each benchmark records its own flags; machine state (sysctl, pressure, swap, display, Flurry, other GPU users) is checked by hand and written into HANDOFF prose. | Reproducibility (prompt section 15) depends on the author's discipline. Two results cannot be compared by a program. |
| **G4 No efficiency scorecard and no common schema.** | Each instrument prints its own columns. `/v1/stats` and `[request]` are shaped for Hamed's reading and Studio, not for cross-run comparison. | No p50/p95/p99 per token, no bytes per token ledger across layers of the hierarchy, no normalized cross-architecture metric. |
| **G5 No standing sweep library.** | Curves exist where a session needed one (budget 36/44/48/52 GiB, GLM 44/48/50, MiniMax budgets 56 to 72). They are prose and ad hoc JSON. | A new session cannot ask "what is the budget curve for model X" and read an answer; it must find the HANDOFF paragraph. |
| **G6 The predictive model covers one term.** | `cache_sim.py` predicts misses and, with two constants, milliseconds per token for DeepSeek. It does not model overlap, the GPU floor as a function of context, the wired-memory cliff, prefetch precision, or the other two models. | The what-if questions of prompt section 7 (2x bandwidth, 32 GB more memory, perfect prefetch) cannot be answered with an error bar. |
| **G7 Storage is characterized by bandwidth plus a miss-cost constant.** | Access size is fixed by the bank format (one record per expert), queue depth is 1 on USB, latency distributions are not recorded. | Prompt section 8 (separate bandwidth, latency, granularity, queue depth, sequential against random) is only partly answerable. Several recorded closures (mirror striping, wider mirror) were priced on constants that have since changed (the SPEED-RESEARCH document says so about its own DeepSeek closures). |
| **G8 Measurement and policy are interleaved in code.** | Instrumentation hooks (`_read_into` wrapping, counters in the store) live next to the cache and prefetch policy. The routing trace and `cache_sim.py` are the clean exception. | Comparing eviction, prefetch or residency policies without touching telemetry is possible offline (`cache_sim.py`, `simulate_policies.py`) but not live. |
| **G9 No cross-model comparison table.** | The three models have separate histories and constants. | Prompt section 10's architecture-level conclusions cannot be drawn without assembling the numbers by hand. |
| **G10 Constants go stale silently.** | The 2026-10-03 research found that DeepSeek's closures were priced at the internal-drive miss cost after the bank had moved to USB, and the 77 ms MiniMax "floor" was a fit intercept. | The ledger needs a date, a regime tag (drive, budget, sysctl, version) and a "reproduced by" field on every constant. |
| **G11 The server is single-flight.** | `src/cachalot/server/engine.py`: "One model instance serves one request at a time"; requests queue on a lock. Prefill is batched over positions; decode is one token of one request. | Batch size, concurrent requests and throughput against latency cannot be measured live. `tokens/s = f(batch size)` is answerable only offline, from routing traces, until a batched decode exists. The recorded speculative-decoding closure (a K-position verify reads K tokens of misses, only the floor amortizes) is the same arithmetic a batch would face and must be re-priced with expert overlap across streams, not assumed. |
| **G12 Predictions are not on file.** | Levers are often "priced" before building, but the prediction lives in session prose and is rarely set beside the result. | The governing question (was the improvement predicted, and did the measurement agree) cannot be audited. Section 7.2 gives the record. |
| **G13 No fixed workload set.** | Each instrument carries its own prompts: `TF_ALTERNATE` decode text, `FILLER_FILE` cold prefill text, the 21-task Pareto battery, Hermes dumps, topic-shift runs. | A lever can be judged on the workload that flatters it. Section 7.1 names the classes. |
| **G14 The floor is not modeled against the chip.** | The floor decomposition exists (section 3.2) but no model maps it to memory bandwidth, GPU compute, CPU time and synchronization count. | "Compute 2x" and "memory bandwidth 2x" what-ifs have no basis. L4(f). |

## 4. The anatomy of one token

This section is the design target for G1. It is a specification of what to measure, not a plan to build it in one change.

### 4.1 Activities

A decode token on any of the three models passes through the following activities. They are not sequential; the dependency edges, not the list order, define the critical path. The list is the prompt's, mapped to what the code has.

Input and token preparation; embedding (and for DeepSeek, the Engram n-gram row reads, which the record shows cost about 3.9 ms of USB reads when not prefetched); attention with KV update and read (sliding window and compressed sparse for DeepSeek, GQA for MiniMax, linear-attention layers for GLM); the router and expert selection; expert lookup in the resident store; cache hit or miss; for a miss, the storage request, its queueing, the transfer and the materialization into a slot (which for MiniMax includes the pair-table indirection); prefetch of the next layer's likely experts; per-layer synchronization (the `mx.eval` the router forces, recorded as 9 to 12 ms across 40 layers on DeepSeek); shared and dense computation; routed-expert computation; output projection; sampling; framework and Python overhead; idle and wait periods.

### 4.2 Resource time against critical-path time

Every activity gets two numbers per token:

- **resource time**: how long a resource (drive, GPU, CPU thread) was busy with it;
- **critical-path time**: how much of the token's wall clock it added, meaning the time during which nothing else useful was making progress because of it.

The relationship the direction states, `exposed_io = max(0, io_time - useful_overlap)`, is a hypothesis about the structure of the dependency graph, not a law. The existing `decode_anatomy.py` already computes the pieces for one reader: summed read time, the union of read intervals, and the main thread's blocked time. A per-token trace extends that to the whole graph.

### 4.3 Minimum viable token trace (specification)

Per decoded token, per layer where it applies, record monotonic timestamps for: router sync start and end; each demand read submitted, started, completed (with expert id, bytes, worker pool); each speculative read submitted, completed, and whether the expert was used by the next layer; kernel dispatch and the next sync; main-thread blocked intervals with the reason. From these a post-processor derives, for each token: wall time, expert wait (critical), expert read busy time (resource), hidden read time, GPU busy time, reads in flight over time, queueing delay (submit to start), and wasted speculative bytes.

Constraints from the repository's own scars, which any implementation must respect (they are in `SKILL.md` section 4):

- nothing that calls `mx.eval()` may run inside an `mx.compile` trace, and any MLX op after an `mx.eval` just to read a host value costs a second GPU round trip (about 21 ms a token on MiniMax). The tracer therefore reads timestamps with `perf_counter` only and reads host values from arrays the existing sync already evaluated;
- the hot path pays one attribute check when no tracer is installed (the existing `RoutingTracer` pattern). A tracer must be measured against itself: its overhead is a measured quantity in the manifest, and an instrumented run is never used as the speed baseline;
- the trace is policy-free: it observes the store and the reader through the same narrow seams the policy already uses, and never decides anything (prompt section 16).

### 4.4 Definitions the lab uses

So that two sessions mean the same thing by the same word. All are per token unless stated, and all are derived from the trace in 4.3, never estimated from aggregates.

- **Resource time** of an activity: the time a resource (the drive, the GPU, one CPU thread) spent on it, summed over overlapping instances.
- **Critical path**: the longest chain of dependent activities from the token's start to its sampled id. Its length is the token's wall time when no idle gap exists.
- **Exposed latency** of an activity: the part of the critical path that the activity occupies while nothing on the path makes progress. For storage, the main thread's blocked time on a demand read.
- **Hidden latency**: resource time minus exposed latency. For storage, read time that ran under GPU work or under other reads.
- **Overlap**: the time two or more resources were busy at once, measured from interval unions (as `decode_anatomy.py` already does for reads).
- **Queueing delay**: submit to start of a request (a read waiting for a worker or for the drive).
- **Dependency delay**: time an activity is ready except for an input still being produced (a layer's experts waiting for its router output).
- **Synchronization delay**: time a thread waits on an explicit barrier (an `mx.eval`) beyond the work the barrier completes.
- **Useful work against waiting**: the fraction of wall time during which at least one resource does work whose result the token uses. Speculative reads that are never used count as waste, not useful work.

### 4.5 Prefill is part of the anatomy

The prompt's object is the generated token, but the user-visible cost of an agent turn is often the prefill (time to first token). The record shows prefill has different bottlenecks: MiniMax prefill at the GPU matmul peak (0.38.1), a cold DeepSeek prefill bound by drive reads (1,131 s for 12k tokens over USB, recorded with memory pressure, contaminated), and Hermes session starts dominated by re-prefill of system blocks (0.58.2, 0.59.0). The same vocabulary applies per chunk instead of per token, and the scorecard carries TTFT and reuse (tokens reused from the prefix cache or a snapshot) beside decode.

### 4.6 The floor as a roofline

For "compute 2x" and "memory bandwidth 2x" to have an answer, the all-resident floor must be split by what bounds each of its parts: bytes moved over the achieved bandwidth (with the chip's 819 GB/s as the ceiling and `mx.sum` over the same bytes as the practical one), arithmetic over the GPU's throughput, CPU work, and the number of synchronizations times their round-trip cost. The recorded decomposition (section 3.2) already suggests that synchronization and dependency structure, not bandwidth, dominate the floor; if so, doubling memory bandwidth would move the floor far less than half. That is a prediction, to be tested by L4(f), not a finding.

## 5. The research programme

Tracks are ordered by dependency, not by size. None of them requires a rewrite. Each says what it produces, what it needs from Hamed, and the stop rule.

### L0. Foundation: the run manifest and the scorecard schema (documentation and tooling, no runtime change)

*Produces:* a manifest written by every benchmark and by the server on request: commit and dirty flag; version; model, quantization and bank format; hardware and OS; storage device and path per component (trunk, expert bank, Engram, mirror); the expert budget and the wired limit; context length; prompt or workload identifier and a content hash; generation parameters; cache state (warm against cold, snapshot reuse, resident-set load); every `CACHALOT_*` variable in the environment; machine state (pressure level, swap, display state, screensaver, other GPU users); the instrument's own overhead. And a scorecard schema (section 7) that results are written into.

*Why first:* G3, G4 and G10 block everything else. A curve without a manifest is anecdote (prompt section 15).

*Needs Hamed:* nothing for the schema (the output format is question 3). A server-side manifest endpoint is a `/v1/stats` addition, which per `RELEASE.md` item 7 triggers a Studio brief.

*Stop rule:* the manifest is done when a result file from any instrument can be re-run from it by someone who has only the file and the repository.

### L1. The constants and bottleneck ledger (derived from existing records, no new measurement)

*Produces:* one table, per model and regime, of every recorded constant with its date, version, drive, budget, sysctl, method and "reproduced by" status: the DeepSeek 70 ms floor and 2.0 ms a miss; the MiniMax 44 to 49 ms floor and 21.1 MiB read at about 6.5 GiB/s; GLM's read-bound token; the USB wall at 1.0 GB/s with queue depth 1; the 6.8 GB/s internal drive; the three cliffs (DeepSeek 52 GiB GPU allocation, MiniMax about 63 GiB under the original GPU working set, MiniMax 70 GiB swap at the raised 86 GiB set); and a bottleneck-migration table (which constraint bound which version, what removed it, what bound next). Stale or contradicted constants are marked, with the section that contradicted them.

*Why second:* it is the cheapest way to meet prompt sections 1 and 6, it exposes which constants a predictive model may rely on, and it needs no runtime and no free machine.

*Needs Hamed:* nothing. It is documentation.

*Stop rule:* every constant used by `cache_sim.py` and by any what-if cites a ledger row.

### L2. The per-token critical-path trace (G1)

*Produces:* the specification in section 4.3, first for DeepSeek's reader (where `decode_anatomy.py` already holds the logic), then MiniMax, then GLM, in that order of Hamed's model priority.

*Gate before building:* a price. How large is the instrumentation overhead against the 70 ms DeepSeek floor? If a token trace costs more than the noise of the A/B method (`TF_ALTERNATE` resolves roughly a few percent), it is sampled, not continuous. Price it on paper and with one micro-benchmark before it touches `src/`.

*Needs Hamed:* an idle machine and no other runtime (section 8). Not available while another session is testing.

### L3. Sweeps as curves (G5)

*Produces:* a library of parameterized sweeps that write scorecard rows, so a curve is one command. First sweeps, ordered by how much existing machinery they reuse:

1. expert budget against hit rate, bytes per token, wait per token, memory pressure, tokens per second (DeepSeek and MiniMax; most of the data exists offline through `cache_sim.py`, so the first version is an offline curve validated at a few live points);
2. storage bandwidth against token latency, using the two real drives (internal 6.8 GB/s, X10Pro 1.0 GB/s) as two measured points and, if Hamed agrees, throttled reads as intermediate points (a hypothesis to validate, not an assumption: throttling changes latency distribution as well as bandwidth);
3. prefetch depth and top-K against exposed I/O and wasted bytes;
4. context length against decode floor and miss rate;
5. quantization (bank bit width) against bytes per token and quality, using the 2-bit, 3-bit and 4-bit banks that exist;
6. reader worker count and queue depth against achieved bandwidth, request latency distribution (p50, p95, p99 per read) and exposed stall, on each drive. The record has queue depth 1 as part of the USB wall; this sweep measures the claim instead of carrying it;
7. batch size and concurrent streams against bytes per token and token latency, **offline first**: the server is single-flight (G11), so the first version replays two or more recorded routing traces in lockstep through `cache_sim.py` and counts the union of experts a batched step would read. It predicts how much a batched decode would amortize misses; only if that prediction clears the cost of building a batched decode path does the question go to Hamed.

Every sweep is run on the workload classes it names from section 7.1, and its rows carry the manifest.

Each sweep reports where the curve saturates (the knee), not the best point. The 36/48/52 GiB DeepSeek result is the template: the interesting finding was the cliff, not the maximum.

*Needs Hamed:* machine time, serialized, one runtime at a time, and for sweeps that change output (miss budget) his existing quality-gate rules.

### L4. Predictive model and what-ifs (G6)

*Produces:* an analytical model, `token_latency = critical_path(compute, memory, storage, synchronization, scheduling)`, built in this order so that every addition is validated before the next is added: (a) the existing misses-times-cost term, extended by the ledger's regime tags; (b) the overlap term `exposed_io = max(0, io_time - useful_overlap)`, fitted from L2 traces and tested on held-out traces; (c) the memory-pressure cliff as an explicit region, not a smooth term; (d) prefetch precision as a parameter; (e) context-dependent floor; (f) the floor as a roofline (section 4.6), so that compute and memory bandwidth enter as parameters; (g) a value layer that turns the model's partial derivatives into milliseconds a token per GiB of fast memory, per GB/s of storage, per unit of GPU throughput and, after L5, per joule. The value layer reports physical resources, not prices: it can say what a resource would buy in latency, and stays silent on what it costs to acquire unless Hamed supplies a price.

*What-if interface:* a function that takes a validated model and a hypothetical (storage 2x, 5x, 20x, infinite; resident memory +32 GB; access latency halved; every selected expert resident; prefetch precision 100 %; compute 2x; memory bandwidth 2x) and returns a predicted token time with an interval and a statement of which assumptions it extrapolates beyond.

*Worked example, derived and unvalidated, to show the form and the discipline required.* Using only recorded MiniMax constants (HANDOFF 18.19, 18.54): 115.6 ms = 72.8 ms of read wait plus 42.8 ms other. If read wait scaled as the inverse of storage bandwidth and overlap were unchanged, doubling bandwidth gives about 36.4 + 42.8 = 79 ms (1.46x) and infinite bandwidth about 42.8 to 44 ms (2.6x to 2.7x), consistent with the recorded 44 to 49 ms all-hit floor. DeepSeek at 48 GiB: 70 + 2.0 x 25 misses = 120 ms; infinite bandwidth is bounded by the 70 ms floor, about 1.7x. The unstated assumptions are exactly what L2 and L3 must test: that miss cost is bandwidth-bound and not latency-bound (the record says USB queue depth 1 is part of the wall), that overlap does not shrink as reads speed up, that the floor does not rise when reads stop competing for the GPU (the record has "reads slow the GPU part of a token about 25 ms", HANDOFF 18.4), and that more bandwidth does not raise hit-rate-independent costs. So this number is a hypothesis, and the lab's first job on it is to try to break it.

*Needs Hamed:* nothing for the model; validation points need the machine.

### L5. Energy (G2)

*Produces:* a measured power trace per run and joules per token and tokens per joule, with the platform limits stated.

*Method options, in order of preference.* (1) `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000`, which reports CPU, GPU and ANE package power on Apple Silicon. It requires root. **Claude does not run sudo; Hamed runs the sampler in a terminal, or grants a narrowly scoped sudoers entry for exactly that command, and Claude reads the file it writes.** (2) Without root: coarse system-level estimates only, clearly labelled estimates. Memory and drive power are not directly reported by `powermetrics`; any figure for them is an estimate and must carry that label (prompt section 11: do not fabricate precision).

*First deliverable:* idle baseline, then one fixed decode workload with the sampler on, reporting watts by component and joules per token, plus the same for a read-dominated token and an all-resident token, so the energy cost of moving weights can be separated from the cost of computing with them.

*Needs Hamed:* running `powermetrics` in his terminal during each energy arm (decided 2026-10-05, section 9). The session writes the exact command, including the output file path, and reads the file afterwards.

### L6. Architecture comparison (G9)

*Produces:* the same scorecard for the three models, with the explanatory variables made explicit: expert size on disk and in a slot, experts per layer, top-k, bytes per token, working set, routing skew and reuse distance (from traces), KV growth per token and attention type, and the measured exposed I/O and hit rate. Conclusions are phrased as "workloads with approximately this expert size and access pattern benefit from this lever because...".

*Needs Hamed:* nothing beyond L0 to L3 data.

### L7. Policy, execution and instrumentation seams (G8)

*Produces:* not a refactor of the runtime. A written map of the seams that already exist (the reader's `_read_into`, the store's admission and eviction, the prefetch predictors, `RoutingTracer`, the Pareto arm definitions) and, where adding a measurement would otherwise force a policy edit, a minimal hook. Rule: policy comparison happens offline in `cache_sim.py` first and live only for a winner.

*Why last:* it is the most likely to turn into premature abstraction. Prompt section 17 says to preserve natural boundaries and not to build speculative ones.

## 6. Standing research questions, mapped to tracks

The prompt's section 19, with the track that will answer each. A question stays "open" until its track has produced a validated answer.

| Question | Track | Existing partial evidence (recorded) |
|---|---|---|
| Where does one token spend its time? | L2 | DeepSeek 70 ms floor plus misses; MiniMax 72.8 wait plus 42.8 (18.19, 18.54) |
| Where does one token move its data? | L0, L2 | DeepSeek about 7.5 GB a token (trunk 5.14 GB plus 240 experts at 9.95 MB), SPEED-RESEARCH section 1.1 (an estimate) |
| What is the minimum required working set? | L3, L4 | hit rate 89.9 % to 99 % table in SPEED-RESEARCH; budget curves |
| Which weights must stay resident? | L3, L6 | hotlist coverage, REAP-style masks (not built) |
| How predictable is expert reuse? | L2, L6 | prediction precision 32 to 48 % (DeepSeek), 59 % of misses found at top 8 (GLM), cross-token prediction 1 to 4 % precise (MiniMax M25, closed) |
| Value of one GB of fast memory? | L3 | about 1.7 % a GiB (GLM 44 to 50), about 2.6 points of decode hit 44 to 52 GiB (DeepSeek) |
| Value of one GB/s of storage? | L3, L4 | derived above; unvalidated |
| At what bandwidth does storage stop being the bottleneck? | L3, L4 | the floor sets it; derived crossover not yet computed against validated overlap |
| At what cache size does more stop helping? | L3 | the cliff found at 52 GiB is a different, harmful saturation |
| How much I/O can computation hide? | L2 | reads in flight and the union fraction in `decode_anatomy.py` (DeepSeek only) |
| What fraction of data movement is avoidable? | L2, L3 | wasted predicted loads about two thirds of speculative bytes |
| How much speculative prefetch is wasted? | L2 | same; a standing report is missing |
| Energy cost of moving weights? | L5 | none |
| Which architectures suit a memory hierarchy? | L6 | expert-size dependence of mirror striping |
| How does context length move the bottleneck? | L3 | MiniMax measured to 64k (decode 2.65 tok/s); attention grows with context (18.2) |
| How do quantizations move the bottleneck? | L3 | 2-bit against FP4 bank halves bytes a token |
| Which improvements raise throughput but hurt energy? | L5 | none |
| Which improvements reduce system cost, not a component benchmark? | L5, L4(g) | none |
| How does throughput scale with batch size and concurrent requests? | L3 item 7 | none live (single-flight server); the speculative-decoding closure is the nearest arithmetic |
| What would 2x compute or 2x memory bandwidth buy? | L4(f) | the floor decomposition; trunk GEMVs at about 45 % of the chip's bandwidth (section 3.2) |
| Does more read parallelism help, and where does queueing start? | L3 item 6 | USB queue depth 1 (recorded as part of the wall, not swept) |

## 7. The efficiency scorecard (schema for L0)

A scorecard row is one run's measured values plus its manifest id. Fields that are not measurable on a given run are absent, never zero.

- **Performance:** TTFT; prefill tokens a second; decode tokens a second; milliseconds a token; p50, p95, p99 of per-token latency where the run has enough tokens (below about 200 tokens a percentile is not reported).
- **Memory:** resident model memory; expert-cache memory; KV memory; MLX active, peak and cache memory (the server already prints these); GPU "Alloc system memory"; paging and pressure level.
- **Storage:** bytes a token; reads a token; mean read size; achieved bandwidth against the drive's recorded wall; hit rate; **exposed stall** (critical-path wait) and **hidden I/O** (read busy time minus exposed stall), once L2 exists; wasted speculative bytes and prefetch precision.
- **Compute:** CPU utilization (the record has about 73 % GPU wait and 27 % CPU for DeepSeek decode, and an all-resident CPU share of 21.5 ms); GPU busy time; synchronization waits.
- **Energy:** watts by component, joules a token, tokens a joule (L5, measured against estimated labelled per field).
- **Efficiency (derived):** tokens per GB moved, useful against physical bytes, compute against bytes moved. Normalized cross-architecture metrics are defined only after L6 shows which normalizations are meaningful.

Every row carries three tags: **measured**, **derived**, or **estimated**.

### 7.1 Workload classes

A lever is reported on the classes it can affect, and a default-on decision needs the default classes (marked *) not to regress. The instrument column names what already exists, so the classes are a naming of current practice plus two gaps, not new machinery.

| Class | What it stresses | Existing instrument or source |
|---|---|---|
| W1 short interactive decode * | decode floor and hit rate at small context | `TF_ALTERNATE` decode runs, `chat.sh` turns |
| W2 long-context decode | attention and KV growth, context-dependent floor | MiniMax runs to 64k (recorded 2.65 tok/s), long Hermes sessions |
| W3 prefill-heavy, cold * | TTFT, prefill read-ahead, GPU matmul peak | cold-text prefill with `FILLER_FILE` and `FILLER_OFFSET` per arm |
| W4 code generation * | output quality where quantization and drops show first | the Pareto 21-task battery; the C# replays that found GLM's corruption |
| W5 routing shift | expert locality broken mid-session, cache recovery | the 0.57.2 topic-shift runs |
| W6 repeated prefix, agent * | prefix cache, snapshots, system-block pins, side requests | Hermes server dumps (`CACHALOT_SERVER_DUMP`) replayed offline |
| W7 concurrent requests | queueing, batching, interference | **gap**: single-flight server (G11); Hermes's back-to-back side requests are the only live proxy |

Classes are fixed by name; their concrete prompts and seeds are pinned in the manifest (content hash), so a later session can re-run the same workload.

### 7.2 The experiment record

Every non-trivial lever, sweep or instrument gets one record, written in the HANDOFF section of the session that runs it. The first three fields are written **before** the build.

1. **Baseline.** The scorecard of the default arm on the named workload classes, with its manifest.
2. **Hypothesis and prediction.** Which resource or dependency is believed to limit, the predicted change in named scorecard fields with an interval, and the measurement that would falsify it.
3. **Intervention.** Exactly what changes, behind which knob, bit-identical or output-changing.
4. **Result.** The scorecard after, same workloads, same method; the gap to the prediction.
5. **Explanation.** Why the result occurred, and if it missed the prediction, which assumption failed.
6. **Generalization.** One of: model-specific, architecture-specific (state the property, for example expert size or top-k), hardware-specific, workload-specific, or a general systems principle; with the evidence for the level claimed.
7. **Bottleneck after.** What limits now, and the evidence.
8. **Kind.** Engineering (it works, unexplained) or research (predicted, measured, explained). Null and negative results use the same record and go to the ledger.

## 8. Rules, and how they combine with the existing ones

The new direction adds rules. It removes none. Where it meets an existing rule, the existing rule wins unless Hamed says otherwise, and the reason is given.

### 8.1 Rules added

1. **Price, then predict, then build.** Before a non-trivial lever, write the predicted effect (a number and an interval) and the measurement that would falsify it. This already happens on paper ("priced and closed"); the change is that the prediction is written down before the build and is compared with the result in the HANDOFF section. A result that misses its prediction is investigated, not rounded.
2. **Every performance claim carries a manifest** (L0) and a tag: measured, derived, hypothesis, prediction, speculation.
3. **After every shipped win, write the bottleneck that moved**: what was reduced, what binds now, and the evidence for the second claim. (Prompt section 6.)
4. **Constants carry dates and regimes.** A constant older than the last change to the drive, budget, sysctl or runtime path that it depends on is marked stale until reproduced. (G10.)
5. **Negative and null results are first-class.** They are kept in the ledger (L1) with the mechanism that closed them, and a closure is re-opened when its constants change, not before. The existing "Before calling a lever closed" rule in `MEASUREMENT.md` still applies.
6. **Say which kind of result it is.** Engineering (faster, no explanation), or research (predicted, measured, explained). Both ship; only the second counts toward the lab's success measure in section 10.
7. **Instrumented runs are not speed baselines.** The tracer's overhead is measured and reported.
8. **Express findings in architectural terms** wherever the data supports it. A per-model percentage goes in the CHANGELOG; the transferable statement goes in the ledger.
9. **Name the workload classes** (section 7.1) every result was measured on. A lever tuned on one class is checked on the default classes before it becomes a default; a class it was not measured on is reported as unmeasured, not as unaffected.
10. **Never present the Mac as a datacenter equivalent.** Unified memory, a USB or internal NVMe drive and one Apple GPU are not HBM, NVLink and a GPU server. Transferable statements name the concept (capacity, locality, overlap, working set) and the regime they were measured in.
11. **Energy figures say how they were obtained.** Measured by `powermetrics` (with the sampler's interval), or estimated (with the method). Memory and drive power are estimates on this platform.
12. **The record comes first for research claims.** A lever counted as research has its experiment record (section 7.2) with fields 1 to 3 dated before the build.

### 8.2 Existing rules that stay exactly as they are

- **Hamed's priority order and model priority.** "Hermes usage first, vision second, speed third" (set 2026-09-22) and "DeepSeek V4.1 Flash, then MiniMax-M3, then GLM-5.3-Flash" (2026-10-03) are unchanged by this document. Hamed decided on 2026-10-05 (section 9, question 1) that the laboratory is the method for every lane and not a lane ahead of Hermes: his stated priorities are about what he uses, and the method serves that use. Lab infrastructure is built inside the speed lane; a Hermes or vision session still follows the research cycle and uses the lab's instruments to explain what it finds.
- **Output-changing defaults are Hamed's call**, Jev included. The new research may find more levers that change outputs (the miss budget is one); each goes through the existing quality gate, and the default decision stays his.
- **One runtime process at a time; never start, benchmark or replay while Hamed's server or another session's test runs.** All of L2, L3 and L5 need the machine. Until it is free, L0, L1 and the offline parts of L3 and L4 proceed.
- **No sudo, ever.** L5's sampler is Hamed's to run, or his to authorize narrowly.
- **No sleeping the Mac or display; no `pmset displaysleepnow`.** Power sampling is done with the display state recorded in the manifest, not changed.
- **Never edit his Hermes config; never delete models, banks, snapshots or `docs/translations/` without his explicit yes; no destructive git on uncommitted work; never edit `src/` while a sweep runs.**
- **Release discipline** (`RELEASE.md`): a semantic bump in `pyproject.toml` and `src/cachalot/__init__.py`, CHANGELOG, README, HANDOFF, the prompt, a Studio brief for any new `/v1/stats` field or knob, memory, skill. A measurement-only session is a patch release.
- **No premature abstraction and no portability sacrifice** (prompt section 17): where a boundary exists (reader, store, tracer, Pareto arm), keep it; do not add platform layers.

### 8.3 Tension worth stating

The direction wants curves and many configurations; the machine allows one runtime at a time, a single run at an agent-sized context can take tens of minutes to hours (a cold 12k DeepSeek prefill over USB was recorded at 1,131 s), and display state, thermal state and Hermes's window shift decode by 13 to 30 % (HANDOFF slow-window notes). The lab therefore leans on **offline models validated at a few live points** (as `cache_sim.py` already does) rather than on dense live sweeps, and treats every live number as having a regime tag.

## 9. What only Hamed can decide

These are asked once, not assumed. The session proceeds with the autonomous items (L0 schema, L1 ledger, the offline parts of L3 and L4) regardless.

**Decided by Hamed on 2026-10-05:** question 1 is (a), the lab is the method for every lane and the lane order "Hermes, vision, speed" stands, with lab infrastructure built inside the speed lane. Question 2: Hamed runs `sudo powermetrics` himself during a benchmark arm; the session hands him the exact command and reads the file it writes. Questions 3 to 6 remain open.

1. **Where does the laboratory direction sit against "Hermes, vision, speed"?** The prompt calls itself "the governing principle for future Cachalot development", which reads as governing *how* all work is done; the existing order governs *what* is worked on. Options: (a) the lab is the method for every lane, the order of lanes is unchanged, and lab infrastructure (L0 to L2) is built inside the speed lane (recommended, because it keeps the tools Hamed uses daily first and still makes every session follow the research cycle); (b) the lab infrastructure becomes the top priority for a stretch of sessions, ahead of Hermes work; (c) the speed lane is replaced by the lab tracks. A concrete difference: under (a) a Hermes-first session that finds a slow turn uses the tracer to explain it; under (b) a session may be spent entirely on L0 and L2 with no user-visible change.
2. **Energy sampling:** will he run `sudo powermetrics` himself during a benchmark, authorize a sudoers entry for that single command, or defer L5?
3. **Output format of the scorecard and manifest:** JSON lines under `benchmarks/results/` (recommended, matches existing results), and whether Cachalot Studio should show them (a Studio brief exists for every runtime-visible change; a new `/v1/stats` manifest field would need one).
4. **Throttled-storage experiments:** may a sweep deliberately throttle reads (for bandwidth curves) on the live drives? This is read-only and reversible, but it is a change to how the runtime is exercised and it needs his machine.
5. **Release shape for this direction:** one patch release containing this charter, README direction section and prompt amendment (recommended), or fold it into the next measurement release. Not decided here because another session is working on the same files.
6. **Concurrency and batching:** after L3 item 7 prices batched decode offline, is building a batched or concurrent decode path in scope? It is the largest runtime change the direction implies, it touches the single-flight engine every Hermes session depends on, and nothing is built before the offline price exists.

## 10. Definition of success

Cachalot meets this direction when, for a given model and configuration, it can produce from its own instruments:

- the per-token anatomy (resource time and critical-path time, exposed and hidden I/O, wasted bytes), with the instrument's overhead stated;
- the curve of token latency against each major resource, with the knees marked;
- a predictive model that, given a hypothetical change, returns a number and an interval, and whose past predictions are on file next to what was later measured;
- joules a token (where Hamed enables the sampler);
- the next bottleneck after each shipped optimization;
- a comparison across the three architectures phrased in architectural terms;
- the value of a unit of each resource (fast memory, storage bandwidth, compute, memory bandwidth) in milliseconds a token, on named workload classes;
- a file of experiment records whose predictions were written before the build.

The governing question from the prompt becomes a release gate for any research claim: **was the improvement predicted, implemented, measured, and found to agree with the explanation?** A "yes" is a research result; a "no" is an equally recorded result with the investigation attached.

## 11. Sequencing, with the machine busy

| Order | Work | Needs the machine? | Needs Hamed? |
|---|---|---|---|
| 1 | L0: manifest and scorecard schema, as a written spec and a helper that benchmarks can call (instrument code, not runtime) | No | Format (question 3) |
| 2 | L1: constants and bottleneck ledger from the existing record; mark stale rows | No | No |
| 3 | L4(a): extend `cache_sim.py` to read ledger constants and emit scorecard rows; produce the offline budget curve with an interval | No (uses recorded traces) | No |
| 4 | L2 price: overhead of a per-token trace against the 70 ms floor, on paper and with a micro-benchmark | Briefly | An idle machine |
| 5 | L2 build for DeepSeek, then MiniMax, then GLM | Yes | An idle machine |
| 6 | L3 sweeps, live validation of the offline curves | Yes | Machine time |
| 7 | L5 energy | Yes | Hamed starts the sampler per arm |
| 8 | L6 comparison, L7 seam map | No | No |
| 9 | L3 item 7: offline batching price from paired routing traces | No (recorded traces) | Only to build, question 6 |
| 10 | L4(f): floor roofline from the recorded decomposition, then checked by one profiled run | Briefly | An idle machine for the check |

Items 1 to 3, 8, 9 and the paper half of 10 are the proposed next work for any session that cannot use the machine. The first session after this charter should start at item 1, and it must begin with the L1 ledger, because L0's schema needs to know which fields the constants actually have.

## 12. Honest limits of this document

- It is a charter and an inventory. It adds no measurement and changes no code.
- The inventory in section 3 was read from the repository and from HANDOFF headlines, `NEXT-SESSION-PROMPT.md`, `SPEED-RESEARCH-2026-10-03.md` and the source files named in the table. HANDOFF is about 1 MB; sections were read selectively, so the gap analysis may understate instruments that live in a section not read. Before any L-track starts, its first step is to grep the repository for an existing implementation of the thing it is about to build.
- The worked what-if in L4 is deliberately labelled a hypothesis and lists the assumptions most likely to break it. It must not be quoted as a Cachalot result.
- The prompt's example metrics (for example "tokens per joule") are goals. Where the platform cannot measure a quantity, the lab reports it as unavailable.
