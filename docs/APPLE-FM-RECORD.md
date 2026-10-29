# Apple Foundation Models as an auxiliary engine: experiment record (predictions before any benchmark)

Written 2026-10-29 against runtime 0.68.5 plus the uncommitted `src/cachalot/auxiliary/` package, **after the
package and its tests were built and before any benchmark in this file was run**. Charter section 7.2 fields 1 to 3
(baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and the
results go in `docs/APPLE-FM-RESULTS.md`. A prediction edited after its run is void. Hamed asked for the work on
2026-10-29 (the `/ca` mission "Integrate Apple Foundation Models into Cachalot and establish a heterogeneous
inference optimization framework").

## 1. Baseline (what is known before the runs)

* **Hamed's own benchmark of the system model** (before this session, Instruments): 42.64 tok/s generation, 0.68 s
  time to first token, asset identifiers indicating an `instruct_3b` sparse variant, 1,562 Neural Engine prediction
  events in the run and none in an idle control, 76.5 % non-overlapping ANE coverage of a 14.0 s activity window, no
  Metal GPU events attributed to the Foundation Models processes. Measured; it does not establish exclusive ANE
  execution or independence from GPU memory contention.
* **This session's probes** (2026-10-29, `apple-fm-sdk` 0.2.1, macOS 27.2, before the package existed; measured,
  single samples, not benchmarks): context size 8,192 tokens; short greedy reply 0.52 s; first `token_count` call
  0.77 s (cold); structured (JSON schema) classification 0.72 s; streamed count to 20: first snapshot 0.62 s, total
  0.83 s; prompt processing 1,149 / 1,226 / 817 tok/s at 802 / 3,202 / 6,402 prompt tokens with 2 output tokens; two
  sessions submitted in parallel took 3.56 s against 1.57 s for one alone (the system model serializes requests);
  a timeout at 0.5 s returned in 0.51 s and the session answered the next request; a 12,802-token prompt failed after
  4.28 s as a generic `GenerationError` (status 255), not `ExceededContextWindowSizeError`; the model's own
  `found` flags in an extraction schema were false for values it had copied correctly (three schema shapes).
* **Target-model constants the cost model reads** (dated, from the repo): DeepSeek V4.1 Flash cold prefill 99 tok/s
  (HANDOFF 18.64, 0.58.2, a 22.5k system block), server decode floor 80 ms a token at 48 GiB (0.60.10), 9.8-10.7
  tok/s at 22-27k context in a live Hermes session (0.61.2); GLM-5.3 prefill ~90 tok/s (0.18.0); a K-position
  verify forward through the only multi-position path costs 243 ms + 27 ms per extra position (HANDOFF section 9.20).

## 2. Hypotheses

* **H-aux:** the system model is a usable auxiliary for short structured tasks: sub-second classification, faithful
  extraction once grounding is enforced, compression that keeps every critical fact it is given.
* **H-cost:** compression pays on a cold target prefill only when the kept share f satisfies
  (1 - f) / R_target > 1 / R_aux_prefill + f / R_aux_decode, f below ~0.27 for DeepSeek (derived from the
  constants above); the auxiliary's 24 ms output token, not its prompt processing, is the expensive side.
* **H-hetero:** the system model runs on the Neural Engine, so it competes with an MLX GPU workload only through
  unified-memory bandwidth and the shared power budget, which slows a bandwidth-bound GPU stream by a few percent.

## 3. Instruments and workloads

`benchmarks/apple_fm_bench.py` (new), arms sequential, fixtures in `benchmarks/apple_fm/` pinned by content hash in
every result row. Workload classes (charter 7.1): W1 (short interactive: overhead, classification, extraction), W3
(prefill-heavy, cold: the compression cost model), W7 (concurrent requests: the contention arms). No target model is
loaded by any arm in this record; the GPU stand-in for W7 is an MLX kernel loop, not Cachalot's runtime.

| arm | what it measures |
|---|---|
| `overhead` | per trial, warm: short greedy generate (inline and through the worker child), streamed first delta, a 300-token generation's rate, a 6k-token prompt's rate, `token_count` |
| `classify` | 40 labelled requests (route: deterministic / auxiliary / target), model and heuristic |
| `extract` | 12 texts, 48 labelled fields (18 absent from their text), model and heuristic |
| `compress` | 3 documents with listed critical facts and constraints, model and heuristic, DeepSeek-tokenizer ratios |
| `memory` | memory before, during and after generation: `vm_stat`, the system service's footprint where readable |
| `contention` | an MLX GPU GEMV (bandwidth-bound) and GEMM (compute-bound) loop alone, the system model alone, both at once; ABBA |

## 4. Predictions (registered before any run)

All **predicted**; probabilities mine. Medians over at least 5 trials unless stated.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| AF-1 | warm short greedy generate, inline | median 0.3 to 0.9 s (0.75) | median above 1.5 s |
| AF-2 | worker-child overhead | the child adds at most 30 ms to the AF-1 median (0.70) | adds more than 100 ms |
| AF-3 | first streamed delta, short prompt | median 0.2 to 0.8 s (0.70) | above 1.2 s |
| AF-4 | generation rate, 300-token reply | 35 to 55 tok/s, counted with `token_count` (0.70) | below 25 or above 70 tok/s |
| AF-5 | prompt processing, ~6k tokens | 700 to 1,300 tok/s (0.70) | below 500 tok/s |
| AF-6 | warm `token_count` | median at most 50 ms (0.55) | median above 200 ms |
| AF-7 | classification route accuracy | model 0.70 to 0.90, heuristic 0.60 to 0.85 (0.55) | model below the heuristic by more than 0.10 |
| AF-8 | false routing (gold target, routed elsewhere) | model at most 0.10 of gold-target items (0.60) | above 0.20 |
| AF-9 | extraction field accuracy (absent fields count as correct when reported missing) | model 0.80 to 0.95, heuristic 0.30 to 0.60 (0.55) | model below 0.70 |
| AF-10 | invented values that pass grounding | at most 1 of 48 fields (0.65) | 3 or more |
| AF-11 | compression critical-fact retention | model 0.70 to 0.90 (0.50); heuristic at least 0.85 (0.60) | model below 0.60 |
| AF-12 | compression constraint retention | 1.00 for both (verbatim by construction; 0.95) | below 1.00 (a defect) |
| AF-13 | compression ratio, DeepSeek tokens | model 0.25 to 0.50 (0.50); heuristic 0.40 to 0.80 (0.55) | model above 0.60 (every document rejected) |
| AF-14 | compression wall time | 3 to 8 s per 1,000 input tokens (0.55) | above 12 s |
| AF-15 | net seconds against DeepSeek's cold prefill (derived from AF-13, AF-14 and 99 tok/s) | between -3 and +4 s per 1,000 input tokens (0.55) | above +6 (compression clearly pays) or below -6 |
| AF-16 | memory: system service growth while generating | 0.5 to 4 GiB (0.40) | above 6 GiB |
| AF-17 | GPU GEMV bandwidth beside system-model generation | slowed 1 to 8 % (0.50) | slowed more than 15 % |
| AF-18 | GPU GEMM throughput beside system-model generation | slowed at most 5 % (0.55) | slowed more than 10 % |
| AF-19 | system-model generation beside the GPU GEMV loop | slowed 5 to 20 % (0.45) | slowed more than 35 % or sped up |

## 5. What each outcome means

* **AF-7/AF-8 held:** the classifier is good enough to feed a routing experiment on real traffic; it is still not a
  routing default (Hamed's call, and a labelled set written by the agent is not traffic). **AF-8 falsified:** routing
  stays off and lever C closes on this model.
* **AF-15 held or below:** compression is not a prefill lever for DeepSeek's cold path on this model; its use is the
  KV footprint of long agent contexts or Hermes's own summary calls (a separate experiment). **Above +6:** a target
  arm (compressed against original context, answer accuracy) is worth Hamed's machine time.
* **AF-17 held:** the ANE is a nearly free side engine for a bandwidth-bound decode, and lever E's next step is a
  real DeepSeek decode arm beside it. **Falsified:** auxiliary work must be scheduled outside decode (between turns).
* Nothing here is a statement about Cachalot's real decode, another Mac, another macOS or SDK version, or a default.

## 6. Stop rules and cost

All arms are short (estimated total under 12 minutes of wall clock, no arm over 5 minutes), need no runtime process,
and run with the machine gate's one-runtime check clear. A defect found during a run is fixed and counted against the
prediction it touches. Target-model arms (DeepSeek or GLM beside the system model; compressed against original
context through the server) are **not** in this record: each is a long run that needs Hamed's go and an estimate first.
