# Policy, execution and instrumentation seams (track L7)

Written 2026-10-07 against runtime 0.61.10, as track L7 of `docs/RESEARCH-DIRECTION.md`. It is a map, not a refactor: it names the boundaries the code already has, where a measurement hook and a policy decision share a class, and which hook a new instrument should use. Nothing in `src/` changed. Line numbers were read from the source on the date above and drift; search by the symbol name.

Rule from the charter (L7): policy comparison happens offline in `cache_sim.py` first and live only for a winner; no new abstraction layers (charter section 8.2, "no premature abstraction"); where adding a measurement would force a policy edit, add the smallest hook.

## 1. The seams that exist

Ordered from the drive up to the request. "Knob" is the environment variable or constant that selects the behaviour.

| layer | seam (symbol, file) | what it is | policy or measurement | knob |
|---|---|---|---|---|
| drive | `ExpertReader.read_expert_into`, `storage/reader.py` | the one place an expert's bytes are read into a slot's buffers (pieces read concurrently, F_NOCACHE) | execution; the throttle inside it is an experiment control | `CACHALOT_READ_THROTTLE_GBPS` (`_throttle`) |
| drive | `CodedBankReader(ExpertReader)`, `minimax/coded_bank.py`; `reader_from_env` | MiniMax's slot-image bank over the internal drive plus the X10Pro mirror, with the adaptive mirror share | execution (placement of reads across drives) | `CACHALOT_MINIMAX_BANK`, `CACHALOT_MINIMAX_BANK_MIRROR`, `CACHALOT_MINIMAX_MIRROR_ADAPT` |
| slots | `ExpertSlotPool`, `SlabSlotPool`, `cache/slots.py` | where an expert's bytes live on the GPU; the slab pool keeps a slot table for GPU-side selection | execution | slab size is a runtime parameter, not an env knob |
| store | `ResidentExpertStore._read_into`, `cache/resident_store.py` | the single chokepoint every read passes through; eight submission sites (predict pool, decode and prefill loads, prefetch, warm-up) all call it | **instrumentation**: counters (`ssd_bytes_read`, `ssd_read_seconds`, `decode_wait_seconds`) update around it | none |
| store | `_evict_lru_locked`, `_evict_slru_locked`, `_admit_reserved_locked` | eviction and admission | **policy** | `CACHALOT_EVICT` (lru, lfu, slru), `CACHALOT_EVICT_SAMPLE`, `CACHALOT_SLRU_PROTECTED` |
| store | `set_capacity` | the actuator memory governors use to give slots back or take them again | execution | none |
| store | `get_many` (decode), `get_many_prefill`, `prefetch_decode` | the request paths: demand, prefill with per-layer quotas, predicted loads | policy and execution together (see section 2) | `max_misses` argument, below |
| store | `ResidentStoreStats` | the counters served at `/v1/stats` and printed on the `[request]` line | instrumentation | none |
| governors | `WiredGovernor.target` (pure logic), `cache/wired_governor.py`, applied by `TextDecodeRuntime._wired_fit` | DeepSeek's give-back above the system wired ceiling | policy (pure) over an execution actuator | `CACHALOT_WIRED_CEILING_GIB` |
| governors | `HostWatch`, `GlmModel._fit_memory`, `_fit_prefill`, `glm/model.py` | the GLM/MiniMax governor: GPU "Alloc system memory" and host pressure | policy, interleaved with reading the system state | `CACHALOT_HOST_GROW_QUIET_S`, `CACHALOT_HOST_SHRINK_EVERY_S` |
| predictors | DeepSeek: `moe_layer_metal.py` (`PREDICT_TOPK`, the block that routes layer L+1 on layer L's input and hands the result to the store) | what to read ahead | **policy** | `CACHALOT_PREDICT_TOPK`, `CACHALOT_PREDICT_AHEAD`, `CACHALOT_PREDICT_LEAD` |
| predictors | GLM: `StreamingSwitchGLU` in `glm/experts.py` (`PREDICT_TOPK`, `PREDICT_LIMIT`, `PREDICT_AFTER_DEMAND`), predictor closure at `glm/model.py` (`predict`) | same | policy | `CACHALOT_GLM_PREDICT_TOPK`, `_LIMIT`, `_AFTER_DEMAND` |
| predictors | MiniMax: `minimax/gpu_select.py` (`_predicted`, `PREFETCH_TOPK`) | reads the predicted layer from the sync the GPU-select loop already ran | policy | the `PREFETCH_*` constants in the file header |
| drop / substitution | DeepSeek: `decode_miss_budget` on the store, read in `moe_layer_metal.py` and parsed in `cli.py` | cap on misses a layer reads; the rest are dropped (outputs change) | **policy, output-changing** | `CACHALOT_DECODE_MISS_BUDGET` |
| drop / substitution | MiniMax: `MISS_DROP`, `PREFILL_MISS_DROP`, `MISS_SUB` in `minimax/gpu_select.py` | a missing expert under a share of its layer's weight is dropped and a resident one substituted | policy, output-changing, on by default | `CACHALOT_MINIMAX_MISS_DROP`, `_PREFILL_MISS_DROP`, `_MISS_SUB` |
| tracer | `RoutingTracer`, `metrics/routing_trace.py` (`record`, `record_predicted`, `mark`, `save`), installed by `TextDecodeRuntime.set_tracer`, which also attaches it to the store as `pred_tracer` | a columnar trace of routed experts, router weights and predicted sets | **instrumentation**, policy-free; the hot path pays one `None` check | `CACHALOT_ROUTING_TRACE` (in `cli.py`) |
| server | `Engine` (`server/engine.py`) | one model instance serves one request at a time; the lock is also what `/v1/stats` reads as `busy` | execution | none |
| offline | `benchmarks/simulate_policies.py` (`Store`), `cache_sim.py`, `pred_gate_price.py`, `batch_union.py`, `d3_layer_hits.py`, `d2_verify_union.py` | models of the store and of levers, replayed over a routing trace | policy comparison without a runtime | CLI flags |
| offline | `benchmarks/pareto.py` (`apply_decode_arm`, arms for env, budget, miss budget, prefill chunk) | quality as a measured axis for an output-changing arm | measurement | arm definitions |
| offline | `benchmarks/decode_anatomy.py`, the `profile_decode_*` family | read intervals against main-thread wait; GPU and sync profiles | measurement, built on the store counters and reader wrapping | CLI flags |

## 2. Where measurement and policy are interleaved (charter gap G8)

These are the places where a new measurement would currently force an edit to a policy file, or where a policy knob is read through an instrument's channel. Each is a fact about the code on this date.

1. **Counters live in the class that evicts and admits.** `ResidentStoreStats` updates inside `ResidentExpertStore` methods next to the eviction code, and `decode_wait_seconds` accumulates inside `get_many`. A new counter is a store edit. This is acceptable (the store is one class and the counters are plain fields) but it means "measure" and "decide" are not separable in the store.
2. **The tracer reaches the store by an attribute and `getattr`.** `set_tracer` writes `store.pred_tracer`, and `moe_layer_metal.py` reads it with `getattr(expert_store, "pred_tracer", None)`. The tracer is policy-free, but the channel is duck-typed rather than a declared field.
3. **The tracer was DeepSeek-only; since 0.62.0 GLM and MiniMax have an unweighted one** (`GlmModel.set_tracer`, `StreamingSwitchGLU._trace`, `GpuSelectDecoder._check`; `CACHALOT_ROUTING_TRACE`). Traces of both have been recorded since (HANDOFF 18.93-18.100); GLM's since 0.62.10 also carries router weights and the predictor's sets (18.102); MiniMax's carries neither, priced in 18.101 (sets free; GLM weights ~0.1 % of a token; MiniMax weights free under the default), not built.
4. **The throttle is inside the reader**, not around it. `CACHALOT_READ_THROTTLE_GBPS` is a global in `storage/reader.py` applied in `_read_expert_into`; it controls experiments by editing the execution path. It is off by default and priced against the real X10Pro [ST-THROTTLE-VALID]. It does not cover MiniMax's coded bank: `CodedBankReader.read_expert_into` serves a bank record through `_read_record` and reaches `ExpertReader` (and so `_throttle`) only for an expert that is not in the bank (`minimax/coded_bank.py`, `grep _throttle` finds nothing there). A throttled MiniMax sweep needed the call added first: **done in 0.61.14** (`CodedBankReader.read_expert_into` now calls `_throttle` for a bank record, test in `tests/test_reader_throttle.py`; HANDOFF 18.91).
5. **Each model has its own predictor and its own drop rule** with no shared interface (three files in section 1). Comparing predictor policy across models means three instruments, which is why `pred_gate_price.py` exists only for DeepSeek.
6. **MiniMax's `MISS_DROP_ARMED` makes the host copy router weights** only so an instrument can see them (`gpu_select.py`); an instrumented run therefore changes the execution path it measures. The charter rule that an instrumented run is never a speed baseline applies here.
7. **Governors read the system and act on the store in one method** (`_wired_fit`, `_fit_memory`). `WiredGovernor.target` is the exception: pure logic, testable without a machine.

## 3. Where the next instrument should attach

| instrument | attach at | why this seam | what to avoid |
|---|---|---|---|
| per-token critical-path trace (charter section 4.3, L2) | `_read_into` entry and exit (one chokepoint, eight submitters), plus each model's router sync (three separate sites) | the read chokepoint is single; per-token reads, queueing and exposed wait can be recorded there with `perf_counter` | any `mx.eval` or MLX op after the sync (a second GPU round trip, ~21 ms a token on MiniMax; `mx.eval` inside `mx.compile`); a tracer that decides anything |
| routing trace for MiniMax and GLM (built in 0.62.0, unweighted) | `RoutingTracer.record_next`, called from `StreamingSwitchGLU` and `GpuSelectDecoder._check` | unlocks `cache_sim.py` and the L6 skew comparison | MiniMax's weight copy (item 6) must be priced before it is on during a speed arm |
| a precision-gated or bandwidth-aware prefetch | the three predictor sites in section 1, behind each model's existing top-K knob | the knob already exists on each model | building it before the offline price: DS-PRED-GATE found no gate beats prediction off on exact outputs |
| a new eviction or admission policy | prototype in `simulate_policies.py` `Store`, then `_evict_*_locked` behind `CACHALOT_EVICT` | offline first (charter L7); the live policy is a closed set of three | a live-only comparison: drift is +-10 % [MC-DRIFT] |
| per-request manifest | `benchmarks/run_manifest.py` `collect`, not the store | machine and config state belong to the benchmark, not to the runtime | adding fields to `/v1/stats` (each triggers a Lab brief, `RELEASE.md` item 7) |
| a throttle for the MiniMax path | `CodedBankReader.read_expert_into` (calls `_throttle` since 0.61.14) | one call, off by default, same knob | changing the bank format |

## 4. What this map does not claim

- It was written from the symbols and file headers, with a search of the code for each name; it does not describe every call path. Before building on a seam, read the current function and run its tests (`tests/` has `test_resident_store.py`, `test_routing_trace.py` and `test_cache_sim_segments.py`).
- It proposes no refactor. A boundary is only worth formalizing when a second user of it appears; section 2 items 2 and 5 are the candidates, and neither has a second user yet.
- Codebase-memory's graph was ready for this session but was used only for orientation; every line number above was read from source.
