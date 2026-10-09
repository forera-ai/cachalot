# Changelog

## 0.62.27 (2026-10-09)

HANDOFF "Start here (2026-10-09, 0.62.27)" and section 18.114. Measurement and one instrument; nothing in `src/` besides the version string changed.

### Added
- `benchmarks/trace_overhead.py` (+ `tests/test_trace_overhead.py`): prices a per-token critical-path tracer on a simulated DeepSeek token (40 layers, 8 read workers, ~331 events), tracer-off against tracer-on tokens alternated, with an off/off control.

### Measured
- Charter L2 price: a column-store tracer adds **0.17 ms a token** (95 % interval -0.02 to +0.38; 0.2 % of an 80 ms floor) and a tuple-list tracer 0.07 ms (-0.08 to +0.22), against the 1.0 ms stop rule; one event costs 151 ns (column) or 56 ns (tuple) alone, 0.5 us inside a contended token. The paper figure of ~100 events was too low (~340-400 are needed); the price clears either way. The in-situ price (the tracer in a real server token) is the next step.

## 0.62.26 (2026-10-09)

HANDOFF "Start here (2026-10-09, 0.62.26)" and section 18.113. Documentation and offline instruments; nothing in `src/` besides the version string changed.

### Added
- `benchmarks/whatif.py` (charter L4): a what-if calculator for decode token time with three small models (GLM, DeepSeek, MiniMax), a band equal to each model's largest held-out error and a note on what each hypothetical extrapolates beyond; `--validate` prints the recorded points. `tests/test_whatif.py` (6 tests).
- `benchmarks/experiment_index.py` and `docs/EXPERIMENTS.md`: an index of the section-18 experiment records from 18.75 (39 sections: 6 research, 9 engineering, 10 measurement, 1 pricing, 13 unlabelled).
- `docs/RESEARCH-DIRECTION.md` section 13: the status of every track on 2026-10-09, the success criteria scored, and an ordered plan.
- `docs/LEDGER.md`: GLM-MISS-COST, GLM-BUDGET-DECODE, GLM-CSHARP-FLAW, GLM-SITE-LOGPROB and a section 7b of what-if validation rows.

### Measured (derived from recorded arms)
- GLM's decode token is `92 + 14.70 x reads` ms, fitted on the two exact arms and checked on the four budget arms it was not fitted to (-0.2 to -2.5 %); 14.70 ms is 13.5 MiB over the X10Pro's 0.963 GB/s, which ST-X10-QD measured independently (0.967). DeepSeek's bandwidth curve fits its four emulated points and over-predicts the real X10Pro by 20 % (it reads the 0.967 GB/s plateau, the live run read ~1.15). MiniMax's model is not validated (+11.9 % on one older point).

## 0.62.25 (2026-10-09)

HANDOFF "Start here (2026-10-09, 0.62.25)" and section 18.112. Measurement and one instrument; nothing in `src/` besides the version string changed.

### Added
- `benchmarks/glm_site_probe.py`: teacher-forces a chosen position of a compile-panel reply and compares GLM's next-token distribution in prefill mode, in decode with no miss budget and in decode with budget 2 (`--site newline|else`, `--replies`, `--window`, `--forced`).

### Measured
- Why 15 of 24 compile-panel replies broke at the same place (`else`, newline, indentation, then a stray `delimiter`): GLM's own top candidate there is ` delimiter` (38-65 %), against 7-13 % for the style-consistent ` {`, in prefill mode and in exact decode; budget 2 moves the top candidate to ` un` in 4 of 6 contexts but the stray word keeps 4-60 %. Prefill mode gives the stray word more probability than decode (0.36 against 0.26), so it is neither the decode path nor the miss budget: it is the checkpoint, as §18.43-18.49 found. Right after `else` the next token is a newline with probability 0.9996 or more in every arm. No default changed; `serve-glm.sh` stays at budget 2.

## 0.62.24 (2026-10-09)

HANDOFF "Start here (2026-10-09, 0.62.24)" and section 18.111. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- GLM decode miss budget, the compile-level C# panel: a short C# task (a quote-aware CSV line splitter with 8 unit asserts), 12 replies an arm built with `dotnet build`. Nothing compiled in either arm (0/12 exact, 0/12 budget 2), so the panel cannot show whether the budget costs anything; decode 1.793 against 1.163 s a token (-35 %). 15 of 24 replies break at the same construct (`else if (c == delimiter)` written as `else` plus a stray `delimiter` line), in both arms, which points at GLM's own code corruption, not the budget. No default changed; `serve-glm.sh` stays at budget 2.

## 0.62.23 (2026-10-09)

HANDOFF "Start here (2026-10-09, 0.62.23)" and section 18.110. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- GLM decode miss budget, the larger blind C# panel: 12 replies an arm of one C# task (exact and budget 2, 46 GiB, 4.3 hours), graded blind reply by reply by one grader. Flawed 9/12 in both arms (Fisher p = 1.0) and severe garbling 3/12 in both; decode 1.794 against 1.184 s a token (-34 %). No difference in C# quality at this size (a gap below about 35 points is not excluded); GLM garbles three of four C# replies either way. No default changed; `serve-glm.sh` stays at budget 2.

## 0.62.22 (2026-10-09)

HANDOFF section 18.109 (addendum). Documentation only; nothing in `src/` besides the version string changed.

### Explained, not measured
- Why arm 3's budget-4 and budget-2 second passes re-prefilled while the exact arm's did not: GLM's in-memory prefix list (3 GiB, oldest used first) refreshes only the longest matching snapshot, so when a sampled reply re-renders identically the prompt-end twin ages out and a repeat then misses, and its extra prefill evicts more. Chance at temperature 0.7, not an effect of the budget; decode timings are unaffected. Snapshot sizes and the list per step are not yet measured. Benchmark replays with a second pass should check `prefill=` or raise `CACHALOT_GLM_PREFIX_GIB`.

## 0.62.21 (2026-10-09)

HANDOFF "Start here (2026-10-09, 0.62.21)" and section 18.109. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- GLM decode miss budget, a second free-running run of arm 2's replay (24 replies on dumped Hermes turns, 46 GiB beside Hamed's applications, 3.0 hours, exact then budget 4 then budget 2): decode 1.585 s a token exact, 1.467 with budget 4 (-7.4 %), 1.090 with budget 2 (-31.2 %), matching arm 2 (1.625 / 1.506 / 1.103); pooled with arm 2, -7.4 % and -31.7 %. Graded blind by one grader (partly blind): every tool call valid, and stories with a slip and flawed C# at the same count in every arm (1/2 each), so nothing visible follows the budget. No prediction was on file before the run, so it is a replicate. No default changed; `serve-glm.sh` stays at budget 2.

## 0.62.20 (2026-10-09)

HANDOFF "Start here (2026-10-09, 0.62.20)".

### Changed
- **`serve-glm.sh` defaults `CACHALOT_GLM_DECODE_MISS_BUDGET=2`** (Hamed's call, output-changing). GLM decode reads at most two non-resident experts per layer and drops the lightest further misses, rescaling the kept outputs. Measured (§18.107, §18.108): a token 1.103 s against 1.625 exact (-32 %); teacher-forced KL 0.024 pooled on three texts; no visible loss on tool calls and stories in 24 blind-graded replies (GLM's C# garbles in every arm). `CACHALOT_GLM_DECODE_MISS_BUDGET=off ./serve-glm.sh` restores the exact path, `=4` the conservative setting (-7 %). The module default in `experts.py` and the chat launcher are unchanged (off).

## 0.62.19 (2026-10-09)

HANDOFF "Start here (2026-10-09, 0.62.19)" and section 18.108. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- GLM decode miss budget, free-running: 24 replies on dumped Hermes turns at 46 GiB beside Hamed's applications, graded blind. Decode 1.625 s a token exact, 1.506 with budget 4 (-7 %), 1.103 with budget 2 (-32 %). No flawed tool call or story in any arm; C# flawed 2/2, 0/2, 1/2 (GLM's own garbling). Two replies per task per arm and one grader: it cannot see a small difference. No default changed.

## 0.62.18 (2026-10-08)

Metadata only; nothing in `src/` besides the version string changed.

### Changed
- README links the Cachalot Lab app at its new address, `https://github.com/forera-ai/cachalot-lab` (it was transferred to forera-ai too; verified with `git ls-remote`).

## 0.62.17 (2026-10-08)

Metadata only; nothing in `src/` besides the version string changed.

### Changed
- The `LICENSE` copyright line reads "Copyright (c) 2026 forera.ai and Cachalot contributors" (was Hamed Prooshani). The licence text (MIT) is unchanged. The `authors` field in `pyproject.toml` is unchanged.

## 0.62.16 (2026-10-08)

Repository move; documentation and metadata only, nothing in `src/` besides the version string changed.

### Changed
- The repository is now owned and maintained by the GitHub organisation **forera-ai**: `https://github.com/forera-ai/cachalot`. README (CI badge, clone URL, a maintainer line under the title, the License section), `pyproject.toml` (Homepage, Repository, Issues), `CONTRIBUTING.md` and the OpenRouter referer in `benchmarks/run_reference.py` point to the new address. The `LICENSE` copyright line and the `authors` field are unchanged. The Cachalot Lab repository (`prooshani/cachalot-lab`) is a separate project and is unchanged.

## 0.62.15 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.15)" and section 18.107. Measurement only; nothing in `src/` besides the version string changed.

### Added
- `benchmarks/glm_miss_budget_quality.py`: teacher-forced paired NLL and KL of the GLM decode miss budget against exact decode, with a determinism control and a prefill-mode reference for GLM's own spread. Picks the expert budget from host availability so it can run beside Hamed's applications.

### Measured
- Three texts (prose, Python, JSON), 99 forced tokens each, 46 GiB, X10Pro bank: budget 4 pools to dNLL -0.004 and KL 0.010 (token -8 to -12 %), budget 2 to dNLL -0.005 and KL 0.024 (token -33 %), budget 1 to KL 0.061 (-60 %), budget 0 to dNLL +0.122 and KL 0.218. GLM's own prefill-against-decode spread pools to KL 0.007. Budgets of 8 or more cannot drop an expert on top-8 routing. Prose tolerates drops worst. No default changed; free-running quality is unmeasured.

## 0.62.14 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.14)" and section 18.106.

### Changed
- **`serve-glm.sh` defaults `CACHALOT_GLM_PREDICT_TOPK=0`** (Hamed's call, 2026-10-08): GLM decode no longer prefetches the next layer's experts. Outputs are unchanged (the prefetch only changed which reads start early). Basis: two live pairs on the X10Pro, both on the off side (-6.5 % in 0.62.13, -3 to -4 % in 0.61.1), each inside drift. `CACHALOT_GLM_PREDICT_TOPK=5 ./serve-glm.sh` restores the old behaviour. The `experts.py` module default (5) is unchanged, so direct users of the library still get K = 5.

### Added
- **A GLM decode miss budget, off by default** (`CACHALOT_GLM_DECODE_MISS_BUDGET=N`, `GlmModel.set_decode_miss_budget`, `StreamingSwitchGLU.miss_budget`). Per decode layer at most N non-resident experts are read, highest router weight first; the rest are dropped and the kept experts' outputs are scaled by total / kept router weight (the DeepSeek rule of 0.57.0). Changes outputs; prefill is untouched. It taps the router gate (as the tracer does) so the weights ride the sync that already reads the indices. The startup line names the budget. `/v1/stats` of the GLM server gains `skipped_experts`.
- Tests: the budget through `StreamingSwitchGLU.__call__` with a stub store (order of drops, zeros for dropped experts, the rescale, exact path unchanged) and the new stats field. 579 pass.

### Not measured
- The budget's speed and quality are unmeasured: it is a knob, not a result. No default uses it.

## 0.62.13 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.13)" and section 18.105. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- **GLM decode prefetch off against the shipped K = 5, one live pair on the same Hermes replay** (`serve-glm.sh --expert-budget-gib 46`, bank on the X10Pro, untraced, six dumped requests at cap 400, run back to back, off first): decode **1.615 s a token with prefetch off against 1.727 with K = 5 (-6.5 %)**; per request, the four requests of 42 tokens or more are faster with it off by 4.3 / 9.3 / 3.4 / 9.3 %; misses a token are the same (104.7 on both arms' long requests). Together with 0.61.1's pair (-4.4 % mixed, -3.3 % repeats) both pairs favour off by 3-9 %; each is inside the machine's drift (up to 16 %) on its own, and this pair ran off first, so order is not balanced. Not a default change.
- The store counters of the K = 5 arm agree with the 0.62.12 simulator: 59.4 predicted loads a decode token at 67.6 % used, 19.3 unused (replay: 61.5, 68 %, 19.5). Bytes read over a whole run (prefill included) were 16 % higher with K = 5. The simulator's bytes-over-rate bound said +18 % on the token; the live token was +6.9 %: prefetch hides part of its extra reads.

## 0.62.12 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.12)" and section 18.104. Instrument and a fix to an offline instrument; nothing in `src/` besides the version string changed.

### Added
- `benchmarks/cache_sim.py` models the store's decode prefetch: `simulate(..., prefetch_gate=)` and `--prefetch-gates off,0,0.5 [--gbps N]`. A predicted expert that is not resident is read into a transient slot and never evicts a resident; it becomes resident only if its layer routes it. The table reports, per decode token, misses, demand reads, predicted loads, how many were used, wasted loads, total reads and a bytes-over-rate lower bound on a token. It works for any model's trace that carries predicted sets (DeepSeek since 0.61.3, GLM since 0.62.10): token boundaries come from the layer order, as in `predicted_used_mask`. It generalizes the DeepSeek-only `pred_gate_price.py` replay; that script stays.
- Three tests (`tests/test_cache_sim_segments.py`): requests that share a decode segment, the prefetch counters on a layers-3-4 trace, and the refusal of a trace without predicted sets.

### Fixed
- `cache_sim.py` grouped decode rows by position within a segment. Two requests with no prefill rows between them (the second request's prompt fully reused) share a segment and repeat positions, so their tokens were merged and their rows reordered. The only recorded trace affected is `pred-trace-0.61.3/smoke.trace.npz` (1,577 tokens counted of 2,407; 37.4 misses a token against the correct 23.2, which now agrees with `pred_gate_price.py` and the measured 23.5). Every other trace in `benchmarks/results` gives the same numbers before and after; no published number besides that one trace's changes.

### Measured (offline)
- DeepSeek (0.61.3 trace, 48 GiB): shipped prefetch 33.3 predicted loads a token at 38 % used, 43.7 reads against the measured 45.6; gate >= 0.2: 16.8 loads at 50 %, 31.6 reads.
- GLM (0.62.11 trace, 46 GiB, 14.16 MB experts): shipped prefetch 61.5 loads a token at 68 % used, 126.1 reads against 106.6 with prediction off (-15.5 % bytes; the live pair measured -11 % on other prompts); a weight gate >= 0.5 keeps 19.4 loads at 88 %, 108.9 reads; >= 0.6: 10.6 loads at 94 %, 107.2 reads. At 1 GB/s the bytes-over-rate bound is 1,509 ms off, 1,543 gate >= 0.5, 1,786 shipped: no gate beats off on exact outputs, the same answer as DeepSeek. Combined with a drop threshold tau 0.10, the shipped prefetch awaits its predicted loads (57.9 misses) and a >= 0.5 gate reaches 43.2 against 42.6 with prefetch off.

## 0.62.11 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.11)" and section 18.103. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- The first GLM trace with router weights and predicted sets (0.62.10's tracer, the Hermes replay at cap 400, 46 GiB with the bank, 793 decode tokens over 42 layers, 38.5 minutes). The new multi-token prefill branch and the real MoE block ran without error; row weights sum to 2.5 (GLM's routed scaling factor), 41 predictions a token (layers 3-43).
- **The predictor's set (top 5) overlaps the next layer's routed set 80.1 % of the time**; by rank 95.7 / 89.3 / 81.0 / 72.0 / 62.5 %. A gate on the predictor's own weight (raw, the top-5 of a sum-2.5 router): >= 0.4 keeps 66 % of predicted loads at 86.7 % precision, >= 0.5 keeps 41 % at 93.5 %, >= 0.6 keeps 24 % at 96.9 %. DeepSeek's equivalent was 72 % overlap and 66 % kept at 82 % precision at its 0.20: GLM's predictions are better.
- **A drop-threshold what-if, offline (`cache_sim.py --taus`, 46 GiB, 3,488 slots): misses a token 106.6 at tau 0 (live 105.6, hit 68.3 % both), 98.8 at tau 0.05 (dropped routing mass 0.8 % a layer), 42.6 at 0.10 (12.8 %), 14.7 at 0.15 (25.3 %), 7.6 at 0.20 (33.7 %).** Output-changing and with no quality measurement: this is how many reads such a rule would remove, not a result. The simulator's millisecond columns are DeepSeek's constants and are not GLM's.
- Tracing cost on the live server: decode 0.57-0.58 tok/s on the short replies against 0.58 untraced (same), 0.735 and 0.484 on the two long ones against 0.765 and 0.509 untraced (-4 %, -5 %); the 2,738-token tool result's prefill took 421 s against 354 and 369 s untraced (+14 to +19 %). One run each; run-to-run drift on this machine is of that size, so these bound the overhead rather than measure it.
## 0.62.10 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.10)" and section 18.102. Instrument only; off unless `CACHALOT_ROUTING_TRACE` is set; no output, speed default or numerics changed.

### Added
- A GLM routing trace (`CACHALOT_ROUTING_TRACE=<file> ./serve-glm.sh`) now also records each layer's **router weights** and, in decode, the predictor's **next-layer set with its weights** (the same files `pred_gate_table.py` and `cache_sim.py --taus` read). The weights reach `StreamingSwitchGLU` through a tapped gate (`tap_gate`, a subclass swap that only stores the gate's own output; bit-identical outputs checked on a real `MoEGate`; parameter tree unchanged) and ride in the sync that already reads the indices (priced in 0.62.9: ~43 us a layer in isolation, 0.1 % of a token). The predicted set is already a host array there, so it costs nothing. MiniMax's trace is unchanged (experts only).
- `predicted_used_mask` finds decode-token boundaries from the layer stopping to ascend instead of from layer 0, so it works for GLM and MiniMax (first MoE layer 3); DeepSeek traces give the same result (tests).
- `RoutingTracer.record_next(..., weights)` and `last_position`. Four tests, including one through the real `StreamingSwitchGLU.__call__` path with a stub store.

### Not done
- Not run on a model: it needs the machine and Hamed's yes (a recording of 25-35 minutes). No `tau` lever is built or proposed; one would be output-changing.
## 0.62.9 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.9)" and section 18.101. Pricing only: one instrument added, nothing in `src/` besides the version string changed.

### Added
- `benchmarks/router_price.py`: an isolated micro-benchmark of what adding a router's weights to the sync that already reads its indices costs (interleaved A/B, no model, about 10 s of GPU).

### Measured (priced, not built)
- What recording router weights and predicted next-layer sets would cost for GLM and MiniMax: GLM decode **predicted sets are free** (the predictor's indices and weights are already host arrays in `StreamingSwitchGLU.__call__`), GLM decode **weights cost about +43 us a layer in isolation (+1.8 ms of a ~1,500 ms token, 0.1 %)** and need a gate wrapper to reach the switch module; MiniMax decode **weights are free under the shipped default** (`MISS_DROP_ARMED` already copies them to the host inside the sync; an exact-picks trace needs `CACHALOT_MINIMAX_MISS_DROP_ARMED=1`; the isolated benchmark gives an upper bound of +353 us a layer, which overstates because the weights are used on the GPU anyway) and MiniMax predicted sets are free too (`_predicted` already reads the scores). `predicted_used_mask` assumes every token starts at layer 0, which is true for DeepSeek only (GLM and MiniMax start at layer 3), a small change.
- Recommendation: build it for GLM only (it is the read-bound model, 91 % of a token, where a precision-gated prefetch could matter); MiniMax's prediction is already ~90 % precise and its substitution already ships.
## 0.62.8 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.8)" and section 18.100. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- A longer GLM-5.3-Flash trace on the same Hermes conversation (six dumped requests of the 0.61.12 session, decode cap raised from 60 to 400, `serve-glm.sh` at 46 GiB with the bank): **748 decode tokens** over 42 layers, 3.3 times the last GLM trace, in 34.5 minutes (estimate 40, up to 50). The busiest 10 % of experts (29 of 288) carry 31.4 % of a layer's decode routes against 14.0 % for uniform routing: excess 17.3 points (layer bootstrap 15.8-18.9). The two earlier GLM traces gave 20.0 (229 tokens) and 19.2 (347): GLM spans 17.3-20.0 across three traces, the new one at the low end. The model ordering on the shared conversation is unchanged and now rests on a long trace for each: DeepSeek 36.1 > MiniMax 21.0 > GLM 17.3 points.
- Live GLM over the six requests: 102.2 misses a token, hit 69.3 %, 0.51-0.76 tok/s; `cache_sim.py` at 46 GiB says 103.1 and 69.3 %: within 1 %.
## 0.62.7 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.7)" and section 18.99. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- A longer DeepSeek trace on the same Hermes conversation the MiniMax traces used (the eight dumped requests of `hermes-live-0.61.1`, decode cap 600, `serve.sh` exact path, 44 GiB): **2,028 decode tokens** over 40 layers, 7 minutes (the saved system-block snapshot was reused: request 1 in 2.9 s). The busiest 10 % of experts (38 of 384) carry 49.1 % of a layer's decode routes against 13.0 % for uniform routing: excess 36.1 points, a layer bootstrap 33.9-38.3. DeepSeek's other three traces give 40.3, 36.8 and 34.7: the new value sits inside their spread, so DeepSeek's value is 34.7-40.3 across four traces. **On the same conversation MiniMax's excess is 21.0 (1,904 tokens): a 15-point gap with the workload held fixed**, the cleanest comparison of the series (DeepSeek wrote its own replies, so the later tokens are not identical text).
- Live DeepSeek over the replay: 28.0 misses a token, hit 88.2 %, 7.1-8.4 tok/s; `cache_sim.py` at 44 GiB says 27.9 and 88.4 %: within 1 %. On the cold-prompt trace the simulator was 18 % pessimistic, so for DeepSeek too its error shrinks with length.
## 0.62.6 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.6)" and section 18.98. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- A longer MiniMax trace: the same eight dumped Hermes requests as 0.62.1, decode cap raised from 120 to 600 tokens, router's own picks, 3,053 slots: **1,904 decode tokens** over 57 layers (2.5 times the first trace), 12 minutes. Busiest-10 % share 33.5 % against 12.4 % for uniform routing at that length: excess 21.0 points; the first trace's 768 tokens gave 21.1. The two agree to a tenth of a point although sampling (temperature 0.7) and the reply lengths differ; a bootstrap over the 57 layers puts about +-1 point on each. MiniMax's cold-prompt excess (25.5, 560 tokens) is outside that interval, so the workload effect for MiniMax (+4.5 points from Hermes to cold prompts) is real; for GLM it was not (+0.6, one trace each).
- Live MiniMax over the replay: 30.9 misses a token, hit 86.4 %, 4.9-8.3 tok/s on the longer replies; `cache_sim.py` at 3,052 slots says 29.9 and 86.9 %: within 3 %. On the first (768-token) Hermes trace the simulator was 22 % optimistic and on the cold trace 10 % pessimistic, so its MiniMax error shrinks with trace length (a short trace is dominated by cache warm-up) and has no fixed sign.
## 0.62.5 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.5)" and section 18.97. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- GLM-5.3-Flash on an agent conversation: six dumped Hermes bodies of the 0.61.12 session (the "Hi", a tool call, the listing, the 2.7k-token tool result and two follow-ups; 21-24k tokens of context, 25 tools) replayed through `serve-glm.sh` at 46 GiB with the bank, 60 tokens a request, 229 decode tokens over 42 layers, 28 minutes. The saved 21,063-token system-block snapshot was reused (request 1: 32 s instead of the 39.7 cold minutes). The busiest 10 % of experts (29 of 288) carry 37.4 % of a layer's decode routes against 17.6 % for uniform routing at that length (excess 19.8 points, 2.1x). On cold prompts the same model read 19.2 points: **for GLM the workload does not change the concentration** (MiniMax moved +4 points between the same two workloads). The model-by-workload table has no empty cell left except DeepSeek and MiniMax on more conversations.
- Live GLM over the six requests: 97.8 misses a token, hit 70.3 %, 0.58-0.64 tok/s on the long ones; `cache_sim.py` at 46 GiB says 100.3 and 70.1 %: within 3 %.
## 0.62.4 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.4)" and section 18.96. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- MiniMax-M3 on the same eight short cold prompts as the GLM and DeepSeek traces (`serve-minimax.sh`, router's own picks, 3,053 slots, 560 decode tokens over 57 layers, 2.5 minutes of replay): the busiest 10 % of experts (13 of 128) carry 40.1 % of a layer's decode routes against 14.5 % for uniform routing at that length (excess 25.6 points, 2.8x). On identical prompts the excess is DeepSeek 34.6, MiniMax 25.6, GLM 19.2 points: **three distinct levels, not two.** 0.62.2's "MiniMax and GLM are alike" compared a Hermes conversation with toy prompts and is corrected: the same MiniMax model reads 21.4 points on Hermes and 25.6 on cold prompts, and GLM sits 6 points below it on the same prompts. The ordering DeepSeek > MiniMax > GLM holds on both workloads where measured.
- Live MiniMax over the eight prompts: 22.3 misses a token, 90.0 % hit, 5.8-11.6 tok/s; `cache_sim.py` at the same slot count says 24.5 and 89.2 %: 10 % pessimistic on this trace, against 20 % optimistic on the Hermes trace.
## 0.62.3 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.3)" and section 18.95. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- DeepSeek V4.1 Flash on the same eight short cold prompts as 0.62.2's GLM trace (`serve.sh`, exact path, 44 GiB, 594 decode tokens over 40 layers, 100 s of replay): the busiest 10 % of experts (38 of 384) carry 50.6 % of a layer's decode routes against 16.0 % for uniform routing at that length (excess 34.6 points, 3.2x). GLM on the same prompts: 35.2 % against 16.0 % (19.2 points, 2.2x). **The workload is not what made DeepSeek more concentrated in 0.62.1-0.62.2: on identical cold prompts it still is, by about 1.8x the excess.** The cause (expert count, top-k, training) remains a hypothesis.
- Live DeepSeek over the eight prompts: 29.6 misses a token, hit 87.6 %, 6.2-8.1 tok/s, prefill of 14-28 token prompts 1.6-2.9 s. `cache_sim.py` at 44 GiB says 35.0 and 85.4 %: pessimistic by 18 % (the simulator has no decode prefetch).
## 0.62.2 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.2)" and section 18.94. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- The first GLM-5.3-Flash routing trace (0.62.0's tracer): eight short varied prompts through `serve-glm.sh` at 46 GiB (bank on the X10Pro), 347 decode tokens over 42 MoE layers, 20 minutes. Routing skew, corrected for trace length against a uniform-routing null: the busiest 10 % of experts carry 35.2 % of a layer's decode routes (uniform at this length: 16 %, excess 19 points); MiniMax 35.1 % (null 13.7 %, excess 21 points); DeepSeek 49.6-53.4 % (null 13 %, excess 37-40 points). The uncorrected 0.62.1 comparison (35 % against 50-53 %) holds, and the correction shows it is not an artefact of MiniMax's shorter trace: DeepSeek routes about twice as concentrated as the other two, MiniMax and GLM are alike.
- `cache_sim.py` on the GLM trace at 46 GiB: 73.9 % hit and 87.7 misses a token against the live 72.5 % and 92.3 (5 % apart; MiniMax's was a fifth optimistic).
- Live GLM decode over the eight prompts: 0.52-0.84 tok/s (1.2-1.9 s a token), 36-73 s of prefill for 23-37 tokens.
## 0.62.1 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.1)" and section 18.93. Measurement plus an instrument fix; nothing in `src/` besides the version string changed.

### Measured
- The first MiniMax routing trace (0.62.0's tracer on a live server): eight requests of the first dumped Hermes conversation replayed through `serve-minimax.sh` with `CACHALOT_MINIMAX_MISS_DROP=0 CACHALOT_MINIMAX_PREFILL_MISS_DROP=0` (the router's own picks), 768 decode tokens and 2.55 M prefill rows over 57 MoE layers. Routing skew: the busiest 10 % of experts (13 of 128) carry 35 % of a layer's decode routes on average; the same measure on DeepSeek's Hermes traces (38 of 384) is 50-53 %. Live decode over the replay: 35.7 misses a token, hit 84 %.
- `benchmarks/cache_sim.py` on that trace is optimistic by about a fifth (68 GiB: 27.7 misses a token and 88 % against 35.7 and 84 % live), because the simulated store is DeepSeek's.

### Fixed
- `benchmarks/cache_sim.py` and `simulate_policies.Store` assumed 40 layers: a MiniMax or GLM trace skipped its layers 40 and above in the prefill replay, divided the prefill quota by 40 and printed a wrong decode-token count. The layer count now comes from the trace (DeepSeek's results are unchanged, checked on `trace_routing_v8_hermes`). One test added.
## 0.62.0 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.62.0)" and section 18.92. Instrument only; off by default; no output, speed default or numerics changed.

### Added
- `CACHALOT_ROUTING_TRACE=<file>` now works for GLM-5.3-Flash and MiniMax-M3 servers too (`serve-glm.sh`, `serve-minimax.sh`), not only DeepSeek: an unweighted trace (phase, layer, position, the routed experts) saved when the server stops, in the format `benchmarks/cache_sim.py` and `load_trace` read. It records from host arrays the model already read (GLM and MiniMax prefill: `StreamingSwitchGLU`; MiniMax decode: `GpuSelectDecoder._check`), so it adds no device read; `RoutingTracer.record_next` costs 2.3 us a layer-token (about 0.1 ms on a 42-layer token, measured by a loop without a model). Positions are a per-(phase, layer) counter. With MiniMax's miss substitution on (the default), prefill rows are the experts after substitution and decode rows the router's own picks; the server prints a warning, and `CACHALOT_MINIMAX_MISS_DROP=0 CACHALOT_MINIMAX_PREFILL_MISS_DROP=0` records the router's picks only.
- `GlmModel.set_tracer`, `RoutingTracer.record_next` and `forced_phase`, three tests in `tests/test_routing_trace.py`.

### Not done
- Not run on a real model: it needs the machine and Hamed's estimate-and-confirm. No predicted sets and no router weights for these two models.
## 0.61.14 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.61.14)" and section 18.91. A one-call instrument fix; no model output, speed default or numerics changed.

### Changed
- `CACHALOT_READ_THROTTLE_GBPS` now also covers MiniMax's coded bank. `CodedBankReader.read_expert_into` held bank records outside the emulated pipe (only the fallback to a checkpoint read reached `ExpertReader`'s throttle), so a throttled MiniMax sweep would have measured nothing. With the variable unset (the default) the path is unchanged. A test (`tests/test_reader_throttle.py`) checks that a 10 MB bank record is held to 10 ms at 1 GB/s and not held when the knob is off. The SEAMS.md finding it closes: section 2 item 4.
## 0.61.13 (2026-10-08)

HANDOFF "Start here (2026-10-08, 0.61.13)" and section 18.90. Measurement only; nothing in `src/` besides the version string changed.

### Measured
- The first live Hermes session on GLM-5.3-Flash with the contiguous bank on the X10Pro (46 GiB, eight requests, 3.7 hours): the cold 21,092-token Hermes block prefilled in 2,379 s (39.7 min, 8.9 tok/s); every later request reused 98-99.9 % of it. Decode 0.53-0.77 tok/s (1.30-1.89 s a token, mean 1.61 s over 1,109 tokens), tracking the hit rate (63-76 %). A 2,696-token tool result cost 356 s, a 389-token image turn 178 s, 18-34 token follow-ups 25-49 s.
- Answer quality, graded against the sources: the Desktop listing and the story are good; the C# snippet is unusable (leaked story text, placeholders, an apology, no code); the image transcription is right except two characters (`129A` read as `29A`, `Tredger` as `Fredger`).
- A client cancel after 15 minutes discarded half a cold prefill (nothing is cached until a request completes).

## 0.61.12 (2026-10-07)

HANDOFF "Start here (2026-10-07, 0.61.12)" and section 18.89. Measurement only; nothing in `src/` besides the version string changed.

### Added
- `benchmarks/short_prefill_trace.py`: per-chunk trace of a short prefill after a context (reads and their busy time, MoE and between-layer time, eval count), the same chunk repeated from one snapshot so repetition 1 (cold reads) and repetitions 2-3 (experts resident) separate reads from everything else. Results in `benchmarks/results/short-prefill-trace-0.61.12/`.

### Measured
- DeepSeek, 44 GiB budget, sysctl 88064, instrumented walls: with every expert resident a 19-token chunk costs 0.55 s and a 34-token chunk 0.78 s at both a 512-token and a 22,000-token context (about 0.26 s fixed plus 15 ms a token); a 69-token chunk 1.17 s at 512 and 1.36 s at 22k; a 246-token chunk is read-bound on every repeat (3,250 reads, 6.7-6.8 s) at both contexts. Cold repetitions add 0.3-1.5 s for 240-1,023 reads. The ~1.5 s a short chunk costs live is mostly expert reads, not a fixed per-chunk cost and not attention over the long cache.

## 0.61.11 (2026-10-07)

HANDOFF "Start here (2026-10-07, 0.61.11)" and section 18.88. Documentation only; nothing in `src/` besides the version string changed.

### Added
- `docs/ARCHITECTURE-COMPARISON.md` (charter track L6): the three models' architecture, cost terms and bottlenecks in one table, every number cited to a LEDGER row or derived from one, and nine findings stated in architectural terms (read share follows bytes a miss over drive rate; prefetch pays by precision; record size decides which I/O levers pay; and so on). Four measurements it names as missing.
- `docs/SEAMS.md` (charter track L7): the policy, execution and instrumentation seams in the code, seven places where measurement and policy share a class, and where each new instrument should attach. Finding: the routing tracer exists for DeepSeek only, and `CACHALOT_READ_THROTTLE_GBPS` does not cover MiniMax's coded bank.

### Removed
- The untracked `benchmarks/lane_cost.py` (residue of a temporary test; Hamed asked for it to be discarded). It was never committed.

## 0.61.10 (2026-10-07)

HANDOFF "Start here (2026-10-07, 0.61.10)" and section 18.87. Instrument only; nothing in `src/` besides the version string changed.

### Changed
- `benchmarks/cache_sim.py` takes `--context-tokens N` and adds the context term of LEDGER row DS-FLOOR-CTX to `--floor-ms`: +4.0 ms over the short-prompt floor by 1.8k tokens (interpolated below that, not measured), then 0.09 ms per 1k tokens. At 25,000 tokens the model gives 84.8-85.1 ms against the measured 84.8 ms (0.61.5). Default 0 leaves every earlier table unchanged. One test added.

## 0.61.9 (2026-10-07)

HANDOFF "Start here (2026-10-07, 0.61.9)" and section 18.86. A default switched on (bit-identical); nothing else in `src/` changed.

### Changed
- `serve-glm.sh` uses the contiguous expert bank at `/Volumes/X10Pro/models/GLM-5.3-Flash-bank` when `bank.json` exists there and `CACHALOT_GLM_BANK` is unset (`CACHALOT_GLM_BANK=` empty or `CACHALOT_GLM_BANK_ENABLED=0` turns it off). One pread an expert instead of 5-9.

### Measured
- GLM-5.3-Flash on the X10Pro, four server arms in ABBA order (shipped, bank, bank, shipped), a cache flush before each, 46 GiB budget: the bank takes a token from 1,631 to 1,531 ms (-6.1 %, 0.613 to 0.653 tok/s); the repeats of a layout agree to 0.1-0.8 %. Every generated text is identical across the four arms and misses, reads and bytes a token are identical to the digit; the drive's busy rate rises 0.952 to 1.011 GB/s.
- The 52 GiB budget was killed twice by the guardian on today's memory state (swap 2.7-4.7 GB, free 0.1 GiB); arms ran at 46 GiB and the absolute token is not comparable with 0.61.1's.

## 0.61.8 (2026-10-07)

HANDOFF "Start here (2026-10-07, 0.61.8)" and section 18.85. A measurement and a correction of 0.61.6 and 0.61.7; nothing in `src/` changed, no default, output or speed changed.

### Changed
- `benchmarks/expert_read_qd.py`: `--skip` (offset into the shuffled experts, for paired slices) and a docstring that says what the page cache does to a run.
- `.gitignore`: `*.textClipping`.

### Measured
- The contiguous GLM bank for layers 3-5, written to the X10Pro (11 GB), against the shipped layout in a paired ABBA run: +6.6 % rate (0.999 against 0.9375 GB/s at two in flight), read time -6.2 %, about -5.6 % of a GLM token (derived), p99 latency halved.

### Corrected
- 0.61.6 and 0.61.7 reported raw 0.25-13.5 MiB block rates of 1.05-1.24 GB/s on the X10Pro and derived from them that DeepSeek's record layout leaves ~18 % of the link unused and that GLM's contiguous layout would take ~16 % off its token. Those rates were page-cache hits (the same instrument on the same shard later reads 0.97-1.0 GB/s); the derived claims are withdrawn. The expert-record queue-depth results (0.92-0.97 GB/s from one to sixteen in flight, internal 5.0 then 7.0-7.3) stand.

## 0.61.7 (2026-10-07)

HANDOFF "Start here (2026-10-07, 0.61.7)" and section 18.84. A measurement and an instrument option; nothing in `src/` changed, no default, output or speed changed.

### Added
- `benchmarks/expert_read_qd.py --glm [--glm-bank DIR] [--layers a,b]`: the same queue-depth instrument on a GLM-5.3-Flash checkpoint, optionally through a contiguous bank, restricted to some layers.

### Measured
- GLM's shipped expert layout (5-9 byte ranges of 256 KiB to 5 MiB an expert) on the X10Pro: 0.906 GB/s with one expert in flight, 0.939 from two to four (the 0.95 a live session reads at). Raw contiguous 13.5 MiB blocks on the same drive: 1.142 GB/s. A contiguous record would take ~18 % off GLM's read time, ~16 % of its 1.5 s token (derived; the stand-in is raw blocks, not a bank file). The contiguous bank on the internal SSD reads 5.3-6.4 GB/s.

## 0.61.6 (2026-10-07)

HANDOFF "Start here (2026-10-07, 0.61.6)" and section 18.83. A new instrument and its measurements; nothing in `src/` changed, no default, output or speed changed.

### Added
- `benchmarks/expert_read_qd.py`: read queue depth against bandwidth and latency (p50/p95/p99) on one expert bank, through the shipped `ExpertReader` with disjoint experts per level, plus a raw random-pread mode. Tests in `tests/test_expert_read_qd.py`.

### Measured
- X10Pro, 9.95 MB expert records: 0.92 GB/s with one expert in flight, 0.966-0.969 from two to sixteen; p50 latency 10.3 ms times the number in flight, tails within 1-3 %. Queue depth is not a lever; the runtime's loaders sit on the plateau.
- Raw preads on the same drive reach 1.14-1.24 GB/s for 1-13 MiB blocks, so the nine scattered ranges of a DeepSeek record leave ~18 % of the link unused.
- Internal drive, same bank: 5.0 GB/s at one expert in flight, 7.0-7.3 at two to sixteen.

## 0.61.5 (2026-10-07)

HANDOFF "Start here (2026-10-07, 0.61.5)" and section 18.82. A measurement only; nothing in `src/` changed, no default, output or speed changed.

### Measured
- DeepSeek's all-resident decode floor against context length, through `serve.sh` (exact, 48 GiB, internal bank), two fresh servers: 79 ms at 27 tokens, 83.4 at 1.8k, 84.8 at 25k in the clean arm (short probes after every context stayed at 79.1 +-0.5): about +4 ms over the first ~2k tokens, then 0.09 ms per 1k. The written prediction (80 + 0.65 ms per 1k, 96 ms at 24k) is falsified.
- The ledger's 95.6 ms at 22k context was drift: the ascending arm reproduced 94-95 ms at 25k but its own end-of-run short probe read 91.6 against 78.8 at the start (+16 % in 11 minutes). A context sweep needs a short probe after every point.
- Cold-prefill requests decode at 117 +-3 ms a token (20-23 misses a token) at every context from 1.8k to 25k, i.e. `80 + 2.0 x misses` with no visible context term.

## 0.61.4 (2026-10-07)

HANDOFF "Start here (2026-10-07, 0.61.4)" and section 18.81. An offline instrument and its price; nothing in `src/` changed, no default, output or speed changed.

### Added
- `benchmarks/pred_gate_price.py`: replays a routing trace that carries predicted sets through the expert store's residency rule (a predicted load fills a transient slot, never evicts, becomes resident only when the next layer routes it) and prices a router-weight gate on decode prefetch at any drive bandwidth with the LEDGER's bandwidth curve, in the exact mode and in budget 0. Tests in `tests/test_pred_gate_price.py`.

### Measured
- Calibration against the measured arms: misses a token 23.2 against 23.5, token 435 against 445.7 ms at 1 GB/s and 118 against 118.6 internal; but the prediction-off token is underestimated by 14 % and budget 0's 1 GB/s token overestimated by 33 %, so the tables are used for ordering and drop counts only.
- A gate is not worth building: on the exact path no gate beats prediction off (`CACHALOT_PREDICT_TOPK=0`, already -11 to -18 % at 1 GB/s); under budget 0 a gate trades dropped experts for bytes (weight >= 0.15: -13 % at 1 GB/s for +3.2 dropped a token; >= 0.20: -43 % for +12.6) with quality measured only at the shipped ~20 drops; on the internal drive >= 0.20 is -7 %.

## 0.61.3 (2026-10-07)

HANDOFF "Start here (2026-10-07, 0.61.3)" and section 18.80. A recording option and one instrument; no default, output or speed changed (the recording is off unless a routing tracer is installed).

### Added
- `RoutingTracer.record_predicted` and the decode hook in `moe_layer_metal.py`: a routing trace (`CACHALOT_ROUTING_TRACE`) now also stores, for every decode layer, the experts the next-layer prediction chose and the predictor's own router weights (`pred_source`, `pred_target`, `pred_position`, `pred_experts`, `pred_weights`). Older traces load unchanged; with no tracer the hot path pays one `getattr`. The predictor's weights are read from the arrays the layer's single router sync already evaluates, so there is no extra GPU round trip.
- `routing_trace.predicted_used_mask`: joins each predicted set to the routes of the same decode token. Decode positions restart with every request, so the join is by token ordinal, not by (position, layer).
- `benchmarks/pred_gate_table.py`: predictor weight against precision and recall, from a trace that carries predicted sets.

### Measured
- One server run (48 GiB, exact, 17 greedy requests of 120 tokens, 2,407 decode tokens, W1): the predicted top-6 overlaps the next layer's routed set 72 % of the time (the recorded recall at top-6 was ~73 %). The weight carries signal: predictions with weight >= 0.20 are 66 % of all and 82 % precise (recall 0.75); >= 0.25 are 38 % and 93 % precise (recall 0.50). A set-overlap screen only; it does not know what was already resident, so a gate is not priced yet.

### Notes
- `guarded_run.sh` refuses to start while the Cachalot Lab app runs (its process matches the runtime pattern); the smoke run used `serve.sh` directly with Lab open and was a correctness check, not a speed arm.

## 0.61.2 (2026-10-07)

HANDOFF "Start here (2026-10-07, 0.61.2)" and section 18.79. Measurement and documentation only; no code, default or output changed.

### Measured
- First live Hermes session on 0.61.1 (budget 0, 48 GiB, 22.4k-token system block, eight requests): decode 9.8-10.7 tok/s at 22-27k context, miss/tok 9-17; the first request was a cold 212 s prefill, as expected (no earlier date or pin existed).
- Server-path replays of the dumped body: **a new day's date line reuses 22,407 of 22,411 tokens (0.95 s against 212 s cold)**, and a changed provider string reuses the 4,096-token pin (saves ~26 s). Both 0.59.0 and 0.60.0 mechanisms work live.
- A prefill of under ~70 tokens costs 1.8-3.8 s whatever its size (a ~1.5 s fixed cost a chunk); 579 tokens 12.7 s, 2,891 tokens 29.5 s.
- Answers read: image transcription exact; the C# importer snippet does not compile (`ClassMap.Map(Type, string)`, one CS1503 under CsvHelper 33.1.0) and its "streaming" method materialises the array. One sample; whether budget 0 caused it is open (a replay with `dotnet build` as the grader was priced, not run).

### Changed
- `docs/LEDGER.md`: DS-PIN-4096 and the new DS-DATE-REUSE and DS-LIVE-0.61.1 measured; DS-PREFILL-SHORT carries its curve.

## 0.61.1 (2026-10-06)

HANDOFF "Start here (2026-10-06, 0.61.1)" and section 18.78. Measurement, one additive stats change; no default changed, outputs unchanged.

### Added
- GLM's and MiniMax's `/v1/stats` carry the expert store's counters under DeepSeek's names (`expert_hits`, `expert_misses`, `predicted_loads`, `predicted_used`, `ssd_bytes_read`, `expert_reads`, `expert_fast_reads`, `expert_read_seconds`) plus `expert_read_busy_seconds`, `decode_wait_seconds` and `decode_waited_misses`. Counters only.
- `benchmarks/micro_read_size_roofline.py --arms sum,gemv,q2,fp8`: a 2-bit `quantized_matmul` arm and the trunk's FP8 GEMV at the same bytes a launch as the bf16 arms.

### Measured
- **GLM-5.3-Flash on the X10Pro (52 GiB, 12 mixed prompts x 120 greedy tokens, W1): 1,518 ms a token (0.66 tok/s)**, 86 misses a token, 91 % of decode waiting on reads with the drive at 0.95 GB/s; short prompts prefill at ~0.7 tok/s. Its K = 5 prefetch roughly breaks even there: off is 1,451 ms (-4.4 %, inside drift from one pair), because 72 % of its predicted loads are used (DeepSeek's 35-40 % loses 11-18 % on the same kind of drive). K = 5 stays.
- DeepSeek at an emulated 1 GB/s under the shipped budget 0: 281 ms a token (exact 446). With prediction off too: 76 ms, but 78 of 240 routed expert uses dropped a token (a timing bound, not a candidate; quality unmeasured).
- Share of 819 GB/s at 3 MiB a launch (a routed expert projection): bf16 GEMV 77 %, 2-bit `quantized_matmul` 56 %, FP8 GEMV 37 %; the quantized kernels never pass ~72 % and ~65 %. In the model the routed experts run at ~29 %: about half the gap is the 2-bit kernel, half what the model adds around it.

### Closed
- Skipping the decode prediction on layers whose predicted experts are resident: priced at ~0 (the store already skips resident keys; the ~5 ms of an all-resident token is unused loads of non-resident guesses).

### Changed
- `docs/LEDGER.md`: GLM-TOKEN-USB measured (GLM-TOKEN-USB-M), GLM-PRED-USB, GLM-PREFILL-USB, DS-MB0-USB, MC-READ-SIZE-Q; open question 6 answered for GLM.

## 0.61.0 (2026-10-06)

HANDOFF "Start here (2026-10-06, 0.61.0)" and section 18.77. The first storage-bandwidth curve (research charter L3 item 2). No default changed; outputs unchanged.

### Added
- `CACHALOT_READ_THROTTLE_GBPS=B` (off by default; unset, empty or 0 = off): emulates a slower expert drive for sweeps. Every expert read the drive served (slower than 1 ms; page-cache hits pass free) occupies one shared pipe of B GB/s and is held until its bytes would have crossed it, so concurrent reads queue as on a queue-depth-one USB drive. Timing only, bit-identical outputs. `src/cachalot/storage/reader.py`, tests in `tests/test_reader_throttle.py`.

### Measured
- DeepSeek, exact path, 48 GiB, 12 mixed prompts x 120 greedy tokens (W1), one fresh `serve.sh` per arm: **118.6 ms a token on the internal SSD, 139.3 at 4 GB/s, 227.4 at 2, 445.7 at 1, 390.1 on the real X10Pro** (the emulation is within 13 % of the drive it imitates). Misses a token are 23.5 in every arm; the all-resident token stays 77-88 ms.
- Below ~2.6 GB/s the token is the drive's time to move every read, `token ~ 453 MB / B` (1 % off at 1 and 2 GB/s), not `80 + misses x cost(B)` (which predicted 318-327 ms at 1 GB/s): 45.6 expert reads a token, of which ~18.6 are speculative loads that are never used (185 MB a token).
- Decode prediction off (`CACHALOT_PREDICT_TOPK=0`) on a 1 GB/s drive: **367.7 against 445.7 ms (-17.5 %) emulated, 345.6 against 390.1 ms (-11.4 %) on the X10Pro**. On the internal SSD prediction nets ~1 ms (section 18.76), so the right prefetch depends on the drive's bandwidth.

### Changed
- `docs/LEDGER.md`: ST-BW-CURVE, DS-PRED-USB and ST-THROTTLE-VALID added; ST-X10's effective rate (~1.15 GB/s on this workload) recorded.

## 0.60.10 (2026-10-06)

HANDOFF "Start here (2026-10-06, 0.60.10)" and section 18.76. Measurement only, no change to inference.

### Measured
- The DeepSeek all-resident token through `serve.sh` at 48 GiB (exact path, sysctl 88064): **79.2-80.3 ms**, against 75-77 ms in-process the same day. A fit of ms a token on misses a token over 12 mixed requests gave intercepts of 93.4 and 73.4 ms on the same workload in two arms, so the ledger's 70 ms floor (DS-FLOOR-48) was a fit intercept; the token model is `80 + 2.0 x misses` (126.1 predicted, 126.2 measured at 23.5 misses).
- Decode prediction costs ~5 ms of an all-resident token (`CACHALOT_PREDICT_TOPK=0`: 72.8-76.8 ms) and nets ~1 ms on tokens with ~24 misses.
- Budget 0 (the shipped decode): 93.3 ms a token on the same mixed prompts (-26 %); 26.4 % of decode layer-calls route to a predicted load still in flight, so GPU-side expert selection (D3) prices at +1 to +11 ms a token and stays held.
- New `benchmarks/micro_read_size_roofline.py`: a batch-1 bf16 GEMV reaches 47 / 71 / 83 / 87-89 % of the M3 Ultra's 819 GB/s at 2 / 4 / 8 / 16+ MiB read a launch; decode launches read 3-11 MiB.

### Changed
- `benchmarks/cache_sim.py --floor-ms` defaults to 80 (LEDGER DS-FLOOR-SRV), was 70.
- `docs/LEDGER.md`: DS-FLOOR-48 contradicted; DS-FLOOR-SRV, DS-PRED-FLOOR, DS-MB0-HOSTLAYERS and MC-READ-SIZE added.

## 0.60.9 (2026-10-06)

HANDOFF "Start here (2026-10-06, 0.60.9)" and section 18.75. The DeepSeek floor re-profiled (charter L4(f)), and a fix to the one-runtime guard. No change to inference.

### Measured
- All-resident decode token in-process at 48 GiB, sysctl 88064: 75.4 ms (median, sync profiler) to 77.4 ms (mean of five passes): 55.5 ms inside `mx.eval`, 17.9 ms of CPU between the 44 host syncs (GPU idle ~27 %); GPU work on the shipped path 52.2 ms (attention 25.6, routed experts 10.1, shared expert 6.0). Predictions written before the run: P1-P2 held, P3 partly, P4 falsified on the generous count (`docs/LEDGER.md` DS-FLOOR-SPLIT-0609, DS-GPU-IDLE, DS-EVAL-DRAIN).
- A resident replay still issues 23 speculative expert loads a token, all unused (218 MiB a token; DS-SPEC-RESIDENT).

### Fixed
- The one-runtime guard in `serve.sh`, `serve-glm.sh`, `serve-minimax.sh`, `chat.sh`, `chat-glm.sh`, `chat-minimax.sh`, `benchmarks/decode_vs_context.sh`, `benchmarks/guarded_run.sh`, `benchmarks/settle.sh` and `benchmarks/run_manifest.py` did not see benchmarks started from the venv, which run as Homebrew's `Python.app/Contents/MacOS/Python`; it does now.
- `guarded_run.sh --force` skipped the other-runtime check along with the memory arithmetic; it now skips only the arithmetic.

## 0.60.8 (2026-10-06)

HANDOFF "Start here (2026-10-06, 0.60.8)" and section 18.74. Measurement only, no runtime change: the decode miss budget 0 (the `serve.sh` default) on a second, independent C# task.

### Measured
- A short CsvHelper + System.Text.Json configuration task, 48 samples per arm, one process, graded blind by two model graders: flawed **exact 2/48, budget 0 2/48 (p 1.00)**; decode a token, paired, **0.776** [0.769, 0.782]. 96 replies in 48 minutes.
- By the decision rule written before the run, 0.60.2's C# cost (26/48 against 11/48) is not reproducible across C# tasks: across three C# results it showed once, on long free-form code in an agent context. The task's base rate is low (4 %), so it rules out a large effect, not a small one.

### Not changed
- `serve.sh` keeps `CACHALOT_DECODE_MISS_BUDGET=0` (Hamed's default). `CACHALOT_DECODE_MISS_BUDGET=off ./serve.sh` is the exact path for long free-form code.

## 0.60.7 (2026-10-06)

HANDOFF "Start here (2026-10-06, 0.60.7)" and section 18.73. Review fixes to 0.60.6's run manifest; no runtime change.

### Fixed
- `benchmarks/run_manifest.py` recorded every `CACHALOT_*` variable by value, so a run with `CACHALOT_API_KEY` set would have written the server's API key into a manifest meant to be shared. Variables ending in `_KEY`, `_TOKEN`, `_SECRET` or `_PASSWORD` are now recorded as `<redacted>`; knobs such as `CACHALOT_QB_MAX_TOKENS` keep their values. The one manifest written so far (0.60.6's C# run) holds no credential.
- `hermes_desktop_running` was true whenever any process line contained "Hermes" and any other line contained ".app/" (for example the Hermes CLI plus any open app); it now needs both in one process's command line.
- `docs/LEDGER.md`'s regime warning records that Hamed re-applied the sysctl (88064) after the reboot.

### Added
- Two tests in `tests/test_run_manifest.py` (redaction that keeps token-count knobs; the one-line Hermes check).

## 0.60.6 (2026-10-06)

HANDOFF "Start here (2026-10-06, 0.60.6)" and section 18.72. Measurement and instruments, no runtime change: decode miss budget 1 on C#, the first two tracks of the research charter (L1 ledger, L0 manifest and scorecard), and batched decode priced offline.

### Measured
- Budget 1 against exact and budget 0 on a C# request rebuilt from Hermes's own store (the 0.58.1 dump was deleted by a reboot): 48 samples per arm, one process, blind sheet graded by three model graders. Flawed: exact 10/48, **budget 1 10/48 (p 1.00), budget 0 13/48 (p 0.63)**. Decode a token, paired against exact: budget 1 **0.973** [0.967, 0.978], budget 0 **0.749** [0.745, 0.754].
- The positive control did not reproduce 0.60.2's budget-0 C# cost (26/48 against 11/48 there), so by the rule written before the run the budget-1 result is inconclusive about safety; the code evidence on budget 0 is now split across two bodies of the same request. Budget 1 buys only ~3 %, so it is not proposed as a default; the choice stays 0 or `off` (Hamed's).
- Batched decode, offline (`benchmarks/batch_union.py`, the 22k-context trace's 13 replies dealt into B lanes): concurrent streams share few experts (96.5 % of routed uses distinct at B = 2) and misses a token rise (24.9 at B = 1, 30.5 at B = 4); the aggregate gain is 1.27-1.42x at B = 4 and comes from spreading the 70 ms floor over the lanes, with each user's tokens ~2x slower. Not built.

### Added
- `docs/LEDGER.md` (charter track L1): every constant the decisions rest on, per model and regime, with tag, source section, status and reproduction; bottleneck-migration tables for the three models; stale constants in code; open questions (no DeepSeek constant at the default sysctl; the floor's split is from the 77-80 ms era).
- `benchmarks/run_manifest.py` (track L0): a run manifest (commit, environment, workload class and hash, hardware, sysctl, pressure, swap, wired and GPU memory, screensaver, other runtimes) and JSON-line scorecard rows from a fixed schema with measured / derived / estimated tags; absent fields are absent, never zero. `tests/test_run_manifest.py`.
- `benchmarks/batch_union.py`, `tests/test_batch_union.py`.
- `quality_blind_ab.py run --budget 1,0`: several capped arms in one process (arms exact, b1, b0; the two-arm path keeps its labels and table); the C# build check keys on the body's text; `run` writes a manifest.

### Changed
- `benchmarks/cache_sim.py` defaults: floor 70 ms and 2.0 ms a miss on the internal bank (LEDGER DS-FLOOR-48, DS-MISS-48; were the internal-drive era's 77 and 1.7).

## 0.60.5 (2026-10-06)

HANDOFF "Start here (2026-10-06, 0.60.5)" and section 18.71. Documentation only, no runtime change: the runtime's docs follow Codex's completed rename of the app.

### Changed
- README's Cachalot Lab link points to `https://github.com/prooshani/cachalot-lab`.
- `docs/lab/CODEX-LAB-PROMPT.md`'s rename notice records the finished rename: repository `prooshani/cachalot-lab`, working directory `/Volumes/X10Pro/Cachalot Lab` (the old Studio path is a compatibility alias), code graph `Volumes-X10Pro-Cachalot-Lab`, and the app's decision (its `docs/RENAME.md`) to keep the bundle identifier `com.prooshani.cachalotstudio`, the Keychain service `com.cachalot.studio.runtime-api-key` and the preference key `cachalot-studio-ui` so existing data is not stranded.

## 0.60.4 (2026-10-05)

HANDOFF "Start here (2026-10-05, 0.60.4)" and section 18.70. Documentation only, no runtime change: the desktop app is renamed from Cachalot Studio to **Cachalot Lab** (Hamed's decision; Codex carries out the app-side rename).

### Changed
- `docs/studio/` moved to `docs/lab/`; the product brief is `docs/lab/CODEX-LAB-PROMPT.md` (was `CODEX-STUDIO-PROMPT.md`) with a rename notice at its top listing the identifiers Codex migrates (Git remote, working directory, bundle identifier, data directory, keychain service names). Per-release briefs keep their file names under `docs/lab/briefs/`; their text is left as sent.
- README section "Cachalot Studio" is now "Cachalot Lab"; the current prompt, the research charter and the session skill use the new name. "Mac Studio" (the hardware) is unchanged.
- `docs/RESEARCH-DIRECTION.md` gains a naming note: "Cachalot Lab" is the app, "the lab" in lower case is the research direction.

### Added
- `docs/lab/briefs/2026-10-05-runtime-0.60.4.md`: the rename brief for Codex.

## 0.60.3 (2026-10-05)

HANDOFF "Start here (2026-10-05, 0.60.3)" and section 18.69. Documentation only, no runtime change: the research-direction charter (`docs/RESEARCH-DIRECTION.md`), the README direction section and the prompt amendment, released together as the charter's own section 9 (question 5) recommends.

### Added
- `docs/RESEARCH-DIRECTION.md`: Cachalot as a measurement-driven inference systems laboratory. The inventory of instruments and results the repository already has (section 3), the gaps against the direction (G1-G14), the anatomy of one token and its definitions (section 4), the programme L0-L7 (run manifest and scorecard, constants and bottleneck ledger, per-token critical-path trace, sweeps as curves, predictive model and what-ifs, energy, architecture comparison, seams), the workload classes W1-W7, the experiment record, twelve added rules and how they combine with the standing ones. Hamed's decisions of 2026-10-05 are in its section 9: the lab is the method for every lane and the order "Hermes, vision, speed" stands; he runs `sudo powermetrics` himself for energy arms; questions 3-6 stay open. The charter was written against 0.60.1 and does not yet carry 0.60.2's finding that the budget-0 default invents C# API members (HANDOFF 18.68); that result belongs in its ledger (L1) and workload class W4.
- README: a short "Research direction" paragraph in the roadmap section.

## 0.60.2 (2026-10-05)

HANDOFF "Start here (2026-10-05, 0.60.2)" and section 18.68. Measurement only, no runtime change: the code-only N = 48 blind check of the decode miss budget (the `serve.sh` default since 0.60.0). **It found a quality cost on C#.**

### Measured
- C# importer and TypeScript importer bodies of the 0.58.1 Hermes dump, 48 samples per arm each (96 + 96 = 192 replies, temperature 0.7, 1,500 new tokens at most; the first 16 per cell are 0.60.1's, 32 new), shuffled, split in four halves and graded blind by four separate model graders, key opened afterwards. Flawed: **C# exact 11/48, budget 0 26/48 (Fisher p = 0.003)**; TypeScript exact 9/48, budget 0 8/48 (p = 1.0); both 20/96 against 34/96 (p = 0.036). The C# excess is invented API members: 15 in budget 0 against 1 in exact (other wrong code 8 against 10, unsupported claims 3 against 0). The 32 new C# samples per arm alone: 18/32 against 6/32. Decode 138.3 -> 103.5 ms a token (-25 %). `dotnet build` of the bare snippet passes 7/48 and 10/48 (it mostly measures missing project context).
- 0.60.1's N = 16 read (9/16 against 4/16 on C#, p = 0.15) was this effect at a quarter of the size.

### Not changed
- `serve.sh` still defaults `CACHALOT_DECODE_MISS_BUDGET=0` (Hamed's decision of 2026-10-05); the evidence now argues against it for code work. Options in HANDOFF 18.68.

## 0.60.1 (2026-10-05)

HANDOFF "Start here (2026-10-05, 0.60.1)" and section 18.67. Measurement only, plus instrument changes: the model-graded N = 16 blind quality check of the decode miss budget (now the `serve.sh` default), run on the six bodies of the 0.58.1 Hermes dump.

### Measured
- 192 replies (6 bodies x 16 samples x 2 arms, temperature 0.7, 1,500 new tokens at most, 48 GiB, one process, seeds matched, order alternating), generated through `Engine.chat`, shuffled and graded blind by six separate model graders (one per body, given only the sheet and the 18.60 rubric; the key opened afterwards). Flawed: **exact 26/96, budget 0 28/96, Fisher p = 0.87**. Per body (exact / budget 0): greeting 0/16 / 0/16; tool call 0/16 / 0/16; summarising an `ls` result 12/16 / 7/16 (p 0.15); 200-word story 4/16 / 5/16; C# importer 4/16 / 9/16 (p 0.15); TypeScript 6/16 / 7/16. No loops, valid tool calls 16/16 in both arms. Decode **121.7 -> 99.4 ms a token (-18.3 %)** (-6 % on 9-100 token replies, -25 % on long ones).
- The one pattern to watch: the C# body had 5 "invented" flaws (made-up API members) in the budget-0 arm against 0 in exact, and 9 against 4 flawed overall; the `ls` summary leaned the other way (7 against 12). Neither is significant at n = 16, and `dotnet build` passed in 3/16 (exact) and 4/16 (budget 0) because the replies' snippets need a project and usings the check does not give them.

### Changed
- `benchmarks/quality_blind_ab.py` reads its dump, rows and token cap from `CACHALOT_QB_DUMP`, `CACHALOT_QB_ROWS`, `CACHALOT_QB_MAX_TOKENS`, turns image parts into a text note, and resumes a stopped run (`CACHALOT_QB_BODIES` limits the bodies of a call).

## 0.60.0 (2026-10-05)

HANDOFF "Start here (2026-10-05, 0.60.0)" and section 18.66. Hamed's three decisions on v108's first job: the decode miss budget is 0 by default in `serve.sh`, an agent's system block is reused across days, and a model (not the author) grades the N = 16 quality check.

### Changed
- `serve.sh` exports `CACHALOT_DECODE_MISS_BUDGET=${CACHALOT_DECODE_MISS_BUDGET-0}`: DeepSeek decode drops every non-resident expert a layer would read (-22 to -27 % a token measured in 18.59-18.61, no sign of harm in the blind checks 18.60-18.61). This changes outputs. `CACHALOT_DECODE_MISS_BUDGET=off` (new spelling; `exact` too) restores the exact path; a direct `cachalot.cli serve` without the variable is still exact.
- A new system-date reuse (`cachalot/server/system_date.py`, on by default, `CACHALOT_SYSTEM_DATE_REUSE=0` off, `CACHALOT_SYSTEM_DATE_REUSE_DAYS` window, default 7): the "Conversation started: <date>" line of the leading system message shows the first true date seen in the last 7 days (kept in `system-dates.json` beside the snapshots), so the first message of a new day reuses the saved block (about 227 s of cold prefill saved) instead of prefilling it. The model sees a date up to 7 days old; after the window the true date is used and starts a new window. Only the system message's date line is replaced; the client's history and dump are untouched. Startup prints `system date reuse: ...`, a replaced request prints `[request] system date: showing X for Y`.

### Added
- Tests: six for the date reuse (window, restart, off switch, no-date and user-message cases, and a server-path check that a new day reuses the whole block), one for the budget spelling. 542 tests pass.

## 0.59.0 (2026-10-05)

HANDOFF "Start here (2026-10-05, 0.59.0)" and section 18.65. The DeepSeek server keeps the 4,096-token chunk snapshot inside an agent's system block on disk, so a restart followed by a changed block (a new day, another model name or provider string) reuses the head instead of prefilling the whole block cold.

### Added
- `PrefixCache.persist_pin` and `SnapshotStore.persist_pin`: the chunk pins inside a system block (until now in memory only, lost at a restart) go to the snapshot directory when they are at most 8,192 tokens (`SnapshotStore.PIN_MAX_TOKENS`), at most 8 pin files (`keep_pins`), pruned by use apart from the 32 system blocks. A pin is indexed at startup but not preloaded into memory; `fetch` loads it (~0.1 s) when a request starts with it and memory holds nothing longer; a request that starts with a pin counts as using it. The server (`cli.py`) attaches the hook.

### Changed
- Nothing else. Outputs are unchanged: a pin is the very snapshot the in-memory cache already took at that chunk boundary and reused within a server run; the numerics tag does not move and saved blocks stay valid. 536 tests pass, five new.

### Measured, not shipped
- Priced from the dumped Hermes sessions: the shared head of two session starts is ~6.2k of 22.5k tokens, so the 4,096 pin saves ~41 s of a 227 s first request at 99 tok/s (a cut at 6,144 would save ~62 s but changes the chunk partition and is not bit-identical by construction). Not yet measured live: that needs a restart and a Hermes session with a changed date or provider line.

## 0.58.2 (2026-10-05)

HANDOFF "Start here (2026-10-05, 0.58.2)" and section 18.64. Measurement and documentation only; no runtime change. Why every Hermes session start prefilled the whole system block, and the first Hermes session on the splice-before-image fix.

### Measured
- The three saved system blocks differ for three reasons, each checked by re-rendering the dumped prompts: Hermes's model name (session 2 said `Model: minimax-m3`, which drops a 178-token "Tool-use enforcement" section at token 1,024: 22,311 against 22,493 tokens), its provider string (`custom` against `custom:cachalot`, +3 tokens: 22,490 against 22,493, re-rendered exactly), and the date (token ~6,227). All sit in the first 6.3k tokens and the 16.2k tokens of tool schemas follow, so any change re-prefills the whole block (~227 s at 99 tok/s). With a stable Hermes configuration the saved block is reused (first request ~1.7 s).
- First Hermes session on 0.58.1 (budget 0, the image going in natively): the image turn prefilled 246 tokens in 8.85 s with `spliced=1` (113 s on 0.58.0), decode 8.97-9.70 tok/s, every request's reuse equals the previous prompt plus reply except the last (Hermes rewrote the old image message: 569 tokens, 13 s).

### Open
- A snapshot at an earlier point of the system block (the shared head of two sessions is ~6.2k tokens of 22.5k) would save ~60 s of the first request after a Hermes configuration or date change; not built.

## 0.58.1 (2026-10-05)

HANDOFF "Start here (2026-10-05, 0.58.1)" and section 18.63. A vision turn no longer re-prefills the whole conversation before its image.

### Fixed
- `Engine._splice_own_replies` takes the prompt's image spans and shifts the ones behind a splice by the token-count difference, so the model's own reply tokens are put back into a client's re-rendered history before an image as well. Until now, with an image in the request only replies after the last image span were spliced, and an image sits in the newest message, so none were: Hermes's re-serialised tool calls diverged at the first assistant reply and every token after it was prefilled again (a long Hermes conversation: 7,531 tokens, 80 s, on the image turn). A reply whose region overlaps a span is still left as the client sent it. Outputs are unchanged; no numerics tag moves. 2 tests in `tests/test_reply_splice.py`.

### Measured
- Server-path check (a 304-token tool-call turn, then the same conversation with the tool call re-ordered and an image in the newest message, greedy): reused prefix 304 tokens before, 360 (the whole earlier reply) after, spliced 0 to 1, prefill 7.51 to 5.82 s, identical answer. The gain grows with the history before the image.

## 0.58.0 (2026-10-05)

HANDOFF "Start here (2026-10-05, 0.58.0)" and section 18.62. Router-share substitution (the best resident expert of the next four ranks replaces a missing one whose router share is under tau, MiniMax's 0.39.0 rule) judged as an arm against the decode miss budget on the topic-shift stream. It loses on both axes and is closed for DeepSeek.

### Added
- `V41Model.set_decode_substitution(tau, ranks=4)` (off by default, decode only, changes outputs): in `moe_layer_metal.moe_layer_forward`, a non-resident expert whose router share is under `tau` is replaced, at its position and with its weight, by the best resident expert among the next `ranks` by selection score (router score plus correction bias). Misses it cannot replace are read, or dropped when a decode miss budget is also set. New module `model/decode_substitution.py` (pure planning functions), `ResidentExpertStore.is_cached`, counters `decode_substitute_tau` and `substituted_experts` in `V41Model.stats()`. No server environment knob, because the result below closes the lever. 7 tests (`tests/test_decode_substitution.py`) and one in `tests/test_resident_store.py`.
- `benchmarks/pareto.py`: arm keys `substitute_tau` and `substitute_ranks`; `parse_decode_arm` / `apply_decode_arm` read the arm strings `exact`, `N`, `sT` and `sTbN`, now used by `topic_shift.py` and `topic_shift_gen.py` (the topic-shift report gains a substituted-experts column).

### Measured, closed
- Topic-shift stream (1,600 teacher-forced steps, 48 GiB, each arm a fresh process), against exact (131.3 ms, 26.2 misses a token): budget 0 98.3 ms (-25 %), mean dNLL +0.013; substitution under 0.25, 111.7 ms, dNLL +0.049; substitution under 1.0 (every miss with a resident candidate), 111.4 ms, dNLL +0.127; substitution under 1.0 then budget 0, 92.5 ms, dNLL +0.134, KL max 3.6-3.9 against budget 0's 0.4-1.3. At equal speed the substitute costs about ten times the log-likelihood of a drop. Substitution is closed for DeepSeek (D4).

## 0.57.2 (2026-10-04)

HANDOFF "Start here (2026-10-04, 0.57.2)" and section 18.61. What the DeepSeek decode miss budget costs when the text changes topic.

### Added
- `benchmarks/topic_shift.py`: one teacher-forced stream that shifts topic twice (Python, prose, JSON) after a short prefill, each arm in its own fresh process, reported per 100-token window and per segment (log-likelihood change, KL, experts read and dropped, step time).
- `benchmarks/topic_shift_gen.py`: six prompts that each ask for three unrelated pieces in one long reply, generated per arm in a fresh process, with a shuffled arm-free grading sheet. Both are live-run instruments built on the tested helpers; they have no unit tests of their own.

### Measured, not shipped as a default
- Budget 0: +0.11 nats a token in the 100 tokens after a shift (top-1 agreement 83 %), fading within about 200; +0.040 [+0.011, +0.070] over the prose segment, about +0.02 over the stream; the step stays 24-27 % faster. The cache is not frozen: predicted prefetch loads still admit experts. Budget 1: about 6 % faster, no measurable cost. Six generated three-topic replies, graded blind: flawed 1/6 against 1/6, no loops, 120.0 to 90.9 ms a token (-24 %). Small samples, one grader, short context.

## 0.57.1 (2026-10-04)

HANDOFF "Start here (2026-10-04, 0.57.1)" and section 18.60. The blind agent-quality check of the DeepSeek decode miss budget.

### Added
- `benchmarks/quality_blind_ab.py` (`run`, `sheet`, `score`) and `tests/test_quality_blind_ab.py` (3 tests): the six request bodies of a dumped Hermes session sampled N times per arm at the dump's temperature through `Engine.chat`, arms interleaved with matched seeds; mechanical checks (tool-call validity, repetition, truncation, C# `dotnet build`); a shuffled arm-free grading sheet with a separate key; an exact (Fisher) comparison of flawed rates.

### Measured, not shipped as a default
- Budget 0 against exact, 8 samples per arm per body, 96 replies graded blind: flawed 6/48 against 14/48 (Fisher p = 0.077; not significant, not read as an improvement); valid tool calls 8/8 in both arms, no loops; long replies 136.7 to 111.9 ms a token (-18 %). One grader, six prompts, 700-token cap (the C# build check failed in both arms because the cap cut the code): no sign of harm, a small harm not excluded. A long reply that changes topic is untested.

## 0.57.0 (2026-10-04)

HANDOFF "Start here (2026-10-04, 0.57.0)" and section 18.59. A server knob for DeepSeek's decode miss budget, off by default, and its server-path check.

### Added
- `CACHALOT_DECODE_MISS_BUDGET=N` on `cachalot.cli serve` (DeepSeek): at most N non-resident experts are read per layer in decode, the rest dropped and the layer's router weights rescaled. Unset, empty or negative is off (the default; `serve.sh` does not set it); a non-integer exits with an error. It changes outputs; prefill is untouched, so no snapshot or numerics tag moves. The startup line says so. 3 tests (`tests/test_decode_miss_budget_env.py`).
- `benchmarks/server_miss_budget_ab.py`: the server-path A/B (`Engine.chat`, a 22k Hermes-shaped context, the budget flipped per turn with swapped parity and an exact-vs-exact control) and its report.

### Measured, not shipped as a default
- Budget 0 through the server path at 22k context and 48 GiB: 140.3 to 108.7 ms a token (-22.6 %, paired -31.7 ms, 95 % interval [-37.7, -25.9]; the control differs by +0.9 ms, sd 7.2), 19.5 experts dropped a token; a second run of the driver at 24 tokens a turn: ratio 0.782. Quality at agent scale is unmeasured, so no default is proposed.

## 0.56.1 (2026-10-04)

HANDOFF "Start here (2026-10-04, 0.56.1)" and section 18.58. The full-size Pareto sweep of DeepSeek's per-layer decode miss budget.

### Changed
- `benchmarks/pareto_arms.example.json` gains a `miss-budget1` arm.

### Measured, not shipped
- 576 teacher-forced positions and 21 tasks an arm, seven arms, 48 GiB: `miss_budget` 4 / 2 / 1 are inside the noise band and 0-4 % faster; `miss_budget` 0 (every missing expert dropped) cuts the decode step 131 to 95 ms (-27 %) with dNLL +0.007 [-0.015, +0.028] (the noise arm: +0.008 [-0.007, +0.024]), but a heavier KL tail (mean 0.027 against 0.014, max 1.31 against 0.23) and one more failed task: outside the noise band. A 36 GiB budget is bit-identical and 13 % slower. The budget is reachable only through `V41Model`, not the server, so there is no server-path number yet; default-on is Hamed's call.

## 0.56.0 (2026-10-03)

HANDOFF "Start here (2026-10-03, 0.56.0)" and section 18.57. The Pareto harness, so output-changing levers can be judged.

### Added
- `benchmarks/pareto.py`, `benchmarks/pareto_tasks.json`, `benchmarks/pareto_arms.example.json`, `tests/test_pareto.py` (13 tests): named arms (environment, expert budget, per-layer decode miss budget, prefill chunk), a sweep that runs each arm in its own process, and a report with paired NLL and a block-bootstrap interval, KL against the reference's top-64 tokens, top-1 agreement, a 21-task checkable battery (Python asserts, JSON, C# `dotnet build`) with Wilson bounds, ms and misses a token, the Pareto frontier and a noise band from a numerically equivalent arm.

### Measured, not shipped
- Quick smoke (144 positions, 6 tasks): the noise arm (96-token prefill chunks) moves dNLL +0.014 [-0.018, +0.045], KL 0.011; dropping every decode miss cuts the step 141 to 113 ms and misses 34.5 to 18.5, KL 0.014 (outside the band), dNLL +0.013 with an interval spanning zero. Too little power for a verdict; a full-size sweep is the next step.

## 0.55.2 (2026-10-03)

HANDOFF "Start here (2026-10-03, 0.55.2)" and section 18.56. D2 (speculative verify for DeepSeek) priced from the weighted trace and held.

### Added
- `benchmarks/d2_verify_union.py`: the expert reads of a verify block of K consecutive tokens (the union per layer) from a routing trace, and the speed-up per accepted token for a per-position acceptance, a marginal cost a position and a draft cost.

### Measured, not shipped
- Misses a token are 28.0 at every K (a union of K tokens' misses equals the sum of the single-token misses), so only the 70 ms floor is amortised. At 48 GiB against a 126 ms token: 1.08-1.12x with Rapid-MLX's 8 ms a position at acceptance 0.78, 0.92-1.00x with the 26.9 ms measured on our kernels, 1.11x only at acceptance 0.9. Under the 10 % stop rule: held.

## 0.55.1 (2026-10-03)

HANDOFF "Start here (2026-10-03, 0.55.1)" and section 18.55. D3 (GPU-side expert selection for DeepSeek) priced from the weighted trace and held.

### Added
- `benchmarks/d3_layer_hits.py`: the share of decode layers whose six experts are all resident, per budget, and D3's net saving under a per-layer saving and rewind cost.

### Measured, not shipped
- At 48 GiB 52.8 % of decode layers are all-hit (45.1 % at 36, 54.2 % at 52). With MiniMax's constants (0.63 ms saved a layer, 0.7 ms rewind) D3 nets +0.1 ms of a 126 ms token; with free rewinds 13 ms. Below the 8 ms stop rule: held.

## 0.55.0 (2026-10-03)

HANDOFF "Start here (2026-10-03, 0.55.0)" and section 18.54. MiniMax's floor re-measured, MiniMax shown safe against a wired-memory holder, and the first weighted DeepSeek routing trace.

### Added
- `CACHALOT_ROUTING_TRACE=path` on the DeepSeek server (`cachalot.cli serve`): installs a `RoutingTracer` on the runtime and saves it, with router weights, when the server stops. The weights cost a device read per layer, so record a trace in its own run, never in a timing run. Outputs are unchanged.
- `tests/test_cache_sim_segments.py`.

### Fixed
- `benchmarks/cache_sim.py` replayed zero tokens on a trace without segment marks (every server trace: nothing calls `tracer.mark`), printing 100 % hit and the floor time. It now splits at every prefill/decode change and refuses a trace with no decode tokens.

### Measured, not shipped
- **MiniMax floor (M0):** an all-hit token is 44-49 ms through the server (0.4-1.1 misses a token); the research's 77 ms was the intercept of a narrow fit. Mixed requests: 82.9 ms at 10.7 misses, fit `27 + 5.2 x misses` (r 0.95).
- **MiniMax with a 4 GiB mlock holder:** 86.2 ms at 11.6 misses against 82.9 at 10.7, floor unchanged; no cliff at 68 GiB.
- **DeepSeek weighted trace** (1,958 decode tokens at a 22k Hermes-shaped context, 13,062 of 15,360 experts used): `cache_sim.py` gives 88.3 % hit, 28 misses, 126 ms at 48 GiB against live 87-93 % and 7.7 tok/s. Miss substitution removes 16 % of misses at tau 0.10 (1.6 % routing mass dropped), 52 % at 0.15 (9.5 %).

## 0.54.0 (2026-10-03)

HANDOFF "Start here (2026-10-03, 0.54.0)" and section 18.53. A wired-memory governor for DeepSeek, built after §18.52 found the edge.

### Added
- `src/cachalot/cache/wired_governor.py` and a hook in `TextDecodeRuntime` (`_wired_fit`): between tokens (the first, then every 16) and before a prefill, the runtime reads the system's wired memory (`vm.page_wired_count`, microseconds) and, above a ceiling, gives expert slots back through
  `ResidentExpertStore.set_capacity`; it takes them back only after 60 s without a shrink and 1.5 GiB under the ceiling. On by default; `CACHALOT_WIRED_CEILING_GIB` (default 76 % of RAM = 73.0 GiB; 0 or negative off), `CACHALOT_WIRED_HYSTERESIS_GIB` (1.5),
  `CACHALOT_WIRED_GROW_QUIET_S` (60), `CACHALOT_WIRED_CHECK_EVERY` (16). A fit prints `wired fit: expert slots A -> B (system wired X GiB, ceiling Y)`. Outputs are unchanged (only which experts stay resident).
- 12 tests (`tests/test_wired_governor.py`).

### Changed
- `serve.sh`'s comment names the real cause. The 0.53.0 changelog, README and HANDOFF suspected the script's 80 GiB wired limit: refuted (a limit of 84 GiB with a 52 GiB budget decoded at the same 4.74 tok/s). The edge is the whole system's wired memory (~74.5 GiB), whoever wires it.

### Measured
- Floor against system wired memory (same greedy prompt, 3 repeats): 12.7 / 12.6 / 12.4 / 11.0 / 6.3 tok/s at 71.7-72.2 / 72.6-73.4 / 73.6-74.2 / 74.3-74.6 / 75.1-75.3 GiB (budgets 48 / 49 / 50 / 51 / 52).
- Another process wiring 4 GiB (mlock), the default 48 GiB server on the 12-request mixed set: governor off **4.85 tok/s** (fit `ms = 161 + 1.78 x misses`), governor on **7.92** (`67 + 2.27 x misses`); no holder, governor on 7.98. 52 GiB under the governor: 8.13 (a ceiling of 73.9 GiB gave 7.57: too close to the edge).
- In-process `decode_resident.py`: all-resident floor 182 ms at a 52 GiB budget (wired 75.5), 76 ms at 36: the floor depends on the pool's wired size alone, not on residents, hits or the server.

## 0.53.0 (2026-10-03)

HANDOFF "Start here (2026-10-03, 0.53.0)" and section 18.52. D0 ran: DeepSeek measured through `./serve.sh` on a settled machine.

### Changed
- `serve.sh` serves DeepSeek at a 48 GiB expert budget (was 52). With the script's 80 GiB wired limit, 52 GiB takes the GPU's total system allocation to 80.5 GiB and an all-resident token costs 164 ms instead of 70: **4.76 -> 7.92 tok/s on a
  mixed 12-request set (+66 %)**, and a cold 4.5k-token prefill 52 s instead of 66 s (n=1 each). Same outputs: the budget changes residency, not arithmetic. `chat.sh` (44 GiB, 72 GiB wired) is unchanged; a chat run by hand should use 48, not
  the 52 the README recommended for 0.9.x.

### Added
- `benchmarks/cache_sim.py`: a trace-driven decode cost model (hit rate, misses a token, ms a token and tok/s per budget and drive, plus the MiniMax-style miss drop with the dropped router mass as a quality proxy). It warns below 1,000 decode tokens;
  the only DeepSeek traces in the repo have 160.
- `RoutingTracer.record(..., weights=)` stores the router weights (saved only when every record has them; older traces load unchanged); the decode and prefill trace hooks pass `route.weights`.

### Measured, not shipped
- Decode as it runs today (internal bank, settled machine): the fit is `ms = 70 + 2.0 x misses` at budgets 36-48 GiB (r 0.95-0.98), floor and miss cost both as in the 0.9.x README. Prefetch precision 39-41 %, 33-42 predicted loads a token.
- Engram row reads over USB: 3.90 ms a token (two layers x 24 random rows, 300 trials), prefetched at the top of the token, so not part of the floor. Prediction off at 48 GiB: 7.65 against 7.87-7.92 tok/s (neutral, kept).

## 0.52.3 (2026-10-03)

HANDOFF "Start here (2026-10-03, 0.52.3)", section 18.51, and `docs/SPEED-RESEARCH-2026-10-03.md` (revision 2: DeepSeek first). No runtime change.

### Changed
- `serve.sh` and `chat.sh` read DeepSeek's 2-bit bank from `~/DeepSeek-V4.1-Flash-q2g128` (internal SSD, ~6.8 GB/s) when it carries the marker `.cachalot-verified`, else from the X10Pro (1 GB/s). The marker is written only after a
  byte-for-byte compare with the X10Pro copy (done: 42 files, 0 differences).
- `serve-glm.sh` and `chat-glm.sh` fall back to `/Volumes/X10Pro/models/GLM-5.3-Flash-MLX-4bit-MTP` when no internal copy exists. The internal GLM copy (169 GiB) was removed after a byte compare with that one (53 files, 0 differences).

### Measured, not shipped
- First DeepSeek run on the internal bank: 4.1-5.1 tok/s at a 90-96 % hit rate, taken at memory pressure level 12 with swap at 5.6 of 7 GiB and the screensaver on: not a valid speed figure (README 0.9.x: 9.4-9.6 tok/s). The server's
  counters showed 47k predicted loads with 16.8k used (36 %).
- Research, revision 2: DeepSeek-V4.1-Flash added. Public REAP checkpoints cost +2.8 % (25 % pruned) and +16.8 % (50 %) wikitext perplexity; Rapid-MLX measured DSpark at 2.02x on an M3 Ultra with the target resident;
  a framework-free engine (ds4) decodes a 2-bit V4 Flash at 25-36 ms a token against our 77 ms all-resident floor.

## 0.52.2 (2026-10-03)

HANDOFF "Start here (2026-10-03, 0.52.2)", section 18.50, and `docs/SPEED-RESEARCH-2026-10-03.md`. No runtime change.

### Added
- `docs/SPEED-RESEARCH-2026-10-03.md`: a literature and project survey (REAP, HOBBIT, SliceMoE, MoE-CORE, residency-aware self-speculation, flash-moe, public REAP checkpoints of GLM-5.3-Flash and
  MiniMax-M3), a review of the project's own method, an estimate of speed against hit rate for both models, and ten ranked untried levers with the measurement each needs first.

### Measured, not shipped
- A regression of ms a token on misses a token over 14 MiniMax replies (this session, default settings): `ms = 77 + 3.42 x misses` (r 0.65, n 14, narrow range). A miss costs its full raw read time
  (21.1 MiB at 6.5 GiB/s = 3.2 ms), so reads are not hidden from the token, and `decode_wait_ms_per_miss` (1.2 ms) understates the cost about three times. The 77 ms intercept against the 35-43 ms
  all-hit floor of 0.38.0 is the open question (up to ~30 % of a token).
- Public REAP checkpoints exist for GLM-5.3-Flash (`pipenetwork/...REAP25/37/50`: perplexity 3.46 -> 4.88 / 6.08 on wikitext-2) and MiniMax-M3 (`JANGQ-AI/...REAP22/32-Coder`); none publishes its keep-list.

## 0.52.1 (2026-10-02)

HANDOFF "Start here (2026-10-02, 0.52.1)", section 18.49. The default model path is unchanged.

### Added
- GLM model directories may carry `nonexpert-sanitized.safetensors`: the non-expert weights of another quantisation in mlx-lm's layout (`language_model.*` names). When present, `GlmModel` loads
  them instead of the shards' non-expert tensors, renames mlx-lm's `forget_gate.*` and fused `conv1d` to the names `LanguageModel.sanitize` fuses from, and reads each module's bit width off its tensors
  (packed columns per scale group), so a mixed 4/8-bit checkpoint loads next to 4-bit expert shards. Unit tests for the renaming.

### Measured, not shipped
- A mixed-precision GLM-5.3-Flash did not stop the garbled C#. The hybrid: this machine's 4-bit experts plus `pipenetwork/GLM-5.3-Flash-MLX-mixed-4_8bit`'s non-expert tensors (attention, delta-rule
  projections, shared experts, lm_head at 8 bits; 8.9 GiB fetched by byte range instead of 170 GiB). Its expert scales and biases equal the local ones byte for byte and its 4-bit codes differ in 0.03 %
  of positions (by one). The same C# replay as 0.51.7: Hermes bodies at temperature 0.7, **12 of 12 syntax garbles** (11 of 12 excluding one reply cut off at 3,000 tokens) against GLM 4-bit's 8 of 12 and
  MiniMax's 1 of 12; one message, 4 of 8 garbled, 3 clean, 1 package-only, against 3 of 4. The 4-bit non-expert weights are not what garbles GLM's code.

## 0.52.0 (2026-10-02)

HANDOFF "Start here (2026-10-02, 0.52.0)", section 18.48.

### Added
- **GLM-5.3-Flash reads video** through `./serve-glm.sh`: a `video_url` (also `video`, `input_video`) content part with a `data:` URI, an http(s) URL or a local path. The part is a port of
  mlx-vlm's `Glm5NextVideoProcessor`: frames are sampled at the checkpoint's 2 fps (`vision.sample_indices`), decoded with `ffmpeg` (needs `ffmpeg` and `ffprobe` on the PATH or in
  `/opt/homebrew/bin`), resized under a token budget for the whole clip, and cut into temporal steps of two frames. Each step is one image-like span through the existing tower, and the
  `<|video|>` marker expands to `<|begin_of_image|>`, the step's `<|image|>` run, `<|end_of_image|>` and `"<seconds> seconds"` per step, as mlx-vlm's processor writes it.
  Prefix keys carry the video's content hash per step, so a resent clip reuses its prefix.
- `CACHALOT_GLM_VIDEO_MAX_TOKENS` (default 4000, the `<|image|>` tokens of a whole clip; the checkpoint allows 240,000) and `CACHALOT_GLM_VIDEO_MAX_FRAMES` (default 128; mlx-vlm 2048).
  At the measured ~50 tok/s prefill, 4,000 tokens is about 80 s.
- Tests: video part records, sampling counts, temporal-step patches, mixed image and video expansion.

### Changed
- The tower's output cache holds 192 images or video steps (was 8), so a clip's steps stay cached for a second question.

### Measured
- Against mlx-vlm 0.7.4 (`~/venvs/mlxvlm-compare`): the sampled frame numbers are identical for eight (frames, fps) cases; the patches and grid of a 14-frame 480 x 270 clip are bit-identical
  (`[7, 20, 36]`, max difference 0.0); the expanded prompt is token-identical to the tokenised mlx-vlm text.
- Live: a 6 s, 448 x 336 test clip (a blue circle sliding right with "ALPHA", a red square sliding right with "OMEGA" from 3 s) is read correctly: the text change at 3 seconds and the
  blue-to-red change; the shape count was wrong ("two overlapping" circles for one). 1,221 prompt tokens, prefill 24.8 s, decode 3.0 tok/s; the same clip again reuses all 1,221 tokens (0.001 s).

## 0.51.10 (2026-10-02)

HANDOFF "Start here (2026-10-02, 0.51.10)", section 18.47. No runtime change.

### Measured, not shipped
- GLM-5.3-Flash reads an image that sits in a **system** message: a 448 x 448 synthetic image (a red square, a blue circle, the text "TIDE 731") sent once in a system
  message and once in the user message (temperature 0, `./serve-glm.sh`) was described correctly both ways ("A red square and a blue circle appear above the black text
  \"TIDE 731\"."). 300 and 286 prompt tokens, prefill 50.9 s and 53.5 s (both include the tower's first load or a cold path), decode 0.30-0.35 tok/s on a freshly started server.
  `vision.image_records` already collects image parts from any message and the chat template renders the image markers for the system role, so no code was needed.
- Video on GLM priced on paper, not built: the vendored tower already takes a temporal grid and the checkpoint's `video_processor` names 2 fps and a temporal patch of 2, but
  `vision.py` says "Videos are not supported" and the server has no video part, frame decoding or `<|video|>` expansion.

## 0.51.9 (2026-10-01)

HANDOFF "Start here (2026-10-01, 0.51.9)", section 18.46. No runtime change.

### Added
- `benchmarks/nand_temp.c`: reads the internal SSD's NAND temperature from the HID sensor (`smartctl` is not installed), one line per period, for lining up with
  `iostat` and a benchmark's `EPOCH T0=`. Idle 35 C; 44-54 C during a prefill.

### Measured, not shipped
- The DeepSeek prefill drift (Job 3, §15.13) cannot be asked as written: the internal DeepSeek expert bank was deleted in 0.48.1, so DeepSeek's prefill streams
  its experts and Engram tables from the X10Pro over USB, and the internal drive carried 16-24 MB/s during the trace. One cold 12,288-token prefill in that
  configuration (distinct 60k-character window, keep-alive 0.5, 52 GiB budget) took 1,131 s (chunks 234 / 536 / 335 s; the internal bank took 124-130 s), the
  X10Pro averaging 164-542 MB/s with bursts to 938, memory pressure level 9 and swap 4.3 to 5.1 GiB during the run, the screensaver running. Contaminated by
  memory pressure, so it shows only that this setup swaps; it says nothing about drift.
- Harness pitfall: `FILLER_OFFSET` is ignored without `FILLER_FILE` (the first sweep prefilled the same text every run: identical logits, 633 s against
  1,131 s for fresh text).

### Closed
- Job 3 as a thermal-drift question on the internal drive (no internal bank to heat). Reopen only with a guarded rerun of six cold prefills at a lower budget
  (about 44 GiB, screensaver off, ~2 h) against the X10Pro-bound configuration.

## 0.51.8 (2026-10-01)

HANDOFF "Start here (2026-10-01, 0.51.8)", section 18.45. No code changed.

### Measured, not shipped
- A higher-precision pipeline for GLM-5.3-Flash does not lower the prefill-against-decode spread. Same 260 tokens of prose (24 prefilled, 236 decoded one at a
  time, 52 GiB budget), layer outputs of one batched pass against one-token decode: library code, a patched generic path with the hyper-connection residual
  stream rounded to bf16 as shipped, and the same path with the stream kept in fp32. Median relative error at layer 23: 0.185 / 0.189 / 0.187; at the last
  layer 0.180 / 0.238 / 0.178; KL of the logits 0.0172 / 0.0213 / 0.0168; argmax agreement 0.949 / 0.966 / 0.957. The linear-attention recurrent state is
  already fp32 in the library (`gated_delta_update` allocates it as float32). Any change of rounding order moves the final hidden state ~20 % (patched bf16
  path against the library, last layer 0.22), so the noise is the model's own sensitivity, not the stream's storage precision.
- The streamed against non-streamed difference of 0.51.5 (9/12 against 0/4) has no code path to come from: `Engine.chat` consumes `Engine.stream_chat`, so both
  run the same generator. It was sampling noise on small n.

### Closed
- Higher-precision residual stream and linear-attention state for GLM, and the streamed/non-streamed lead. Open: another GLM-5.3-Flash quantisation (Hamed's
  call; a download), video and a system-message image on GLM, the DeepSeek prefill drift.

## 0.51.7 (2026-10-01)

HANDOFF "Start here (2026-10-01, 0.51.7)", section 18.44. No code changed.

### Measured, not shipped
- The C# replay of 0.51.5 on MiniMax-M3 (same bodies, same `dotnet build` harness, ImplicitUsings on for every saved block of both models): syntax garble 1
  of 19 replies against GLM-5.3-Flash's 12 of 21 (8 of 12 against 1 of 12 on the same Hermes arm, Fisher p about 0.005). MiniMax's failures are wrong API
  calls (5 of 12 Hermes replies), not broken tokens. GLM's corruption is therefore GLM's (4-bit weights or architecture sensitivity), not the runtime's cache,
  prompts or harness. MiniMax decoded at 7.5-7.6 tok/s at 23.7k context, hit 93-94 %.

## 0.51.6 (2026-10-01)

HANDOFF "Start here (2026-10-01, 0.51.6)", section 18.43. No code changed.

### Measured, not shipped
- GLM-5.3-Flash's corrupted code (0.51.5) is not a decode-path fault. Same 900 tokens through batched prefill and one token at a time: KL mean 0.0153,
  median 0.00002, max 1.96, identical with `CACHALOT_GLM_PREDICT_TOPK=0` (the prefetch is exonerated); relative hidden-state error 1.2 % in layer 0 (no MoE)
  growing to ~10 % median by layer 23. Teacher-forced NLL of neutral text is the same under both modes (decode minus prefill -0.015 on Python, -0.003 on
  prose, -0.003 on the greedy C# text, every 95 % interval spanning zero).

### Closed
- The decode-path hypothesis of 0.51.5. Open: a MiniMax baseline on the same C# replay, a higher-precision pipeline, fidelity to a reference implementation.

## 0.51.5 (2026-10-01)

HANDOFF "Start here (2026-10-01, 0.51.5)", section 18.42. No code changed.

### Measured, not shipped
- GLM-5.3-Flash's vision tower and image preprocessing against mlx-vlm 0.7.4 (own venv `~/venvs/mlxvlm-compare`, mlx 0.32.2): patches and grids
  byte-identical, tower rows bit-identical (max difference 0) on seven image sizes up to 2160x3840 (7,973 rows), real checkpoint weights.
- Three Hermes sessions on `./serve-glm.sh` at `--expert-budget-gib` 44 / 48 / 50, one per arm: decode 2.79 / 3.00 / 3.10 tok/s over six turns
  (about 1.7 % per GiB), hit rate rising, a ~4.7 s floor for a tiny follow-up prefill at every cap, the screenshot read correctly at all three (the image
  goes in natively, `images=1`). Swap flat at 44 and 48; at 50 one tool-result prefill peaked MLX at 72.2 GiB and swap rose 0.7 GiB.

### Found, open
- GLM-5.3-Flash's C# output is corrupted: 9 of 12 non-streamed replays of the Hermes request failed to compile, 3 of 4 with no system prompt and no
  tools, and temperature 0 gives the same wrong text twice (a code fence closing mid-class, a stray line in a `for` header, a cut-off snippet). Not the
  budget, the sampling temperature or the 21k prefix. Whether the 4-bit weights or a decode-path fault in the runtime cause it is not yet known.

## 0.51.4 (2026-09-30)

HANDOFF "Start here (2026-09-30, 0.51.4)", section 18.41.

### Measured, not shipped
- GLM-5.3-Flash vision on real images through `./serve-glm.sh` (two 3,840 x 2,160 wallpapers, each resized to ~8,000 image tokens, greedy,
  thinking off). A misty-forest photograph: scene, colours, moss, "no people or text" all correct. A cyberpunk illustration: called an
  illustration, "Origin" and "Behring" signs read correctly, one brand name ("Weyland") not visible in the image (a hallucinated detail).
  Prefill of an ~8k-token image prompt 178-287 s, decode 3.2-3.5 tok/s; the same image resent reuses all 8,004 tokens (prefill 0.00 s, same text).
  No code changed.
## 0.51.3 (2026-09-30)

### Documentation
- `docs/next-session-prompts/v84-2026-09-30.md` added: the current prompt also lives in that folder (as v77's did: the file to paste is
  the versioned copy and is identical to `docs/NEXT-SESSION-PROMPT.md`). 0.51.2 had archived v78-v83 but not the current v84.
## 0.51.2 (2026-09-30)

HANDOFF "Start here (2026-09-30, 0.51.2)".

### Documentation
- The release round of the 0.48.0-0.51.1 session: the next-session prompt is v84 (the superseded v78-v83 are archived in
  `docs/next-session-prompts/`, rebuilt from the commits that wrote them, because each had been edited in place), with one consolidated account
  of the session, the GLM levers closed or held, the new rules (per-turn alternation for ~10 % decode levers, cold pages for cache benchmarks,
  `MiniMaxModel` subclasses `GlmModel` without `__init__`, Hermes self-updates and its vision timeout) and the new instruments; HANDOFF has one
  start-here block for the whole session and a refreshed header; the README lists the three releases (items 60-62).
## 0.51.1 (2026-09-30)

HANDOFF section 18.39 (Hermes screenshot session).

### Measured
- **A Hermes session with a real screenshot on GLM** (Hermes CLI, an isolated home, `./serve-glm.sh`; a 2,364 x 2,016 Desktop image = 6,120
  image tokens, 6,168-6,188 prompt tokens): GLM's reading was accurate against the image (a Cachalot logo: concentric light-blue, royal-blue and
  navy rings with a slot cut into the left side, a four-pointed star in the centre, "CACHALOT" in bold navy capitals, a light-grey background with a
  faint grid, no numbers). Each vision pass prefilled the image in 62-163 s and decoded 485-728 tokens at 2.5-3.5 tok/s (140-215 s).
- **Hermes routes `--image` through its `vision_analyze` tool** (the aux vision task on the same server) whose default 120 s timeout is
  shorter than one pass, so the first run timed out on every attempt and looped (five image requests); with `auxiliary.vision.timeout: 1500` in
  the Hermes config the session answered (it also downscaled the file after two timeouts).
## 0.51.0 (2026-09-30)

HANDOFF sections 18.39-18.40.

### Added
- **GLM-5.3-Flash reads images** (`cachalot.glm.vision`, the vendored `glm5_next` vision tower, `image_url` parts through
  `./serve-glm.sh` and the engine). The checkpoint's 24-block ViT (`model.visual.*`, ~0.9 GiB) loads on the first image; images are
  preprocessed by a port of mlx-vlm's processor (an aligned canvas under an 8,000-token budget, PIL bicubic, 2x2 merge groups); each
  `<|image|>` marker becomes its image's run of tokens and the tower's rows replace those embeddings in prefill. Saved prefixes are
  keyed on the image's content hash, so a second turn that resends the image reuses it (338 of 349 tokens, 2.4 s) and a different
  image of the same size does not. `/v1/stats` carries `images_served`. Videos are not supported.
- Checked: two charts of one size read exactly (every label and value), an HTTP request with a base64 image answered correctly, 476
  tests (`tests/test_glm_vision.py` covers the resize, the patch layout, normalisation, marker expansion and cache keys).

### Fixed
- (Caught before release) the shared engine called a `has_vision()` that MiniMax's model, a `GlmModel` subclass that skips its
  `__init__`, could not answer; `has_vision` now returns False without a config and a test pins it.

### Measured, not shipped
- **MTP for GLM priced and closed on arithmetic** (HANDOFF 18.40): consecutive tokens share 21-28 % of a layer's experts, so a
  two-token verify reads ~1.75x the experts for 1.58 tokens at the published 58 % acceptance: ~11 % more reads per token on a
  drive-bound decode, before the MTP block's own 288 experts.
## 0.50.4 (2026-09-30)

HANDOFF section 18.38.

### Measured
- **The first Hermes sessions on GLM-5.3-Flash through `./serve-glm.sh`** (an isolated `HERMES_HOME`, `CACHALOT_SERVER_DUMP`, the
  usual four prompts: hi, the Desktop listing, a C# snippet, a 200-word story; 8 server requests each), the next-layer prefetch
  on (`CACHALOT_GLM_PREDICT_TOPK=5`) against off (0): decode 3.01 tok/s (1,899 tokens) against 2.62 (2,117 tokens), +14.5 %, faster
  in all eight matched requests (+4 to +25 %); the same hit rates and misses a token. Hermes's 14,282-token first request prefilled
  in 156 s (91 tok/s) and a restarted server reused it in 1.3 s. Replies were sound (8 folders and 22 files in the listing, a
  working C# importer without a NuGet dependency, a clean story). One pair of live sessions: replies differ (temperature 0.6), so
  this is an observation next to the per-turn alternation's -8.4 %, not a rate.
## 0.50.3 (2026-09-30)

HANDOFF section 18.37.

### Measured, not shipped
- **GPU-side expert selection for GLM (S1c) closed on the hit-rate arithmetic.** With the prefetch off at a 52 GiB budget only
  16.3 % of decode layer-calls have every expert resident (34 % have at most one miss; mean 2.6 misses a layer, 1,968
  layer-calls). MiniMax's loop keeps the GPU running only through all-hit layers; GLM's would rewind and re-run the next
  layer's attention (and, for its 34 linear-attention layers, restore a recurrent state) at ~84 % of layers.
- **The X10Pro mirror for GLM decode** (the checkpoint copy on the X10Pro, `CACHALOT_MIRROR_PATH`), per-token alternation on
  swapped pairs, 100 teacher-forced tokens: `split` mode at 0.10 cuts store wait 5.2 % in both orders (172.9 vs 182.4 and 175.3
  vs 185.0 ms; wait per miss 1.94 vs 2.06 and 1.98 vs 2.08) and a token 0 % and 6.8 % (noisy); the default `pieces` mode at 0.13
  1-3 % of wait. Not made a default: GLM has no adaptive share, and a busy X10Pro cost MiniMax 3x on short prefills (0.45.0).
  To try it: `CACHALOT_MIRROR_PATH=/Volumes/X10Pro/models/GLM-5.3-Flash-MLX-4bit-MTP CACHALOT_MIRROR_FRACTION=0.10
  CACHALOT_MIRROR_MODE=split ./serve-glm.sh`.
## 0.50.2 (2026-09-30)

HANDOFF section 18.36.

### Measured, not shipped
- GLM's 16-bit (scale, bias) pair index for expert slots (G6) priced and held. Every one of the 12,384 experts fits a fixed
  2,048-entry table per projection (most pairs in one projection: 1,405), so a slot would shrink from 13.50 to ~12.77 MiB
  (5.75 % more slots). The same gain emulated with a 55 GiB budget cut misses a token by 4-7 % (mean 5.5 %, three rounds), which
  is ~2.5 % of a token at best before the cost of rebuilding scales and biases in the matmul kernels, below what this Mac's decode
  measurements can resolve (+-5 % with per-turn alternation, +-10 % between processes).
## 0.50.1 (2026-09-30)

HANDOFF section 18.35.

### Measured
- 0.50.0's GLM next-layer prefetch confirmed through `GlmModel.stream()` with agent-shaped turns (a 3,000-token prompt, then
  turns that append the reply and 150 new tokens; 48 greedy tokens a turn), the knob alternating per turn inside one process
  and the parity swapped in a second: on the same turn's context K=5 was faster in all six turns, -5 to -12 %, mean 375 against
  409 ms a token (-8.4 %), identical tokens. Four separate processes in a row (K 0, 5, 5, 0) showed only -1 % (416 against
  420 ms), because the run-to-run drift is +-10 %; that method cannot see a 9 % effect.
## 0.50.0 (2026-09-30)

HANDOFF section 18.35.

### Added
- **GLM-5.3-Flash decode reads the next layer's likely experts early, on by default** (`cachalot.glm.experts`,
  `CACHALOT_GLM_PREDICT_TOPK=5`, `CACHALOT_GLM_PREDICT_LIMIT`, `CACHALOT_GLM_PREDICT_AFTER_DEMAND=1`; `PREDICT_TOPK=0` turns
  it off). Each decode MoE layer scores the next layer's router on its own MoE input in the same sync as its routing and
  starts reading that layer's five best-ranked non-resident experts once its own misses are in. Same tokens: greedy
  decode of 64 tokens is identical with and without it. Measured with in-process alternation on swapped pairs (80
  teacher-forced tokens, 52 GiB): store wait -18 to -25 %, a token -9 to -13 %; in separate processes on the same text
  404 -> 325 ms a token (2.48 -> 3.08 tok/s). Wider prediction (8 and up) gains nothing: the drive is the wall and the
  extra reads are experts nobody asks for.

### Measured, not shipped
- Scoring the next layer's router on this layer's MoE input finds 67 % of the next layer's experts and 59 % of its misses
  at the top 8 (82 % and 79 % at 16), against 28 % and 0 % for the previous token's set (1,968 layer-calls).
## 0.49.0 (2026-09-30)

HANDOFF section 18.34.

### Added
- **A contiguous GLM-5.3-Flash expert bank, off unless selected** (`cachalot.glm.bank`, `benchmarks/glm_bank.py`,
  `CACHALOT_GLM_BANK=<dir>`, `CACHALOT_GLM_BANK_ENABLED=0` ignores it). One 13.5 MiB record per expert in the slot's own
  tensor order, one `layer-NNN.bin` per MoE layer, so a read is one pread instead of up to nine (the checkpoint scatters an
  expert over 1-9 ranges). A bank may cover some layers only. Bit-identical: each written layer is byte-compared with the
  checkpoint on 16 random experts, and greedy decode through a 3-layer bank gave the same 32 token ids and the same 103.3
  misses a token as the checkpoint.

### Measured, not shipped
- The bank's speed effect is not measured: a read timing of bank against checkpoint on this Mac served both from the file
  cache (0.6-1.7 ms for 13.5 MiB), so it says nothing, and decode through three of 42 layers moved nothing (2.92 against
  2.91 tok/s). On paper the gain is small: the reader already issues a multi-range expert's pieces concurrently. Decode is
  ~70 % store wait at the drive's 5.3-5.8 GiB/s, so bytes per token, not read count, are the limit.
## 0.48.1 (2026-09-30)

HANDOFF section 18.33.

### Changed
- **The launch scripts follow the models' new places.** GLM-5.3-Flash is back on the internal SSD
  (`~/GLM-5.3-Flash-MLX-4bit-MTP`, copied from the X10Pro; `serve-glm.sh` and `chat-glm.sh` default to it, and
  `CACHALOT_GLM_PATH` overrides). The internal DeepSeek expert bank was deleted to make room (Hamed's call; a
  checksum compare against the X10Pro copy found no difference) and `serve.sh` / `chat.sh` now default to the X10Pro
  copy (`Flash4-1/DeepSeek-V4.1-Flash-q2g128`), so DeepSeek reads over USB; `CACHALOT_EXPERT_BANK` overrides.

### Measured, not shipped
- GLM's bias-code idea (MiniMax's bank) does not carry over: biases take 9 values of k, and 8.6 % of 608.7 million
  sampled groups do not satisfy `bias = bf16(k x scale)`. GLM baseline on the internal SSD, 52 GiB budget, 2k-token
  rounds: prefill 72-100 tok/s, decode 2.0-2.9 tok/s (hit rate 60-70 %, ~100-135 misses a token, 5.3-5.8 GiB/s).
  A 64 GiB budget cuts misses a token by 16-26 % but pushes wired memory to 87-90 GiB and makes tokens 1.5-2.3x
  slower (paging): closed on this Mac.
## 0.48.0 (2026-09-30)

HANDOFF section 18.32.

### Changed
- **MiniMax's terminal chat samples at temperature 0.7** (`chat-minimax.sh`, was the checkpoint's 1.0; Hamed's call,
  matching the server default since 0.47.0). `./chat-minimax.sh --temperature 1.0` restores 1.0. top_p stays 0.95.
  No numerics change: prefill KV is unaffected and saved snapshots stay valid.

### Measured
- Hamed's first chat session at 0.7 (four turns, display on): decode 12.5 tok/s on the story, 9.0 on C#, 8.7 on
  TypeScript, expert hit 94-96 %, follow-up prefills 1.4-2.5 s for 19-41 new tokens, drive 6.62 GiB/s. The first
  turn's 164-token prefill took 7.7 s while the warm set was still reading back (9.9 s). The story is 206 words. The
  TypeScript snippet is correct; the C# snippet does not compile (`value` is never assigned) and reads dynamic
  records through reflection. One sample each, not a rate.

## 0.47.0 (2026-09-30)

HANDOFF section 18.31.

### Changed
- **MiniMax's server samples at temperature 0.7 when a request sends none** (`serve-minimax.sh`, was the
  checkpoint's 1.0; Hamed's call). A client's own temperature always wins and top_p stays 0.95. Measured in 0.46.0:
  a long tool result derailed agent replies 5 times in 12 at 1.0 and once at 0.7, and tool calls under Hermes's
  system prompt went from 30/32 to 32/32, with reasoning and code tasks unchanged. `./serve-minimax.sh
  --default-temperature 1.0` restores the old default. The terminal chat (`chat-minimax.sh`), GLM and DeepSeek are
  unchanged.

## 0.46.1 (2026-09-30)

HANDOFF section 18.30.

### Documented
- Three Hermes sessions on 0.46.0 with the same prompts. The first two still sampled at temperature 1.0: the
  `temperature: 0.7` in Hermes's config sat under the provider's `models:` mapping, where Hermes reads it as a model
  name. With the entry corrected, the third session's requests all carry 0.7. At 1.0 the directory-listing summaries
  invented details (a wrong size ordering, a folder that does not exist, "240+ files" for 134 lines) and the
  replies wrote "Wait —" asides; at 0.7 the summary was right apart from a screenshot count. Speed was clean in all
  three (saved system block reused in 1.6-1.8 s, decode 6.8-11.2 tok/s). The first request of a new day prefills
  Hermes's 21k-token block again (101 s), because Hermes writes the date into its system prompt.

## 0.46.0 (2026-09-30)

HANDOFF section 18.29.

### Added
- **A loop guard for runaway lists that count up** (`CACHALOT_LOOP_GUARD_INCREMENTING`, default 64, 0 off): a
  GLM/MiniMax reply that ends with 64 list items sharing one template and differing only in one integer that goes up
  by one (`noto 1`, `noto 2`, ...), none of them in the prompt, stops there with `finish_reason: stop` and a
  `[loop guard] ... list items counting up by one` log line. The existing guard only catches exact repeats, so the
  4,829-token runaway of section 18.28 ran to its end; replayed through the tokenizer it now stops at token 1,258.
  None of 1,279 other replies and tool outputs from the dumps and this session's runs trips it; a listing the model
  copies from the prompt never does.

### Measured, not shipped
- **A lower agent temperature for Hermes.** Hermes sends none, so MiniMax samples at the checkpoint's 1.0. The
  directory-listing turn that derailed live, 12 samples each through the server, graded blind: derailed 5/12 at 1.0,
  1/12 at 0.7, 0/12 at 0.5. Tool prompts under Hermes's real system prompt: 30/32, 32/32, 32/32. Reasoning and code
  tasks 35-36/36 at every temperature. With a weak one-line system prompt a low temperature makes the model answer
  some tool requests from memory (3, 6, 7 of 36). Where to set it: an `extra_body: {temperature: 0.7}` on Hermes's
  `cachalot` custom provider, or `./serve-minimax.sh --default-temperature 0.7`. Hamed's call; the default is
  unchanged.

## 0.45.3 (2026-09-29)

HANDOFF section 18.28.

### Documented
- Hamed's second Hermes session on 0.45.x, with a server dump: the reply to a long directory listing derailed (it
  invented a list of forbidden categories) and the next two replies carried it on. Replayed from the dump: the live
  prompts are token-identical to the replay's, and the same turn sampled eight times at the live temperature 1.0
  derailed 0 of 8 times with the default settings and 3 of 8 on the exact path. The cause is temperature-1.0
  sampling on a long tool result, not the miss substitution or the runtime's state. A 44 s first-request prefill
  replays at 4-6 s. Hermes sends no temperature; a lower agent temperature is open.

## 0.45.2 (2026-09-29)

HANDOFF section 18.27.

### Documented
- Hamed's first Hermes session on 0.45.1: the first request after a restart reused the saved system block in 1.75 s
  (the warm set was back in 10 s, before the first message), short turns 1.9-3.1 s, a 5,988-token tool result
  prefilled at 148 tok/s, decode 6.8-9.5 tok/s at 21-29k context, no stalls. One 25-token follow-up took 7.35 s
  (open). The tool call, code and story were clean; the summary of a 134-entry directory listing invented a few
  names and counts. The session ran without `CACHALOT_SERVER_DUMP`, so nothing could be replayed; the next one
  should set it.

## 0.45.1 (2026-09-29)

HANDOFF section 18.26.

### Measured, not shipped
- **Pair-index prefill kernels for MiniMax.** MLX's `qmv_wide`, split-K `qmm_t` and `qmm_t` copied into Metal
  kernels that read each weight group's (scale, bias) from the slot's one-byte index instead of rebuilding them per
  expert: bit-identical to `mx.quantized_matmul` for every row count from 1 to 130, 20-30 % faster per matmul at
  12-64 rows in isolation, but 4-7 % slower over whole layers once the down projection consumes their outputs.
  The rebuild they would remove costs 0.15-0.2 s of a 128-2,048-token prefill chunk (0.6 s of 8k).
- **One grouped launch per projection** (MLX's sorted `gather_qmm_rhs` reading slots straight from the slab pool,
  bit-identical to `mx.gather_qmm`): -13 % at 1,000 tokens, -10 % at 2,048, +23 % at 128, +96 % at 16, 0 % at 8k;
  not bit-identical to the shipped path.

### Closed
- Both kernel designs above, and the per-expert (scale, bias) rebuild as a speed lever.

### Added
- `benchmarks/minimax_prefill_kernels/`: the kernels and the benchmarks that measured them (`rb_price.py`,
  `pm.py`, `lay2.py`, `qmm_m.py`, `sg_test.py`).

## 0.45.0 (2026-09-29)

HANDOFF section 18.25.

### Changed
- **The X10Pro mirror's share of each MiniMax expert read follows both drives' speed.** Every record's weight
  pieces are timed, and the share moves towards the split where the internal SSD's and the mirror's pieces finish
  together: never above the configured 13 % (the optimum on an idle X10Pro), never below 2 %. With another reader
  on the X10Pro (Spotlight, a copy), a fixed 13 % made every read wait for its USB piece; now decode reads are 12-17 %
  shorter and decode 7 % faster in that case, and nothing changes on an idle drive. Same bytes, same outputs.
  `CACHALOT_MINIMAX_MIRROR_ADAPT=0` keeps the share fixed; a `[bank] mirror share A -> B` line appears at most once a
  minute when it moves.

### Fixed
- HANDOFF's 0.44.0 "Start here" block had lost the name of `CACHALOT_HOST_GROW_QUIET_S=0`.

### Documented
- Hamed's first Hermes session on 0.44.0: the first request after the restart reused the saved block (7.2 s),
  short follow-ups 1.9-7.2 s, decode 7.3-10.2 tok/s at 21-33k. Hermes's approval check (2,207 tokens) took 86.8 s
  live against 12-16 s in replays, and Hermes timed out on it; part of that reproduces with the X10Pro busy.

## 0.44.0 (2026-09-29)

HANDOFF section 18.24.

### Fixed
- **A restart reuses the agent's system block again.** Snapshot files were named by their tokens alone, so a block
  already saved under another numerics tag (the exact path, an older switch setting) kept the current configuration
  from ever writing its own: since 0.43.0 every restart of `serve-minimax.sh` prefilled Hermes's ~21k-token system
  block cold (98-133 s here, 323.6 s in Hamed's sixth session). The runtime identity is now part of the file name's
  hash; the first request after a restart takes 2-10 s again. Applies to every model's disk snapshots.
- **Parking whole slabs no longer takes the transient slots.** The governor could drive the expert capacity below
  zero (`memory fit: expert slots 1892 -> -16`) and park the slots a prefill's misses load into; it now stops at a
  capacity of 1.

### Changed
- **MiniMax's memory governor watches macOS memory pressure between its fits.** With other apps holding host memory,
  a fit right after a long prefill could read a momentarily high availability and take back 13.6 GiB of expert
  cache at once; swap grew 6 -> 19 GiB and short follow-ups at 30k context took 30-41 s instead of 8-18. A watcher
  now samples the pressure level and available memory every 0.5 s: the cache grows only after 60 s without warning
  pressure and by the lowest availability seen in that window, and each pressure event gives a slab back at the next
  decode token (at most one every 10 s). Server-path replays of Hamed's sixth Hermes session with 2 GiB of GPU and
  12 GiB of host memory held by other processes: pressure samples halved in all four pairs, total decode -5.6 to
  -10.6 % in three pairs (+10.4 % in one), the 120 s follow-up stall gone. Inert on a quiet machine.
  `CACHALOT_HOST_GROW_QUIET_S=0` restores the previous rule; `CACHALOT_HOST_SHRINK_EVERY_S` sets the interval.

### Documented
- The sixth session's `read=` 4.5-5.3 ms a miss and GPU "Alloc" 86.5 GiB are normal on this machine (a clean replay
  shows the same); the follow-up stalls came from host memory pressure.

## 0.43.2 (2026-09-29)

HANDOFF section 18.23 item 8.

### Fixed
- **A reply cut inside an unclosed tool call no longer reaches the client as text.** In Hamed's sixth Hermes
  session a MiniMax reply looped inside a `terminal` call's command until the loop guard stopped it; the unclosed
  call cannot be parsed, and ~6,000 characters of raw call markup went to Hermes as the reply. At a loop-guard or
  length stop the unclosed block is now dropped (logged as `[tool call] unclosed block dropped`).

### Documented
- Hamed's sixth Hermes session on 0.43.0 (both substitution switches on): tool calls right, decode 6.6-9.6 tok/s at
  27-34k context; short follow-ups stalled 28-66 s with every read at 4.5-5.3 ms a miss, and all processes' GPU
  allocation stood at 86.5 GiB against the 86.0 GiB working set afterwards (the paging zone; not attributed to the
  switches).

## 0.43.1 (2026-09-29)

### Fixed
- `cachalot.__version__` (the server's startup line) still said 0.41.0 through 0.42.0 and 0.43.0; it now follows
  `pyproject.toml`.

## 0.43.0 (2026-09-29)

HANDOFF section 18.23 item 7.

### Changed
- **MiniMax miss substitution is on by default, decode and prefill** (Hamed's call after the gates in HANDOFF
  18.21-18.23): `CACHALOT_MINIMAX_MISS_DROP` 0.20, `CACHALOT_MINIMAX_MISS_SUB` 4, `CACHALOT_MINIMAX_PREFILL_MISS_DROP`
  0.20. Not bit-identical. `CACHALOT_MINIMAX_MISS_DROP=0 CACHALOT_MINIMAX_PREFILL_MISS_DROP=0` restores the exact
  path. The numerics tag changes, so the first request after upgrading re-prefills a system block once (snapshots
  written by the exact path stay on disk for it). Benchmark arms that need the exact path must now set both to 0.

## 0.42.0 (2026-09-29)

HANDOFF section 18.23.

### Added
- **MiniMax prefill miss substitution, opt-in** (`CACHALOT_MINIMAX_PREFILL_MISS_DROP=0.20`; 0, the default, is
  exact). In a prefill chunk, a missing expert is not read when every token routed to it gives it under 20 % of its
  routing weight and has a resident runner-up among the next `CACHALOT_MINIMAX_MISS_SUB` ranks (4 when unset);
  only those tokens change, and the prefill read-ahead skips the same experts. About 70 % of a short follow-up's
  prefill waits on reads and a third of its missing experts serve one token. Through the server path, 200-token
  follow-ups at 10-12k context: **5.80 -> 4.61 s (-20.5 %)**. Teacher-forced NLL on three texts: two inside a
  rounding-noise arm's band, one +0.0055 [-0.0007, +0.0114] against the noise arm's +0.0016. Off by default
  because it changes outputs; disk snapshots written with it carry their own numerics tag.
  `CACHALOT_MINIMAX_PREFILL_SUB_MAX_ROWS` (32): experts routed to more tokens are always read.

### Measured
- Decode miss substitution (`CACHALOT_MINIMAX_MISS_DROP=0.20 CACHALOT_MINIMAX_MISS_SUB=4`) on 36 greedy tasks with
  thinking on and tool calls (thinking off and on): exact 31/36, rounding noise 31/36, the switch 33/36 at +26 %
  decode; it fails no task that both others pass. Still off by default.
- A finer memory-governor grain (shrinking a slab) priced and closed: it needs a 2.7 GiB copy peak at the GPU
  ceiling or drops the slab's cached experts.

## 0.41.0 (2026-09-29)

HANDOFF section 18.22.

### Changed
- **MiniMax decodes with a 0.25 GiB MLX buffer cache** (`CACHALOT_MINIMAX_DECODE_CACHE_GIB`; prefill keeps 2 GiB,
  a negative value restores 2 during decode). The memory governor sizes the expert cache with the buffer cache
  emptied, and decode refilled it to 2 GiB, which put all processes' GPU memory above the 86 GiB working set: at a
  30k agent context the next 200-token follow-up prefill took 15-32 s instead of 5-7 in 12 of 35 turns, and in 0
  of 20 with the cap. Mean follow-up prefill 17.2 -> 6.1 s, decode unchanged, same tokens.
- **MiniMax drops persisted system blocks from memory after each request** (`CACHALOT_MINIMAX_SPILL_BLOCKS`, 0
  keeps them): a Hermes session held a 2.4 GiB copy of its system block through every decode token; a new
  conversation now reloads it from disk in ~0.4 s. Bit-identical.

### Measured
- Miss substitution (`CACHALOT_MINIMAX_MISS_DROP=0.20 CACHALOT_MINIMAX_MISS_SUB=4`) at 24k context on three texts:
  every paired-NLL interval against exact spans zero (+0.021 / -0.003 / +0.005), mean KL 1.4x a rounding-noise
  arm's, misses -30 to -55 %, ms a token -14 to -21 %. Still off by default.

## 0.40.1 (2026-09-29)

### Documented
- HANDOFF 18.21 item 10: Hamed's fifth Hermes session on 0.40.0, supervised (memory healthy, no loops). The C#
  request that flailed through nine tool turns replays as a direct answer in 20 of 20 samples (10 with miss
  substitution, 10 without): a rare sampled branch compounding on its own history, not the runtime.

## 0.40.0 (2026-09-29)

HANDOFF section 18.21 item 9.

### Added
- **Runaway-loop guard for GLM/MiniMax replies** (`CACHALOT_LOOP_GUARD_REPEATS`, default 6, 0 off): a reply whose
  last six blocks of 10-200 tokens are the same block, back to back, ends there with `finish_reason: stop` and a
  `[loop guard]` log line. In Hamed's Hermes sessions four MiniMax replies looped (1-6k tokens each; two were the
  compression summary, 3,768 and 7,305 tokens, which is why compression timed out); on all 112 dumped replies the
  guard stops exactly those four (at 839 / 1,592 / 578 / 1,175 tokens) and no other.

### Fixed
- The GLM engine's stop-string path created its cancel event after the model stream had started, so a stop string
  did not end generation; the event now exists before the stream.

### Measured
- Loops are not the miss substitution: one of the four was on the exact path, and 24 samples per arm of the
  looping request (temperature 1.0, top_p 0.95, 23.5k context) looped in neither arm.

## 0.39.3 (2026-09-28)

### Documented
- HANDOFF 18.21 item 8: Hermes compresses early because of `compression.threshold_tokens: 30000` in the Hermes
  config (the system block alone is 21.3k tokens), and the local summary (3,768 tokens, 583 s) hits Hermes's
  timeout. Decode at Hermes's 30k context measured: 5.8-6.2 tok/s exact, 7.7 with miss substitution; short
  prose 13.5-14.0 (no regression against 0.33.0's 12.7-12.9).

## 0.39.2 (2026-09-28)

HANDOFF section 18.21 item 7.

### Fixed
- **MiniMax tool calls opened with a plain `<tool_call>` were lost.** Sampled at temperature 1.0, MiniMax-M3
  sometimes writes the `<tool_call>` token without its `]<]minimax[>[` namespace token in front (one of the two
  tool calls in Hamed's Hermes session). The server only opened a tool block on the namespaced form, so the call
  streamed out as text, Hermes read an empty reply and injected "You just executed tool calls but returned an
  empty response", and every later turn answered that nudge ("I haven't run any tools for that request..."). The
  splitter now opens the block on either form (`TOOL_STARTS`); the parser already accepted both. Two tests, one
  built from the dumped reply.

## 0.39.1 (2026-09-28)

### Added
- `docs/studio/`: the build brief for Cachalot Studio, the desktop app for this runtime
  (`CODEX-STUDIO-PROMPT.md`, runtime contract as of 0.38.1), and `docs/studio/briefs/2026-09-28-runtime-0.39.0.md`,
  the Codex brief for 0.39.0's MiniMax miss-substitution knobs (contract edits, UI changes).

## 0.39.0 (2026-09-28)

HANDOFF section 18.21 (the twentieth MiniMax-M3 speed session).

### Added (off by default; not bit-identical)
- **MiniMax decode miss substitution:** `CACHALOT_MINIMAX_MISS_DROP=0.20 CACHALOT_MINIMAX_MISS_SUB=4`. A missing
  routed expert whose share of its layer's routing weight is under the threshold is not read; the best-scored
  resident expert among the next `MISS_SUB` ranks replaces it with the router's own renormalised weights (without
  one it is left out and the others rescaled). The speculative prefetch skips experts that would be skipped.
  Misses per token -37 to -40 %; teacher-forced decode -14 to -21 % a token, server path (`stream_agent.py`, six
  agent turns, ABAB) decode 104.2 -> 94.6 ms mean (-9 %; -13.5 % within one display state).
- Quality, three texts x 300 teacher-forced tokens: mean KL to exact 0.016 / 0.025 / 0.019 against 0.013 /
  0.018 / 0.030 for a numerically equivalent prefill-chunk change; paired NLL -0.005 / -0.002 / +0.017 (95 %
  bootstrap intervals all include 0). A 24-task greedy battery: 24/24 exact, 24/24 substitution, 24/24 the
  rounding-noise arm; 16 vs 17 of 24 answers byte-identical to exact.
- `cachalot.minimax.gpu_select.miss_plan` (tested) and `CACHALOT_MINIMAX_MISS_DROP_ARMED=1` (weights to the host
  at 0, for `TF_ALTERNATE` arms and the `MISS_SHARES` histogram). Disk snapshots are keyed by the setting.
- `glm_prefill_timeline.py`: `TF_OUT` also saves the target ids; `--compare` prints the paired per-token NLL
  difference with a bootstrap interval; `MISS_SHARES` line.

### Measured and closed
- Dropping missing experts without substitution: a KL 15.7 position on one text. Threshold 0.25 with
  substitution: KL 9-15 positions and NLL +0.085 on one text. Threshold 0.15: fewer misses saved (-13 %) and one
  KL 10.8 position (the rounding-noise arm itself had one of 4.2).

## 0.38.1 (2026-09-28)

HANDOFF section 18.20 (the nineteenth MiniMax-M3 speed session).

### Added
- `benchmarks/glm_prefill_timeline.py` `FIT_PREFILL=1`: the server's slot give-back before a long prefill
  (without it a 16k MiniMax prefill runs out of Metal memory at the 68 GiB cache).

### Measured and closed
- MiniMax prefill is compute-bound: an 8k chunk is 30.0 s (36.4 s at 8-16k) with 0-0.5 s of store wait; MLX's
  3-bit `quantized_matmul` runs 17.5 TFLOPS at MiniMax's shapes, the bf16 peak; ~10 % of a chunk is overhead.
- The per-expert (scale, bias) rebuild costs 1.44 s an 8k chunk (4.0 %, timing-only ablation, ABAB); experts
  with few tokens 0.5 s.
- Queuing the shared expert before the routed experts' sync: no change (bit-identical). Queuing a layer's
  rebuilds first as one batch: +2.4 % slower (bit-identical). Neither shipped.
- M27b (the interrupted warm set evicting the first request's experts): at 68 GiB the free slots already take
  back 1,911 of 3,176 saved experts; evicting for the rest lifts hits 0.948 -> 0.951 and does not shorten the
  next turn. Closed.

## 0.38.0 (2026-09-28)

HANDOFF section 18.19 (the eighteenth MiniMax-M3 speed session).

### Added (off by default)
- **`CACHALOT_MINIMAX_FUSED_ROUTE=1`:** MiniMax decode's routing after the gate matmul (sigmoid, correction bias,
  top-4 in MLX's stable-sort order, weights, slot lookup) as one Metal kernel instead of ~12. Bit-identical
  (`tests/test_gpu_select.py`, 400 random cases with ties; the same NLL and misses per arm in the model).
- **`CACHALOT_MINIMAX_COMPILE_SWIGLU=1`:** the clamped SwiGLU's seven elementwise ops through `mx.compile`
  (bit-identical at decode and prefill shapes).
- In one process (`TF_ALTERNATE`, swapped pairs, 2k context, 68 GiB) they cut the non-read part of a decode token
  by 0.65 and 1.25 ms; through the server path (agent turns, ABAB) decode was 99.3 / 95.5 ms against 94.9 / 94.7
  with the same `ids_hash`: no measurable gain, so both stay off.

### Measured and closed
- Decode read wait is the drives' bandwidth (misses x 21.1 MiB / ~6.5 GiB/s); speculative reads only reorder it.
- The GQA decode kernel is limited by its matrix multiplies (~5 TFLOPS fp32 at 45k); fragments loaded straight
  from device memory are bit-identical and no faster; more or fewer threadgroups no faster.
- INT8 KV (S3): no attention speedup (compute-bound), ~125 slots at 45k (~2.5 %), a quality risk: closed.
- Prefill attention at 45k context runs at 17 TFLOPS in MLX's kernel (a 2k tool result: ~11 s of attention).
- Partial reuse of on-disk system blocks: Hermes's two stored blocks share only their first 1,966 tokens.

## 0.37.2 (2026-09-28)

HANDOFF section 18.18 item 8 (Hamed's second Hermes Desktop session, replayed from its dump).

### Fixed
- **MiniMax's server no longer over-commits memory at startup.** With 0.37.1's 8 GiB prefix budget, startup kept
  the four most recently used snapshots (two 21k-token system blocks, one of them stale: 2.5 GiB) and the whole
  68 GiB expert pool before the first request fitted the capacity: every process's GPU memory peaked at 91.3 GiB,
  swap grew 2.6 GiB, warning pressure. `serve-minimax.sh` now preloads two snapshots
  (`CACHALOT_SNAPSHOT_PRELOAD`, new for every server; the rest load from disk when a prompt starts with them), and
  the capacity is fitted right after they load, before the warm set: startup peak 87.2 GiB, swap flat, normal
  pressure; the first request reused the 21,318-token system block as before.
- **A GPU overshoot smaller than a quarter slab now gives back a slab.** The slab pool keeps up to a quarter slab
  (0.66 GiB) of excess, so the governor's GPU term (18.17) never corrected an overshoot of a few hundred MiB; it
  now asks for at least one whole slab.

### Found
- Three turns of the live session decoded at 2.85-3.51 tok/s (the others 5.2-7.7). Replaying the session's nine
  requests with the display asleep, on 0.37.1 and on this version, decoded them at 5.5-10.7 tok/s: the slow turns
  were not the memory configuration. They were long streamed replies with Hermes Desktop's window on screen, the
  slow window of HANDOFF 15.7/15.8 (not proven here).

## 0.37.1 (2026-09-28)

HANDOFF section 18.18 item 7 (Hamed's Hermes Desktop session on 0.37.0).

### Fixed
- **An agent conversation's snapshot no longer falls out of the MiniMax server's prefix cache when a short side
  request arrives.** MiniMax's KV cache is ~120 KiB a token, so a 43-49k-token Hermes conversation's snapshot
  (5.0-5.7 GiB) alone exceeded the 5 GiB in-memory budget; each of Hermes's short side requests (1,871, 736 and 801
  tokens) evicted it, and the next turn re-prefilled everything after the system block: 169, 211 and 219 s, ~600 s
  of a 33-minute session. `serve-minimax.sh` now keeps 8 GiB (`CACHALOT_GLM_PREFIX_GIB`). Checked through
  `stream()`: 45,000 tokens, a 1,871-token side request, then the next turn reused 45,064 tokens and prefilled in
  8.4 s; normal pressure, no swap. `tests/test_prefix_budget.py` replays the session's order.

### Measured
- 0.37.0 beside Hermes Desktop (M1b): every process's GPU memory peaked at 86.4 GiB against the 87.0 ceiling,
  available memory at 6 %, swap flat, pressure normal. The governor parked to 2,521 slots for every long prefill
  and decoded at 2,654-3,186; decode 5.5-8.8 tok/s at 42-49k context, 8.0 tok/s on a 1,469-token reply.

## 0.37.0 (2026-09-28)

HANDOFF section 18.18.

### Changed
- **MiniMax-M3 caches 68 GiB of experts when macOS's GPU working set is raised to 86 GiB.** With
  `sudo sysctl iogpu.wired_limit_mb=88064` (Hamed's setting; it resets at reboot) Metal's recommended working set
  is 86 GiB instead of 77.76, and the budget can grow past 18.17's 62 GiB ceiling. Agent turns through the server
  path, same `ids_hash` in every arm: **decode 104.7 → 93.0 / 93.6 ms a token (-11 %), turns 53.1 → 46.7 / 47.0 s
  (-12 %), short prefills 23.0 → 19.9 s (-13 %)**; 64 GiB 99.0-99.2, 66 GiB 97.8. 70 GiB grew swap and 72 hit
  warning pressure, so 68 is where the GPU ceiling and host memory meet. Long contexts (8k → 33k through one
  conversation) stay at normal pressure with no swap (peak 75.3 GiB), decode -1 to -7.5 %.
- The scripts pass `--expert-budget-gib 68` and the whole working set as MLX's wired limit
  (`CACHALOT_MLX_WIRED_LIMIT_GIB` 96, capped at the working set). Without the sysctl the startup budget is capped
  at the working set minus 15.75 GiB (`CACHALOT_MINIMAX_GPU_RESERVE_GIB`), i.e. 62 GiB as before; the startup
  reserve of available memory is 16 GiB (was 20).
- **MiniMax's embedding table lives in host memory.** A token needs one row of the 0.5 GiB quantized table, so it
  is read from a map of the checkpoint's own bytes and dequantized by the same `mx.dequantize` (bit-identical).
  At the default working set, with other apps holding 7.3 GiB of GPU memory, this kept the governor from parking
  a slab: decode 113.0 / 112.1 → 107.1 / 104.1 ms, turns 58.2 / 60.5 → 54.5 / 52.9 s (direct benchmark, same
  `ids_hash`). `CACHALOT_MINIMAX_HOST_EMBED=0` keeps it on the GPU.
- Slab pools grow their slabs instead of their count past 27 slabs (Metal binds at most 31 buffers per kernel), so
  any budget builds the GPU-select kernels.

### Found
- A lower available-memory floor (6 GiB instead of 8) changes nothing at 68: the GPU ceiling binds first.

## 0.36.0 (2026-09-28)

HANDOFF section 18.17.

### Changed
- **MiniMax-M3's memory governor now counts every process's GPU memory.** The GPU's memory is shared: with the
  62 GiB expert cache, 1 GiB more of anyone's GPU allocations falls off a cliff. Another process holding 2 GiB of
  GPU memory slowed agent turns from 55 to 79-86 s (short prefills 2.3x, decode +6 %, same tokens) and itself hit
  Metal's out-of-memory error. The governor reads the GPU driver's "Alloc system memory" (all processes, ~20 µs
  through IOKit) and gives back slots, or refuses to grow, while it exceeds Metal's recommended working set plus
  1 GiB (78.76 GiB on the 96 GiB M3 Ultra). Beside the same 2 GiB holder, agent turns take **57.6 s instead of
  78.8-86.1**, and the holder no longer fails; alone, nothing changes (55.5 s, 3,009 slots, same `ids_hash`).
  `CACHALOT_GPU_ALLOC_SLACK_GIB` (1.0; negative turns it off).

### Found
- **Expert budgets above 62 GiB are closed by the same cliff, not by prefill memory.** Through the server path,
  63 GiB (26 slabs) decoded 138.5 / 138.2 / 120.3 ms against 62's 110-114; 64 GiB 192-193 ms (144 with a 0.5 GiB
  MLX buffer cache). Misses fell 8 %, but every token paid for GPU paging. 66 GiB also exceeds Metal's 31 kernel
  buffers (28 slabs).
- Taking back the partial last slab (81 slots) after a long prefill works but decodes no faster (114-117 against
  114.7-115.1 ms); not shipped.

## 0.35.0 (2026-09-28)

HANDOFF section 18.16.

### Changed
- **MiniMax-M3 reads its experts from a slot-image bank (M24).** Each record now holds exactly what a slot holds
  (one byte per weight group indexing a per-projection table of (scale, bias) pairs, then the weights): 21.11 MiB
  instead of 22.16, byte-identical outputs (checked against the original download). Agent benchmark with the
  display asleep, three runs each: **decode 126.9 → 120.5 ms a token (-5.0 %)**, turns -3.3 %, same `ids_hash`.
  The head goes through a scratch buffer (`CACHALOT_MINIMAX_SLOT_HEAD_SCRATCH`, default 1): read straight into the
  slab it cost 3.33 instead of 2.84 ms a miss. `heads.zst` shrank from 4.0 to 3.2 GiB.
  `benchmarks/minimax_coded_bank.py --to-slot BANK` rewrites a bank in place, layer by layer, each layer checked
  byte for byte before the old file goes; `--to-coded` goes back.
- **The X10Pro mirror only holds what the reader takes from it.** `--mirror-tail BANK MIRROR --tail 0.25` writes
  sparse layer files with the last 25 % of each weight piece (36 GiB in a minute instead of a 154 GiB copy);
  the reader caps its mirror fraction at the mirror's `tail` and skips records the mirror does not hold at the
  same place.
- **MiniMax-M3's expert cache is 62 GiB (was 56).** Agent benchmark: 56 → 58 → 60 → 62 GiB decode 119.6 → 111.6 →
  105.5 → **101.7 ms** (-15 %), same `ids_hash`, normal memory pressure; 64 ran out of GPU memory. Prefill chunks
  above 512 tokens (was 2,048) now hand slots back first, down to 52 GiB at 8,192 tokens. Through the server path
  agent turns decode 105 against 122 ms at 56; prompts up to 33k tokens peak at 74.8 GiB of the 80 GiB limit.
  `./serve-minimax.sh` and `./chat-minimax.sh` pass 62; `CACHALOT_MINIMAX_PREFILL_FULL_TOKENS` (512).
- **The first turn after a restart no longer waits for the whole warm set (M27).** A request stops the startup
  reads (one batch at most) and the rest is read after it, while idle: the chat's first "Hi" 11.8 → 4.8 s.
  `CACHALOT_WARM_SET_YIELD=0` restores the wait.

### Found
- macOS's screensaver (Flurry, after 10 idle minutes) renders on the GPU and slows decode ~13 %. Benchmarks sleep
  the display per arm; for long unattended agent runs, set the screen saver to Never.

## 0.34.1 (2026-09-28)

Documentation only (HANDOFF section 18.15 item 9).

- Hamed's chat session on 0.34.0 read against his 0.33.0 one on the same prompts: decode +7 to +17 % (prose 13.6 /
  11.8 tok/s, code 6.1-6.4), short follow-ups 2.3-3.3 s (3.1-4.3), code hit rates +1-2 points; the chat head
  snapshot is reused after a restart ("Hi" prefilled 7 of 164 tokens: M1c done); `/stats`' new fields read as
  intended (1.79 ms of decode wait per miss, the drive at 6.65 GiB/s). The first turn waits 8.8 s for the 56 GiB
  warm set: M27 in next-session prompt v63. Answers clean; their faults are the 3-bit model's at T = 1.

## 0.34.0 (2026-09-28)

HANDOFF section 18.15.

### Changed
- **MiniMax-M3 caches experts in 56 GiB instead of 52, under a memory governor (S2).** On 0.33.0 a 56 GiB expert
  budget is no longer slower than 52 (18.1 and 18.5 measured it slower on older code): agent benchmark, four
  ABAB processes, **decode 162.6 → 141.1 ms a token (-13 %)**, turns 73.6 → 65.0 s (-12 %), short prefills -9 %,
  the same `ids_hash`. A fixed 56 is not safe, though: an 8,192-token prefill chunk at 56 reached warning pressure
  (30.6 → 48.7 s) and then Metal's out-of-memory. So a prefill now gives back slots first: chunks up to 2,048
  tokens keep the full capacity, an 8,192-token chunk prefills at 52 GiB's (linear in between), and the first decode
  token's memory fit takes the slabs back. The capacity grows only at normal pressure and with at least 8 GiB
  available (`kern.memorystatus_level`) afterwards; at warning pressure, or under 4 GiB available, a slab is given
  back. At startup the budget is capped at what macOS reports available minus 20 GiB. Through `stream()` (the
  server and chat path), 8k then 16.5k prompts: prefill the same as at 52, decode -5 to -6 %, same tokens, no
  pressure. `./serve-minimax.sh` and `./chat-minimax.sh` pass 56; `CACHALOT_MINIMAX_PREFILL_BUDGET_GIB` (52),
  `CACHALOT_HOST_AVAILABLE_FLOOR_GIB` (8), `CACHALOT_MINIMAX_STARTUP_RESERVE_GIB` (20).
- **`/stats` in the GLM/MiniMax chat (M26):** `read_ms_per_expert` (the mean duration of one read, which grows with
  the queue depth) is replaced by `decode_wait_ms_per_miss` (what a decode miss cost the token) and `drive_gib_s`
  (bytes read over the wall time with at least one read in flight, `drive_busy_seconds`); `ssd_gib_read` counts
  every read; `borrowed_transient_slots` explains `resident_experts` above `expert_slots`. The startup line prints
  "(none)" instead of "(none tokens)".

### Measured and not kept (HANDOFF 18.15)
- M25, predicting the first miss after an all-hit layer: 8.1 of 34.2 misses a token, spread over every layer; the
  previous token's runner-up experts are 1-4 % precise.
- Fusing the decode token's small kernels (`mx.compile` of the swiglu, routing tail and output combine): bit-equal,
  -1.1 ms of a 30 ms all-hit floor. The expert matvec in other threadgroup shapes, with wider or software-pipelined
  loads: bit-equal, none faster (380-420 GB/s against ~640 for plain reads).
- M24 priced: a compact record (the slot image) reads 4.1-4.6 % faster a miss; needs Hamed (bank rewrite).

## 0.33.1 (2026-09-28)

Documentation only (HANDOFF section 18.14 item 9).

- Hamed's chat session on 0.33.0 read against his 0.29.0 one on the same prompts: prose 12.7-12.9 tok/s at 95 %
  hits (8.4-9.2 before), code 5.5 tok/s (4.3), no stray sentences, every earlier token reused. Found: short
  follow-ups still cost a fixed 3.1-4.3 s; `/stats`' `read_ms_per_expert` is a per-read duration that rises with
  queue depth, not a miss's cost; `resident_experts` exceeds `expert_slots` (borrowed transient slots); the startup
  line prints "(none tokens)". Next-session prompt v62 gains M26 (display fixes) and M1c (check the chat head
  snapshot on the next restart).

## 0.33.0 (2026-09-27)

HANDOFF section 18.14.

### Changed
- **MiniMax-M3 decode reads the next layers' missing experts from the routing its GPU loop already computed.**
  When the host finds a miss at layer i, the 0.31.0 loop has already run layer i+1's attention and routing on
  layer i's output. The slab kernels now give a not-yet-read expert (slot -1) zero rows, so that output is layer
  i's without its misses and its routing matches layer i+1's real one ~91 % of the time. As soon as layer i's own
  reads are submitted, the host issues reads for layer i+1's non-resident experts, then runs layer i+1's hit
  experts and layer i+2's attention and routing on that output (its KV write rewound at once) and issues layer
  i+2's (~88 % of the speculated reads are used). The drive no longer idles while the host recomputes a layer and
  its queue is deeper: read time per miss 3.5 → ~3.0 ms. Byte-identical (TF log-probs and prefill logits
  `cmp`-equal against 0.32.0 at 1k and 8k context; the agent benchmark's `ids_hash` unchanged). Agent benchmark,
  ABAB: **decode 187.1 → 158.4 ms a token (-15 %)**, turns 80.7 → 72.4 s (-10 %); 8k context 280 → 238 ms.
  `CACHALOT_MINIMAX_SPEC_PREFETCH` (reads per layer, default 8; 0 restores 0.32.0), `CACHALOT_MINIMAX_SPEC_DEPTH`
  (2), `CACHALOT_MINIMAX_SPEC_AFTER_DEMAND` (0: issued alongside the layer's own reads).
- `ResidentExpertStore.get_many` takes a callable `prefetch` (called only when the prefetch is issued),
  `prefetch_limit` and `prefetch_after`; `prefetch_decode` takes a per-call `limit`.

### Measured and not kept
- Entropy-coding the 3-bit expert weights: 2.68 bits of entropy a weight (a static Huffman code 2.72, 9.4 % fewer
  bytes), but a bit-exact decode inside the matmul (lanes reading one interleaved stream in lockstep) takes 79 µs
  against 14 µs a projection, ~+90 ms of GPU time a token against at most ~25 ms saved; a CPU decoder would need
  ~13 G symbols/s.
- A wider mirror: the X10Pro reads 1.00 GB/s at every queue depth and piece size (USB 10 Gb/s); the 0.13 split is
  already ideal.
- Six workers for predicted reads instead of two (no change); speculating three layers deep (+1.7 %); building the
  depth-2 graph before layer i's reads are submitted (it delayed them).

## 0.32.0 (2026-09-27)

HANDOFF section 18.13.

### Changed
- **MiniMax-M3 expert slots keep one byte per weight group (M23b).** A scan of every expert of the bank (7,296
  records, 21,888 projections) found at most 166 distinct bf16 scales and at most 205 distinct (scale, bias) pairs
  per projection. A slot now holds, per group, a byte index into its projection's 256-entry table of (scale, bias)
  pairs (`w?.pidx`, `w?.plut`) instead of the bf16 scale and the 4-bit bias code: 21.10 MiB a slot instead of
  22.36, 2,523 resident slots in 52 GiB instead of 2,381. The reader builds the table while the weights stream in
  (`codes_qmv.pair_from`, O(n), ~0.9 ms a projection on its own thread pool; a projection with more than 256 pairs
  is refused). The decode kernels (`codes_qmv`'s copy of MLX's qmv_fast and the slab kernels of 0.31.0) take scale
  and bias from the table; prefill rebuilds both in one launch per projection. Byte-identical: 40 teacher-forced
  log-probs and the prefill logits `cmp`-equal against 0.31.0's slots; the agent benchmark printed the same
  `ids_hash` in eight runs. Agent benchmark, ABAB against 0.31.0's slots: decode 174.0 / 168.8 → 157.4 / 160.5 ms
  with a byte scale index and the codes (mode 1), then mode 1 against the pairs 158.5 / 161.7 → 156.0 / 155.3 ms;
  in all **-9 %** decode, turns 79.5 / 76.1 → 71.2 / 71.1 s, short prefills -6 %. `CACHALOT_MINIMAX_SLOT_SIDX`:
  2 (default) pairs, 1 byte scale index + codes, 0 the 0.31.0 slot.
- `minimax.gpu_select`: `CACHALOT_MINIMAX_SELECT_PREFETCH` (top-k, default 0) and `_AHEAD` read the predicted
  experts of a later layer from inside the GPU-select loop; measured slower (below) and left off. An all-hit
  layer now advances the store's decode walk, so stale predictions expire (S1d).

### Measured and not kept
- The 0.28.0 decode prefetch inside the GPU-select loop (S1b): 19.4 predicted reads a token, 13.9 used, 219.4 →
  229.3 ms (in one process, token by token). The prediction is known about 0.6 ms before the layer needs it and a
  read takes ~3.5 ms. The best non-resident prediction is right 46 % of the time one layer ahead, 38 % two, 21 %
  six, so reaching further ahead mostly spends a saturated drive on wrong experts.
- The compact slot form in a side file on the SSD (7.5 GiB, 0.84 MiB fewer bytes a read): 3.49 → 3.77 ms a
  miss in both orders. The bank's head is contiguous with the weights; a second location is slower than the
  bytes it saves. Removed.

## 0.31.0 (2026-09-27)

HANDOFF section 18.12 item 5.

### Changed
- **MiniMax-M3 decode selects its experts on the GPU (S1; the pattern of Splash's GPU-side MoE).** The shipped
  decode layer waited on the host for each MoE layer's routing before its experts ran (~0.4-0.65 ms of GPU idle
  per layer). Now the expert slots live in slabs of 128 (`cache.slots.SlabSlotPool`: one contiguous, page-aligned
  record per slot, slot arrays and views are views into the slab, so reads and prefill are unchanged), the store
  keeps a (layer, expert) -> slot table in step with every admission and eviction (`track_slots`), and a decode
  token (`minimax.gpu_select.GpuSelectDecoder`) submits each layer as attention + routing + slot lookup, then the
  routed experts straight from the slabs (the codes kernels with the slot base computed from a GPU index), while
  the host reads the previous layer's slots one step behind. A layer with a miss is fixed: the store reads the
  missing experts (the same `get_many`), only their rows are recomputed, and the next layer's attention is rewound
  and rerun. Byte-identical: 40 teacher-forced log-probs and the prefill logits `cmp`-equal with it on and off;
  same greedy ids at 20k and 40k context; the agent benchmark printed the same `ids_hash` in four runs.
  Agent benchmark ABAB: decode 194.4 / 185.8 → 173.0 / 171.1 ms per token (**-9.5 %**), turns -6 %. Chat-like
  text in one process, token by token against the shipped path: prose (93 % hits) 95.4 → 75.4 ms (-21 %), code
  (81 %) 224.7 → 213.9 ms (-5 %); a 79 %-hit filler text even (the shipped path's decode prefetch, 13 used reads
  a token, has no counterpart yet). 40k context: 207.6 → 190.7 ms. `CACHALOT_MINIMAX_GPU_SELECT=0` restores the
  per-layer sync (it also runs on the slab pool, so `TF_ALTERNATE` can flip it per token).
- The memory fit parks and unparks whole slabs with a slab pool: shrinking leaves at most a quarter slab of
  excess, growing needs a slab and a quarter of room; a victim slab's surviving residents are copied to free slots
  elsewhere, so the resident set is the one slot-granular parking would keep.

## 0.30.0 (2026-09-27)

HANDOFF section 18.12.

### Fixed
- **GLM and MiniMax sample with their checkpoints' top_p (M21).** The terminal chat had no `--top-p` and the
  server filled a request's missing `top_p` with 1.0; both checkpoints' `generation_config.json` ask for 0.95, and
  Hermes's chat-completions transport never sends `top_p`. Measured on MiniMax with the 0.29.0 chat session's
  prompts, two seeds each: at top_p 1.0, 24 of ~290 story tokens and 44-61 of ~1,200 code tokens were drawn from
  outside the 0.95 nucleus (4-8 %), and one story ended "Your feedback is appreciated."; at 0.95, none. Now the
  chat's `--top-p` and the server's new `--default-top-p` default to the checkpoint's value (GLM/MiniMax; DeepSeek
  keeps 1.0), a request's own `top_p` still wins, and the server prints the defaults at startup.
- **GLM/MiniMax chat parity (M22).** `chat-minimax.sh` / `chat-glm.sh` attach the serve scripts' snapshot
  directory (`--snapshot-dir`, `CACHALOT_SNAPSHOT_DIR`): the resident expert set is read back at startup and saved
  after every turn, and the prompt head (template and system message) is snapshotted to disk. A restart's first
  "Hi": prefill 14.3 s at 21 % hits before, 2.0 s (157 of 164 tokens reused) at 88 % after. The agent snapshots
  in that directory are indexed but not preloaded by the chat (they would take memory from the expert cache).
  `/stats` (session tokens, hit rate, reads, ms per read, slots, sampling), `/clear` and `/help` are answered
  locally; any other single `/word` is reported as unknown and never sent to the model. `idle warm:` and
  `warm set:` lines from background threads go into the next turn's summary instead of the line being typed
  (`model.notice`; the server still prints them). `--seed` seeds the chat's sampler.

### Measured, not changed
- **S0, the numbers behind GPU-side expert selection (HANDOFF 18.11 S1).** All-hit floor at 2k, one process,
  bit-identical logits in every arm: shipped 71.5-74.6 ms (min 60-66), host-known routing 39 ms, routing on the
  GPU through a (layer, expert) -> slot table into stacked copies with `gather_qmm` 35 ms, and the same with each
  layer submitted asynchronously and the previous layer's routing read on the host one step behind **34.2 ms**.
  In the chat session's replay ~43 of 57 MoE layers per prose token (93 % hits) and ~26 per code token (81 %)
  have no miss: expected savings 17-28 ms of ~133 (13-21 %) on prose and 10-17 ms of ~233 (4-7 %) on code.

## 0.29.0 (2026-09-27)

HANDOFF section 18.10.

### Changed
- **MiniMax-M3 expert slots hold 4-bit bias codes instead of bf16 biases (M13b).** Every group bias of the
  checkpoint is bf16(k x scale) with k in -7..-3, so a slot keeps k + 7 as a nibble: 22.36 MiB a slot instead of
  23.62, 2,381 resident slots in the 52 GiB budget instead of 2,253 (+128), and 128 x 1.26 MiB less for the
  transient slots. Decode multiplies with MLX's own 3-bit `qmv_fast`, copied from the installed MLX headers into an
  `mx.fast.metal_kernel` with one change (the bias rebuilt from the code in the kernel, with the bank's rounding),
  gate and up in one launch; prefill rebuilds the biases on the GPU (one small kernel per projection) and calls
  `mx.quantized_matmul` as before. Byte-identical: a 1,024-token prefill's logits and 40 teacher-forced decode
  log-probs, codes on and off, compare equal with `cmp`; the agent-turn benchmark printed the same `ids_hash` in
  six runs. Agent-turn benchmark, ABABAB: decode 211.9 / 201.4 / 208.7 → 190.4 / 190.5 / 194.3 ms per token
  (**-7.5 %**; misses per token 37.1 → 33.9), short prefills 30.8 / 30.0 / 31.1 → 29.1 / 28.8 / 29.3 s (-5 %).
  The kernel is checked against `mx.quantized_matmul` when the model loads; a mismatch (a future MLX) falls back
  to the rebuild path for every matmul. Needs the bank covering every expert. `CACHALOT_MINIMAX_SLOT_CODES=0`
  keeps bf16 biases; `CACHALOT_MINIMAX_CODES_KERNEL=0` forces the rebuild path.

### Added
- `src/cachalot/minimax/codes_qmv.py`: the codes kernels, the GPU bias rebuild, the slot-side nibble packing and
  the load-time bit check. `ResidentExpertStore.peek(key)`: a resident without touching the LRU order.
- `CACHALOT_DECODE_ASYNC_OUT` (off): submit a decode layer's routed output as soon as it is built. Measured, not
  enabled (below).

### Measured, not changed
- **The all-hit decode floor is ~22 ms of host round trips.** Replaying one token with the routing known in advance
  (no per-layer sync) takes 37.1 ms against 59.3 shipped, identical logits. Filling that gap without dropping the
  sync: the shared expert queued before the routing wait (-0.9 ms, noise); the previous layer's predicted top-8
  experts computed speculatively before the wait (floor -8 ms, identical, but +1-2 ms with real misses: the extra
  encode delays the reads); the layer output submitted early (`DECODE_ASYNC_OUT`: floor -3.3 ms, non-read part -7 ms
  in one process, but the agent benchmark 205.7 → 204.1 ms, noise); building the miss experts and the next layer's
  attention while the reads run (250.6 → 249.9 ms, noise). The four experts of a layer in one `gather_qmm` per
  projection: bit-identical, not faster (76.1 against 74.6 ms). Decode is read-bound; only the first two ideas'
  code is gone.

## 0.28.0 (2026-09-27)

HANDOFF section 18.9.

### Changed
- **MiniMax-M3 decode reads one predicted expert per layer while the drive would idle.** Each decode layer
  applies the next MoE layer's own norm and router to this layer's residual (evaluated in the same sync as its own
  routing, ranked on the host) and, once its own misses have arrived, reads the best-ranked of the top-2 that is not
  resident into a transient slot, while the GPU finishes the layer (~1.5 ms per layer in which the drive did
  nothing). A correct prediction is awaited instead of read from scratch; a wrong one is dropped at the next layer.
  Same token ids. In one process, token by token (`TF_ALTERNATE`, 2k context): 216.5 → 203.1 and 205.0 → 202.8 ms
  per token on two texts; the agent-turn benchmark, ABAB: decode 241.0 / 244.2 → 233.2 / 235.1 ms per token
  (-3.4 %), short prefills unchanged. `CACHALOT_MINIMAX_PREDICT_TOPK` (default 2, 0 off);
  the store's `CACHALOT_DECODE_PREFETCH_AFTER_DEMAND` and `CACHALOT_DECODE_PREFETCH_LIMIT` (MiniMax sets 1 / 1;
  DeepSeek keeps 0 / 0, the old behaviour).
- A prefill expires decode's leftover predictions first (`GlmModel.prefill`), so they give their transient slots back.

### Fixed
- **Why 18.1's one-layer-early prediction lost.** It read the prediction back with an MLX op after the routing sync
  (`argsort` and `reshape`), a second GPU round trip per layer: +21-24 ms per token before a single byte was read,
  and its reads competed with the layer's own misses. Both are gone.

### Added
- `benchmarks/glm_prefill_timeline.py`: `TF_ALTERNATE` prints decode prefetch loads and uses per token.
- `benchmarks/minimax_followup_turns.py`: `HEARTBEAT=S` evaluates a one-element op every S seconds of a `PAUSE`, as
  the server's heartbeat does.

### Measured, not changed
- **18.8's two unexplained slow-downs were the benchmark's missing heartbeat (M17 closed).** A 4 s pause without
  it: short prefills 38.6 / 38.8 s; with it 31.9 / 31.6 s, the same as no pause (32.3 s). The server always had
  it. Idle warming re-measured with it: decode 227.4 / 228.7 → 212.1 / 211.4 ms (-7 %), short prefills 31.7 /
  31.8 → 30.1 / 29.8 s (-5 %), no prefill slower after warming.
- A prefetch that yields the drive to demand reads (piece by piece, promoted when awaited, cancelled when wrong):
  the same as without. Predicting two layers ahead: no gain (wait 132.2 ms both arms). Top-4 with two per layer:
  +12 ms (worse). A folded one-matmul predictor: the same as the exact one. Not kept.

## 0.27.0 (2026-09-26)

HANDOFF section 18.8.

### Changed
- **MiniMax-M3 prefill chunks read ahead only the next layer's predicted experts.** From 100 to 3,072 tokens a
  chunk applies the next layer's norm and router to this layer's residual (top-3 per token), and queues the union,
  most-picked first, behind its own misses, instead of the whole next layer from 768 tokens (or nothing below).
  The prediction is ~90-97 % precise and 78-95 % complete. Same text, same token ids in every arm, ABAB: 150-token
  turns 9.4-9.8 → 8.3-9.1 s, 500 13.2-13.7 → 11.4-12.0 s, 800-1,500 14.5-15.1 → 12.5-13.4 s, 3,000 16.1 → 15.0 s;
  a run's short prefills 77.0 / 79.0 → 68.0-72.8 s and 75.7 / 75.7 → 70.6 / 70.6 s. Below ~100 tokens nothing
  changes; above 3,072 the whole layer is read again (4,096: 17.7 s whole against 19.4 predicted).
  `CACHALOT_MINIMAX_PREFILL_PREDICT` (1/0), `_PREDICT_MIN` (100), `_PREDICT_MAX` (3072), `_PREDICT_TOPK` (3).
  `StreamingSwitchGLU` takes an explicit `speculate` list; GLM is unchanged.
- **MiniMax-M3 warms its expert cache between requests.** 0.2 s after a request the server moves the resident set
  towards the experts this process has requested most (`ResidentExpertStore.warm`: missing ones read into the slots
  of residents outside that set, never more residents than before, most wanted most recently used); the next
  request cancels it and waits one batch of reads at most (a cancel measured 0.3 s in). 0.3-2.2 s of reads per
  pause. With a 4 s pause between agent turns, ABAB, same token ids: decode 207.6 / 208.8 → 183.7 / 182.2 ms per
  token (-12 %), short prefills 52.1 / 52.4 → 49.0 / 50.7 s. `CACHALOT_MINIMAX_IDLE_WARM=0` turns it off; GLM
  has it off (`GlmModel.IDLE_WARM`). The server prints `idle warm: N experts in S s`.

### Added
- `benchmarks/minimax_followup_turns.py`: prediction recall and precision per turn, `WARM`, `WARM_SECONDS`, `PAUSE`.

### Measured, not changed
- Prediction top-2 / top-4 / top-6 against top-3: 70.2 / 70.0-72.8 / 78.9 against 68.0-74.2 s (top-6 reads too
  much); predicting from 16 tokens: the same as not below 100. Predicted experts first, then the rest of the layer,
  above 3,072 tokens: the same as the whole layer.

## 0.26.0 (2026-09-26)

HANDOFF section 18.7.

### Changed
- **MiniMax-M3 prefill chunks speculate from 768 tokens, not 128.** From `SPECULATE_MIN_TOKENS` a prefill chunk
  queues every expert of the next layer behind its own misses, which suits GLM (top-8 of 288) but not MiniMax, whose
  skewed top-4 of 128 reaches ~74 experts of a layer at 150 tokens: a 150-token prefill read 4,965 experts for a
  demand of ~2,200, and the speculative transients evicted decode's borrowed cache. Same text, same tokens, ABBA:
  150-token turns 14.7 → 8.8-10.6 s, 250 15.1/14.7 → 12.1 s, 500 14.8 → 13.8 s, 800-1,000 and cold 2k unchanged;
  all short prefills of the run 99.8 / 94.7 → 82.0 / 79.3 s. `CACHALOT_MINIMAX_SPECULATE_MIN_TOKENS` sets it;
  `StreamingSwitchGLU.speculate_min_tokens` is the per-layer override (GLM keeps 128).
- **MiniMax prefill reads take each record's head from a compressed copy.** A bank may hold `heads.zst` +
  `heads.json`: every record's bf16 scales and bias codes (or biases), zstd level 19, 13.68 → 4.01 GiB. A prefill
  (bulk) read takes the head from it and the weights from the record: 6 % fewer bytes, the same bytes in the slot.
  Short prefills 39.9 / 39.9 → 37.2 / 37.6 s, cold 2k 23.5 → 22.1 s, decode unchanged. Decode reads keep the plain
  head: there the ~1.5 ms decompression is on the latency path and measured slower.
  `CACHALOT_MINIMAX_ZHEADS` = 1 (default, bulk only), 2 (always), 0 (never). The store sets `reader.bulk`.

### Added
- `coded_bank.write_zheads` and `benchmarks/minimax_coded_bank.py --zheads BANK [--level 19]` (a few minutes;
  each blob round-trip checked).

### Measured, not changed
- Speculative decoding for MiniMax: verifying G tokens per forward reads exactly G times the misses (37.2 / 74.5 /
  151.2 for G = 1 / 2 / 4); a draft could only save the non-read part. Closed.
- M14 idle-time warming, priced from a trace: a popularity preload between turns cuts a follow-up's prefill misses
  5-15 % (2-30 %). Not built yet.

## 0.25.0 (2026-09-26)

HANDOFF section 18.6.

### Fixed
- **MiniMax-M3 decode at an agent's context (M15).** At 25k context the KV cache and a second copy of it came on top
  of the 52 GiB expert set sized at 2k; with the display on, decode stalled at the memory ceiling (0.90-1.88 tok/s,
  single tokens up to 3.4 s). Now, same text, same tokens, display on: 3.55-5.03 tok/s on the same turns.
- **A short prefill could deadlock** (0 % CPU, forever) when its own layer's residents were the LRU end of the cache
  and decode's borrowed slots held the rest: `get_many_prefill` evicted experts it had counted as hits. The shrink
  now skips the layer it serves, and a blocked scan evicts a borrowed resident instead of waiting.

### Changed
- **The MiniMax expert capacity follows the KV cache.** After a request's first decode token (and every 512 tokens)
  whole expert slots are parked (their memory freed) or unparked so MLX's active memory stays at its level right
  after loading plus `CACHALOT_MINIMAX_KV_ALLOWANCE_GIB` (default 1.0; negative disables). Short contexts are
  unchanged; at 25k the first request gives back ~200-330 slots. Display-on A/B at 25k, two rounds: turn 2
  1.84/1.88 → 3.29/3.30 tok/s, turn 3 0.90/0.90 → 4.37/4.36. New: `ExpertSlotPool.park/unpark`,
  `ResidentExpertStore.set_capacity`.
- **MiniMax holds one KV copy instead of two.** A request consumes the conversation snapshot it continues instead
  of copying it, and the prompt snapshot (for a retry) is taken after the reply over the same buffers
  (`CACHALOT_MINIMAX_CONSUME_SNAPSHOTS=0` restores the old path; GLM unchanged). Same token ids (hash-checked over
  five turns at 8k); MLX active during decode at 8k 66.6-75.3 → 64.5 GiB; at 25k, with the capacity fit on in both
  arms, +3-6 % per turn and fewer misses.

### Added
- The server's `[request]` line ends with `mlx=active/peak/cache GiB` (peak since the previous request).
- `benchmarks/minimax_codes_qmv.py`: MLX's 3-bit `qmv_fast` as a custom kernel that rebuilds each group's bias
  from a 4-bit code, bit-identical to `mx.quantized_matmul` on MiniMax's expert shapes; +4.6 ms per token in a
  decode-shaped chain against ~9-13 ms of fewer reads. Priced and deferred (HANDOFF 18.6 item 6).

### Measured, not changed
- MiniMax short follow-up prefills (M14) read experts at 5.5-6.8 GiB/s, the two drives' wall: read-ahead or
  overlap cannot shorten them (HANDOFF 18.6 item 5).

## 0.24.1 (2026-09-26)

Documentation only; no runtime change.

- HANDOFF 18.5 item 8: MiniMax-M3 in a live Hermes Desktop session. Follow-up prefills 3.4-3.7 s on a reused
  22k system block; decode at that context 1.8-2.0 tok/s, with the time outside the expert reads; Hermes's
  auxiliary requests (`provider: auto`) each a cold 17-20 s prefill that also takes decode's borrowed cache.
- Next-session prompt v53: M15 (decode at an agent's context) and M14 (short follow-up turns) added.

## 0.24.0 (2026-09-26)

HANDOFF section 18.5.

### Changed
- **MiniMax-M3 expert reads bypass the page cache** (`serve-minimax.sh` / `chat-minimax.sh` now set
  `CACHALOT_PAGE_CACHE=0`, i.e. `F_NOCACHE`). With the bias-free bank's one contiguous record per expert a direct
  read is faster, where it was 13 % slower with the checkpoint's nine pieces (18.4). In one process, token by token,
  on two texts: read wait per miss 3.74 → 3.62 ms on both, the non-read part of a token 84.7 → 79.9 and 89.2 →
  84.4 ms. Cold 2k prefill unchanged (24.6-25.0 s either way). Same bytes, so same outputs.
  `CACHALOT_PAGE_CACHE=1` restores the old path.
- **A MiniMax decode token's hit experts run while its misses are read.** `ResidentExpertStore.get_many` takes an
  `on_hits` callback, called after the misses' reads are submitted; `StreamingSwitchGLU` queues the hits' matmuls
  there (`mx.async_eval`) and computes the misses after. Same matmuls in the same order: prefill logits and decode
  log-probs byte-identical (`cmp`). In one process on two texts: the non-read part 89.4 → 83.8 and 96.9 → 90.6 ms,
  read wait per miss unchanged. On for MiniMax (`hit_overlap`), off for GLM; `CACHALOT_DECODE_HIT_OVERLAP=0/1`
  forces it.
- **MiniMax mirror fraction 0.10 → 0.13** (the tail of each weight piece read from the X10Pro's bank copy). With the
  internal reads faster, 0.13 beat 0.10 in two in-process A/Bs (3.58 vs 3.68, 3.87 vs 4.01 ms per miss) and 0.16
  (3.53 vs 3.82); 0.07 lost to 0.10.
- Together (0.23.0's configuration against this one, same text, same tokens, separate processes ABAB, both with
  mirror 0.10): 248.0 / 253.5 → 230.2 / 243.6 ms per token (-5.5 %). Through the server: 500-token replies at
  4.61 and 4.93 tok/s, a tool-result turn at 5.33 tok/s.

### Added
- Experiment knobs for in-process A/Bs (`TF_ALTERNATE`): `cachalot.storage.reader.BYPASS_OVERRIDE` (page cache per
  token, descriptors cached per setting), `cachalot.minimax.coded_bank.MIRROR_FRACTION`,
  `cachalot.glm.experts.DECODE_HIT_OVERLAP`.
- `benchmarks/minimax_bias_kernel_price.py`: M13's GPU bias rebuild (bit-identical on 201.7 million groups) and its
  cost in a decode-shaped chain. `minimax_policy_replay.py` gains `tlfu` (frequency admission) and `--decay`.
  `glm_prefill_timeline.py` `ROUTE_TRACE` now records teacher-forced decode steps too.

### Measured, not shipped
- The expert budget at 56 GiB with the page cache off: still slower (misses 41.9 → 38.5, the non-read part +50-90
  ms); the pressure was not page-cache churn.
- TinyLFU admission: 63-78 misses per token against LRU's 54 at 52 GiB (Belady 23).
- Read threads at `QOS_CLASS_BACKGROUND`: reads and the GPU part both slower.
- M13 (2-bit codes in the slot, biases rebuilt on the GPU per use): +149 slots, about -3 to -4 misses per token,
  against +4-8 ms per token of rebuild kernels: net 1-4 %, deferred.

## 0.23.0 (2026-09-26)

HANDOFF section 18.4.

### Changed
- **MiniMax-M3 reads its routed experts from a bias-free bank** (`cachalot.minimax.coded_bank`). MLX's affine bias
  of every expert group in this checkpoint is exactly `bf16(k * scale)` for a whole number k (checked on all 6.46
  billion groups), so the bank stores each expert as one contiguous 22.16 MiB record with 2-bit codes in place of
  the bf16 biases (23.62 MiB and nine pieces in the checkpoint); experts with any k outside -6..-3 (109 of 7,296)
  keep their biases. The reader issues the weight preads first and rebuilds the biases into the slot through a
  lookup table while the weights arrive, so the slot holds the checkpoint's exact bytes: decode log-probs and
  prefill logits are byte-identical to the stacked checkpoint's. Same text, same tokens: store wait per token
  211.9 → 192 ms (-9 %), a token 301 → 282 ms (-6 %), a cold 2k prefill 25.9 → 24.6 s (168.3 → 157.9 GiB read);
  through the server, short tool turns decode at 4.75-4.87 tok/s with 5.3 ms reads (0.22.0: 4.10-4.71, 5.6-6.3).
- **The internal MiniMax checkpoint holds only the non-expert weights** (5.3 GiB; Hamed's choice, to fit the bank):
  the model index then comes from the bank (`index_from_bank`). The X10Pro keeps the full download, and a copy of
  the bank for mirror striping (the tail 10 % of each weight piece). `serve-minimax.sh` / `chat-minimax.sh` set
  `CACHALOT_MINIMAX_BANK` (default `~/MiniMax-M3-coded-bank`) and `CACHALOT_MINIMAX_BANK_MIRROR`. MiniMax disk
  snapshots are prefilled once more (the checkpoint's shard sizes are part of their identity); numerics unchanged.

### Added
- `benchmarks/minimax_coded_bank.py` (`--scan`, `--write MODEL OUT [--layers a-b]`, `--verify`),
  `benchmarks/minimax_trim_checkpoint.py`. `minimax_floor_replay.py` `IDLE_MS` / `IDLE_SPIN` / `IDLE_READ` /
  `IDLE_GPU_MS`: the all-hit floor with waits, spins or real reads injected; `glm_prefill_timeline.py`
  `TF_PROFILE=1` and each `TF_ALTERNATE` arm's read wait per miss.

### Checked
- The GQA decode kernel's quality at 32k (v51's M7b): a second text, 400 positions, KL against the old path 0.0060,
  below the model's own chunking noise (0.0071); top-1 97.8 %. It stays on.

### Measured, not shipped
- Keeping the GPU busy during a layer's reads (a tiny matmul every 0.5 ms): the floor with injected sleeps went
  85 → 61 ms of non-wait time, the real decode not at all (89.3 vs 89.4 ms). Readahead off: no change. Page cache
  off: 13 % slower. Darwin role and thread QoS on the floor with real reads: no change. Real reads alone, into a
  buffer no kernel touches, raise the all-hit floor's GPU time 55 → 79 ms at 28 reads per token (section 18.4).

## 0.22.0 (2026-09-26)

HANDOFF section 18.3.

### Added
- **Mirror striping for multi-piece experts; on for MiniMax-M3.** An expert of nine pieces (MiniMax's, GLM's, a
  stacked DeepSeek bank) used to fall back to serial reads whenever `CACHALOT_MIRROR_PATH` was set. Its pieces are
  now read concurrently, with about `CACHALOT_MIRROR_FRACTION` of the bytes from the mirror drive: whole pieces,
  smallest first (`CACHALOT_MIRROR_MODE=pieces`, default), or the tail of every piece (`=split`). A failed mirror
  read is retried on the primary and switches the mirror off. `serve-minimax.sh` / `chat-minimax.sh` use the
  X10Pro's copy of the checkpoint when it is mounted (`CACHALOT_MINIMAX_MIRROR=` turns it off): decode 320 → 303
  ms per token on the same text with the same tokens (-5 %), a cold 2k prefill 28.9 → 26.5 s (-9 %). Fraction
  0.10 (the four scales/biases pieces); 0.15 is slower than off.
- **A GQA decode attention kernel for MiniMax-M3** (`cachalot.minimax.gqa_decode`, simdgroup matrix multiplies,
  split-K with an online softmax): each KV head is read once for its 16 query heads. 60 layers of decode
  attention: 44 → 23 ms at 64k, 24 → 13 ms at 32k, 9 → 6 ms at 8k. Used from 4,096 cached tokens
  (`CACHALOT_MINIMAX_GQA_DECODE_MIN`, 0 turns it off). In one process, alternating token by token: a decode token
  304 → 281 ms at 64k (-8 %), 298 → 282 ms at 32k (-5 %). Rounding only: KL against the old path on 800 positions
  at 8k is below the model's own chunking noise (0.009-0.012 vs 0.012-0.015); at 32k on 160 positions it was
  above it (0.046 vs 0.027) with a lower NLL. Snapshots are unaffected (prefill does not use it).
- `glm_prefill_timeline.py` `TF_ALTERNATE=module:NAME:A:B`: an in-process A/B of a decode switch, token by token,
  immune to the drift between processes. `benchmarks/minimax_gqa_decode.py`: the kernel against MLX's SDPA.

## 0.21.0 (2026-09-25)

HANDOFF section 18.2.

### Changed
- **MiniMax-M3: fewer GPU kernels per decode token.** q/k/v and the shared expert's gate/up each run as one quantized
  matmul, stacked at load time with the originals dropped (bit-identical, no extra memory); GemmaRMSNorm runs as
  `mx.fast.rms_norm` (rounding only: its KL against the old path on the same text equals the model's own chunking
  noise). The non-read part of a token 93 → 85 ms. `CACHALOT_MINIMAX_FAST_NORM=0`, `CACHALOT_MINIMAX_FUSE_QKV=0`,
  `CACHALOT_MINIMAX_FUSE_SHARED=0` turn each off. MiniMax snapshots are prefilled once more (new numerics tag).
- **A short prefill gives back only as much of decode's borrow as it reads**, not all of it: 8-14 % fewer misses on
  30-120-token follow-up turns, same tokens (`CACHALOT_PREFILL_SHRINK_ALL=1` restores 0.20.0). Applies to GLM too.

### Added
- **Warm restart for `serve-minimax.sh` and `serve-glm.sh`:** the resident expert set is saved after every request
  (`resident-set.json` in the snapshot directory) and read back in the background at startup. The first 22-token
  turn after a restart prefilled in 4.8 s instead of 8.7 (MiniMax; GLM shares the code, not measured live).
  `CACHALOT_WARM_SET=0` turns it off.
- `benchmarks/minimax_floor_replay.py` (decode floor with every step identical), `benchmarks/minimax_followup_turns.py`
  (an agent's short follow-up turns), `glm_prefill_timeline.py` `TF_DECODE`/`TF_OUT` (teacher-forced decode: NLL,
  speed and log-probs on the same text).
- MiniMax-M3 measured at 32k and 64k tokens: 64k fits (wired 78.5 GiB), prefill 148 tok/s, decode 2.65 tok/s.

## 0.20.0 (2026-09-25)

HANDOFF section 18.1.

### Changed
- **MiniMax-M3 decode 2.89 → 3.51 tok/s on the same text, same tokens.** One GPU round trip per MoE layer instead of
  three: the routing is evaluated once and cast on the host, the shared expert is queued while the misses are read,
  and the routed output is evaluated by the next layer's routing sync (`CACHALOT_MINIMAX_DECODE_OVERLAP=0` restores
  the old step). A single decode token skips the row gathers. Both host-side changes also apply to GLM.
- **MiniMax decode borrows the prefill's idle transient slots** (272 slots, 6.3 GiB) as residents; a prefill evicts
  back to the base capacity before it needs them. 52.7 → 45.6 misses per token at the same memory footprint
  (`CACHALOT_MINIMAX_DECODE_BORROW=0` turns it off).
- **MiniMax prefill chunk 8,192 (was 2,048) and the lm_head on the last position only.** A 2,048-token chunk already
  reads nearly every expert, so the longer chunk reads the same bytes for 4x the tokens: 16k tokens at 240 tok/s
  (was ~85), a 17k-token agent system block in 101 s (was ~220). NLL over the last 1,024 tokens unchanged (0.7940 vs
  0.7945). The chunk is part of the snapshot identity: MiniMax snapshots from 0.19.0 are prefilled once more.

### Added
- `CACHALOT_MINIMAX_PREDICT_TOPK`: one-layer-early routing prediction for MiniMax decode, off (measured slower on a
  saturated drive).
- `benchmarks/minimax_decode_floor.py` (all-hit decode floor and a profile), `benchmarks/minimax_policy_replay.py`
  (cache policies on a decode trace), `benchmarks/expert_read_speed.py` (expert read speed by concurrency).
- `benchmarks/glm_prefill_timeline.py`: decode store wait, read time per expert and page-cache share; `ROUTE_TRACE`,
  `DECODE_IDS_OUT`; NLL for MiniMax.

### Fixed
- `benchmarks/glm_prefill_timeline.py` with `ROUNDS=R` fed every round the same N tokens; each round now gets its own.

## 0.19.0 (2026-09-25)

HANDOFF section 18.

### Added
- **MiniMax-M3 as a third model.** `./serve-minimax.sh` and `./chat-minimax.sh` run
  `pipenetwork/MiniMax-M3-MLX-3bit` (text only, 3-bit, ~427B total, 128 experts top-4) from the internal SSD, on
  the same port and API (model id `minimax-m3`). The conversion's mlx-lm model file is adapted in
  `cachalot.minimax.language`; the stacked experts are cut into per-expert byte ranges and streamed through the
  same store and scan-resistant prefill as GLM. Measured with a 52 GiB cache: prefill 80-100 tok/s, decode 2.5
  tok/s on 4k contexts and 3.4-4.0 tok/s on short chat turns. Tool calls (M3's XML format, typed by the tool's
  schema), thinking (`<mm:think>`) and disk prefix snapshots work through the server.
- `--family minimax` (auto-detected from `model_type`).
- `CACHALOT_SNAPSHOT_KEEP`: how many snapshot files the GLM/MiniMax disk store keeps (default 32;
  `serve-minimax.sh` uses 8, since M3's full-attention cache is ~2.4 GB at 20k tokens).

### Changed
- `GlmEngine` and `cachalot chat` take the chat template, the reasoning/tool markers and the tool-call parser from
  the model (`render_chat`, `splitter_cls`, `parse_tool_calls`). GLM's behaviour is unchanged.
- **GLM reads from the X10Pro.** Its internal copy was removed to make room for MiniMax-M3; `serve-glm.sh` and
  `chat-glm.sh` point at `/Volumes/X10Pro/models/GLM-5.3-Flash-MLX-4bit-MTP` (`CACHALOT_GLM_PATH` overrides), which
  is much slower until it is copied back.

## 0.18.0 (2026-09-25)

HANDOFF sections 17.1 and 15.13.

### Changed
- **GLM prefill is 48 % faster, bit-identical** (60.4-61.1 to 89.5-90.3 tok/s at 6,144 tokens; 90.8 tok/s at
  16,384). A 2,048-token chunk routes to nearly every expert of every layer, 3.5x what the cache holds, so the LRU
  path evicted every resident before the next chunk asked for it. Prefill now goes through a store path that never
  evicts (residents kept, misses through transient slots, the next layer's experts read while this one computes):
  8,152 reads per chunk instead of ~11,700, and the drive stays busy. `CACHALOT_GLM_PREFILL_SCAN=0` restores the old
  path. Section 17's "12.5 tok/s" was a 248-token prompt; the rate was 62.7 tok/s before this change.
- `benchmarks/prefill_unwire_timeline.py` prints the wall-clock time of its start, so its events line up with an
  `iostat` trace.

### Fixed
- **GLM left MLX's buffer cache uncapped.** Freed prefill activations piled up past the wired set and macOS
  swapped them (6-13 GiB of swap-outs per chunk once the transient slots were added). `GlmModel` now caps it at
  2 GiB, as the DeepSeek runtime always has (`CACHALOT_GLM_MLX_CACHE_GIB`).

### Added
- **Disk prefix snapshots for GLM.** `./serve-glm.sh` keeps system-block snapshots in
  `~/.cache/cachalot/prefix-snapshots-glm` (`CACHALOT_GLM_SNAPSHOT_DIR`; empty disables), with DeepSeek's policy
  (32 files by last use, 4 preloaded, the rest loaded on demand). Live: a 2,449-token system block prefilled in
  49.5 s on the first request and in 6.25 s on the first request after a restart. ~370 MB per file at 20k tokens.
- `benchmarks/glm_prefill_timeline.py`: per-chunk GLM prefill timeline (seconds, store wait, misses, GiB read,
  memory), teacher-forced NLL over the last K tokens (`NLL_LAST`), per-position log-prob comparison (`--compare`),
  decode after prefill (`DECODE_TOKENS`) and agent-like turns on one cache (`ROUNDS`).

## 0.17.0 (2026-09-25)

HANDOFF section 17.

### Added
- **GLM-5.3-Flash as a second model.** `./serve-glm.sh` and `./chat-glm.sh` run
  `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` from the internal SSD on the same port and API as DeepSeek V4.1 Flash;
  `./serve.sh` / `./chat.sh` are unchanged. The model code is mlx-vlm's `glm5_next`, vendored unmodified
  (`src/cachalot/third_party/mlx_vlm/`, commit `ad4a3cc`, MIT); Cachalot replaces each MoE layer's routed
  experts with SSD streaming through its wired expert store (`cachalot.glm`). Non-expert weights 5.5 GiB.
  Measured with a 52 GiB expert cache: decode 3.3-3.6 tok/s at 68-76 % hits, 14.3 tok/s all-resident, prefill
  12.5 tok/s. Tool calls, thinking on/off and an in-memory prefix cache work through the server.
  Not yet for GLM: vision, MTP, disk snapshots, hotlist, prediction.
- `--family auto|deepseek|glm` on `serve` and `chat` (auto reads the checkpoint's `model_type`).

### Changed
- `/v1/completions` answers 400 when the loaded model is GLM.
- `benchmarks/coding_quality.py` and `attention_mass_probe.py` point their FP4 reference bank at the X10Pro
  checkpoint; the internal duplicate was removed to make room for GLM.

## 0.16.0 (2026-09-24)

HANDOFF section 15.12.

### Fixed
- **A batch of Hermes subagents no longer pushes the main agent's system block off the disk.** Every system
  block is written to the snapshot directory, and 0.15.0 kept the 8 newest files by creation and loaded all of
  them at startup; one batch of six subagents plus a smoke test had already removed Hamed's 22,082-token block, so
  the next restart would have prefilled it cold (~4.5 min). `SnapshotStore` now keeps up to 32 files (~2 GB),
  pruned least recently *used* first (use times in `index.json`), loads only the 4 most recently used at startup,
  and reads the token ids of the others from their file headers; when a request starts with one of those, it is
  loaded from disk then (`PrefixCache.fetch`). Replayed with restarts (`benchmarks/snapshot_store_replay.py`): after
  7 or 12 more subagents and a restart the main agent's turn reuses its 22,082-token block, against 0 before. Live:
  a subagent request whose block was on disk but not preloaded reused 19,491 tokens and prefilled 396 in 13.5 s.
  Scheduling only; `NUMERICS_VERSION` stays.

### Added
- `CACHALOT_ENGRAM_LOOKAHEAD=1` (off by default): reads the next prefill chunk's Engram rows during the current
  chunk, bit-identical. It removes two thirds of the layer-1/14 wait, but on rows not in the page cache the reads
  compete with expert streaming and a 12k prefill is no faster (8-arm A/B, fresh text per arm), so it stays off.
- `benchmarks/snapshot_store_replay.py`: the dump through the prefix cache and the disk store, with restarts.
- `benchmarks/prefill_unwire_timeline.py`: `lookahead_hits`, and `FILLER_FILE` / `FILLER_OFFSET` for cold arms.
- `docs/hermes-subagent-context.md`: why each Hermes subagent's first turn prefills ~15.6k tokens (its task
  context sits in the system prompt, in front of ~13.5k identical tool-schema tokens), what moving it to the user
  turn would save (~76k tokens, ~15 min per six-subagent batch, replayed), and the Hermes change that would do it.

### Documentation
- **Local context compression does not fit Hermes's budget; point `auxiliary.compression` at a hosted model.**
  The 6,258-token summary prompt that timed out on 2026-09-24, replayed on 0.16.0: 103 s of prefill and a
  3,386-token summary at 4.5 tok/s, 850 s against a 120 s timeout. Manual test §5 and README.

## 0.15.0 (2026-09-24)

HANDOFF sections 15.10-15.11, from Hamed's long Hermes Desktop session with six parallel subagents.

### Fixed
- **The prefix cache no longer evicts each parallel agent's previous turn.** 0.13.0's chunk snapshots
  pinned inside a system block outranked every unpinned snapshot; with six subagents, each with its own
  ~20k-token block, the pins filled the 20-entry cache and every subagent turn re-prefilled 15-31k tokens
  (190-411 s), and the main agent's first turn after the dispatch re-prefilled all 40,679 tokens. `PrefixCache`
  now has a byte budget (`prefix_cache_bytes`, 1.5 GiB; `prefix_cache_entries` rises to 64 as a ceiling) and
  evicts by tier, least recently used within a tier: snapshots contained in a later snapshot, then in-block
  chunk pins, then each conversation's latest state, then system blocks. `benchmarks/prefix_pin_replay.py` on
  the session's dump: 479,899 tokens prefilled before, 256,597 after (~46 minutes at 12.3 ms/token). Which
  snapshot is restored changes, never what it holds: outputs are unchanged and `NUMERICS_VERSION` stays.
- **`serve.sh` defaults to 8,192 new tokens (was 2,000).** Hermes sends no `max_tokens`; at 2,000 two context
  summaries and a nine-task `delegate_task` call were cut short, the half tool call reached Desktop as raw
  DSML, and three of the subagents were never dispatched. A request without `max_tokens` is now shortened to
  what fits in `max_seq_len` instead of refused; an explicit `max_tokens` that does not fit is still refused.

- **Prefill no longer lets macOS un-wire the model at every chunk.** A 4,096-token chunk's Engram rows (~100k
  random reads) arrive seconds after layer 1 needs them; the prefill thread waited with the GPU idle, and macOS
  un-wired the working set (77 to 6 GiB wired, up to ~42 GiB compressed) and paged it back over 10-15 s. The wait
  now evaluates a one-element probe every 0.5 s (`CACHALOT_PREFILL_KEEPALIVE`, 0 disables). 12,342-token
  prefill, ABBA: 305.9 s to 260.6 s (-15 %), wired never below 74.7 GiB, identical logits. HANDOFF section 15.11.

### Changed
- New `benchmarks/prefill_unwire_timeline.py`: one chunked prefill against wired memory, per-chunk seconds.
- `benchmarks/prefix_pin_replay.py` charges each replayed snapshot its real size and takes `--gib`.

## 0.14.0 (2026-09-24)

HANDOFF section 15.9.

### Changed
- **`serve` and `chat` run with shared-memory Metal fences** (`MLX_METAL_FAST_SYNCH=1`, exported by
  `serve.sh`/`chat.sh` and defaulted by `cachalot.cli` before any model work). Over the focused-app Darwin role
  it won every slow- and mid-window block measured, +8 to +25 % tok/s (`decode_anatomy.py`, ABBA), and is a
  null in a fast window; prefill unchanged. Output is bit-identical (`decode_fingerprint.py`, 24 greedy
  steps, same token ids and fp32 logit sums). `MLX_METAL_FAST_SYNCH=0` disables; benchmarks do not set it.
- **Prefix snapshots on disk survive a release.** Their identity used the package version, so every bump,
  documentation-only ones included, discarded an agent's saved system prompt and cost one cold re-prefill
  (~6-7 minutes at Hermes Desktop's 22k tokens). It now uses `snapshot_store.NUMERICS_VERSION` ("0.13.0"),
  bumped only with a change that can move KV bits. Snapshots written by 0.13.0 load under 0.14.0.

### Added
- `benchmarks/vision_ablation.py`: three dense-layout cases (a 6x6 letter grid, a ten-row table, 15 px text)
  and `--cases`; `vision_ablation.sh ARM` passes extra arguments through. Zeroing the learned delimiters or
  removing `bias_vl` each misreads a grid row (0.83), so all three vision fixes of section 16.4 are
  load-bearing; the shipped model scores 1.0 on all three.

## 0.13.1 (2026-09-24)

Documentation and one instrument fix. HANDOFF section 15.8.

- **The slow-decode window's main trigger is the visible Hermes Desktop window.** Hamed hid it (Cmd-H) for a
  second chat and decode went from 4.2-5.2 to 5.5-6.7 tok/s at identical expert reads, with WindowServer and
  Electron's GPU process roughly halved. The manual test now says to hide Hermes during generation.
- A cold prefill after a restart that had loaded the right snapshot was Hermes's `Provider:` line flipping
  between `custom` and `custom:cachalot`; the manual test's troubleshooting table has the config fix.
- `slow_window_sampler.py` counted page-cache residency in a Python loop over 9.3 M pages, one full core
  while it ran; it now counts in C and scans every 60 s.

## 0.13.0 (2026-09-24)

HANDOFF section 15.7.

### Added
- **`serve` and `chat` run with the focused-application Darwin role** (`src/cachalot/darwin_role.py`,
  `setpriority(PRIO_DARWIN_ROLE, 0, 1)`, no privilege needed). In the slow-decode window it won every pair
  measured, +6 to +40 % tok/s; with the display off or in a fast window it is a null. Scheduling only, never
  numerics. `CACHALOT_DARWIN_ROLE=0` disables it; benchmark scripts do not apply it.
- **Every expert read is tallied**: count, reads under `CACHALOT_FAST_READ_MS` (1 ms, i.e. page-cache hits),
  and summed read time. The `[request]` line gains `read=` and `fast=`, `/v1/stats` gains `expert_reads`,
  `expert_fast_reads` and `expert_read_seconds`, and `decode_vs_context.py` reports them per arm.
- `benchmarks/slow_window_sampler.py`: bank page-cache residency, memory, swap, GPU utilization, display
  power, top processes and the server's read counters every 10 s beside a server.
- `CACHALOT_VISION_ABLATE=delims,engram_mask,bias_vl` undoes one of section 16.4's vision fixes at a time
  (the server warns when it is set); `benchmarks/vision_ablation.sh` / `.py` score five verifiable image
  cases per arm.
- `benchmarks/prefix_pin_replay.py`: replays a request dump through the real `prepare_prompt` and
  `PrefixCache`, with and without pinning the in-block chunk snapshots.

### Changed
- **The chunk-boundary snapshots inside a system block are pinned** (in memory, not on disk), so a client
  that changes its system block part-way through, as Hermes compression does, reuses up to the last chunk
  before the change even after a long session: ~8k tokens (~100 s) once per compressed session.
  `prefix_cache_entries` 16 to 20, `max_pinned` 8 to 12.

### Measured
- **The slow-decode window is compute, not reads**: `rest` doubles (97-121 to 203-208 ms/token) at
  identical misses, read latency and page-cache share. It appeared only with the display on (10 arms in a
  row at 7.1-7.3 tok/s after the display turned off, back at once when it was woken), but display-on alone
  is not sufficient; the trigger is not named yet.
- **Vision ablation**: only the Engram image mask is load-bearing on five verifiable cases (mean 0.86
  without it: "9 red circles" for 3, a chart value lost); zeroed delimiters and the text gate bias scored
  1.00 like the shipped model.

## 0.12.3 (2026-09-24)

Documentation only. Hamed's retest of Hermes Agent Desktop on 0.12.2: every check correct, no errors. A new
chat reused the 22k-token system block in 3.8 s, and images went to Cachalot natively (HANDOFF §15.6, end).
Next-session prompt v41: the slow-decode window stays Job 1, and the long Desktop session and compression are
Jobs 2 and 3.

## 0.12.2 (2026-09-24)

**Found by Hamed's first Hermes Agent Desktop session.** HANDOFF section 15.6.

### Fixed
- **"There is no Stream(gpu, 12) in current thread" on every retry after a cancelled request.** Each
  request ran on a new thread, and MLX 0.32 ties an array that is not evaluated yet to the thread that built
  it. A request cancelled between prefill chunks left its last chunk snapshot holding unevaluated arrays
  (the windows and the published indexer state are kept by reference); every retry of the same prompt
  restored that snapshot on another thread and failed, six times in the Desktop session. All generation now
  runs on one persistent thread (`create_app`'s `generate_pool`), and `snapshot()` materializes everything it
  refers to. Reproduced in isolation; `tests/test_server.py` checks that every request runs on one thread.
- **Hermes dropped a long prefill after 900 s.** Hermes's stale-stream detector for local endpoints
  (`agent.local_stream_stale_timeout`, 900 s) only counts parsed chunks, and an OpenAI SDK never surfaces SSE
  comments, so the `: keep-alive` comments did not count. A 39,279-token cold prefill was cancelled at
  917 s. The server now also sends an empty-delta `chat.completion.chunk` every 15 s while it prefills.

### Measured, for the manual test
- Hamed's Hermes system block is 22,281 tokens (MCP servers and memory included): 420 s cold.
- Hermes's system prompt changes under a running session: the `browser_exec` tool description switches
  between "Screenshots are attached…" and "Your model cannot view images…" depending on whether Hermes thinks
  the *profile's configured* main model has vision, and the `Provider:` line changes when the provider is
  re-selected. Each change is a cold prefill of the whole block. Declaring `supports_vision: true` for the
  `cachalot` model keeps the first stable (the official encoding accepts images in tool results).

## 0.12.1 (2026-09-23)

### Fixed
- **`serve.sh` and `chat.sh` refused to start whenever any command line contained "cachalot"**, including
  the `tee /tmp/cachalot-serve.log` of the manual test's own start command. Hamed's first Hermes Desktop run
  timed out after 300 s with "Connection error" because the server had never started. The guard now matches
  only a running runtime (`deepseek-v41/bin/python` or `cachalot.cli`).
- `docs/manual-tests/hermes-desktop.md`: a readiness check (`curl …/v1/models`) before the first message,
  Hermes profiles having their own `config.yaml`, a troubleshooting table, and the compression cost corrected
  (the request after it is one cold prefill, since Hermes adds a tool).

## 0.12.0 (2026-09-23)

**Agent turns reuse the model's own reply, images in history are not re-encoded, and a long Hermes
session was run end to end.** HANDOFF section 15.5.

### Added
- **Reply splice** (`Engine._splice_own_replies`): after every reply the prefix cache holds a snapshot of
  prompt + reply, but Hermes sends each reply back re-serialized. Tool-call arguments come back with their
  keys in a different order than the model wrote them (`{"content":…,"path":…}` for a `path`-first call),
  and an empty thinking block comes back as `reasoning_content: " "`. Either way the re-rendered history
  diverges a few tokens into the reply and the whole reply is prefilled again. When the client's copy of a
  reply parses to the same message (same content and reasoning up to surrounding whitespace, same tool
  names, same argument values), the server now puts the model's own tokens back in its place, so the
  snapshot matches. Anything that does not parse to the same message is left as sent. On the replay of a
  10-turn Hermes session it removed 499 of 4,453 warm prefill tokens (11 %), up to 111 on one turn; live,
  every turn after a tool call reused prompt + reply (`spliced=N` on the `[request]` line). The saving grows
  with reply length: a `write_file` whose body is re-serialized re-prefilled all of it before.
- **Image span rows cached by content digest** (`VisionEncoder`): an agent resends every image of its history
  on every turn, and the ViT and aligner ran again each time (0.14-1.16 s per image). The rows are a pure
  function of the image bytes and are now kept (16 images). `/v1/stats` reports `vision_rows_reused`.
- **`miss/tok` and `hit` on the `[request]` line**: expert-cache misses per decoded token and the hit rate,
  which is what decode speed follows. They separate "this content routes to cold experts" from "the machine is
  slow right now" (section 15.5 has a case of the second).
- **`benchmarks/decode_vs_context.py`** (+ `.sh`): decode speed against context length with the task held
  fixed, one arm per process. 54 vs 16,054 tokens of context: 6.23/8.45 vs 7.38/7.45 tok/s at 23.7 vs 29.1
  misses/token. Context length does not slow decode.
- **`benchmarks/reply_splice_replay.py`**: replays a request dump offline (tokenizer and encoding only) and
  counts prefill tokens with and without the splice; its baseline reproduces the live `reused` numbers.
- **`docs/manual-tests/hermes-desktop.md`**: a step-by-step Hermes Agent Desktop test with what to expect
  at each step and what to report.
- `CACHALOT_SERVER_DUMP` also records each reply's token ids, so a prefix-cache miss can be traced to the
  exact token where a client's history left the model's output.

### Fixed
- **An agent's system-block snapshot was evicted by one long session.** The prefix cache keeps 16 snapshots
  in LRU order and every request adds two or more, so after ~8 requests the system block was gone from memory
  (it stayed on disk, which is read only at startup). The next new session paid the whole cold prefill
  again (185.9 s, measured). Boundary snapshots are now evicted only after every per-turn snapshot
  (`PrefixCache.max_pinned`, 8), and those loaded from disk at startup are pinned too. Verified live: a new
  session after a 15-request session with a compression reused 13,702 of 13,711 tokens, 1.07 s.

### Changed
- Disk snapshots: the eight newest are kept instead of four. Hermes writes its working directory into the
  system prompt (token 3,924 of 13,698), so each project has its own system block and its own snapshot.

### Measured, no change needed
- Hermes, 10-turn session through the CLI, context 13.7k → 26.3k tokens: every answer correct, `reused` never
  fell back to 0 after the first request, swap flat. Two images in one message, and follow-ups that resend
  them: correct answers, 511/532 and 548/564 tokens reused.
- Hermes (updated today) now sends `reasoning_effort: "medium"`, so its sessions run in thinking mode.
- Hermes compresses at ≥ 75 % of a window under 512k tokens (85 % when its 64k floor binds), ~49-56k on
  Cachalot's 65,536, whatever `compression.threshold` says; `compression.threshold_tokens` is the knob
  that lowers it. Triggered at 18k, twice: the summary request (2,145-token prompt) decoded 1,595 and 1,774
  tokens, 249 s and 343 s, the second past Hermes's 300 s budget (the turn still completed). Compression also
  adds a `skill_manage` tool mid-list, which changes the system block, so the next request prefills it cold
  once (254 s).

## 0.11.0 (2026-09-23)

**An agent's system prompt is reused whole, and survives a server restart.** Both levers of v38's Job 2,
measured live with the stock Hermes CLI against `./serve.sh`. HANDOFF section 15.4.

### Added
- **Snapshot where the system prompt ends** (`Engine.system_prefix_len`, `prefill_chunks(cuts=...)`): the
  server renders the leading system message (tools and reasoning-effort header included) on its own, and
  when its tokens are a prefix of the prompt's, prefill ends a call there and snapshots. A new Hermes
  session's first request now reuses 13,456 of 13,468 tokens and prefills in **1.09 s instead of 18.9 s**;
  the whole session took 15 s against 46 s. Checked across chat/thinking modes, reasoning efforts, with and
  without tools: the rendered system block is a token prefix in every case.
- **Prefix snapshots on disk** (`src/cachalot/model/snapshot_store.py`, `cachalot serve --snapshot-dir`,
  `CACHALOT_SNAPSHOT_DIR`; `serve.sh` sets `~/.cache/cachalot/prefix-snapshots`): boundary snapshots are
  written as safetensors (46 MB for Hermes's 13.5k-token system block, 10 ms) and loaded at startup (20 ms).
  Each file carries an identity of the runtime version, `max_seq_len`, the checkpoint's config and index,
  and the expert bank's files (size and mtime); a mismatched file is ignored. The four newest are kept. The
  first request after a restart prefilled in **3.31 s instead of 163 s**. Restore from disk is
  bit-identical to restore from memory on the real bank at 3,000 and 9,000 tokens
  (`benchmarks/prefix_snapshot_exactness.py`, new disk arm).
- `tests/test_system_boundary_snapshots.py`, 10 tests. 300 pass.

## 0.10.0 (2026-09-23)

**Hermes Agent drives Cachalot, images work end to end, and long prompts no longer run out of memory.** Job 1
was run from the session with the stock Hermes CLI and an isolated `HERMES_HOME`. Hamed's own Hermes config
was not touched. Parallel tool calls, a file write, a resumed session and an image all come back correct.
Four server breakages found on the way are fixed. Vision piece 4 wires image content through the server.
HANDOFF sections 15.1-15.3 and 16.4.

### Added
- **Image input through the server** (`src/cachalot/model/vision_prompt.py`): OpenAI `image_url` content
  parts (URL, path or base64 data URI) go through the official encoding, the MLX preprocessing, the ViT +
  aligner, and the piece 3 splice. The tower loads lazily on the first image (~1 s, 0.9 GiB). Checked on a
  real chart: every value read correctly.
- **Chunked prefill** (`generation.prefill_chunks`, `CACHALOT_PREFILL_CHUNK`, default 4096): long prompts
  are prefilled in chunks that never split an image span, with `cancel` checked between chunks. Quality was
  checked against token-by-token decode on 64 teacher-forced tokens: NLL 1.918 chunked vs 1.938 whole vs
  1.868 sequential at 1,536 tokens, within noise at 8,192.
- **Query-row chunking of the prefill indexer's scores** (`CACHALOT_INDEX_Q_CHUNK`, default 1024, balanced
  slices never below half a chunk). Bit-identical to the unchunked scores in the tests.
- **Prefix-cache snapshots at every prefill chunk boundary**, and least-recently-used eviction. A new Hermes
  session's first request reused 12,288 of 13,495 tokens: 18.9 s instead of 206.3 s.
- Server: one `[request]` line per request on stderr (prompt, reused, prefill seconds, decode tok/s,
  images), `CACHALOT_SERVER_DUMP=path` to append request bodies as JSON lines, SSE `: keep-alive` comments
  every 15 s of silence, and `images_served` / `vision_loaded` in `/v1/stats`.
- Benchmarks: `prefill_memory_sweep.py`, `prefill_chunk_check.py`, `prefill_chunk_quality.py`,
  `prefix_snapshot_exactness.py`.
- Tests: `test_vision_prompt.py`, `test_prefill_chunks.py`, `test_index_query_chunking.py`, and server tests
  for content parts, image errors and `reasoning_effort` aliases. 290 pass.

### Fixed
- **`serve.sh` served a 32,768-token context, and Hermes refuses anything below 64,000.** Now 65,536
  (+~84 MB of compressed-KV cache).
- **A 13.5k-token prompt ran the Metal heap out in prefill** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`).
  Fixed by the chunking above.
- **`reasoning_effort: "none"` returned a 500.** OpenAI-style values are now mapped (`none`/`minimal`
  switch thinking off, `medium` is 50, `xhigh`/`max` is `max`), and unknown ones are a 400.
- **A disconnected client still cost a full prefill.** Disconnects are polled every second, and queued
  cancelled requests are dropped.
- **Vision piece 3 against the reference**: image spans now get the learned `image_start`/`image_newline`/
  `image_end` rows, image positions are DEAD in the Engram hash with the gate shut, and the prefix cache
  keys image spans by content hash so two pictures of one size never share KV.
- `cachalot.__version__` was stuck at 0.9.8.

### Changed
- **Snapshots keep only the written rows of position-indexed caches** and restore pads zeros back:
  bit-identical on the real bank at 3k and 9k tokens, 14.7 / 33.0 MiB instead of ~215 MB. 12 live entries
  held 514 MB instead of 2.59 GB.

## 0.9.11 (2026-09-23)

**Vision phase 1 piece 3 is complete: per-token `bias_vl` routing and `image_mask` threading ship, wired
into `TextDecodeRuntime`'s prefill path and live-smoke-tested on the real bank.** Runtime change for prefill
only: every prefill call now threads two new optional keyword arguments end to end; omitting them (every
caller today) takes the exact old code path, confirmed bit-identical by the widened test suite and by a live
smoke test's unchanged first decode token. Separately, the FP8 GEMV family's post-fusion gap is re-measured,
after a discarded first attempt that mixed two scripts' incompatible measurement methods.

### Added
- **`src/cachalot/model/moe_prefill_batched.py`**: `route_topk_rows` gains `bias_vl`/`image_mask`, both
  optional and enforced both-or-neither. When given, each row selects `bias_vl` instead of the uniform
  `bias` where `image_mask` is set (`mx.where(image_mask[:, None], bias_vl[None, :], bias[None, :])`) before
  the top-k argsort — selection only, matching the official semantics' "correction bias affects selection
  only". Bit-identical to before when neither is given. This is the one router function that needed a real
  signature change (HANDOFF section 16.3): the decode router (`route_topk_fused`) and prefill's per-token
  fallback (`route_topk`) already take one bias per call, one token at a time, and a decode token is never
  an image position.
- **`src/cachalot/model/moe_prefill_grouped.py`**: matching `gate_bias_vl`/`image_mask` kwargs, forwarded to
  `route_topk_rows`; raises `NotImplementedError` if `image_mask` is given with `batched=False` (the
  per-token Python-loop fallback no production caller sets), before touching `expert_index`/`expert_store`.
- **`block_layer0_prefill.py`, `block_sliding_window_prefill.py`, `block_compressed_source_prefill.py`,
  `block_compressed_reuse_prefill.py`, `block_compressed_index_source_prefill.py`**: the same two
  keyword-only parameters, threaded straight through to each module's `moe_prefill_grouped` call.
- **`src/cachalot/model/text_decode_runtime.py`**: `prefill_tokens`/`_prefill_tokens_impl` gain
  `image_rows: mx.array | None = None` and `image_token_id: int = IMAGE_TOKEN_ID` (`129264`, now a module
  constant). The old `mx.stack([embed_token_decode(...) ...])` line is replaced unconditionally by piece 3
  step 1's `merge_image_embeddings` (0.9.10), bit-identical when `image_rows` is `None`. A local
  `image_mask` (`None` when there is no image) is built once and passed to all five layer-block call sites
  through a closure that returns `None` — not the tensor — when there is no image, so a plain text prefill
  never fetches `ffn.gate.bias_vl` at all.
- **`tests/test_route_topk_bias_vl.py`**: `route_topk_rows`'s new behavior checked against `router_mlx.
  route_topk` called once per row with the bias that row should have used — bit-identical with neither arg,
  both-or-neither and wrong-shape guards raise, text rows use `bias` and image rows use `bias_vl` exactly.
- **`tests/test_moe_prefill_grouped_image_mask.py`**: the `batched=False` guard.
- **`benchmarks/vision_piece3_prefill_smoke.py`**: live smoke test on the real 4-bit bank, same pattern as
  `qkv_fusion_live_smoke.py` — a plain text prefill (unchanged first decode token), then the same prompt with
  three interior token ids overwritten to `IMAGE_TOKEN_ID` and random `[3, 5120]` `image_rows` (no vision
  encoder is wired into the server yet, piece 4 is not started), through all 40 layers' `bias_vl` selection
  and eight follow-on decode tokens, finite output throughout, no crash.
- **HANDOFF sections 16.3, 9.36.**

### Fixed (documentation)
- HANDOFF/README's "43 MoE layers carry a `bias_vl`" corrected to 40: three of the 43 `bias_vl` tensors in
  the checkpoint index are unused `mtp.{0,1,2}.ffn.gate.bias_vl`, already excluded from this text-only
  runtime by `resident_trunk.py`'s existing filter. Confirmed by reading the shard header directly:
  `bias`/`bias_vl` are both `F32 [384]`, one pair per real layer, 40 pairs.

### Changed
- **The FP8 GEMV family's post-fusion gap, re-measured** (HANDOFF section 9.36). A first attempt summed
  numbers from `micro_fp8_gemv_kernel.py`'s isolated raw-kernel-launch measurement and
  `micro_shared_expert_roofline.py`'s real-function measurement for the same unfused shapes and found them
  ~20% apart — discarded before publishing, not reported. The self-consistent number, each shape group
  diffed only within the one script that measures its own before/after: **14.00 ms/token shipped today
  against 14.54 ms/token for the same shapes unfused, both measured the same way this session** — a 0.54
  ms/token combined win, the same order of magnitude as the two fusions' individually documented deltas.
  Gap against `mx.sum`: **2.36 ms/token today, against 2.90 ms/token unfused, measured the same way.** The
  original 3.48 ms figure (section 7.1.10) used a third methodology and is not directly comparable to
  either of these; if quoting a before/after, use 2.90 → 2.36.

### Verified
- 253/253 project tests pass (247 plus 5 new `test_route_topk_bias_vl.py`, plus 1 new
  `test_moe_prefill_grouped_image_mask.py`).

## 0.9.10 (2026-09-22)

**Two changes: a second FP8 GEMV fusion candidate is screened and correctly rejected, and vision phase 1
piece 3's first step ships.** No runtime behavior change on either path — the rejected fusion was never
wired in, and the vision splice is written and tested but not called from `TextDecodeRuntime` yet.

### Added
- **`src/cachalot/model/vision_mlx.py`**: `merge_image_embeddings`, piece 3 step 1 of the vision plan
  (HANDOFF section 16.2). Splices `vision_embed()`'s aligner rows into the embedded sequence at
  `image_token_id` positions, in reading order, for a text position falling back to the existing
  `embed_token_decode` unchanged. The official input boundary (`h = self.embed(input_ids);
  h = h.unsqueeze(2).repeat(..., hc_mult, ...)`) is a bare broadcast, not a learned expansion, so an image
  row needs the identical broadcast, not a new numerical path — the architectural risk flagged when piece 3
  was scoped did not materialize at this step. Raises `ValueError` if the prompt's `image_token_id` count
  and the supplied `image_rows` count disagree in either direction. Not wired into
  `TextDecodeRuntime._prefill_tokens_impl` — piece 3's other two steps (per-token `bias_vl` in the router
  kernel, threading `image_mask` through the prefill path) still need to exist first.
- **`tests/test_vision_prefill_splice.py`**: an all-text prompt is bit-identical to the unmodified
  `embed_token_decode` stack; an interleaved text/image prompt places each image row correctly and leaves
  text positions untouched; too few image rows, too many image rows, and image rows supplied for a prompt
  with no `image_token_id` positions all raise.
- **`benchmarks/micro_qb_indexer_fusion_roofline.py`**: screens whether attention's `wq_b` `[32768, 1280]`
  and the indexer's `wq_b` `[4096, 1280]` fuse the same way `wq_a`/`wkv` (0.9.9) and the shared expert's
  `w1`/`w3` (0.9.8) did — both read `qr`, the low-rank query, independently, a same-activation,
  independent-output pair the code's own comment names. Fused output is bit-identical to the shipped
  two-call path, but the indexer only runs on 8 of 40 layers and `wq_b` is not occupancy-bound (already
  456 GB/s, near its own `mx.sum` ceiling), so the recovered cost is 0.038 ms/token — two orders of
  magnitude below what a live session can confirm. **Rejected, not shipped**; the FP8 GEMV family's
  remaining ~3.1 ms gap (`wq_b`, `wo_b`, shared `w1`/`w3`/`w2`) now has no fusion candidate left unexamined.
- **HANDOFF sections 16.2, 9.35**.

### Verified
- 247/247 project tests pass (242 plus 5 new).

## 0.9.9 (2026-09-22)

**Two changes: the attention `wq_a`/`wkv` fusion ships, and vision phase 1 pieces 1-2 (the ViT+Aligner port
and the image preprocessing port) are numerically checked against the official reference.** Runtime change:
attention decode now issues one GEMV over `wq_a`/`wkv` instead of two, on all five decode call sites. Vision
change: two new modules, unwired, no runtime behavior change.

### Changed
- **`src/cachalot/model/attention_qkv_fusion.py`** (new): `wq_a` `[1280, 5120]` and `wkv` `[512, 5120]` read
  the same quantized activation and neither depends on the other's output, the same shape as 0.9.8's shared-
  expert `w1`/`w3` fusion, so they concatenate into one `[1792, 5120]` GEMV instead of two, cached per layer
  by weight-object identity (`fused_qr_kv_linear`, `_fused_wqkv`). These were the two worst-throughput
  shapes in the FP8 GEMV family (HANDOFF section 7.1.10: 171 and 87 GB/s against `wo_b`'s 383), occupancy-
  bound — 512 and 1280 output rows launch too few simdgroups to fill the GPU. Measured 2.01 → 1.66 ms/token
  across forty layers against a 1.55 ms `mx.sum` ceiling (`benchmarks/micro_qkv_fusion_roofline.py`),
  0.35 ms/token recovered, bit-identical by construction. No kill switch — attention is never called from
  inside an `mx.compile`d trace (only the MoE block is), so this fusion carries none of the shared expert's
  eval-inside-a-trace hazard.
- **`src/cachalot/model/attention_compressed.py`, `attention_layer0.py`, `attention_sliding_window.py`**:
  all five decode call sites (`compressed_attention_decode_source`, `_reuse`, `_index_source`,
  `layer0_attention_decode`, `sliding_window_attention_decode`) switched from two `fp8_linear` calls to
  `fused_qr_kv_linear`.

### Added
- **`src/cachalot/model/vision_mlx.py`**: MLX port of the official ViT (patch embed, 2D-RoPE bidirectional
  attention, SwiGLU MLP, RMSNorm) and `Aligner` (space-to-depth downsample reproduced by reshape/transpose,
  MLX has no `unfold`). Not wired into `TextDecodeRuntime`; `resident_trunk.py`'s filter is unchanged.
- **`src/cachalot/model/image_processor_mlx.py`**: near-verbatim port of the official resize-ratio solver
  and patchify, PIL replacing torch, `ml_dtypes` supplying the BF16 patches. Adds `pillow` as the new
  `vision` optional dependency extra.
- **`benchmarks/vision_parity_check.py`**, **`vision_parity_check_fp32.py`**: diff the MLX vision port
  against the official PyTorch reference on a real image, dynamically loaded from the checkpoint's own
  `inference/` directory. Preprocessing is bit-identical; the FP32 forward diff is at machine precision
  (`6.7e-6`), confirming the BF16 forward's `3.8e-2` max diff is accumulated rounding noise across 32
  layers, not a defect.
- **`benchmarks/micro_qkv_fusion_roofline.py`**, **`qkv_fusion_live_smoke.py`**: the `wq_a`/`wkv` fusion's
  roofline measurement and a live sanity check on the real 2-bit bank.
- **`tests/test_attention_qkv_fusion.py`**: bit-identical output between the fused and unfused paths on the
  checkpoint's real shapes; the concatenation is cached rather than rebuilt on every call and two layers'
  caches do not collide.
- **HANDOFF sections 16.1, 9.34**.

### Verified
- 242/242 project tests pass (238 plus 4 new).
- `wq_a`/`wkv` fusion: live smoke test on the real 2-bit bank (`benchmarks/qkv_fusion_live_smoke.py`),
  prefill plus twelve greedy decode tokens, no crash, correct completion (`'Hello'`, `'!'`, EOS for "Say
  hello in one short sentence.").
- Vision port: `benchmarks/vision_parity_check.py` against the official reference on a real image (a
  700x486 downsize of `assets/dsv41_kv_cache.png` — full resolution drives the ViT to ~9,200 patches and a
  CPU-only dense-bidirectional-attention reference forward over that many tokens does not finish in
  reasonable time). Grids match exactly, patches bit-identical, FP32 forward diff at machine precision.

## 0.9.8 (2026-09-22)

**The shared expert's `w1`/`w3` fusion is shipped, unconditionally, after one live crash and one fix.**
Runtime change: the shared FP8 expert now issues two GEMVs per layer instead of three on every decode
token. Bit-identical output, no quality change.

### Changed
- **`src/cachalot/model/shared_expert_metal.py`, `src/cachalot/model/moe_layer_metal.py`**: `w1` and `w3`
  read the same quantized activation and neither depends on the other, so they now concatenate into one
  `[2I, H]` GEMV instead of two `[I, H]` ones (`shared_expert_forward_fused`, `_fused_w13`), cached per layer
  by weight-object identity. Measured at 0.115 → 0.106 ms/layer, 4.60 → 4.23 ms per token across forty
  layers (`benchmarks/micro_shared_expert_roofline.py`), first identified and left unshipped in 0.9.2's
  ranking (HANDOFF section 9.27) because 0.4 ms is below what any live session can confirm.
- `shared_expert_forward` (the original two-GEMV form) stays for `moe_prefill_grouped.py`'s separate
  per-token prefill fallback, which was never benchmarked fused and is unchanged.

### Fixed
- The first version of the fusion called `mx.eval` inside `shared_expert_forward` itself, which
  `moe_layer_metal._compiled_moe_block`'s `mx.compile`d trace calls on the shipped 2-bit affine bank — MLX
  refuses `mx.eval` mid-trace (`ValueError: [eval] Attempting to eval an array during function
  transformations like compile or vmap is not allowed`). Every offline check before this was caught,
  including a bit-identical unit test and the full 238-test suite, called the fused path eagerly, never
  from inside `mx.compile`, so nothing caught it before a live `./chat.sh` session did, on the first decode
  step. Fixed by moving `w1`/`w3` concatenation into eager Python before the compiled call, and by giving
  `_compiled_moe_block` a stable cache key again once the code path was unconditional.

### Added
- **`tests/test_shared_expert_fused.py`**: bit-identical output between the fused and unfused paths on the
  checkpoint's real shapes; the concatenation is cached rather than rebuilt on every call and two layers'
  caches do not collide; a test that calls the fused path from inside an actual `mx.compile`, reproducing
  the crash; a test pinning that `mx.eval` inside a trace still raises if the bug is reintroduced.
- **HANDOFF section 9.33**: `kernel_consts.py:39`'s previously-unexplained store-blocked time (carried since
  0.9.5's next-session prompt) is explained — a profiler attribution artifact, not a real or recurring cost.

### Verified
- 238/238 project tests pass.
- `V41Model.from_pretrained` run directly against the real 2-bit affine bank (`/Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128`)
  at a 24 GiB budget: warmup plus eight further `decode_token` calls, no traceback, the same eight greedy
  token ids before and after the fix, and again after the kill switch was removed.
- Tried live by Hamed: a 642-token story and a 1,744-token Objective-C json-to-CSV turn both read correct,
  9.91 and 8.27 tok/s, 92.35 % hit rate — in line with the shipped-52-GiB baseline (different prompts than
  the fixed four-prompt set, so not a controlled A/B, and 0.4 ms/token was never going to show up in a live
  read either way).

## 0.9.7 (2026-09-22)

**Server bug fix, budget investigation continued, vision scoped.** No change to the shipped runtime path.

### Fixed
- **`src/cachalot/server/app.py`**: `ServerConfig.default_frequency_penalty` was still 0.2, citing the
  62 %/12 % repetition-collapse claim that section 9.9 retracted (it was the transposed residual mix, not
  real repetition; the fixed runtime is 0 of 12 with the penalty off). An unmodified OpenAI client — the
  default case for a new Hermes connection — was the one remaining path still paying a penalty nothing
  causes any more. Now 0.0, matching `cachalot chat`. Two `tests/test_server.py` assertions that pinned the
  stale default updated to match; a third pins the knob is still configurable.

### Added
- **`serve.sh`**, mirroring `chat.sh`: the shipped 52 GiB / 80 GiB wired configuration as an HTTP server on
  port 8011. Smoke-tested this session against the real checkpoint: `/v1/models`, non-streaming and
  streaming chat completions, and a tool-calling round trip (valid JSON arguments, correct `finish_reason`).
- **HANDOFF section 7.2.10**: six budget arms (46-56 GiB) with `memwatch.sh` running beside them. No OS-level
  memory-pressure event at any budget; the one 54 GiB session's Objective-C-turn collapse (4.83 tok/s against
  8.4-8.8 at 48-52) has no memory signature and is turn-4-specific, not session-wide — refuting the prior
  physical-memory-cliff hypothesis for this run. 52 GiB stays shipped.
- **HANDOFF section 15**: the Hermes HTTP server audited, the frequency-penalty bug above, and what is not
  yet checked (Hermes's actual client behaviour against a single-flight engine).
- **HANDOFF section 16**: vision scoped. The checkpoint carries a real 263-tensor ViT + aligner (~480 M
  params, under 1 GiB) that `resident_trunk.py` currently filters out by name; a PyTorch reference
  implementation and image preprocessor ship in the checkpoint directory. Four-piece port plan, ordered by
  what can be checked before anything is wired into the text model.

## 0.9.6 (2026-09-22)

Measurement tooling and one live reading, no runtime change. **Hamed's first 54 GiB session decoded at 4.9-6.2
tok/s against 9.4-9.6 at 52, with 0.33 points more hit rate, and a 60 GiB attempt was abandoned as unusably
slow; a guarded replay of the same four prompts at 50 GiB on a clean machine then decoded at 9.58 and 8.52
tok/s with memory pressure normal.** The shipped configuration stays at 52 GiB. Why 54 is slow is not yet
established, and the tool that will establish it is new.

### Added
- **`benchmarks/memwatch.sh`**: samples free, available, wired, compressor, anonymous and file-backed memory,
  swap, pressure level, pageouts and decompressions once a second beside an interactive chat, never kills
  anything, and prints a summary on Ctrl-C. `guarded_run.sh` samples the same counters only for benchmarks it
  launches and kills on pressure, which a chat cannot be run under.
- `benchmarks/chat_turns.py --turns-file` and `--temperature`, so a live session's prompts can be replayed at
  the shipped temperature under `guarded_run.sh`. The 2026-09-22 session is in `benchmarks/sessions/`.

### Learned
- `CACHALOT_MLX_WIRED_LIMIT_GIB=90` does nothing: `resolve_wired_limit` takes the minimum of the request and
  the device's recommended working set, 77.8 GiB, and the banner printed 77.8 at 54 GiB exactly as at 52.
- The 50 GiB replay (`benchmarks/results/guarded/replay50_20260922-013337.*`): story 9.58 tok/s, Objective-C
  8.52, swap flat at 490 MB, compressor flat at 3.7 GiB, pressure 1, peak system wired 71.7 GiB. Section 7.2.9.

## 0.9.5 (2026-09-22)

Quality-gate change and no runtime change: **the coding gate now contains Objective-C, compiles it, and runs
it.** Two live Objective-C turns had failed a compiler and a third compiled only after a one-line repair and
still contradicted its own documented column order, while the 40-case gate contained no Objective-C and
executed nothing.

### Added
- **`benchmarks/code_validity.py`**: `check_objc` (clang `-fobjc-arc -fsyntax-only`, so a selector Foundation
  does not declare fails exactly as in a full build) and `run_objc`, which builds a block, runs it with a
  20 s timeout and no stdin, and diffs stdout against the text the task said it must print. `score()` runs a
  block only if it compiled; `summarise()` reports `executed`, `ran_clean` and `output_matches` next to the
  compile rate. A harness failure (no clang, no Foundation) is `valid: False` and never scored against the
  model. The C/C++ path was refactored into one shared `_check_clang` with no change in behaviour.
- **Six Objective-C tasks in `benchmarks/coding_tasks.json`** (26 tasks, up from 20), each a complete
  program with no input and an `expected_stdout`: CSV header in first-seen key order (the exact defect of the
  third live turn), word frequency, interval merge, LRU cache, Roman numerals, matrix transpose.
  `benchmarks/objc_reference/` holds a reference program per task; a test builds each one and requires its
  output to equal `expected_stdout` and to appear verbatim in the prompt.
- `coding_quality.py` recognises `objc` fences and writes `expected_stdout` into `rows.json`.
- Four tests (233 total): the live turn that calls `-map:` is rejected, the repaired turn compiles, running
  distinguishes right output from wrong output, every reference matches its expectation, and `score()` runs
  a block only when it compiled.

### Not done
- No model was run against the new cases; the corpus hash changed, so `--resume` will not continue a run
  begun on the 20-task corpus. Running the three banks on the 26 tasks (about 2 h) is the next quality job.

## 0.9.4 (2026-09-22)

Benchmark-instrument fix and one measurement; no runtime code changed. **Job 1 is answered: the miss is at
the drive's rated wall, and wired memory is a null.**

### Fixed
- **`benchmarks/expert_read_scaling.py`'s `--wire-gib` ballast-heartbeat thread crashed on its first tick
  on MLX 0.32.2** (`RuntimeError: There is no Stream(gpu, 0) in current thread`) because the pinged array
  was never `mx.eval`'d on the thread that created it, and 0.32.2 cannot resolve a default GPU stream for
  an array's first materialization from a different thread. The exception printed to stderr and the thread
  died; `main()` never saw it and the script's own exit code stayed 0, so three sweeps in a row silently
  measured an unwired machine after the first arm or two. Fixed with one `mx.eval()` call on the main
  thread before the heartbeat thread starts.

### Measurement
- **The miss is at the drive's rated wall, and wired memory pressure has no measurable effect.** At
  `io_workers=8`, 42.9 GiB wired (45 % of the machine): 6.81 GB/s, 1.46 ms/expert. Unwired: 6.76 GB/s,
  1.47 ms/expert. Both match section 3.1's "6.6-6.8 GB/s cold" rating, and the runtime's own 1.41 ms
  blocked-per-miss component sits inside that same band rather than above it. Tested at 45 % of the
  machine's RAM wired, not the shipped 52 GiB budget's 80-83 %, because available memory this session
  (60-64 GiB free) did not admit a bigger ballast. **The budget is the only lever left on the miss.**
  §7.1.11, §9.31.

## 0.9.3 (2026-09-21)

Documentation and one archived live turn; no code and no numerics changed. **The shipped 52 GiB
configuration was read a second time and replicates, and every MLX peak this project has ever quoted was
in GB against a limit in GiB.**

### Measurement
- **The second live session at 52 GiB, on a build with no runtime change against the first.** Prose
  **9.59 tok/s** against 9.42, Objective-C **8.15** on 1,484 tokens against 8.53 on 1,493, session hit
  rate **92.31 %** against 92.37 %, 5,276 residents against 5,314, and an MLX peak **identical to the
  byte**. The configuration replicates, so **52 GiB has two sessions and not one**. What is still unrun is
  section 9.24's A/B: this session had the Engram change on, so it is a second reading of the good arm.
  §7.2.7.
- **Section 7.1.9's price of a miss, checked against a conversation for the first time.** 502,435 expert
  requests over 240 per token is ~2,093 token-equivalents and 38,661 misses, so **18.5 misses per token**;
  at 1.7 ms each on the 79.6 ms floor that predicts **111 ms**, against **104.3 ms** observed on the prose
  turn and **122.7** on the coding one. The prediction lands between the session's own two long turns.
- **About 48 % of the session's drive traffic was speculative.** 755.07 GB read; the 38,661 demand misses
  account for 384.8 GB, the hotlist for 8.6 GB, and prefill was almost entirely prefix-cache reuse (13
  hits, 4,192 tokens, 619 of 623 on the coding turn). Section 9.10's 42.7 % was measured at 44 GiB on a
  different shape of session, and no instrument in the repository can say what fraction of the 360 GB
  earned its place at this budget.

### Corrections
- **`mlx_peak_bytes` is bytes, and four sections divided it by 10^9 while dividing the wired limit by
  2^30.** Every "peak against limit" pair before section 7.2.8 understates the headroom by 7.4 %: the
  52 GiB peak is **67.74 GiB, not 72.73**, against a 77.8 GiB wired limit, and the 44 GiB peaks are
  55.1-55.6 GiB rather than 59.2-59.7. The arithmetic settles which reading is right — 5,276 residents is
  48.91 GiB of experts, and the trunk, transient slots and MLX cache are about 14.4 GiB more, which 67.74
  clears and 72.73 would need 23.8 GiB of unaccounted memory to reach. **Nothing about any speed or
  quality conclusion moves**; the peaks were only ever used to decide whether a budget fits and the error
  was in the safe direction. What moves is the headroom, and therefore the next budget worth screening:
  **10.1 GiB free at 52, and 60 GiB projects to about 75.2 GiB** against a replay that says 52 → 60 is
  worth another 2.3 points of hit rate. §7.2.8.
- **`predicted_used` is not the numerator of a precision.** It increments in one place
  (`resident_store.py:629`), when a demand request catches a prediction **still in flight**; a prediction
  that lands before it is demanded is counted as an ordinary hit. Two sections called
  `predicted_used / predicted_loads` "prediction precision"; it is a lower bound by an unknown margin.
  §7.2.7.

### Quality
- **The third live coding turn put through a compiler, and the third to fail** — but the useful one.
  `[NSMutableArray map:]` is not declared by Foundation; one line repairs it and the program then compiles
  clean, runs, exits 0 and writes a valid CSV. **Its column order still contradicts both its own Notes and
  its own worked example**, because the header is built from `flat.allKeys`, which is unordered. This is
  the first case in the project where compiling is not the check either, and it is the argument for a gate
  that *runs* what it builds. Archived with the compiler output, the repair and the real output in
  `docs/live-turns/2026-09-21-json2csv-2/`.
- A bare "Hi" was answered in Chinese again, as in the first 52 GiB session, and did not recur once the
  session asked for English. A model behaviour on a multilingual checkpoint, reproducible across sessions,
  and nothing in the 40-case corpus looks for it.

229 tests pass. Version 0.9.3.

## 0.9.2 (2026-09-21)

Measurement and documentation; **nothing under `src/cachalot/` was touched and no numerics changed.**
**The last block in the ranking with no mechanism turned out not to be a block: it is the miss.** With
that settled, the three GPU blocks nobody had screened were screened, and all three came back at or near
the machine's own limit.

### Measurement
- **A streaming token at a 100 % hit rate is the all-resident floor.** `profile_decode_sync.py` gained
  `--stream-passes`, which replays the same continuation from the same snapshot; decoding is greedy, so
  pass 2 runs the identical tokens through the identical graph at the identical positions over the same
  240-experts-per-layer working set scattered across a 40 GiB slot pool, with everything already resident.
  It costs **79.6 ms against the all-resident arm's 79.6**, 58.5 ms inside `mx.eval` against 58.0, 19.3 ms
  of CPU against 19.3. So the "+20 ms inside `mx.eval`" that three prompts called unexplained is part of a
  miss, worth **0.31 ms of each one** on top of the 1.41 ms the store already charges. §7.1.9.
- **`io_workers` is a null from 2 to 16**, on eight interleaved arms with two reps of each width: every
  whole token inside 141.5–148.5 ms, and the two reps of the shipped width span 5.5 ms of that on their
  own. The in-eval / blocked split does not move. §9.26.
- **`CACHALOT_PAGE_CACHE=0` is a 16 ms loss**, in the blocking and the CPU, with the GPU getting nothing
  back. The shipped `=1` is confirmed from a direction nothing had tried. §9.26.
- **Attention is 86 % weight streaming.** A reuse layer's 0.441 ms is 0.270 ms of FP8 GEMV over `wq_a`,
  `wq_b`, `wkv` and `wo_b`, 0.107 ms of BF16 grouped matmul over `wo_a`, and **0.064 ms** of the sparse
  attention whose shapes three prompts wanted attacked. The weight inventory is read off the checkpoint
  headers: 126.6 MB per layer on disk, 160.2 MB resident. §7.1.10.
- **The FP8 GEMV family is the largest GPU path in the runtime** — four attention projections on forty
  layers plus all three shared-expert GEMVs, **5.14 GB per token against the routed experts' 2.3 GiB**,
  16.61 ms — and it runs at **79 % of `mx.sum` over the same bytes**. §7.1.10.
- **The shared expert, screened for the first time** after three prompts carried it unscreened: 307 GB/s
  against 385 for `mx.sum` over its own bytes, and the screen reproduces the profiler to 4 %. The one
  structural idea its shape allows — stacking `w1` and `w3` into one [4608, 5120] GEMV, which is
  **bit-identical by construction** because `K` and therefore the lane split are unchanged — is worth
  0.4 ms per token and is **not shipped**, because 0.4 ms is a tenth of what a live session resolves and
  this project does not ship on a screen. §9.27.
- **`wo_a` is held dequantized in BF16 and that is the right call.** It reads 67.11 MB per layer where the
  checkpoint ships 33.55, but MLX's BF16 grouped matmul runs it at **625 GB/s** and is faster than every
  FP8 form measured, including a one-launch upper bound that ignores the indexing a grouped FP8 kernel
  would need. §9.28.
- **The FP8 GEMV kernel's lanes-per-row policy, written for load balance with `N` not an input, picks the
  fastest split on five of six shapes.** Best split per shape is 16.56 ms against the shipped 16.61. §9.29.

### Instruments
- `benchmarks/profile_decode_sync.py`: `--stream-passes N` and `--io-workers N`, and the ready line now
  prints the reader width.
- `benchmarks/micro_fp8_gemv_kernel.py`: the largest GPU path priced per shape against `mx.sum`, every
  legal lanes-per-row split, each checked for bit-identical output against the shipped split.
- `benchmarks/micro_shared_expert_roofline.py`: the shared expert against the memory wall on forty
  distinct layers, with the `w1`/`w3` fusion and the activation-quantization launches priced separately.
- `benchmarks/micro_wo_a.py`: `wo_a` in BF16 against every FP8 form of the same projection.

### Pitfalls
- **Read which function an instrument calls before ranking a lever off it — fifth occurrence.** A
  four-variant kernel screen was written, run and found bit-identical against
  `fp8_gemv_metal.fp8_gemv_quantized`, which the runtime does not call: `fp8_linear_quantized` dispatches
  to `fp8_fused_metal.fp8_gemv_decoded` whenever `CACHALOT_FUSED_FP8` is set, and it is set by default.
  The shipped kernel was already 30 % faster than the best variant of the retired one.
- **`--mode both` and `--mode stream` do not produce the same streaming arm**: the all-resident arm leaves
  240 experts pinned and changes what the continuation evicts, which is 7 ms.
- **Interleave the arms of a sweep and run each twice.** The shipped `io_workers` width spanned 5.5 ms
  across its own two reps, the whole range of the sweep.

229 tests pass. Version 0.9.2.

## 0.9.1 (2026-09-21)

Documentation and one archived artifact; no code and no numerics changed. **0.9.0 was read live at a 52 GiB
expert budget: 9.42 tok/s on prose, 8.53 on 1,493 tokens of Objective-C, a 92.37 % session hit rate.**

### Measurement
- **The live session.** Against 0.7.0's four sessions at a 44 GiB budget — 7.78–7.92 tok/s on prose,
  6.90–6.93 on Objective-C, 89.92–90.00 % hit rate — 0.9.0 at 52 GiB runs at **9.42 and 8.53 tok/s with a
  92.37 % hit rate**, 5,314 resident experts and an MLX peak of 72.73 GiB against the 77.8 GiB the wired
  flag set. That is 22–27 ms off a token; the miss arithmetic gives the larger budget about 6 ms and the
  Engram change of 0.9.0 the remaining 16–21. **The two causes were not separated in one session.**
- **`simulate_policies.py` was validated against a live run for the first time.** It predicted +2.6 points
  of decode hit rate for 44 → 52 GiB; the session moved the session hit rate by **+2.37**.
- **52 GiB fits on a 96 GiB machine** with an 80 GiB wired limit and no memory-pressure event. `chat.sh`
  still defaults to 44 pending a second session.
- **The live coding turn does not compile.** `CSVEscape` sends `-stringValue` to an `NSString`, which the
  class does not declare; repairing that one line compiles the program and it then aborts on the model's
  own example input, because the same mistake appears again behind an `id` from `-allKeys`. Everything else
  in the program — RFC 4180 quoting, nested-value serialisation, the build line, the worked example — is
  correct. It is a model-level type error of the same class as 0.7.0's turn, and not this runtime's defect
  class. The turn, the compiler output and the one-line repair are archived in
  `docs/live-turns/2026-09-21-json2csv/`.

### Documentation
- `docs/HANDOFF.md` section 7.2.6 (the live session, what it proves and what it does not), with section 9.4
  closed live, section 9.24's open live question answered, and section 12.1 rewritten around the 52 GiB
  session shape.
- **Two standing rules revised.** A live session cannot resolve 5–8 ms, which is what the old rule was
  calibrated on, but it resolved 22–27 ms here; and an offline simulation should be checked against a live
  run the first time one is possible.
- README: the interactive numbers now carry both budgets. Next-session prompt v28; v27 archived.

## 0.9.0 (2026-09-21)

**The Engram row reads came off the decode thread: +15.2 % on the benchmark decode rate, with the numerics
bit-identical.** This is the first change to move the rate since 2026-09-19, and what it removes is the
block three successive analyses called the largest unexplained cost in the project. 229 tests.

### Performance
- **Engram row reads go through the reader's worker pool.** `EngramRowReader` only used its sixteen workers
  for batches of 64 rows or more, a threshold chosen for prefill. A decode token asks for 24 rows twice, so
  every decode batch took the serial branch: 96 `pread`s issued one after another from the decode thread,
  each waiting behind a queue the expert stream was filling. `CACHALOT_ENGRAM_PARALLEL_MIN` (default 8) is
  the threshold; every worker writes into its own slice of the output buffer, so the assembled rows do not
  depend on the scheduling.
- **Both Engram layers' reads are issued at the top of the token.** The row ids come from the token being
  decoded, which is known before layer 0 runs, so layer 14's read has thirteen layers of compute to hide
  behind and layer 1's has one. `CACHALOT_DECODE_ENGRAM_PREFETCH` (default 1).
- **Measured.** `decode_anatomy.py`, 36 GiB budget, 96 tokens of continuation: **5.65 → 6.51 tok/s**,
  177 → 154 ms per token, of which 23.7 ms comes out of `rest` — at an identical 81.6 % hit rate, an
  identical 694-695 MiB read per token, 44.1 misses per token either way and 45 % prediction precision
  either way. Repeated at a 40 GiB budget: **5.69 → 6.61 tok/s** (+16.2 %), 176 → 151 ms per token,
  `rest` 120.1 → 95.9, with the hit rate (83.7 %), the bytes (632 MiB), the misses (39.2) and the
  precision (43 %) identical to the digit. `profile_decode_sync.py`, same budget, median streaming token: 188.3 → 162.4 ms, with the
  Engram column falling from 28.4 ms to 2.4 and every other column unchanged. The two mechanisms compose:
  parallel reads alone are 167.8 ms, the prefetch alone 165.3.
- **Numerics unchanged and shown to be.** `decode_fingerprint.py`, 16 greedy tokens: identical ids and
  identical fp32 logit sums and maxima to six decimals against the previous shape. Both flags restore it
  (`CACHALOT_ENGRAM_PARALLEL_MIN=1000000 CACHALOT_DECODE_ENGRAM_PREFETCH=0`) for an A/B.

### Instruments
- **`benchmarks/profile_decode_sync.py --mode both` runs a streaming arm beside the all-resident one** and
  splits every interval between two `mx.eval`s into time inside eval, time blocked in the expert store,
  time inside an Engram `pread` and what is really CPU. This is what found the Engram reads: of the 111 ms
  a streaming token costs over an all-resident one, 63 are the priced miss cost, 27 were Engram, 20 are
  inside eval and 4 are CPU.
- **`benchmarks/profile_decode_gpu.py` times the compressor and the indexer**, the two pieces inside a
  source layer's extra millisecond, the indexer at four `index_topk` widths.

### Measurement
- **A source layer's 5.5 ms per token is decomposed**: about two thirds is the indexer, a tenth the
  compressor and the rest the compressed-KV write.
- **`INDEX_TOPK` is not a lever.** The indexer costs 0.485, 0.474, 0.501 and 0.540 ms per layer at widths
  128, 256, 512 and 1024 — an eightfold change is worth 0.066 ms per layer and is not monotone below the
  shipped 512.

### Tests
- `tests/test_engram_reader_parallel.py`: the parallel and serial row paths against each other and against
  the table itself at seven batch sizes, repeated ids keeping their positions, and the default threshold
  being at or below a decode batch.

### Documentation
- `docs/HANDOFF.md` sections 7.1.7 (where a streaming token's time goes), 7.1.8 (the source layers'
  extra), 9.24 (the lever) and 9.25 (the ranking). Roadmap items 8 and 10 close.

## 0.8.1 (2026-09-21)

Documentation only; no code, no numerics, no version of the runtime's behaviour changed.

### Documentation
- **The changelog now covers every released version.** 0.4.0 through 0.7.0 were tagged and pushed with no
  entry; they are backfilled from the release commits and `docs/HANDOFF.md`, newest first.
- **The README carries the shipped configuration's numbers** instead of the FP4-era ones. Performance,
  Status, Roadmap, the storage table and the mirror-striping guidance were all quoting a runtime that
  decoded at 2.8–2.9 tok/s; the shipped one runs at 7.6–7.9 with an expert hit rate near 90 %, and mirror
  striping is a loss at the expert size that now ships.
- Roadmap re-ordered by measured size in a token, matching `docs/HANDOFF.md` section 9.23.

## 0.8.0 (2026-09-21)

Measurement only: the shipped configuration and every numeric it produces are unchanged. The GPU side of a
decode token is now accounted for layer by layer, a synchronisation has a price, and two levers were
screened and closed. 220 tests.

### Measurement
- **The GPU side of a decode token is fully accounted for.** `profile_decode_gpu.py` now covers the ten
  layers it never did — layer 0 and layer 1's sliding-window attention, the four compressed sources, the
  four index-only sources — plus the head and both Engram forwards, and it prices what an `mx.eval` costs
  to drain. The shipped pieces sum to 41.6 ms against the 54.8 ms a token spends inside `mx.eval`, and the
  remainder is the round trip itself. Attention is 22.5 ms per token, the largest GPU block by a factor of
  three. HANDOFF section 7.1.5.
- **`benchmarks/micro_eval_floor.py`** (new, no model): one `mx.eval` costs about 0.20 ms whatever it
  evaluates, and every way of reading a value back costs the same, so a token's 44 synchronisations are
  about 9 ms of fixed cost. HANDOFF section 7.1.6.
- **`benchmarks/micro_compile_attention.py`** (new): `mx.compile` on decode attention is closed three ways
  — `shapeless=True` cannot infer a custom Metal kernel's output shapes, a plain trace is not
  bit-identical, and it retraces on every token for a net loss. HANDOFF section 9.21.
- **`benchmarks/decode_fingerprint.py`** (new): 16 greedy tokens with their ids and fp32 logit checksums,
  for diffing two arms of a speed change.

### Runtime
- **`CACHALOT_PRELAUNCH_SHARED`** (new, default `0` = off): issues the shared expert before the layer
  blocks on its routing, so the GPU has work during the round trip. `1` submits it with `mx.async_eval`,
  `2` adds it to the routing's own `mx.eval`. Arm 1 moves 10 ms per token out of `mx.eval` and puts 13 ms
  back on the CPU; arm 2 costs nothing on the CPU and 5 ms inside eval. 79.1-79.6 ms shipped against
  81.8-82.6 and 83.5-85.7. Every arm's fingerprint is identical to the shipped one. Off until an arm wins.
  HANDOFF section 9.22.

## 0.7.0 (2026-09-21)

Measurement only: no numerics changed and the shipped configuration is unchanged. Every named way of making
routing prediction cheaper was measured and closed, and the 44 GiB configuration that ships was profiled for
the first time. 219 tests.

### Measurement
- **The shipped budget has a profile.** 44 GiB, 512-token context: 170 ms per token at an 83.5 % expert hit
  rate reading 627 MiB, of which 54.0 ms is blocked on the store and 46.4 of that is misses no prediction
  covered, with the drive idle 45 % of the time. Reproducible to ±0.3 %. HANDOFF section 7.1.3.
- **Three ways to reclaim the prediction's waste, all closed.** Admitting a mispredicted load has a 4.3 %
  ceiling; a blocklist on re-reading a dropped expert trades 8.1 wasted reads for 3.4 demand misses; the
  submission bookkeeping is 0.040 ms per token, not the 3.6 two prompts had carried. HANDOFF section 9.19.
- **Four nulls at the shipped budget**: `CACHALOT_PREDICT_AHEAD=2`, top-4, top-8 and every
  `sys.setswitchinterval`, which rules reader-thread scheduling out of the 33 ms a streaming token spends
  above the all-resident floor.
- **`benchmarks/predict_ghost.py`** (new): what happens to a mispredicted expert after it is dropped, and
  what a blocklist of any lifetime would have done.
- **`benchmarks/micro_predict_submit.py`** (new): the prediction submission bookkeeping, four arms, no model.
- **`profile_decode_gpu.py` corrected**: two of its rows were timing paths the runtime had stopped taking,
  which had put routed experts at 19.8 ms per token instead of 6.7 and the router at 7.5 instead of 0.9.

## 0.6.0 (2026-09-21)

The release that restored quality and took the first ~8 ms off the compute floor. Output is equal to the
hosted reference on every column of the coding gate, and the decode path gained two shipped changes.

### Fixed
- **The hyper-connection residual mix was applied transposed.** `hc_post` computed `comb @ residual` where
  the official implementation does `comb.T @ residual`, in both the MLX path and the fused Metal kernel.
  Re-gated through the fixed runtime, the 2-bit bank compiles **20 of 20** C++ blocks, parses 18 of 18
  Python blocks and emits **0 of 101** malformed `#include` lines — every column equal to the hosted
  reference, against 0/42, 5/26 and 49/154 before. `tests/test_hyper_connection.py` pins the contraction.
  HANDOFF section 7.4.8.
- **The repetition collapse was the same defect, not the sampler.** 0 of 8 against the original 5 of 8 on
  the same bank and protocol, p = 0.026, so `--frequency-penalty 0.2` is no longer needed. HANDOFF 9.9.

### Runtime
- **The decode MoE block is traced once instead of rebuilt forty times a token.** `mx.compile` over the six
  routed experts, the shared expert and their sum takes the all-resident token from 82-85 ms to **77 ms**
  and the CPU side from 27 to 21.5, with mean NLL identical to four decimals. HANDOFF section 9.16.
- **The fused Metal kernels' scalar parameters are built once** (`model/kernel_consts.py`, about 1,600
  constructions per token): **1.2-1.3 ms** of CPU per token, numerics unchanged. HANDOFF section 9.17.
- **Mirror striping is off on this bank.** It is a property of the expert size, not of the runtime: an 18 %
  loss at 9.49 MiB per expert where it was a gain at 17.93. HANDOFF section 9.11.1.
- **`./chat.sh`** (new): exports the shipped environment, refuses a second runtime and passes flags through.
  The multi-variable one-liner it replaces does not survive line wrapping, and the failure is silent — the
  runtime serves FP4 off the USB drive at 0.6 tok/s. The expert-bank banner now prints unconditionally.

### Measurement
- **The compute floor was being measured wrong.** The hyper-connection kernels are **4.6 ms** of a token,
  not 68.7; the 68.7 was an artifact of a profiler that evaluates each piece behind its own barrier. An
  all-resident token is about 65 % GPU wait and 35 % CPU. HANDOFF sections 6.3, 6.3.1.
- **Context length is not what makes a long turn slow.** Worth 4.7 ms of `rest` across a fourfold change
  and not monotone; the live spread between turns is working-set variation. HANDOFF section 6.4.

## 0.5.0 (2026-09-17)

A measurement release: five levers worked, three of them settled against what the handoff expected, and the
quality gate's own statistics were repaired.

### Measurement
- **The compute floor is 93 ms per token, not 120.** 49 % of a decode token is expert streaming and the
  arithmetic half is roughly 400 GPU dispatches rather than anything a faster kernel would fix.
- **The `mtp.*` layers are DSpark**, not plain multi-token prediction: five drafted tokens per forward and
  2.85 accepted, but verification reads W/T times the bytes, so the lever is 1.20x rather than 1.5x.
- **The 512-token mean NLL has a paired standard error wider than every difference it was asked to rank.**
  The gate reports the paired median and the sign test instead, and the operating rules say to judge by them.
- **The coding gate was scoring failed compilations as clean.** `check_cpp` grepped stderr and never read
  the compiler's exit status, so every C++ ratio older than 2026-09-19 is withdrawn. HANDOFF section 7.4.1.
- **In-flight predictions had no lifetime** and were discarded for finishing early; fixed and shipped.
  HANDOFF section 9.13.

## 0.4.0 (2026-09-17)

Decode 275 -> **182.5 ms per token** at a 36 GiB budget, 3.64 -> **5.48 tok/s**, from a 2-bit routed-expert
bank built here out of the FP4 checkpoint and a prediction width the smaller expert made worth raising.
Interactive chat at a 44 GiB budget runs at **6.0-7.5 tok/s** against 4.3-5.6 before, with an 87.3 % session
hit rate and sub-second follow-up prefills.

### Runtime
- **A 2-bit affine g128 expert bank**, 9.49 MiB per expert against FP4's 17.93, built from the checkpoint by
  `benchmarks/build_affine_bank.py`.
- **`CACHALOT_PREDICT_TOPK` raised from 3 to 6**, which only becomes worth doing once experts are small
  enough that the extra reads stay under compute.

### Note added in 0.6.0
The quality cost this release reported against the 3-bit bank — +0.019 nats and 6.3 points of top-1, with
occasional mangled tokens — **was the transposed hyper-connection residual mix, not the bank**. Re-gated
through the fixed runtime the 2-bit bank is equal to FP4 and to the hosted reference on every column.

## 0.3.0 (2026-09-16)

Interactive use on a 96 GB Mac: a chat follow-up turn now answers in about a second instead of ten, and decode
is roughly 40 % faster end to end. Two kernel panics on 2026-09-15 are also addressed at the root.

### Runtime
- **Alternative expert banks.** `CACHALOT_EXPERT_BANK=<dir>` serves the routed experts from a different
  checkpoint than the trunk, Engram and tokenizer. `storage.index.detect_expert_bank` recognises oMLX-converted
  checkpoints (`omlx_deepseek_v41` in `config.json`), whose experts are stacked per layer, and indexes them into
  per-expert byte ranges; `model/expert_affine.py` evaluates them with `mx.quantized_matmul`. Validated against
  `Jundot/DeepSeek-V4.1-Flash-oQ3e-mtp` (calibrated affine 3-bit, 14.77 MiB per expert against FP4's 17.93):
  teacher-forced NLL 2.3004 -> 2.3080 nats, and decode 2.89 -> 3.36 tok/s at an equal 28 GiB budget because the
  same budget holds 1,941 experts instead of 1,599. The FP4 path is unchanged.
- **Idle heartbeat.** Within about six seconds of an idle Metal queue macOS un-wires the whole working set despite
  `mx.set_wired_limit` (wired 50 -> 6 GiB, then 44 GiB in the compressor), and the next turn pays the
  decompression: a 13-token follow-up prefill took 10.5 s after a typing pause against 2.9 s back to back. A daemon
  thread now evaluates a one-element op every `idle_heartbeat_seconds` (default 0.5,
  `CACHALOT_IDLE_HEARTBEAT_SECONDS=0` disables) while no forward pass runs.
- **One-layer-early routing prediction.** Layer L+1's router is applied to layer L's router input and the top-k
  predicted experts load into free transient slots while the GPU runs layer L. Decode 2.26 -> 2.55 tok/s at a
  28 GiB budget; `CACHALOT_PREDICT_TOPK` defaults to 3 (top-2 and top-4 were both worse, the SSD being
  bandwidth-bound).
- **Environment overrides applied when a value is auto.** `TextDecodeRuntime` replaced the environment-loaded
  config unconditionally, so the default 0 discarded `CACHALOT_EXPERT_CACHE_BUDGET_GIB` and
  `CACHALOT_MLX_WIRED_LIMIT_GIB`. Benchmarks asking for 32 GiB ran with the 47-50 GiB auto budget, which wires
  ~67 GiB on a 96 GB machine and, with other applications open, drove it into swap and two kernel panics.
- **Short prefills keep the cache.** Prefill admission evicted every resident of a layer the prompt did not route
  to, so the 2-3-token speculative prefills below shrank the resident set from 1,599 to 453 experts. Quota room a
  prompt's own misses do not fill now keeps that layer's current residents, most recently used first.
- **Concurrent piece reads.** An expert split across several tensors (any stacked bank) has its pieces read in
  parallel: nine scattered reads of a 15.5 MiB expert take 4.2 ms serially, 2.8 ms concurrently, at the same
  aggregate bandwidth. Mirror striping stays on the serial path and should be left off for stacked banks.

### Chat
- **Typing-time prefill.** On an interactive terminal the chat reads keystrokes raw and, after 0.4 s without a key,
  prefills the template head plus the finished words of the message through the prefix cache; Enter then pays only
  for the last word and the template tail. Measured with a pseudo-terminal typing at human pace, the follow-up turn
  reused 50 of 57 prompt tokens instead of 37 and waited 2.0 s instead of 5.5 s. `--no-typing-prefill` disables it
  and `/stats` reports the speculative work.

### Benchmarks
- `benchmarks/guarded_run.sh` runs any benchmark under a memory guardian: it refuses to start when another runtime
  is alive, when memory pressure is not normal, when the boot volume is nearly full or when free plus reclaimable
  memory cannot hold the planned wired set; it samples `vm_stat`, swap, pressure, RSS and the compressor every
  second into a CSV; and it kills the command on critical pressure, sustained warning pressure, swap growth, low
  disk, a timeout, or a runtime that reports a larger budget than requested.
- `benchmarks/nll_expert_precision.py` gates an expert bank on teacher-forced NLL with identical dense reference
  math on both sides; `benchmarks/check_oq3e_shards.py` compares two banks weight by weight without loading the
  model; `benchmarks/chat_turns.py` replays a six-turn chat with optional idle gaps;
  `benchmarks/chat_pty_typing.py` drives the CLI through a pseudo-terminal.

## 0.2.0 (2026-09-15)

First public release as **Cachalot** (package renamed from `v41runtime`).

### Runtime
- `CACHALOT_MIRROR_PATH` / `CACHALOT_MIRROR_FRACTION`: a second identical checkpoint copy on another drive serves
  the tail of every expert read concurrently (byte striping, offsets identical). 10 % on a 1 GB/s USB mirror:
  decode 2.86 -> 3.00 tok/s, 512-token cold prefill 32 -> 28.5 s.
- `TextDecodeRuntime.warmup()` compiles all kernels at load (`V41Model.from_pretrained(warmup=True)`); the first
  decoded token no longer pays ~0.6 s of Metal compilation. Opt-in `CACHALOT_EVICT=lfu` eviction (no measurable gain).
- Prefill loads the next layer's most-used experts speculatively while that layer's router is still being
  computed (cancelled if unneeded, promoted if needed; experts consumed in arrival order) and reads both Engram
  layers' rows in the background from prefill start. 2048-token prefill 55 s -> 44 s cold / 46 s -> 38 s warm,
  512 tokens 32 s / 23 s, i.e. at the SSD floor. `CACHALOT_SPECULATIVE_PREFILL=0` disables speculation.
- The auto expert budget is also capped by memory available at start (free + purgeable + reclaimable file
  cache minus trunk, MLX cache and 12 GiB headroom).
- Prefill attention processes 256-token chunks (`CACHALOT_ATTN_CHUNK`), `wo_a` runs as per-group GEMMs, and FP8
  linears use exact bf16 operands; 2048-token prefill 57 s -> 55 s cold / 49 s -> 46 s warm, 512 tokens 33 s / 25 s.
- Prefill routed experts run on simdgroup-matrix FP4 kernels (`fp4_sgmm_metal.py`: dequantize once, bf16 MMA,
  fp32 split-K) for up to 64 rows per expert, `quantized_matmul` above; 2.5x less GPU time per expert.
  `CACHALOT_PREFILL_SGMM=0` restores the affine-8 path.
- Fused decode path (`decode_fused_metal.py`, `router_fused_metal.py`, `fp8_fused_metal.py`): single-launch
  router top-k, sparse attention, hyper-connection mixes, RoPE, RMSNorm, hc_pre/hc_post, FP8 activation quantization,
  and a vectorized FP8 GEMV; MoE output dispatched asynchronously. All-resident decode token 0.10 s -> 0.068 s.
  Every fused kernel switches off with `CACHALOT_FUSED_DECODE=0` / `CACHALOT_FUSED_FP8=0`.
- Batched compressor/indexer source layers (2/8/14/20, 24/28/32/36); Engram rows via parallel pread instead of
  mmap faults (9-18 s -> 0.2 s per layer); miss-aware prefetch lookahead. 512-token prefill 35 s cold / 29 s warm.
- Routed experts in prefill use an exact FP4 -> affine-8-bit repack and `mx.quantized_matmul`.
- Batched prefill: chunked attention for sliding-window and reuse layers, batched hyper-connection mixes,
  router, routed-expert dequantize+GEMM, shared expert, Engram; `wo_a` dequantized at load.
  512-token cold prefill 86 s -> 55 s on the internal SSD.
- Opt-in decode miss budget (`set_decode_miss_budget`), measured and documented as unusable for quality; off.
- Fused top-k routed-expert Metal kernels (two launches per layer) and a bf16-reading fp32 head GEMV;
  all-resident decode token 0.10 s.
- Checkpoint path discovery (`~/DeepSeek-V4.1-Flash`, `~/models/`, `/Volumes/*`, or `CACHALOT_MODEL_PATH`).
- Pre-allocated, wired expert slot pool; experts are `preadv()`'d from the shard straight into MLX unified memory
  (no per-expert allocation, memcpy, or Metal residency churn).
- `mx.set_wired_limit` keeps trunk + experts resident; macOS no longer compresses cold expert buffers
  (decode 2.3–3.4 s/token → SSD bytes + 0.15 s).
- Expert shard reads bypass the page cache (`F_NOCACHE`).
- Auto-sized expert budget from unified memory (`expert_cache_budget_bytes=0`), `system_reserve_bytes`.
- Prefix cache: multi-turn requests prefill only the new suffix; snapshot/restore of sequence state.
- Decode misses of a layer are loaded concurrently and admitted in logical order.
- Streaming generation core with cancellation; `top_k`/`top_p` layered on the official sampler.
- `max_seq_len` default 32,768.

### Serving and tooling
- OpenAI-compatible server: `/v1/chat/completions` (SSE, tools, `response_format`, thinking mode with
  `reasoning_content`, `stop`, usage with prefix-cache stats), `/v1/completions`, `/v1/models`, `/v1/stats`,
  `/health`, optional Bearer auth.
- `cachalot serve | chat | doctor | bench` CLI; `CACHALOT_*` environment overrides.
- Routing tracer, trace analysis, policy replay simulator, decode profilers under `benchmarks/`.
- Checkpoint-free test suite (store, prefetch, prefix cache, config, server) and GitHub Actions CI.

### Inherited from the pre-release runtime
- Exact DeepSeek V4.1 Flash text path in MLX + Metal (mHC, CSA2 attention, Engram, FP4/FP8 kernels).
- Layer-major prefill with expert-major MoE scheduling and deterministic per-layer admission.
- MLX free-buffer cache capped at 2 GiB.
