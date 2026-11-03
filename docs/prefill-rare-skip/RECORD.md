# Prefill rare-expert skip: experiment record

Written 2026-11-03 against runtime 0.78.0, before any run on the machine. HANDOFF section 18.259. Lever id **P-RARE**. Hamed chose the speed lane ("Speed", 2026-11-03); the knob is built and off by default, every number below is a price or a prediction until the runs in section 5 exist.

## 1. Baseline (recorded)

- A short or mid prefill is bound by expert reads: a 246-token chunk paid about 3,250 reads, 5.2 s of drive time inside a 6.8 s chunk (HANDOFF 18.89, 0.61.12, first-release bank, 44 GiB). Each prefill call reads every routed expert that is not resident, once a layer (`ResidentExpertStore.prepare_prefill_layer`, `moe_prefill_grouped`).
- Live follow-up prefills on the shipped 1017 defaults: 6,624 tokens in 33.4 s, 2,292 tokens in 24.2 s (LEDGER DS1017-DI2-LIVE, 0.75.1). Cold through the server: 16,040-token Hermes block 90.06 s, 4,277-token prompt 28.68 s (DS1017-DI2-SRV, keep 140).
- Shipped prefill: DI-1 encoder-only (layers 20 to 39 run once over the last rows of a prompt of more than one call) with DI-2 keep 140 (`serve.sh`), chunk 4,096, 48 GiB, hotlist on, decode miss budget 0, tool-exact decode.

## 2. Hypothesis and price (derived offline, 2026-11-03)

**Hypothesis.** In a prefill call of many rows, a large share of the expert reads go to experts that only one or two rows route to, and those experts carry a small share of the routing mass. Not reading them saves drive time in proportion, at a quality cost close to that of dropping their mass.

**Instruments.** `price_rows.py` (distinct experts by rows a chunk, and their mass) and `price_reads.py` (a per-layer LRU of `slots / 40` with prefill and decode sharing it, the rule applied with the shipped DI-1/keep 140 eligibility) on the two 1017 Hermes routing traces of HANDOFF 18.246 (`~/cachalot-routing-1017-s1.trace.npz`, `-s2`, 19 and 13 requests, router's own picks with weights). The residency model is approximate (no hotlist pinning, no cross-layer reclaim): the read counts are a price, not a measurement.

**Rows.** In follow-up prefills of 1,000 to 6,500 tokens, experts routed by one row of a 4,096-position chunk are 11-17 % of the distinct experts a layer touches and carry 0.2-0.5 % of the routing mass; by at most two rows, 20-25 % and 0.4-1.2 %. Below 1,000 tokens the mass share rises fast (422 tokens: 1.7 % at one row; 28 tokens: 17 %), which is why the rule has a minimum call size.

**Reads (session 1, 48 GiB = 5,179 slots, min rows 1,024).** Of all follow-up reads (13 follow-ups, 90,938 reads), k = 1 saves 6.8 % and k = 2 11.3 %. On single-call follow-ups of 1.3-2.4k tokens it saves 16-18 % (k = 1, mass 0.21-0.34 %) and 28-30 % (k = 2, mass 0.51-0.81 %); on multi-call prompts only the first call's layers 0 to 19 are eligible (4.8-8.9 % at k = 1); the 16k cold block 5.0 % (k = 1) and 8.7 % (k = 2), mass 0.06 % and 0.15 %. Prompts under 1,024 tokens are untouched. In the model every skipped expert was a miss (rare experts are not resident), so the residency-blind rule costs no extra drops in this trace.

**Time (derived).** At 1.0-1.6 ms of exposed drive time a saved read (18.89's 5.2 s / 3,250 = 1.6 ms is the fully exposed case), k = 2 saves 2.4-3.9 s on a 2.4k follow-up and 10-16 s over session 1's follow-ups; session-wide that is single-digit percent of prefill time, consistent with the honest ceiling in the session skill.

## 3. Design

`CACHALOT_PREFILL_RARE_SKIP=k` (default off), `CACHALOT_PREFILL_RARE_MIN_ROWS` (default 1,024). In `moe_prefill_grouped`, batched path only, after the routing and before the layer's store planning: every expert that at most `k` rows of the call route to is removed from the work list, and each affected row's remaining router weights are multiplied by its original total over its kept total (the decode miss budget's rescale, `moe_layer_metal`). The rule is blind to residency, so a call's result depends only on the call: snapshots taken with it on equal a cold prefill with the same setting and are keyed `-rare{k}m{min}` (`snapshot_store.numerics_version`). The router result the caller sees (and the routing tracer) is unchanged. A startup line names the setting; the `[request]` line gains ` rare_skip=<experts skipped>/<mass dropped %>` (cumulative) once the rule has applied.

## 4. Predictions (registered before any run)

Quality gate (`run-gate.sh`, the DI-2 long gate: three 8,192-token texts in two 4,096 calls, 192 teacher-forced positions each, against the 0.74.0 reference and noise arm `noise-chunk3000` KL 0.0116):

- **R-1.** The same-day `keep140` control reproduces 0.74.0: KL mean 0.0015-0.0025, top-1 98.0-99.2 %. If not, the gate is not comparable and R-2/R-3 are void.
- **R-2.** `rare1`: KL mean 0.0020-0.0040, verdict inside noise, top-1 at least 97.5 %.
- **R-3.** `rare2`: KL mean 0.0025-0.0060, verdict inside noise, top-1 at least 97.0 %.
- **Falsifier (quality).** Either arm's KL mean at or above the noise arm's 0.0116, or a dNLL interval wholly above +0.01 nats: that setting is closed.

Server ABBA (`run-all.sh`, `RARE_K` = the largest k that passed the gate, default 2; four fresh `serve.sh` with shipped defaults; the Hermes block then a 2,300-token single-call prompt, both cold):

- **R-4.** The rare line appears in B arms only; `rare_skip=` counts are non-zero in B arms after the short prompt.
- **R-5.** Short prompt prefill (k = 2): -12 to -30 % against exact (k = 1: -7 to -18 %). The mean of the B arms below both A arms.
- **R-6.** Hermes block prefill (k = 2): -3 to -9 % (k = 1: -2 to -5 %); within the pair spread (1.5-3 %) at the low end, so a null here does not falsify.
- **R-7.** Replies: the 16 greedy tokens equal across arms on the Hermes block (a tool call or the same first words); the short prompt's may differ in wording.
- **Falsifier (speed).** The short prompt saves less than 5 % at the chosen k, or the B arms do not both sit below both A arms: the read-exposure assumption is wrong and the lever closes as priced.

Runtime estimate: gate about 20 minutes (three arms of about 5 minutes and settles), server ABBA about 20 minutes. Both need `iogpu.wired_limit_mb` at 88064 (it read 0 on 2026-11-03) and no other runtime.

## 5. Results

Not run yet.

## 6. Kind

Planning, an instrument and a default-off knob (predictions committed first). No default changed.
