# L2 in-situ price: the per-token trace inside a real DeepSeek decode token (record opened 2026-10-11, before the build and the run)

Charter references: `docs/RESEARCH-DIRECTION.md` section 4.3 (the per-token critical-path spec), section 7.2 (experiment record), section 13.4 item 1. The paper and simulated prices are HANDOFF 18.114 (0.17 ms a token on a simulated token, 95 % interval -0.02 to +0.38, stop rule 1.0 ms).

Written before any code in `src/` exists for this tracer and before any model run. Hamed's go: "go, option 1, L2 in-situ price" (2026-10-11), after the estimate "about 2 hours for the price".

## Baseline

DeepSeek's server token is 79-80 ms all-resident (DS-FLOOR-SRV) and about 118-126 ms at 23.5 misses a token (DS-TOKEN-48). The server-path A/B that alternates whole turns resolves about 3 % and its control reads +-7 ms a token (HANDOFF 18.59), which cannot see a 0.2 ms effect. `CACHALOT_TOKEN_TRACE` does not exist in `src/`. The only per-layer tracer in the runtime is the routing tracer (a host read of arrays the sync already evaluated).

## What is built (the smallest tracer that answers the question)

`cachalot.metrics.token_trace.TokenTrace`: per-thread preallocated `array('q')` columns (timestamp, code, a, b), no lock and no allocation on the hot path, merged only when drained. Off means the store's `token_trace` attribute is `None` (one `getattr` and one `is not None` test at each site). Events, DeepSeek only:

| code | emitted from | meaning |
|---|---|---|
| TOKEN | start of `_decode_token_impl` | token boundary (a = position) |
| SYNC_START / SYNC_END | around the routing `mx.eval` of `moe_layer_metal` decode | the layer's router sync (a = layer) |
| GET_START / GET_END | around `expert_store.get_many` | main thread blocked on the layer's experts (a = layer, b = entries) |
| READ_START / READ_END | inside `ResidentExpertStore._read_into`, on the worker thread | a drive read (a = layer, b = expert; end: b = bytes) |

Derived per token (drained after a request): `sync_ms`, `blocked_ms` (exposed I/O: the main thread waiting in `get_many`), `read_busy_ms` (union of read intervals over all threads), `hidden_read_ms` (read busy that falls outside the blocked intervals), `reads`, `bytes`, `events`.

Out of scope: GLM, MiniMax, prefill, the demand-submit event (the submit time is the first READ_START of the layer; queueing is READ_START minus GET_START, which this design does read off the data).

## Method for the price

The unit is the decode token, not the turn, because turn-level A/B cannot resolve a sub-millisecond effect. `benchmarks/token_trace_price.py` loads the model once, runs `Engine.chat` turns, and flips the tracer **per decode token** by parity (a token's parity is its position, so tokens alternate on / off; the swapped-parity pass starts on the other phase). Each token's wall time is the interval between consecutive `TOKEN` stamps, which are written in both states (a stamp is not tracing; it is the instrument). Adjacent tokens are paired (on, off). Two workloads: **resident** (the same short prompt repeated, no misses, the case where the main-thread cost shows with no I/O noise) and **mixed** (different prompts, about 20 misses a token, where worker threads emit events while the main thread waits). A control run toggles a **second, empty** attribute (off against off) over the same positions to show the noise floor. Paired differences, block-bootstrap 95 % interval (blocks of 40 pairs, because token times drift). Event counts per token are measured, not assumed. A bit-identical check: the generated token ids of an on turn equal those of an off turn.

Context: `serve.sh` environment (`pareto.serve_defaults()`), 48 GiB expert budget, sysctl 88064, display on, no other runtime process, no sampler. The machine is read-bound only in the mixed arm.

## Prediction (written before the build)

1. Events a token in situ: 340-420 on the resident workload (40 layers x 2 sync + 2 get_many + 2 for a layer's reads; fewer reads), up to 520 on the mixed workload (about 45 reads x 2 more).
2. Overhead per decode token: **+0.15 ms resident, +0.3 ms mixed**, planning interval 0.0 to +0.6 ms (0.2 to 0.8 % of an 80 ms floor). The micro-benchmark's 0.17 ms was a simulated token with a calibrated Python loop; the real token has more GIL contention from MLX's host thread but fewer Python-side events than the simulation's pool model.
3. The control (off against off) will read within +-0.15 ms of zero over 400 pairs (token-to-token noise sd about 1.2 ms on the resident workload, so the standard error of the mean over 400 pairs is about 0.06 ms).
4. Exposed I/O on the mixed workload: `blocked_ms` is 60-95 % of `read_busy_ms` per token (prediction hides only a minority of reads: 38 % of predicted loads are used, DS-PRED-USB); resident workload: `blocked_ms` 0 +-0.1 ms.
5. Decode output unchanged (token ids identical on and off).

## Falsifier and stop rule

- The paired overhead's upper 95 % bound exceeds **1.0 ms a token** (1.25 % of 80 ms) on either workload: the tracer is sampled (every Nth token), not continuous, and the build is shelved behind a sampling knob.
- The control's interval excludes zero by more than 0.15 ms: the alternation itself perturbs the step; the price is void, and the design (toggling inside the token) is the first suspect.
- Token ids differ between on and off: the tracer changed an output; stop and find the hook that did it.
- Mixed `blocked_ms` below 40 % of `read_busy_ms` falsifies prediction 4 (prediction hides most reads); that is a finding about prefetch, not about the tracer, and is recorded as such.

## Result

Run 2026-10-11 13:48-14:07, runtime 0.63.2 plus the uncommitted tracer, `./serve.sh` defaults through `Engine.chat`, exact decode, **38 GiB expert budget** (the memory guard refused 48 and 40 on the day's free memory, 69.4 GiB available; it passed 38), sysctl 88064, display on, no sampler, stale swap 6.0 GB. The machine is not the one the prediction assumed: at 38 GiB neither workload was resident. Raw rows: `benchmarks/results/l2-price/run.json` (git-ignored); the analysis is `benchmarks/token_trace_price.py --report`.

**Workloads as they ran (measured).** "Resident" (one prompt, 160 tokens, repeated): 24.8 misses a token, 127.7 ms a token. It was not resident. §18.131 had already found that a 160-token reply goes read-bound at 48 GiB; I did not apply that. "Mixed" (twelve prompts): 42.4 misses a token, 144 ms. Adjacent-token pairs therefore have a standard deviation of 24-28 ms (planned: 1.2), because a token's time is set by its misses.

**Estimator.** The planned adjacent-token pairing could not resolve the effect: on-minus-off was +2.37 ms (p0), -2.12 (p1) and +1.08 (control) on the resident workload and -2.29, +3.22, -3.24 on the mixed one. The sign follows which parity records, so a **parity effect of about +-2.5 ms** (even-position tokens differ from odd-position ones, cause not found; it is common to both states) swamps the tracer. Averaging p0 and p1 cancels it (+0.12 and +0.47 ms) but with intervals of about +-1 ms. What does resolve the effect is matching tokens **by position across passes**: the same prompt gives the same tokens and routes at the same positions in every pass, p0 records even positions and p1 odd ones, so `T = (mean_even(w_p0 - w_p1) - mean_odd(w_p0 - w_p1)) / 2` cancels content, parity and pass-level drift. This estimator was chosen after seeing that the planned one failed; it is labelled post hoc and its code is in the report. The pass-to-pass difference it cancels is large: two unrecorded passes (p0 odd positions against the control pass) differ by -6.0 ms (-8.0 to -4.0) resident and -8.1 ms mixed, about 5 % of a token, which is why turn-level A/B cannot see this effect.

| quantity | resident workload (24.8 misses) | mixed workload (42.4 misses) |
|---|---|---|
| turns, positions a turn | 6, 132 | 11, 130 |
| **tracer cost a decode token, position-matched** | **+0.095 ms (95 % turn bootstrap -0.276 to +0.417)** | **+0.449 ms (-0.239 to +1.214)** |
| pooled, both workloads (post hoc) | +0.324 ms (-0.170 to +0.838), 17 turns | |
| events recorded a token | 256 | 280 |
| reads a token (union of intervals) | 47.7 | 59.5 |
| router sync a token (the `mx.eval`, includes queued GPU work) | 59.2 ms | 63.0 ms |
| main thread blocked in `get_many` (exposed I/O) | 41.0 ms | 51.6 ms |
| read busy (union of read intervals, all threads) | 72.4 ms | 89.8 ms |
| hidden read (read busy outside the blocked intervals) | 32.6 ms | 39.7 ms |
| blocked / read busy | 0.57 | 0.57 |
| generated text equal to the tracer-free text of the same prompt | 3 of 3 | 12 of 12 |
| events dropped (buffer full) | 0 | 0 |

**Against the predictions.**
1. Events a token (340-420, up to 520): **missed low**, 256 and 280. The build records two events per read and no submit event; the prediction counted three. Reads came out at 48-60 a token, not 45.
2. Overhead (+0.15 ms resident, +0.3 mixed, interval 0.0 to +0.6): the point estimates **held** (+0.095 and +0.449). The mixed interval reaches +1.21, wider than planned, because 11 turns with a turn-to-turn range of -1.6 to +3.2 ms is a small sample. The resident interval (upper 0.42) excludes the 1.0 ms stop rule. The pooled interval (upper 0.84) also stays under it, post hoc.
3. Control within +-0.15 ms: **not applicable** as written. The control pass does not carry a tracer difference to measure; the quantity that mattered was the pass-to-pass difference (-6 to -8 ms), which the matched estimator cancels and the planned one did not.
4. Blocked 60-95 % of read busy: **missed narrowly**, 57 % in both workloads (43 % of read-busy time is hidden). Resident workload: blocked 0 +-0.1 ms: **falsified by the workload's own misses** (41 ms).
5. Decode output unchanged: **held** (15 of 15 turns identical).

**Verdict on the stop rule, plainly.** Resident workload: passes (upper 0.42 < 1.0). Mixed workload: the interval **cannot exclude 1.0 ms** (upper 1.21), and the rule as I wrote it says that means sampling. The point estimate (0.45 ms, 0.3 % of a 144 ms token) is far under the rule and the pooled interval clears it, but the pre-registered condition is not met on the mixed arm, so this record does **not** certify the continuous tracer as proven under 1 ms; it says the most likely cost is 0.1-0.5 ms and that a 30-turn mixed arm would settle it (about 12 minutes of decode). Nothing is enabled by default and no CLI knob exists yet, so nothing is exposed that the price would need to guard.

**What the instrument shows (derived from one run, 38 GiB, no sampler).** At 38 GiB a DeepSeek decode token of about 127-144 ms spends roughly 59-63 ms inside the router `mx.eval` (a layer's queued GPU work plus the sync), 41-52 ms with the main thread blocked on expert reads, and the remaining 27-30 ms elsewhere (attention graph building, Python between syncs). The drive is busy 72-90 ms of that on 48-60 reads, of which 43 % is hidden behind other work. That is the first measured exposed-against-hidden split for DeepSeek; it is the input L4(b) needed.

## Limits (known now)

One machine, one build, one display state; the drift check is the control and the swapped parity. Exposed-I/O definitions are the tracer's (main-thread wait in `get_many`), not a proof of the critical path: a wait inside `get_many` can overlap with GPU work still queued by `async_eval` of the previous layer. The tracer does not see GPU time. This is a price of the instrument and a first reading of exposed against hidden read time, not the L4(b) overlap fit.

## Kind

Research (predicted, measured, explained in part): the point estimates of the overhead held, the event count and the workload precondition were missed, the planned estimator failed and was replaced by a position-matched one (post hoc), and the pre-registered stop rule is not met on the mixed arm by interval width. Also an instrument (`TokenTrace`, hooks in the DeepSeek decode path, off unless installed).
