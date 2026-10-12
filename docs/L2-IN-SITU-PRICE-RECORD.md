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

## Addendum 2026-10-11 (second session hour): the certifying arm, prediction written before the run

**Baseline.** The first run's mixed arm gave +0.449 ms (95 % turn bootstrap -0.239 to +1.214) from 11 turns; turn-to-turn range -1.6 to +3.2 ms.

**Question.** Is the matched cost on a miss-bearing workload under the 1.0 ms stop rule, with the upper bound of its interval under 1.0?

**Method.** Same script and estimator (`token_trace_price.py`, position-matched), mixed workload only: 30 turns in each of p0 and p1 (twelve prompts, the first eighteen turns repeat prompts 0-5 on a warmer cache), a 10-turn control pass, the 4-turn tracer-free text check, 160-token greedy replies. Budget: the largest the memory guard allows (the first run passed at 38 GiB and was refused at 40). The previous estimate of "about 12 minutes" omitted the second parity pass and the control; the arm is about 27 minutes plus the model load.

**Prediction.** Matched cost +0.35 ms (planning interval 0.0 to +0.8); 95 % turn-bootstrap upper bound between +0.6 and +0.95 (the turn-to-turn sd of about 1.3 ms divided by the square root of 30 gives a half-width of about 0.5 ms). Generated text identical in all traced turns that have a tracer-free twin.

**Falsifier / stop rule (unchanged).** Upper bound above 1.0 ms: the tracer is sampled, not continuous. Any text difference: stop and find the hook. Null contrast (p0 odd positions against the control pass) outside +-12 ms: the machine drifted too much and the arm is void.

### Addendum result (run 2026-10-11 18:43-19:07, 34 GiB; the memory guard refused 38 that evening with 64.3 GiB available)

Mixed workload, 30 turns planned in each of p0 and p1 (27 usable: three replies ended before 20 positions matched), a 10-turn control pass, 4 tracer-free text turns, 160-token greedy replies. Misses a token 44.2, token 153.7-155.1 ms, 294 events and 66.3 reads a token.

| quantity | value |
|---|---|
| **matched tracer cost a decode token** | **+0.554 ms (95 % turn bootstrap +0.051 to +1.114)**, median +0.233, 27 turns |
| pass-to-pass difference cancelled (p0 odd positions against the control pass) | -2.36 ms (-5.31 to -0.09); inside the +-12 ms limit |
| text equal to the tracer-free text of the same prompt | 12 of 12 |
| router sync / blocked in `get_many` / read busy / hidden read | 62.4 / 62.8 / 104.4 / 43.1 ms |
| blocked / read busy | 0.60 |
| events dropped | 0 |

**Against the prediction.** Point estimate +0.55 against +0.35 (inside the planning interval 0.0 to +0.8): held. The upper bound **+1.11 against the predicted 0.6-0.95: missed**, because the turn-to-turn spread was larger than the 1.3 ms I assumed. Text: held. Null contrast: held.

**Verdict on the falsifier, plainly.** The upper bound (+1.11) is above the 1.0 ms stop rule, and the lower bound (+0.05) excludes zero. The rule as written says the tracer is sampled, not continuous. The cost is now known to be real and about half a millisecond (0.36 % of a 154 ms token, 0.7 % of the 80 ms floor), which is 1.9 microseconds an event at 294 events, 4 times the 0.5 microseconds the simulated token showed under contention. The pre-registered rule is not met; I am not reinterpreting it. What follows from it: when the `CACHALOT_TOKEN_TRACE` knob is built, the default should be sampled (every 4th decode token would cost about 0.14 ms a token averaged), with continuous recording as an explicit opt-in whose cost is stated as +0.55 ms (up to 1.1). That default is Hamed's call.

Exposed against hidden at 34 GiB: 60 % of read-busy time is exposed (first run 57 % at 38 GiB), the same within the run-to-run spread.

# Addendum 2: the in-situ price of the 1-in-4 default (opened 2026-10-12, before the script ran against the model)

Hamed's go: "go, record the in-situ price of the 1-in-4 default" (2026-10-12). Estimate: model load about 3 minutes, 4 warm turns, 30 prompt pairs of two 128-token runs (about 40 s a pair, 20 minutes) and 10 control pairs (7 minutes): about 30-35 minutes, 34 GiB, exact decode, under `guarded_run.sh`.

## Baseline
Continuous recording (`every=1`) costs +0.554 ms a decode token (95 % +0.051 to +1.114, 27 turns, 34 GiB; 0.63.4, 18.133), 1.9 microseconds an event over 294 events. The shipped default records one token in four, so the derived average is about 0.14 ms a token (0.4 % -> 0.1 % of ~150 ms). It was never measured: `benchmarks/token_trace_price.py` alternates tokens by parity and cannot represent a 1-in-4 phase.

## Method
`benchmarks/token_trace_price_sampled.py`, `Engine.chat`, `serve_defaults()` environment, 34 GiB, greedy, 128 new tokens, the 12 mixed prompts cycled. Each of 30 prompts is run twice back to back, once with `TokenTrace(every=4)` and once with a stamp-only tracer, the order alternating by prompt (so a prompt's second run, with a warmer cache, falls on each side equally). Position-matched difference in differences: T = mean over sampled positions (k % 4 == 0) of (wall_e4 - wall_off) minus the same mean over the other positions; the cost of a sampled token is T and the average over all tokens is T / 4. Interval: bootstrap over prompts. Control: 10 pairs of a second stamp-only tracer against the stamp-only one, same labels, for the noise floor of T. Text identity of the pair and dropped events are checked. The stamp-only tracer's own cost (a `perf_counter_ns` and two array appends a token) is in both arms and not priced.

## Prediction (written before the run)
1. The cost of a sampled token T is +0.55 ms (the continuous per-recorded-token cost), with a planning interval 0.0 to +1.1.
2. The average over all tokens, T / 4, is +0.14 ms, 95 % interval within -0.05 to +0.30.
3. The control pairs give T within +-0.4 ms of zero.
4. The e4 and off texts are identical in 30 of 30 pairs; 0 events dropped; about 300 events a sampled token (about 75 a token averaged).
Registered rule: the sampled default is certified if the upper end of the 95 % interval of T / 4 is below 0.5 ms (about 0.3 % of a token).

## Falsifier
Point estimate of T / 4 above 0.30 ms (twice the derived cost), or an upper bound at or above 0.5 ms: the sampled default is not certified and the cost is not what the continuous figure scaled by a quarter predicts (a per-sampled-token cost larger than the per-recorded-token cost, for instance from a cold buffer or a drain).

## Limits
One machine, 34 GiB, one workload; the estimator resolves a few tenths of a millisecond a sampled token, so a true 0.14 ms average (0.55 a sampled token) sits near the interval's width; the server's `TokenTraceMeter` bookkeeping between requests is outside the decode window and not priced here.

## Addendum 2 result (measured, 2026-10-12, runtime 0.64.8): the design could not certify the default

`benchmarks/token_trace_price_sampled.py --budget-gib 34`, 30 prompt pairs (27 usable, the others had fewer than 20 shared positions or too few sampled positions) and 10 control pairs (9 usable), 39.2 misses a token, tokens about 145 ms (off runs), 103.7 tokens a run on average, pressure 1, swap flat. Text identical in 30 of 30 pairs, 0 events dropped. Results are in `benchmarks/results/l2-price/run-e4.json` (local).

| estimate | mean | 95 % bootstrap over pairs | median |
|---|---|---|---|
| T, the cost of a sampled token (27 pairs) | +1.097 ms | -0.261 to +2.580 | +0.682 |
| T / 4, the average over all tokens | +0.274 ms | -0.065 to +0.645 | |
| control T, a second stamp-only tracer against off (9 pairs) | +1.843 ms | -1.471 to +5.282 | |
| control per token (T / 4) | +0.461 ms | | |

## Against the prediction

1. T = +1.10 ms against the predicted +0.55 (planning interval 0.0 to +1.1): at the upper edge of the planning interval, with a confidence interval that contains the prediction.
2. T / 4 = +0.274 ms against +0.14, interval predicted within -0.05 to +0.30: the point is inside, **the upper bound (+0.645) is not**; missed.
3. Control T within +-0.4 ms of zero: **missed**, +1.84 ms (-1.47 to +5.28). The noise floor of this estimator at 27 pairs is larger than the effect it is meant to resolve; the control, which cannot contain any tracing cost, reads +0.46 ms a token on average.
4. Texts identical 30 of 30 and 0 dropped: held. Events a sampled token were not summarized in this run (the stamps and positions were kept, the events were not): unverified.

**The registered rule is not met.** The default is certified if the upper end of T / 4 is below 0.5 ms; it is +0.645. The falsifier's upper-bound clause fired. The point estimate (+0.274) is below the falsifier's 0.30 point threshold.

## Explanation (hypothesis)

The estimate is too noisy for the question. The per-pair standard deviation of T is several milliseconds (a quarter of the positions are treated, about 26 positions a run, against a token-to-token sd of several ms), and the control shows the estimator's bias or noise at the same scale as the effect. Neither "0.14 ms" nor "0.65 ms" can be excluded. What is established is a bound: with 95 % confidence the average cost of the 1-in-4 default is below about 0.65 ms a token (0.45 % of a 145 ms token); the point estimate +0.27 ms is consistent with the derived 0.14 and with the continuous per-recorded-token cost (+0.55 x 1/4).

To certify under the registered rule the half-width of T would need to fall from about 1.4 to about 0.6 ms: roughly five times the pairs (about 140 pairs, about 90-100 minutes of machine time), or longer runs of the same prompts, or an ABBA order at the token level. Not done.

## Limits

The control's non-zero mean (+1.84) means the estimator may carry a bias at this size; the order alternation was meant to cancel it; the control pairs were also alternated and show it did not shrink below the noise. One machine, one budget, one workload. Interval widths: the bootstrap is over pairs (27 and 9).

## Decision

The sampled default stays (off unless `CACHALOT_TOKEN_TRACE=1`, one token in four). Its cost is bounded, not certified: below about 0.65 ms a token at 95 % confidence, point +0.27, derived +0.14. No change to the knob or its default. A tighter price needs a design with much more power; the cost is small against the question it answers (a trace of 40 layers at +0.1 % of a token) and the knob is off by default, so more machine time is not justified unless a use appears that needs the number.
