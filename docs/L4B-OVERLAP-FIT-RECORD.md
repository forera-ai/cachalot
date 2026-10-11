# L4(b): the overlap term fitted from the L2 traces (record opened 2026-10-11, before the fit script exists and before any number is computed)

Charter references: `docs/RESEARCH-DIRECTION.md` section 4.6 L4(b) ("the overlap term `exposed_io = max(0, io_time - useful_overlap)`, fitted from L2 traces and tested on held-out traces"), section 7.2 (experiment record), section 13.4 item 2. Hamed's go: "/ca read the handoff and session prompts and do the tasks" (2026-10-11), First job A(3) of the v183 prompt. Offline only: no model is loaded, no server runs, nothing in `src/` changes.

## Baseline

The only token-time model that uses read counts is the straight line of `benchmarks/whatif.py` (DeepSeek: floor plus misses times a per-miss cost, and below about 2.6 GB/s bytes over rate). It has no overlap term, so it cannot say how much of a token's read time hides behind compute. The L2 in-situ price (HANDOFF 18.132-18.133) recorded per-token rows with `wall_ms`, `sync_ms`, `blocked_ms` (main thread waiting for experts), `read_busy_ms` (union of read intervals), `hidden_read_ms` (read busy outside the blocked intervals), `reads` and `bytes`. Those rows are kept in `benchmarks/results/l2-price/run.json` (the 0.63.3 run, 38 GiB) and `run30.json` (the 0.63.4 30-turn run, 34 GiB). They are per-token summaries; the raw per-layer rows were not saved, so a layer-level overlap fit is not possible from them.

Descriptive facts already seen while opening this record (measured, over 5,939 token rows pooled, before any model is fitted): mean wall 148.2 ms, sync 62.2, blocked 57.1, read busy 96.5, hidden 40.9, reads 62.1 a token. `blocked_ms` tracks `read_busy_ms - hidden_read_ms` (correlation 0.998), which is close to a definition of the instrument, not a finding. A straight line of hidden on read busy reads 11.7 + 0.30 x busy; wall on reads reads 65.6 + 1.33 x reads with a residual sd of 22 ms.

## The model

Per decode token, with `x` the read busy time:

- overlap form A, constant: `hidden = min(x, H)`
- overlap form B, proportional: `hidden = f x`
- overlap form C, affine: `hidden = clip(h0 + h1 x, 0, x)`

and `exposed = x - hidden`, `wall = w0 + exposed` where `w0` is the part of the token that is not waiting on experts (router syncs and everything else). `x` itself is modelled from the read count (`x = k reads`, through the origin). The fit is least squares on the training run; the test is the other run, in both directions, plus a split by turn inside each run. A held-out score is the root mean square error in ms of `blocked_ms` and of `wall_ms`, and the bias.

## Prediction (written before the fit)

1. The affine form C has the lowest held-out `blocked_ms` error of the three, and the constant form A the highest: the hidden share rises with read busy time but not in proportion (intercept 8-15 ms, slope 0.25-0.35).
2. Held-out `blocked_ms` root mean square error of form C is 8-14 ms (the straight-line fit of hidden on busy leaves 8.1 ms inside the sample).
3. The wall-time model (reads to busy to exposed to wall) is **not much better** than the existing straight line of wall on reads: held-out wall error within 10 % of it either way. Reason: 60 % of the wall's variance sits in sync and the unexplained 29 ms (sd 10), which no overlap term touches.
4. The two runs differ by a bias in `w0` of 5-15 ms because their budgets differ (38 against 34 GiB) and the machine drifted; the overlap parameters (h1 in particular) agree between the runs to within 0.1.

## Falsifier and stop rule

- If the held-out wall error of the overlap model is **at least 15 % lower** than the straight line's, the overlap term earns its place and goes into `whatif.py`'s DeepSeek model as a validated term, with the interval.
- If it is within +-10 %, as predicted, the overlap term is recorded as a validated description of the exposed read time (what hides) but **not a better token predictor**, and `whatif.py` is not changed.
- If form A or B beats C held out, prediction 1 is falsified and the shape of the overlap is written down as such.
- If the two runs' h1 disagree by more than 0.15, the overlap is not a stable property of the machine across these two sessions and no constant goes into the what-if.

## Limits (written now)

Two runs, one model, one machine, two budgets, the mixed workload (different short prompts, about 20-50 reads a token); the resident arm has too few reads to inform an overlap fit. Per-token rows are not independent (adjacent tokens share cache state and drift), so the interval is a block bootstrap by turn, not a per-token standard error. `hidden_read_ms` and `blocked_ms` are defined by the same instrument from the same events, so the exposed share is a property of the tracer's definition and of the decode schedule together. Nothing here is a statement about GLM or MiniMax. A layer-level overlap fit (which layers hide reads and which expose them) needs a new recording with raw rows (`TokenTrace.drain()`), and that needs the machine and Hamed's go.

## Result (measured offline on the saved L2 rows; `benchmarks/overlap_fit.py`, 2026-10-11)

Rows: `run.json` and `run30.json` per-token summaries (counts in HANDOFF 18.135). Four held-out directions: train on one run and test on the other (both ways), and even turns against odd turns inside each run.

| held-out direction | C blocked RMSE (ms) | A / B blocked RMSE | wall RMSE C / line (ms) | C against the line |
|---|---|---|---|---|
| run to run30 | 9.4 | 15.6 / 12.1 | 27.3 / 26.0 | +5.3 % (+3.7 to +7.9) |
| run30 to run | 7.0 | 14.7 / 8.4 | 15.1 / 13.9 | +8.5 % (+5.9 to +11.6) |
| run, even to odd turns | 7.4 | 13.2 / 9.3 | 15.1 / 13.7 | +10.2 % (+6.7 to +14.3) |
| run30, even to odd turns | 9.3 | 16.0 / 11.4 | 23.2 / 22.6 | +2.4 % (+0.5 to +4.6) |

Fitted affine overlap: hidden = 9.6 to 13.7 ms + 0.28 to 0.33 x read busy (h1 0.330 on `run.json`, 0.293 on `run30.json`); read busy = 1.53 to 1.58 ms a read; w0 91 to 95 ms.

## Against the prediction

1. **Held.** C has the lowest held-out blocked error in all four directions, A the highest in all four; the proportional form B sits between.
2. **Mostly held.** C's blocked error is 7.0 to 9.4 ms; the predicted 8 to 14 was missed at the low end in two directions (7.0 and 7.4).
3. **Held narrowly, and in an unexpected direction.** The wall model is not better than the line; it is 2.4 to 10.2 % **worse** (every bootstrap interval is above zero). The 10.2 % is at the edge of the predicted +-10 %. The falsifier for adoption (15 % better) is not met.
4. **h1 held, the w0 bias missed.** h1 differs by 0.037 between the runs (inside 0.1). w0 differs by 2.1 ms between the runs, not the predicted 5 to 15.

## Explanation (hypothesis)

The overlap is real and stable (about 30 percent of extra read time hides behind compute, plus a fixed 10 to 14 ms that hides regardless) and the affine form describes the exposed time well, but the wall adds two steps that the line does not have, a reads-to-busy step and a w0 held constant, and each adds error. In this data the straight line of wall on reads already holds the same information, because read busy time is almost a straight line in reads (1.5 ms a read). Hidden time is an effect of the schedule, not an independent lever: a model of it describes where the time goes without predicting the token any better. This is a hypothesis; the per-layer rows that would separate it were not saved.

## Decision

`benchmarks/whatif.py` is **not** changed: the overlap term does not beat the line as a token predictor. The overlap parameters are recorded as a description of the sampled 1-in-4 and continuous traces on this machine (hidden = about 10 to 14 ms + 0.28 to 0.33 x busy; DS-OVERLAP in `docs/LEDGER.md`). L4(b) is closed at the token level. What would reopen it: a recording with raw per-layer rows to see which layers hide reads; that needs the machine and Hamed's go.
