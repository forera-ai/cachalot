# CUDA lane: a routing trace with predictions at several leads (plan and experiment record; nothing run)

Written 2026-10-20 against runtime 0.66.36, **before any recording and before any lead-aware analysis code**. Hamed asked for "a tracer knob (off by default) that records router scores two layers ahead, predictions first; do not run the server until I say." Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in a separate file. A prediction edited after its run is void. Labels follow plan section 0. **The server is not started by this work.**

## 1. Finding: the knob already exists, so no `src/` change is planned

`docs/CUDA-PREDICTOR-PRICE-RESULTS.md` ended on "the next input is a trace with the router's predictions two layers ahead". Reading the runtime (`src/cachalot/model/moe_layer_metal.py`, `src/cachalot/metrics/routing_trace.py`), that input needs no new code:

* The decode path already computes, per layer L, the router of layers `L + CACHALOT_PREDICT_LEAD` up to `L + LEAD + AHEAD - 1` applied to layer L's router input (`route_topk_fused`, top-`CACHALOT_PREDICT_TOPK`, default 6), in the same sync as the layer's own routing.
* With `CACHALOT_ROUTING_TRACE` set, every such prediction is recorded with its source layer, target layer, token position, experts **and router weights** (`record_predicted`, host values of arrays the sync already evaluated: no extra GPU round trip).
* `CACHALOT_PREDICT_SUBMIT=0` computes and records the predictions but does not hand them to the store ("diagnostic only"), so no prefetch alters residency; routing and the real experts are unaffected by it.

So one recording with `CACHALOT_PREDICT_AHEAD=3` (leads 1, 2 and 3 in one run, `PREDICT_LEAD` left at 1) and `CACHALOT_PREDICT_SUBMIT=0` yields three predictions of every target layer from three different source layers for the **same** tokens, which is the controlled comparison the question needs. The existing trace format already carries `pred_source` and `pred_target`; the lead is `pred_target - pred_source`. The offline code (`prefetch_coverage.py`, `overlap_arena.py`, `predictor_price.py`) keys predictions by `(position, target)` and would silently overwrite leads with each other; it needs a lead selector (offline Mac work, section 4).

Known limits of such a run, stated before it: it is a diagnostic configuration (no prefetch, so decode is read-bound, about 10 % slower per token than the shipped path, plus two extra router passes per layer); the output text differs from earlier traces (temperature 0.7, non-streaming replay), so absolute numbers are read against `R1` on the **new** trace, never against the 0.465 and 0.365 of the old one; the earlier offline recall table (`benchmarks/predictor_recall.py`, HANDOFF section 8) already says lead 2 costs 6.5 points of recall (65.0 % against 71.5 % at top-6), measured on different data.

## 2. The question and what it cannot claim

Does a prediction made one or two layers earlier keep enough coverage to be worth the extra window it buys? Stage B showed the window matters at short layer times (`P40` 0.17 below `P0L` at 0.3 ms) and that wasted copies do not; stage C and the predictor price showed nothing built from expert ids beats the recorded predictor. A lead-2 prediction is available one full layer earlier; the offline bound says what it costs in coverage, and a later GPU stage would say what the window buys. It cannot claim any time, any GPU behavior, other traces or other models.

## 3. Recording specification (for Hamed; the agent does not run it)

Same conversation as `routing-trace-deepseek-long` (§18.99): `benchmarks/results/routing-trace-deepseek-long/source-requests.jsonl` and `replay_trace.py` (cap 600, temperature 0.7). Estimate before the run (Hamed's rule: estimate and confirm every lengthy test): about **10 to 15 minutes** (the 2026-10-08 recording took 7 minutes with prefetch on; no prefetch adds about 10 % per decode token, the two extra router passes a little more); worst case about 25 minutes if a 22k-token system block is not snapshot-warm. Machine state as then: sysctl 88064, display on, nothing else running, swap steady. Trace size about 3 times the old predicted-set part: expected under 60 MB. Exact commands are in `docs/CUDA-LEAD-TRACE-RUNBOOK.md` (written with this record). Output goes to `benchmarks/results/routing-trace-deepseek-leads/` (gitignored, like the old one).

## 4. Offline work done before the recording (Mac, no server)

`predictor_price.py` and `prefetch_coverage.py` gain a lead selector (`--lead N`, default 1, selecting records with `pred_target - pred_source == N`), `predictor_price.py` the predictors `R2`, `R3` (the recorded sets at leads 2 and 3) and `R12` (lead-1 set followed by the lead-2 experts not in it, up to twelve), each with tests that tripping a lead mix-up fails (a synthetic trace with distinguishable leads). Existing numbers must not move: the lead-1 default reproduces the old outputs exactly.

## 5. Baseline (recorded, old trace)

Held-out window (decode tokens 1000 to 1399), 17 % resident: `R` bound 0.465, recall of the real set 0.667; first 400 tokens 0.365. The new trace has different replies and so a different window; its own `R1` is the baseline.

## 6. Predictions (registered before any recording)

All **predicted**, on the new trace, window = decode tokens 1000 to 1399 if the trace has at least 1,400 decode tokens (otherwise the last 400 after a training prefix of half the tokens), 17 % resident, serial bound, tables none (only recorded predictions are used). Probabilities are my own.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| LT-1 | `R1` on the new trace | bound between 0.30 and 0.50 (the two windows seen read 0.365 and 0.465) (probability 0.75) | outside 0.25 to 0.55 |
| LT-2 | lead 2 against lead 1 | `R2` bound below `R1`'s by 0.02 to 0.12 (the offline recall table loses 6.5 recall points) (probability 0.6) | `R2` at or above `R1`, or more than 0.20 below |
| LT-3 | lead 3 against lead 2 | `R3` bound below `R2`'s by 0.02 to 0.10 (probability 0.6) | `R3` at or above `R2` |
| LT-4 | recall of the real set (any residency, layers with a lead-n source) | lead 1 between 0.60 and 0.72, lead 2 between 0.54 and 0.66, lead 3 between 0.50 and 0.62, each below the previous by 0.02 to 0.10 (probability 0.5 for all three intervals) | lead 2 recall at or above lead 1's |
| LT-5 | union | `R12` bound above `R1`'s by 0.00 to 0.06, with loads a token at least 1.6 times `R1`'s (probability 0.5) | above `R1` by 0.08 or more (the union would be a real lever), or below `R1` |
| LT-6 | the diagnostic run does not disturb routing | the new trace's real experts (`experts`) have the same shape rules as the old (2,000 +- 300 decode tokens, six experts a row); predicted rows per token equal (layers minus lead) a lead (probability 0.9) | any mismatch |
| LT-7 | recording cost | the run completes inside 25 minutes, the trace is under 60 MB, and the new decode tokens are at least 1,400 (probability 0.8) | beyond 35 minutes, or fewer than 1,400 decode tokens |

## 7. Decision rules (fixed here)

* **A GPU stage for lead 2 is justified** if `R2`'s 17 % bound is at least `R1`'s minus **0.05** (a lead-2 predictor keeping at least 95 % of the bound buys a full layer of window, which stage B showed matters at 0.3 ms). If lower, the extra window does not pay at this coverage and the lead lever is closed offline.
* **The union `R12` justifies a GPU stage** only if its bound is at least `R1`'s plus **0.05**.
* **Any GPU stage** needs Hamed's go, a harness change (predictions keyed by lead and a lead-2 arm), a CPU rehearsal and predictions first.

## 8. Needs Hamed

A go and a quiet machine for one recording (the agent starts nothing). The commands are in the runbook. Nothing else.
