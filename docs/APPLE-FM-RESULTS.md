# Apple Foundation Models as an auxiliary engine: results (AF-1 to AF-19)

Written 2026-10-29 against runtime 0.69.0 (the package as committed with this file). Record: `docs/APPLE-FM-RECORD.md`
(AF-1 to AF-19, committed first in eb92d02 before any benchmark ran). Instrument: `benchmarks/apple_fm_bench.py`; raw
rows and manifests under `benchmarks/results/apple-fm/` (local, not in the repository). Machine: Mac Studio M3 Ultra,
96 GB, macOS 27.2, `apple-fm-sdk` 0.2.1, system model "AFM 3 Core Advanced"; no Cachalot runtime process running,
screensaver off, memory pressure normal, swap 4.4-4.6 GiB used and flat, 5-minute load 3-4 (Hamed's applications open).
All numbers measured unless tagged derived; greedy sampling everywhere, so the model's outputs were identical across
trials and only times vary.

## 1. Scores

| id | result | score |
|---|---|---|
| AF-1 | short greedy generate, inline: median 0.333 s (p25 0.331, p75 0.339, n 7) | held |
| AF-2 | through the worker child: 0.315 s; the child adds -17 ms (it ran second in each trial) | held |
| AF-3 | first streamed delta of a ~300-word reply: median 0.890 s (0.838-0.971) | outside (0.2-0.8), under the 1.2 s falsifier |
| AF-4 | generation 51.7 tok/s (49.9-52.7, counted with `token_count`) | held |
| AF-5 | prompt processing at ~5.7k tokens: 1,019 tok/s (1,002-1,025; includes the context pre-check) | held |
| AF-6 | warm `token_count`: median 30 ms (28-36, n 21); the first, cold, call took 0.77 s in the probe | held |
| AF-7 | route accuracy, registered run: model 0.90, heuristic 0.90 | held |
| AF-8 | false routing (gold target sent elsewhere), registered run: model 3 of 20 (0.15) | outside (at most 0.10), under the 0.20 falsifier |
| AF-9 | extraction field accuracy, registered run: model 0.917 (44 of 48), heuristic 0.375 | held |
| AF-10 | invented values that passed grounding, registered run: 2 of 48 | outside (at most 1), under the falsifier (3) |
| AF-11 | critical-fact retention, model: 11/13, 12/12, 2/11 by document, pooled 25/36 = 0.69; heuristic 9/13, 12/12, 11/11 (0.89 pooled) | model outside (0.70-0.90), above the 0.60 falsifier; heuristic held |
| AF-12 | constraint retention 1.00 for both, every document | held |
| AF-13 | ratio in DeepSeek tokens, model: 0.766, 0.823, 0.284; heuristic 0.542, 1.00, 0.991 | **falsified** for the model (two of three above 0.60) |
| AF-14 | compression wall time: 10.9, 15.3, 11.1 s per 1,000 input tokens (the model's tokens) | outside (3-8); bench_log past the 12 s falsifier |
| AF-15 | net seconds against DeepSeek's 99 tok/s cold prefill (derived): -8.9, -16.0, -4.9 per 1,000 target tokens | **falsified** (below -6 for two documents): compression loses |
| AF-16 | memory while generating: the service's RSS 1.57 to 1.96 GiB (+0.39), the whole machine's wired + active + compressed +1.56 GiB (includes the benchmark process) | inconclusive: `footprint` of the service needs root |
| AF-17 | GPU GEMV 659.4 GB/s alone, 645.8 beside the system model: -2.1 % | held |
| AF-18 | GPU GEMM 18.21 TFLOP/s alone, 18.11 beside: -0.5 % | held |
| AF-19 | the system model 52.9 tok/s alone, 51.6 beside the GEMV (-2.3 %); 0.0 % beside the GEMM | outside (5-20 %), not past the falsifier |

**Tally of 19: 10 held (AF-1, 2, 4, 5, 6, 7, 9, 12, 17, 18); AF-11 split (heuristic held, model outside); 5 outside their intervals and short of their falsifiers (AF-3, 8, 10, 14, 19; AF-14's bench_log document alone passed its 12 s falsifier); 2 falsified (AF-13, AF-15; AF-13's heuristic half was also outside, 1.00 and 0.99 on two documents); 1 inconclusive (AF-16).**

## 2. What the classification and extraction runs found, and what was changed after them

The registered runs are the scores above. Each found defects in my code, fixed afterwards; the re-runs below are
**post hoc, on the same labelled sets, and tuned to them**, so they are not evidence of generalization.

* **A refusal was a generic error.** t9 (translate a contract clause and flag ambiguity) failed with SDK status 255,
  "LanguageModelError:3 - The model refused to answer", reproducible under greedy sampling. The provider now maps that
  text to `AuxRefused` (the SDK's own `RefusalError` was not raised).
* **The fallback routed instead of deferring.** After that failure the heuristic's route (auxiliary) was used. A model
  failure now always routes to the target.
* **The agreement rule was narrower than its docstring.** t17 (a significance question over decode times): the model
  said simple, the heuristic routed it to the target for its task type, and the rule only treated the heuristic's
  *complexity* as disagreement. Any heuristic target route now makes the decision `uncertain` (answered by the target).
* **An arithmetic pattern missed a leading parenthesis** (d5, `compute (12+8)*3`, sent to the model instead of a rule).
* Post hoc: model route accuracy 0.95, false routing 1 of 20 (t20, "My Mac panics when the model loads with a 64 GiB
  budget", called simple chat by both model and heuristic), missed offloads 1; heuristic 0.925, 2, 1. Median latency
  0.818 s a model call (35 calls; 5 arithmetic items take no call).
* **Extraction: the model's self-report was useless and grounding had a numeric bug.** Before the run, three schema
  shapes showed `found: false` on correctly copied values and invented values for absent fields ("password":
  "password"), so the schema is flat and only grounding decides. The registered run then found `1,240.50` failing to
  ground `1240.5` (string comparison of a float; now numeric) and a "not stated" placeholder 0 grounded by the 0 in
  "0.69.0" (placeholders now always read as missing). Post hoc: 0.958 (46 of 48), one invention left: the departure
  city copied into `arrival_city`, a value present in the text, which grounding cannot catch. Median 0.91 s an item.

## 3. Compression: why it loses, and the failure it cannot see

* **Time.** The model writes its facts at ~52 tok/s, 19 ms a token, while DeepSeek prefills a token in ~10 ms cold.
  It copied long passages nearly verbatim as "facts" (bench_log: 22 facts, ratio 0.82; service_spec: constraint
  segments are verbatim by policy and make up most of the document), so it paid output time for little reduction.
  Break-even needs the kept share under ~0.27 (H-cost, derived) and two of three documents were near 0.8.
* **Omission is invisible to the gates.** On the real results document (`docs/CUDA-PIN-IDLE-RESULTS.md`) the model kept
  2 of 11 critical numbers (ratio 0.28) and skipped the score table and the findings section entirely; every fact it
  did return was grounded, so the result passes `max_ratio` and `max_ungrounded` and would be accepted at the
  defaults. Grounding detects invented facts, not missing ones; nothing in the pipeline can, without a reference.
* **Mis-citation drops true facts.** On service_spec the model copied a sentence exactly but cited the wrong segment;
  the grounding check rejected it, losing two of the throughput targets.
* The heuristic baseline kept every critical fact on two documents but compressed nothing there (ratio 0.99-1.00:
  every paragraph of a log or results file contains a number).

**Verdict:** lever B is not a prefill lever for DeepSeek or GLM (90-99 tok/s cold prefill) on this model, and it is not
safe to accept unattended. It stays off. What would reopen it: a target whose cold prefill is far slower than 50 tok/s
per output token saved (for example GLM read-bound from the X10Pro), a compression prompt that returns a much shorter
output, and an omission check (questions answered from the compressed text against the original, offline).

## 4. Contention: the GPU stand-in beside the system model (AF-17 to AF-19)

Cells gpu, fm, both, both, fm, gpu for each GPU load, 20 s windows, separate processes (`docs/APPLE-FM-HETEROGENEOUS.md`
has the table). GEMV (bandwidth-bound, decode-shaped): 659.4 GB/s alone, 645.8 beside the system model (-2.1 %; the
arms separate by 13.6 GB/s against 0.6-1.5 within). GEMM (compute-bound, prefill-shaped): -0.5 %, barely above its
spread. The system model: 52.9 tok/s alone, 51.6 beside the GEMV, unchanged beside the GEMM. The interference follows
memory bandwidth (derived); the Neural Engine is a nearly free side engine for compute-bound GPU work and costs a
bandwidth-bound stream about 2 %. Not yet measured on Cachalot's real decode (`docs/APPLE-FM-HETEROGENEOUS.md` E-1).

## 5. Limits

One Mac and macOS, one SDK version, the agent's own labelled sets (40 routing items, 48 fields, 3 documents), fixes
tuned on them. The GPU stand-in is a kernel loop, not Cachalot's decode, which also streams experts from the SSD and
synchronizes with the host 44 times a token. No energy numbers: `powermetrics` needs root (Hamed's command).
