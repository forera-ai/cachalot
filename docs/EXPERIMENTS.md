# Experiment index (charter section 7.2 and section 10)

Written 2026-10-09 against runtime 0.62.39; regenerate with `benchmarks/experiment_index.py`. The charter asks for "a file of experiment records whose predictions were written before the build". The records themselves live in `docs/HANDOFF.md` section 18 (each with Baseline, Prediction, Method, Result, Explanation, Limits, Kind); this file is the index: where each record is, whether it carries a written prediction, and what kind of result it is, as its own **Kind** line says. It does not restate results. Sections before 18.75 predate the record format and are not listed; sections without a **Kind** line are listed as unlabelled and have no claim made for them here.

Counts of the 52 sections listed (18.75-18.126): 15 research, 13 measurement, 13 unlabelled, 9 engineering, 1 planning, 1 pricing.

Research means predicted, measured and explained (charter section 8.1 rule 6); a falsified prediction is still research and is marked so in its Kind line. Engineering means it works without a written prediction, or an instrument was built.

"Prediction written" means the section has a **Prediction** paragraph; whether it was written before the run is what that section says (the 0.62.23 and 0.62.25 ones were, in the earlier section they cite). Only the Kind column is the record's own verdict.

| section | title | prediction written | kind (as recorded) |
|---|---|:---:|---|
| §18.126 | Power accounting, step E3d: does the unexplained decode power stay when the drive reads are switched off? (0.62.39) | yes | Research (predicted, measured, explained in part). |
| §18.125 | Power accounting, step E2i: what the internal SSD costs when it reads (0.62.38) | yes | Research (predicted, measured, explained in part; the hypothesis not supported). |
| §18.124 | Power accounting, step E3c: does a synthetic read pattern reproduce the decode excess? (0.62.37) | yes | Research (predicted, measured, explained in part; the hypothesis falsified). |
| §18.123 | Power accounting, step E3b: where the decode excess sits (0.62.36) | yes | Research (predicted, measured, partly explained; P3 missed, the P1 metric was flawed, H-channels not supported). |
| §18.122 | Power accounting, step E3: joules a token with every component (0.62.35) | yes | Research (predicted, measured, partly explained; most predictions missed, one fired a falsifier). |
| §18.121 | Power accounting, step E2: what the X10Pro costs when it reads (0.62.34) | yes | Research (predicted, measured, partly explained; every prediction held narrowly). |
| §18.120 | Power accounting, step E1b: what the system-power residual is (0.62.33) | yes | Research (structure predicted and held; every coefficient hypothesis missed). |
| §18.119 | Power accounting, step E1: what a byte of DRAM traffic costs (0.62.32) | yes | Research (predicted, measured, explained in part; the rate prediction missed by 2x). |
| §18.118 | Power accounting, step E0: what this Mac exposes (0.62.31) | yes | Measurement plus instruments. Not research: no prediction was written first. |
| §18.117 | Power accounting for SSD reads and DRAM transport: the plan, and the next session's jobs (0.62.30) | no | Planning (no run). |
| §18.116 | Energy (charter L5): joules a token, all-resident against read-bound (0.62.29) | no | Measurement (first energy numbers) plus instruments. Not research by the charter's definition: no prediction was written first. |
| §18.115 | Energy (charter L5): the parser and the first idle reading (0.62.28) | no | Measurement (first reading) plus an instrument. |
| §18.114 | L2 per-token trace: the overhead priced on a simulated token (0.62.27) | yes | Research (predicted, measured, explained; the unit-cost prediction was missed by 3-10x in the safe direction). |
| §18.113 | Offline charter work: a validated what-if calculator, the ledger, an experiment index, and the plan (0.62.26) | yes | Engineering (instrument and model; validation done in a scratch check before the module existed, so no prediction on file). |
| §18.112 | GLM's same-site break: a teacher-forced probe, prefill against decode against budget 2 (0.62.25) | yes | Research (predicted, measured, prediction falsified, explained). |
| §18.111 | GLM decode miss budget: the compile-level C# panel (0.62.24) | yes | Research (predicted, measured), with a null result on the budget question and a new lead on the corruption. |
| §18.110 | GLM decode miss budget: the larger blind C# panel (0.62.23) | yes | Research (predicted in §18.108, measured, explained). |
| §18.109 | GLM decode miss budget: free-running replay, arm 3 (replicate) (0.62.21) | yes | Engineering (a replicate with no prediction on file). |
| §18.108 | GLM decode miss budget: free-running replay, arm 2 (0.62.19) | yes | Research (predicted, measured). |
| §18.107 | GLM decode miss budget: teacher-forced quality, arm 1 (0.62.15) | yes | Research (predicted, measured, explained as hypothesis). |
| §18.106 | GLM prefetch default off, and a GLM decode miss budget (0.62.14) | no | Engineering (a default shipped on measured pairs; a knob built, unmeasured). |
| §18.105 | GLM decode prefetch off against K = 5, live (0.62.13) | yes | Research (predicted, measured, explained); a default is not changed. |
| §18.104 | A simulator store with decode prefetch (0.62.12) | yes | Engineering (instrument), one bug fixed. No default, output or numerics changed. |
| §18.103 | The GLM trace with weights and predicted sets (0.62.11) | yes | Measurement. No default, output or numerics changed. |
| §18.102 | The GLM side of weights and predicted sets, built (0.62.10) | no | Engineering (instrument). No default, output or numerics changed; off unless a trace is requested. |
| §18.101 | Pricing router weights and predicted sets for GLM and MiniMax (0.62.9) | no | Pricing (engineering). No default, output or numerics changed; one instrument added. |
| §18.100 | A longer GLM trace (0.62.8) | yes | Measurement. No default, output or numerics changed. |
| §18.99 | A longer DeepSeek trace on the MiniMax conversation (0.62.7) | yes | Measurement. No default, output or numerics changed. |
| §18.98 | A longer MiniMax trace (0.62.6) | yes | Measurement. No default, output or numerics changed. |
| §18.97 | GLM on an agent conversation (0.62.5) | yes | Measurement. No default, output or numerics changed. |
| §18.96 | MiniMax on the same short cold prompts (0.62.4) | yes | Measurement. No default, output or numerics changed. |
| §18.95 | DeepSeek on the same short cold prompts (0.62.3) | yes | Measurement. No default, output or numerics changed. |
| §18.94 | The first GLM routing trace (0.62.2) | yes | Measurement. No default, output or numerics changed. Bottleneck: unchanged. |
| §18.93 | The first MiniMax routing trace (0.62.1) | yes | Measurement plus an instrument fix. No default changed. |
| §18.92 | A routing tracer for GLM and MiniMax (0.62.0) | yes | Engineering (instrument), off by default; no default, output or numerics changed. Lab brief: `docs/lab/briefs/2026-10-08-runtime-0.62.0.md`. |
| §18.91 | The read throttle reaches MiniMax's coded bank (0.61.14) | yes | Engineering (instrument). No default, output or numerics changed; no snapshot bump. Bottleneck: unchanged. |
| §18.90 | The first live Hermes session on GLM-5.3-Flash with the contiguous bank (0.61.13) | yes | Measurement (tagged measured; quality reading one sample each). **No default changed.** Bottleneck: GLM on the X10Pro is read-bound at every prefill chunk and every miss (§18.78, §18.86); this session adds the live cost of a cold block and of an image. What it recommends is a conversation with Hamed, not a lever: keep GLM for vision, prose and short tool turns; use DeepSeek or MiniMax for code (already the project's rule, GLM-CODE). |
| §18.89 | What a short prefill costs, split by a per-chunk trace (0.61.12) | no | Engineering plus measurement (tagged measured; the reading derived). **No default changed.** Lever it points to: nothing new for the chunk's compute; short-prefill latency is a residency question (prediction and read-ahead for prefill chunks), which §18.50-18.62 already covered for decode, not for short prefill. Priced on paper only: reading ahead a short chunk's experts needs a prediction of them from the prompt's first tokens, which the repo does not have. |
| §18.88 | L6 architecture comparison and L7 seam map (0.61.11) | no | not labelled |
| §18.87 | The context term of the decode floor in the cache simulator (0.61.10) | no | not labelled |
| §18.86 | The contiguous GLM bank through the server: -6.1 % a token, identical text, on by default (0.61.9) | no | not labelled |
| §18.85 | The real contiguous GLM bank on the X10Pro, a paired run, and a correction of the raw-rate claims (0.61.8) | no | not labelled |
| §18.84 | GLM's expert record layout on the X10Pro, priced from the contiguous bank and a raw-block stand-in (0.61.7) | yes | not labelled |
| §18.83 | Queue depth on the X10Pro: reads plateau at 0.97 GB/s from two in flight, and the record layout costs ~18 % (0.61.6) | no | not labelled |
| §18.82 | The DeepSeek decode floor against context length (W2) (0.61.5) | no | not labelled |
| §18.81 | Pricing a router-weight gate on decode prefetch, offline (0.61.4) | no | not labelled |
| §18.80 | Predicted sets in routing traces, and a first look at the predictor's weight (0.61.3) | no | not labelled |
| §18.79 | The first live Hermes session on 0.61.1: the pin and the date reuse checked, short prefills, and the answers read (0.61.2) | no | Engineering plus verification (the two live checks, which had predictions on paper in §18.65 and §18.66, held). **No default changed.** **Open:** the C# replay above (Hamed's call whether it is worth the machine time); what the ~1.5 s fixed prefill cost is. |
| §18.78 | Prefetch on a slow drive: GLM on the X10Pro, DeepSeek at 1 GB/s under budget 0, and quantized kernels on the read-size curve (0.61.1) | no | not labelled |
| §18.77 | Storage bandwidth against the DeepSeek token: an emulated-drive knob, the curve, and prefetch on slow storage (0.61.0) | no | not labelled |
| §18.76 | The server floor measured the same day: 70 ms was a fit intercept; D3 under budget 0 priced; the read-size curve (0.60.10) | no | not labelled |
| §18.75 | The DeepSeek floor re-profiled, and the runtime guard that missed venv runs (0.60.9) | no | not labelled |
