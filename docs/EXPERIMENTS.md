# Experiment index (charter section 7.2 and section 10)

Written 2026-10-09 against runtime 0.66.22; regenerate with `benchmarks/experiment_index.py`. The charter asks for "a file of experiment records whose predictions were written before the build". The records themselves live in `docs/HANDOFF.md` section 18 (each with Baseline, Prediction, Method, Result, Explanation, Limits, Kind); this file is the index: where each record is, whether it carries a written prediction, and what kind of result it is, as its own **Kind** line says. It does not restate results. Sections before 18.75 predate the record format and are not listed; sections without a **Kind** line are listed as unlabelled and have no claim made for them here.

Counts of the 104 sections listed (18.75-18.178): 48 research, 20 unlabelled, 19 measurement, 13 engineering, 3 planning, 1 pricing.

Research means predicted, measured and explained (charter section 8.1 rule 6); a falsified prediction is still research and is marked so in its Kind line. Engineering means it works without a written prediction, or an instrument was built.

"Prediction written" means the section has a **Prediction** paragraph; whether it was written before the run is what that section says (the 0.62.23 and 0.62.25 ones were, in the earlier section they cite). Only the Kind column is the record's own verdict.

| section | title | prediction written | kind (as recorded) |
|---|---|:---:|---|
| §18.178 | N2 stage 1 harness built (0.66.22) | no | Instrument and tests. No default changed, nothing measured on a GPU. |
| §18.177 | N2 stage 1 designed (0.66.21) | no | Planning and an offline instrument. No default changed, nothing measured on a GPU. |
| §18.176 | CUDA session V1: the V100 contrast (0.66.20) | no | Measurement (GPU microbenchmarks). No default changed. |
| §18.175 | CUDA session V1 prepared (0.66.19) | no | Instruments, tests, documents. No default changed, nothing measured. |
| §18.174 | CUDA session S2: the second GPU session (0.66.18) | no | Measurement (GPU microbenchmarks) and a cell fix. No default changed. |
| §18.173 | CUDA session S2 prepared (0.66.17) | no | Instruments, tests, documents. No default changed, nothing measured. |
| §18.172 | CUDA session S1: the first GPU measurements (0.66.16) | no | Measurement (GPU microbenchmarks only). No default changed. |
| §18.171 | CUDA session S1 prepared (0.66.15) | no | Instruments, tests, documents. No default changed, nothing measured. |
| §18.170 | Why the SLRU model collapses at small resident fractions (0.66.14) | no | Measurement. No default changed. |
| §18.169 | CUDA lane N0a on a second trace, MiniMax-M3 (0.66.13) | no | Measurement. No default changed. |
| §18.168 | CUDA lane N0a: the offline resident-fraction policy table (0.66.12) | no | Measurement. No default changed. |
| §18.167 | The CUDA N0/N1 scaffold (0.66.11) | no | Instruments and tests. No default changed, no measurement. |
| §18.166 | The NVIDIA CUDA research plan (0.66.10) | no | Planning (nothing measured; the plan's hypotheses are proposed and carry falsifiers, and their intervals are written before each phase's run). **No default changed.** |
| §18.165 | CloudRift acknowledged in the README (0.66.9) | no | not labelled |
| §18.164 | A better 2-bit fit on the NLL gate (0.66.8) | yes | Research (predicted, measured, explained as hypothesis); the first attempt was an engineering miss on the time estimate. |
| §18.163 | The 3-bit expert bank run live (0.66.7) | yes | Research (predicted, measured, explained as hypothesis): the internal-drive predictions held, the X10Pro 3-bit token missed low. |
| §18.162 | The expert-bank bit width as a curve (0.66.6) | yes | Research (predicted, measured offline and by screen, explained as hypothesis): the quality predictions held, the cost predictions missed on a wrong assumption. |
| §18.161 | The resident GLM energy arm, 12 tokens (0.66.5) | yes | Research (predicted, measured, explained as hypothesis). |
| §18.160 | The resident GLM energy arm: not resident (0.66.4) | yes | Research (predicted, measured, explained as hypothesis); void as a resident arm. |
| §18.159 | The energy meter's cost on a decode token (0.66.3) | yes | Research (predicted, measured, explained as hypothesis). |
| §18.158 | The bf16 control (0.66.2) | yes | Research (predicted, measured, explained as hypothesis). |
| §18.157 | The prefetch pair at 48 GiB (0.66.1) | yes | Research (predicted, measured, explained as hypothesis). |
| §18.156 | `CACHALOT_FP8_GEMV_DECODE` on by default (0.66.0) | no | Engineering (a default on a prior research result). |
| §18.155 | The 2-bit routed-expert dequantization priced: not the limiter (0.65.2) | yes | Research (predicted, measured twice, explained as hypothesis). |
| §18.154 | The FP8 half-decode GEMV on the other shapes (0.65.1) | yes | Research (predicted, measured, explained as hypothesis): the additional token saving was predicted inside its range; one shape missed at the default threadgroup. |
| §18.153 | A faster FP8 GEMV behind a knob (0.65.0) | yes | Engineering with a research front half (predicted, measured twice, explained as hypothesis): stop rule met, saving prediction missed high. |
| §18.152 | The DeepSeek floor roofline (0.64.18) | yes | Research (predicted, measured offline and on the machine, explained as hypothesis): rate and layer-time predictions held, the ceiling prediction was falsified, the whole-floor numbers missed on a stale input; the floor's bandwidth range is now a number. |
| §18.151 | The prefetch pair on the first prompt set, rotated (0.64.17) | yes | Research (predicted, measured, explained as hypothesis): the numeric predictions held but the order-effect bound; a replicate that tightens the prefetch number rather than moving it. |
| §18.150 | The position-in-pass test (0.64.16) | yes | not labelled |
| §18.149 | The prefetch pair on a second prompt set (0.64.15) | yes | Research (predicted, measured, explained as hypothesis): most predictions held, the gap missed its planning interval and the registered upper falsifier fired narrowly. |
| §18.148 | The drift of one exact configuration (0.64.14) | yes | Research (predicted, measured, explained as hypothesis): most numeric predictions held, the window and blocked-time predictions missed, and a correction made two sections ago was itself corrected. |
| §18.147 | The kept-expert count test: the budget-0 excess was run-to-run variation (0.64.13) | yes | Research (predicted, measured, explained): the central prediction fell with its falsifier and, more usefully, the premise of the question did not reproduce; a bound used in earlier sections was too small. |
| §18.146 | The budget-0 router-eval excess: partly characterized, unexplained (0.64.12) | yes | Research (predicted, analysed offline, explained as hypothesis): three predictions held, the central one fell with its falsifier, and the excess is characterized but unexplained. |
| §18.145 | The alternated prefetch pair (0.64.11) | yes | Research (predicted, measured, explained as hypothesis): arm 8 void by its registered falsifier (the mechanism was named, its size underestimated); arm 8b held on all but the interval width. |
| §18.144 | Decode prefetch on and off on the 22k Hermes context (0.64.10) | yes | Research (predicted with a rule fixed beforehand, measured, explained as hypothesis): most numeric predictions missed narrowly and the discrimination came out ambiguous, with a result between the two hypotheses. |
| §18.143 | The in-situ price of the 1-in-4 default (0.64.9) | yes | Research (predicted, measured, explained as hypothesis): identity held, the interval predictions missed, the registered rule not met and no larger cost shown; the instrument's noise floor was the finding. |
| §18.142 | The router sync reconciled: it is GPU compute, not a synchronization cost (0.64.8) | yes | Research (predicted, measured offline, explained): three predictions held, one held in direction only, and an earlier section's interpretation corrected. |
| §18.141 | Decode prefetch on and off at 46 GiB (0.64.7) | yes | Research (predicted, measured, explained as hypothesis): all six predictions held; the budget was changed from 48 to 46 GiB before the run for the guard. |
| §18.140 | The raw per-layer rows at 22 GiB: the expert budget slope continues (0.64.6) | yes | Research (predicted from the measured slope, measured, explained as hypothesis): predictions held but one; the L4(c) question is answered for this workload as a slope, not a region. |
| §18.139 | The raw per-layer rows at 28 GiB: the expert budget slope (0.64.5) | yes | Research (predicted, measured, explained as hypothesis): the structure predictions held, the size was over-predicted, no falsifier fired. |
| §18.138 | The raw per-layer rows with the shipped decode miss budget 0 (0.64.4) | yes | Research (predicted, measured, explained as hypothesis): all six predictions held; plus an instrument fix after a void first run. |
| §18.137 | The raw per-layer rows with decode prefetch off (0.64.3) | yes | Research (predicted, measured, explained as hypothesis): two of six predictions held, two fell with the registered falsifier firing, one was badly framed, one missed narrowly. |
| §18.136 | The raw per-layer rows of the DeepSeek decode token (0.64.2) | yes | Research (predicted, measured, explained as hypothesis): three of four predictions held and one was falsified in direction. |
| §18.135 | L4(b): the overlap term fitted offline from the L2 rows (0.64.1) | yes | Research (predicted, measured offline, explained as hypothesis): three of four predictions held, the wall comparison went the wrong way for adoption, and the term is closed as a token predictor. |
| §18.134 | `CACHALOT_TOKEN_TRACE`: the sampled per-token trace as a server knob (0.64.0) | yes | Engineering (instrument): it works and was checked live; no speed or output change, and the cost of the shipped default is derived, not measured. |
| §18.133 | The tracer's cost on 30 turns: +0.55 ms a decode token, the stop rule not met (0.63.4) | yes | Research (predicted, measured, explained as hypothesis); the prediction's upper bound missed and the registered stop rule is not met. |
| §18.132 | L2 per-token trace for DeepSeek decode: built, and priced inside a real server token (0.63.3) | yes | Research (predicted, measured, explained in part) plus an instrument. |
| §18.131 | Power accounting, step E3h: the meter's tail fix live, and its cost (0.63.2) | yes | Research (predicted, measured): tail-fix predictions held; the cost question unanswered by a design that failed its precondition. |
| §18.130 | Power accounting, step E3g: the energy arms on GLM and MiniMax (0.63.1) | yes | Research (predicted, measured, explained in part). |
| §18.129 | Power accounting, step E4, route 1: the energy meter in the server (0.63.0) | no | Engineering (instrument). No prediction on file, no default and no output changed. |
| §18.128 | Power accounting, step E3f: the model's own kernels, run alone (0.62.41) | yes | Research (predicted, measured, explained in part). |
| §18.127 | Power accounting, step E3e: the kernels in-process, without the server (0.62.40) | yes | Research (predicted, measured, explained in part). |
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
