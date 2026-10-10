# E3c record: does irregular GPU access reproduce the decode power excess? (charter L5b, follows `docs/E3B-REFIT-RECORD.md`)

Fields 1-4 written 2026-10-10, before the run, against runtime 0.62.36. Result and explanation are appended after the run.

## 1. Baseline

E3b (HANDOFF 18.123): under real DeepSeek decode `PSTR` exceeds the E1b component fit by **+12.6 W** (all-resident) and **+13.9 W** (read-bound), on the SMC rails `PVCC` (+11 W), `PSVR` (+8 W) and `PMVR` (+5 W); not a missing Energy Model channel and not the fans. On the synthetic steps (sequential stream, cache-resident matmul, CPU load) the same fit errs by -6.5 to +4.2 W, rms 3.0 W. Hypothesis from E3b: irregular access (small reads at scattered addresses, many synchronizations) costs more per byte in DRAM, the fabric and the cache than the sequential stream the DRAM figure of 49.9 pJ/B was calibrated on.

## 2. Hypothesis and predictions (tags: prediction)

No model is loaded. Eight GPU read conditions, each 30 s, against sequential references at the same session:
- **SEQ25, SEQ50, SEQ100:** sequential 9.5 MiB blocks (contiguous, wrapping over a 4 GiB buffer), duty 0.25, 0.5, 1.0 (the reference line).
- **R95-100, R95-50:** 9.5 MiB blocks (one expert's size) at random block-aligned offsets, 64 sums chained per evaluation, duty 1.0 and 0.5.
- **R1-100:** 1 MiB blocks at random offsets, 64 sums per evaluation, duty 1.0 (small reads).
- **DL-100:** decode-like: groups of 8 random 9.5 MiB blocks, each group evaluated and waited on before the next (many synchronizations, as the 40 per-layer syncs of a token), duty 1.0.
- idle steps between passes. Two passes in opposite order.

Metric: the old fit's error against `PSTR` for each step (decode: +12.6/+13.9 W; sequential steps: about 0 +-3 W), and the rail residuals of `PVCC`, `PSVR`, `PMVR` against a fit of each rail on the sequential and idle steps.

Predictions:
- **R95-100 and R95-50** (long runs of 9.5 MiB at random places: still row-buffer friendly) show an error within **+3 W** of the sequential steps' (probability 70 %).
- **R1-100** (small scattered reads) shows an error **at least +3 W** above the sequential steps' (probability 45 %).
- **DL-100** (many synchronizations) shows an error **at least +2 W** above (probability 40 %).
- No step reaches **+8 W** (the decode excess is mostly not produced by a synthetic read pattern; probability 70 %).

## 3. Falsifier

The access-pattern hypothesis, as these synthetic reads model it, is falsified if **none** of R95, R1 and DL shows an error 3 W or more above the sequential steps': the decode excess then comes from something these reads lack (the real quantized kernels, attention, the CPU-side work, the unified-memory pressure of 67 GB resident). It is strongly supported if any step reaches **+8 W** above. A run in which the sampler was not running is void. Small-block and synchronized steps are launch-bound and reach a fraction of the sequential rate (a smoke test without a sampler read 22 % and 35 % of the sequential 100 % step's rate); that is the point of those steps, not a reason to void them, because the metric (the fit's error against `PSTR`) uses the measured components, not the rate. Each step is reported with its achieved rate.

## 4. Intervention

None to the runtime. New instrument `benchmarks/gather_test.py` (full-channel logging; `--analyze` re-reads a run). Needs Hamed's root `powermetrics` beside it for the run (about 10 minutes), Lab closed, the machine as cool as practical: the session records fan RPM and idle watts with every run.

## 5. Result, explanation, generalization, bottleneck after, kind

To be written after the run.
