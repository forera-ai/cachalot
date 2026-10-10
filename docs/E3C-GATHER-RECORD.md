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

## 5. Result (measured; 2026-10-10, 15:52:33-16:01, Cachalot Lab closed, fans 1,000-1,014 RPM throughout; `benchmarks/results/energy/e3c-gather.jsonl` and `.header.json`, sampler `e3c-sampler.txt`)

Per step (achieved GB/s, PSTR W, the E1b fit's error against PSTR W; A then B pass): idle 0.0, 26.2, -0.8 / 0.0, 26.2, -0.2; SEQ25 82.9, 40.6, +1.1 / 87.7, 50.2, +1.2; SEQ50 175.8, 55.9, +1.1 / 178.0, 59.1, +1.5; SEQ100 349.2, 81.3, +3.1 / 346.9, 85.2, +1.6; R95-100 354.1, 82.5, +1.1 / 352.4, 85.0, +1.7; R95-50 169.8, 55.4, +0.8 / 160.2, 58.0, +2.4; R1-100 85.5, 67.1, +3.7 / 85.0, 71.6, +5.0; DL-100 141.7, 53.9, +3.2 / 139.4, 57.0, +2.6. Means by condition: sequential +1.6 W (range +1.1 to +3.1); R95-100 +1.4; R95-50 +1.6; **R1-100 +4.3**; **DL-100 +2.9**; real decode (E3b) +12.6 and +13.9 W. The largest error of any step is +5.0 W.

Rates: the random 9.5 MiB blocks read as fast as the sequential ones (354 against 349 GB/s); 1 MiB blocks reach 85 GB/s and groups of 8 blocks with a synchronization after each 140 GB/s (launch-bound, as intended).

Against the predictions: R95-100 and R95-50 within +3 W of the sequential steps (+1.4 and +1.6 against +1.6): **held**. R1-100 at least +3 W above: +2.7 above the sequential mean, **missed narrowly**. DL-100 at least +2 W above: +1.3 above, **missed**. No step reaches +8 W: **held** (maximum +5.0). Falsifier: none of R95, R1 and DL shows an error 3 W or more above the sequential steps': **fired** (the nearest, R1-100, is +2.7).

Rails: residuals of `PVCC`, `PSVR`, `PMVR` against a fit on the sequential and idle steps move a long way in steps whose `PSTR` is fine: R95-100 `PVCC` -8.7, `PSVR` -8.1 (while `PSTR` -1.1); R1-100 `PVCC` -14.7, `PSVR` -13.4 (`PSTR` -4.9); DL-100 `PVCC` +5.8, `PSVR` +5.6 (`PSTR` +2.6). A rail residual of 10-15 W therefore does not by itself locate extra power: the rails trade load among themselves as the components change.

## 6. Explanation

A synthetic GPU read with scattered 9.5 MiB blocks, scattered 1 MiB blocks or per-group synchronization does not reproduce the 13 W by which real decode exceeds the component fit: the largest error is +5.0 W (R1-100) and the small-block and synchronized steps add only +2.7 and +1.3 W over the sequential steps. So the access-pattern explanation, as these reads model it, is not supported (R1-100 is the nearest, and small scattered reads could still explain a few watts). What real decode has that these steps lack is not identified. Candidates, none tested: (a) **the internal drive**: E3 showed it reading 1.82 GB/s in the all-resident phase (speculative prefetch loads) and 3.56 GB/s in the read-bound one, and the excess was +12.6 and +13.9 W; the idle phases without reads had none; the drive's controller sits on the SoC and no Energy Model channel reports it, so its power would show on a SoC supply rail; (b) the real quantized matrix-multiply and attention kernels (a different mix of ALU, cache and memory work than a sum); (c) CPU-side work of the server process and the threads that issue reads; (d) 67 GB of resident, wired memory. The E2 of the internal drive tests (a) directly.

## 7. Generalization

Hardware-specific. The transferable statement: the energy of a data-movement pattern should be measured at the pattern, not inferred from a stream; a synthetic GPU read of scattered or synchronized blocks costs within a few watts of a sequential one on this machine.

## 8. Bottleneck after

Speed: unchanged. Energy: the 13 W of real decode that no tested synthetic load reproduces; the next test is the internal drive (hypothesis a).

## 9. Kind

Research (predicted, measured, partly explained): the access-pattern hypothesis falsified by its own criterion; two of four predictions missed narrowly.
