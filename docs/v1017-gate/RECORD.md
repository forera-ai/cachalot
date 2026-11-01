# DeepSeek-V4.1-Flash-1017 against the old checkpoint: the quality gate (runbook step 5)

Written 2026-11-01, before any arm ran, against runtime 0.72.4 (git 1ad0cc0). The predictions below are committed
before the run; the results go in `docs/HANDOFF.md` section 18.247 and in `report.txt` next to this file.

## Question

Does 1017 predict the gate's fixed texts (prose, code, markdown) and pass its 21 checkable tasks at least as well as the
checkpoint served until now, and does the 1017 bank behave like the old one under the runtime (its rounding arm inside
the usual band)? This decides runbook step 8 (1017 as the default), which stays Hamed's call.

## Baseline (recorded)

- `pareto.py` band on the old checkpoint, internal bank, 48 GiB: `noise-chunk96` KL 0.0144 (0.56.1), 0.0146 after the
  MLX 0.32.4 upgrade (0.68.0, HANDOFF 18.223); tasks 20 of 21 in both builds.
- `act_quant(o)` on the old checkpoint (0.71.0, HANDOFF 18.238): dNLL +0.0047 [-0.0156, +0.0261], KL 0.0171 against
  noise 0.0140, neutral.
- Reference speed on the internal bank at 48 GiB: about 140 ms a step (runbook). 0.56.1 arms took about 5 minutes each.
- The old checkpoint's bank is X10Pro-only since 0.72.1 (HANDOFF 18.243); 1 GB/s storage made a token about 3.5 times
  slower than internal in the 0.61.0 curve (446 against 119-139 ms).

## Method

`benchmarks/pareto_arms.v1017.json`, four arms, one `guarded_run.sh` process each at 48 GiB with `settle.sh` between,
in the runbook's order: `v1017` (reference), `noise-chunk96`, `v1017-actq`, `old`. No hotlist in any arm. 384-token
prefills, 192 teacher-forced positions a text, the 21-task battery. Workload classes W1 (prose), W4 (code), W6 (JSON /
markdown) as the gate's texts are. Machine: llama-server stopped by Hamed before the run; Cachalot Lab closed;
display as it is.

## Predictions (written before the run)

- **G-1** `noise-chunk96` against `v1017`: KL mean 0.008 to 0.022, dNLL 95 % CI contains 0. Falsifier: KL above 0.03
  (the 1017 bank rounds differently from the old one under the runtime).
- **G-2** `v1017-actq` inside the band: |dNLL| below 0.01 and its CI contains 0, KL within 1.5 times the noise arm's.
  Falsifier: CI excludes 0.
- **G-3** `old` against `v1017`: KL mean 0.05 to 0.40 (different weights, not rounding: well above the band).
  Falsifier: KL inside the band (the weights changed too little to matter on these texts).
- **G-4** `old` paired dNLL (positive = old worse): -0.01 to +0.05 nats, centre positive (hypothesis: a later release
  of the same model predicts ordinary text no worse). Falsifier: CI entirely below -0.01 (the old checkpoint predicts
  these texts better).
- **G-5** Tasks: `v1017` passes 19 to 21 of 21; `old` reproduces 20 of 21 (+-1); the two differ by at most 2.
- **G-6** Speed: `v1017` 125 to 160 ms a step; `old` 2.5 to 4.5 times that (X10Pro reads), not comparable for quality.
- **G-7** Wall time: each internal arm 4 to 9 minutes, the old arm 15 to 40 minutes, the run 40 to 75 minutes with settles.

## Kind

Research if the predictions are scored as written; the default decision that follows is Hamed's.

## Erratum before any arm ran (2026-11-01 21:31)

The first launch waited in `settle.sh` at 71.6-72.2 GiB available against the 77 GiB a 48 GiB arm needs (the
llama-server was gone; the rest is ordinary applications) and was stopped before any arm started. The gate runs at
**40 GiB** (needs 69 GiB) from `docs/v1017-gate/arms.json`, a copy of the arms file with only the base budget changed.
Outputs do not depend on the budget on this gate (32, 36 and 48 GiB were bit-identical), so G-1 to G-5 stand as
written. G-6 is scored against itself only: 1017's ms a step at 40 GiB is not comparable to the ~140 ms 48 GiB
baseline, and the old arm's ratio to it still is a ratio within the run. G-7 stands.
