# Experiment record: DI-2, bounded decoder replay, on the long quality gate (keep 140 to 1,024)

Written 2026-11-02 before the first arm. Runtime 0.73.2 plus the default-off knob `CACHALOT_PREFILL_DECODER_KEEP`, committed together with
this record. Hamed: "speed", then "DI-2 quality curve".

## 1. Question

DI-1 (on by default in `serve.sh`) runs the encoder half (layers 0-19) over every prompt row and the decoder half (20-39) over only the last
2,541 rows, which is exact. DeepInfer's bounded replay (DI-2, `docs/DEEPINFER-LEVER-LANE.md`) replays far fewer rows (140 by default) and
cuts each decoder window at the replay start, which is approximate; the report says post-training simulated it. How much does the model's
next-token distribution move as the replay shrinks from the exact 2,541 rows to 1,024, 512, 256 and 140?

## 2. Intervention

`CACHALOT_PREFILL_DECODER_KEEP=N` (N in [128, 2,541]; unset = the exact tail, bit-identical to 0.73.2 by construction). `finish_decoder`
replays the last N rows and sets `attention_prefill_batched.WINDOW_FLOOR` to the replay start, so no decoder-layer window key before it is
seen. Without the floor the zeroed ring would take part in the softmax as zero keys; that is harmless for the exact tail (those rows never
reach the end) but not DI-2's semantics (tests: `tests/test_di2_bounded_replay.py`). Snapshots get a `-keepN` key. Only the encoder-only path
changes; the knob is off everywhere by default.

## 3. Baseline (recorded)

0.72.3, HANDOFF 18.245, the same gate (texts hash `05a1148ccc1ff338`, 1017, 34 GiB): noise-chunk3000 KL 0.0116, dNLL -0.0042 [-0.0245,
+0.0167]; DI-1 exact KL 0.0011, dNLL -0.0018 [-0.0055, +0.0018], top-1 98.6 %. The lane's draft prediction (DI-2 section): dNLL inside
noise for keep >= 256, top-1 at least 90 % at keep 140.

## 4. Predictions (fixed before the run)

- D-1. noise-chunk3000: KL 0.008-0.020, CI contains 0 (reproduces 0.72.3).
- D-2. di1: KL at most 0.003, inside noise.
- D-3. keep1024 and keep512: inside noise by `pareto.py`'s rule (|dNLL| within the band, KL at most the noise arm's).
- D-4. keep256: dNLL CI contains 0 and KL at most 1.5 times the noise arm's.
- D-5. keep140: dNLL under +0.05 and top-1 at least 90 %. Falsified by dNLL above +0.05 or top-1 below 88 %.
- D-6. KL does not fall as keep shrinks: KL(140) >= KL(256) >= KL(512) >= KL(1024) >= KL(di1), one inversion of at most 0.001 allowed.
- D-7. Time: each arm 5-7 minutes, the run 40-55 minutes.

Stop rules: the guard's own; one attempt per arm; a crash or a guard kill is reported, not retried, without Hamed.

## 5. Method

`docs/v1017-di2-curve/run.sh`: `benchmarks/pareto.py run` per arm (ref, noise-chunk3000, di1, keep1024, keep512, keep256, keep140), each in
its own `guarded_run.sh --budget-gib 34` process with `settle.sh --budget-gib 34` before it; then `pareto.py report`. `arms.json`: the
0.72.3 long gate's base (8,192-token prefill in two 4,096-token calls, 192 teacher-forced positions a text, three texts, 576 positions, no
battery), 1017 with its internal bank, `CACHALOT_HOTLIST` empty. Raw `.npz` in `benchmarks/results/pareto-di2/` (gitignored); the report and
a result table go here and in HANDOFF.

## 6. Limits known before the run

The gate scores the 192 positions after the prefill; DI-2 changes only the state at the prefill's end, so its effect is largest at the first
decoded positions and fades as decode refills the windows (it scores the regime of a reply's start, which is what a user sees first). The
8,192-token prompts are two calls; DI-2's continuation case (a short suffix after a snapshot) is not exercised here. Prefill time is not
measured by this gate: the speed side is derived (section DI-2 of the lane: -26 points of prefill compute at 4.5k tokens, -5 at 22k) and
is measured only if quality holds. Any default change is Hamed's.
