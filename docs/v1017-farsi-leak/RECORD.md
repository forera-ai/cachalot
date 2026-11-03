# Does the decode miss budget put English words into Farsi? Teacher-forced probe record

Written 2026-11-03 against runtime 0.78.2, before any run. Hamed: "Build Farsi probe" (after HANDOFF 18.261). HANDOFF section 18.262
when scored. Instrument: `benchmarks/farsi_leak_probe.py` (tests: `tests/test_farsi_leak_probe.py`).

## 1. Baseline (recorded, live, all under the shipped decode miss budget 0)

Six story replies in the three local server dumps carry a Farsi translation section (900-1,220 tokens each); two of them hold one stray
English word inside the Farsi: "mattered" (2026-11-02, F1) and "heard" (2026-11-01, F2). The four desktop-listing replies with Persian
text are excluded (they legitimately hold Latin file names). No exact-decode sample exists. A free-running replay would cost about six
minutes a reply (HANDOFF 18.261), so this probe forces the recorded text instead.

## 2. Hypothesis

Under budget 0 a Farsi token's dropped experts leave probability on Latin tokens that exact decode would not; at temperature 0.7 that
shows as an occasional English word. Alternative: the model itself leaves that mass, and the budget does not change it.

## 3. Method

For each section F1-F6 (pinned by dump file, reply row and the sha256 of its prompt and reply token ids): one process, 48 GiB, the
`serve.sh` environment (`pareto.serve_defaults`, DI-1 + keep 140), max_seq_len 49,152. Prefill the recorded prompt plus the reply up to
the line where Persian script begins (31-37k tokens, encoder-only in 4,096 calls, then the decoder tail), snapshot, then teacher-force
the recorded Farsi tokens under each arm, restoring the snapshot between them: exact (no budget) and budget 0. **Amended before the
full run (smoke finding):** budget 0 goes first in every section. The registered alternation let exact warm the cache for budget 0 (the
smoke, exact first, saw 0 drops on 40 tokens); exact reads every expert, so its output does not depend on the cache the first arm leaves,
and budget 0 first meets the cache the prefill left, as the live decode did. Per position: the probability mass on tokens with two or more ASCII
letters ("leak"), the recorded token's log-probability, the top 1,024 log-probabilities (KL between arms on that support), the experts
budget 0 dropped. Image rows are not rebuilt: the conversations' one image has its placeholder tokens removed from the context in both arms (the runtime refuses placeholders without rows; found by the smoke run, fixed before the full run; the count removed is in the manifest).

## 4. Predictions (registered before any run)

- **L-1 (instrument).** Exact leak mass, pooled over the six sections, is small: 0.05-1.0 % a position.
- **L-2.** Budget 0's pooled leak mass is higher: ratio budget 0 / exact 1.2-3.0, and higher in at least 4 of 6 sections.
- **L-3.** At the two recorded stray tokens (F1 "mattered", F2 "heard"), budget 0 gives the recorded token a higher log-probability
  than exact.
- **L-4.** Budget 0 drops 5-30 experts a forced Farsi token (W1's 15.9 a token, DS1017-TOKEN-48, with a warm conversation cache).
- **L-5.** KL between the arms over Farsi positions is at least the long gate's noise level for budget 0 decode (KL 0.01-0.10 a position).
- **Falsifier.** A pooled ratio at or below 1.05, or budget 0 lower in 3 or more of 6 sections: the budget does not raise the leak, and
  the stray words are the model's at temperature 0.7.

Runtime estimate: a smoke run (`--only F1 --limit 40`, about 6 minutes) to check the instrument, then the full run, about 50 minutes
(model load, six prefills of about 3.5 minutes, 2 x 900-1,220 forced tokens a section at about 0.1 s each). Needs
`iogpu.wired_limit_mb` at 88064, no other runtime.

## 5. Results

**Smoke run (2026-11-03, `--only F1 --limit 40`).** The first attempt failed: the runtime refuses image placeholder tokens without image
rows; the probe now removes them. The second: context 35,185 tokens prefilled in 177 s, both arms forced 40 tokens, leak 0.0034 % in
each, 0 experts dropped by budget 0 (exact had gone first; see the amendment in section 3). Footprint 205 s, peak wired 76.0 GiB, peak
swap 104 MB. The report's bootstrap needed at least 64 positions; fixed.

**Full run (2026-11-03, commit 39c1b2f, 2,306 s).** One guarded process, 48 GiB, sysctl 88064, peak wired 75.7 GiB, peak swap 104 MB,
pressure 1. Contexts 32,591-38,530 tokens prefilled in 164-194 s; 6,014 forced Farsi tokens an arm. `report.txt` holds the table; the
per-position arrays are in the gitignored `benchmarks/results/farsi-leak/`.

| section | tokens | leak exact | leak budget 0 | ratio | dropped a token (budget 0) | KL a position | stray token: log p exact / budget 0 |
|---|---|---|---|---|---|---|---|
| F1 | 1,065 | 0.0930 % | 0.0841 % | 0.90 | 14.8 | 0.0107 | " mattered" -0.07 / -0.18 |
| F2 | 1,220 | 0.0220 % | 0.0169 % | 0.77 | 12.2 | 0.0058 | " heard" -3.08 / -4.16 |
| F3 | 970 | 0.0065 % | 0.0054 % | 0.84 | 14.6 | 0.0085 | |
| F4 | 951 | 0.0109 % | 0.0231 % | 2.12 | 14.8 | 0.0033 | |
| F5 | 901 | 0.0054 % | 0.0055 % | 1.03 | 15.0 | 0.0074 | |
| F6 | 907 | 0.0010 % | 0.0013 % | 1.26 | 15.3 | 0.0016 | |

Pooled: leak exact 0.0247 %, budget 0 0.0239 %, ratio **0.97** (block bootstrap 95 % 0.82-1.52), budget 0 higher in 3 of 6 sections;
positions with leak above 5 %: exact 1, budget 0 2 of 6,014.

At the stray positions (top three, probabilities): F1 exact " mattered" 0.934, " اهمیت" 0.032, " مهم" 0.025; budget 0 " mattered"
0.839, " اهمیت" 0.088, " مهم" 0.057. F2 exact " شن" 0.925, " heard" 0.046; budget 0 " شن" 0.973, " heard" 0.016.

**Scored.**
- **L-1 missed (low):** exact leak 0.0247 % a position, under the predicted 0.05-1.0 %.
- **L-2 falsified:** ratio 0.97, budget 0 higher in 3 of 6 (predicted 1.2-3.0 and at least 4 of 6). The registered falsifier fires on
  both counts (ratio at or below 1.05; budget 0 lower in 3 of 6).
- **L-3 falsified:** exact gives both recorded stray tokens a higher probability than budget 0 does.
- **L-4 held:** 12.2-15.3 experts dropped a forced token.
- **L-5 missed (low):** KL 0.0016-0.0107 a position (predicted 0.01-0.10); only F1 reaches 0.01.

**Explanation (measured at the two positions; the general reading a hypothesis).** The stray English words are the model's own: with
every expert read, it puts 93 % on " mattered" at F1's position, where its Persian synonyms get 3 % and 2.5 %; at F2's it ranks the
Persian word first and " heard" second (4.6 %). At temperature 0.7 such a word is sampled at about those rates, under either arm.
Budget 0 changes the distribution by a small KL (mean about 0.006) in no consistent direction for Latin tokens.

**Closed.** The decode miss budget as the cause of English words in Farsi. Not studied: whether a prompt instruction or a sampling
rule (lower temperature for the translation turns) removes them; that is a Hermes/prompt question, not a runtime one.

## 6. Kind

Research: predicted, measured and explained (L-4 held; L-1 and L-5 missed low; L-2 and L-3 falsified, the falsifier fired). No default
changed.
