# G5: GLM decode miss budget 1, a blind replay against exact

Written 2026-11-17 against runtime 0.85.2, before any run. HANDOFF section 18.279 when scored. Lane: `docs/GLM-SPEED-LANE.md` row G5. Hamed, 2026-11-17: "run G5 blind replay at budget 1"; he chose to rebuild the internal bank and the plan below (n = 4, about 1.5 h then; re-estimated at about 2 h in section 5).

## 1. What is known (measured, earlier records)

- Arm 1, teacher-forced (18.107, X10Pro bank, 46 GiB, 99 forced tokens on each of three texts). Budget 1 against exact: prose dNLL +0.029 [-0.071, +0.125], KL 0.139, top-1 0.83; Python dNLL -0.019 [-0.087, +0.048], KL 0.038, top-1 0.96; JSON dNLL +0.031 [-0.016, +0.093], KL 0.007, top-1 0.99. GLM's own prefill-against-decode spread was KL 0.016 / 0.005 on prose / Python. Budget 2, the shipped default: KL 0.051 / 0.016 / 0.005. Budget 1 dropped 68-92 experts a token, budget 2 41-54.
- Arm 2 and arm 3 (18.108-18.109), free-running and blind: exact, budget 4 and budget 2 showed no visible difference on tool calls and stories at n = 2 a task an arm; GLM garbles C# at every arm (the checkpoint, 18.112).
- On the internal bank (18.272) a Hermes decode token at budget 2 is 254-295 ms at 57-75 misses: about 63 ms of compute plus about 3.1 ms a read (18.278's fit).

## 2. Instrument and method

`benchmarks/quality_blind_ab.py` gains a GLM path (`CACHALOT_QB_FAMILY=glm`): `GlmModel` and `GlmEngine.chat`, the server's code path, in one guarded process; the arm sets `GlmModel.set_decode_miss_budget(None | 1)` before each sample. Sample j of a body uses seed `1000 x body + j` in both arms and the arm order alternates with j. `docs/glm-speed/g5_run.sh` exports `serve-glm.sh`'s environment (internal verified bank, prefetch off, short-prefill threshold 512, fast synch, wired limit 80) without its miss budget, a snapshot directory of its own, 46 GiB, the 700-token cap, under `benchmarks/guarded_run.sh` after `benchmarks/settle.sh`.

Bodies (`docs/glm-speed/g5_bodies.py`): Hermes's own request dump of the 2026-11-02 live session (Hamed's standard script; 21 tools; `reasoning_effort` none, so thinking off; `extra_body.temperature` 0.7; top_p from the checkpoint's generation config, as the server does), cut after six turns: 0 the Desktop tool call, 1 the answer to its tool result, 2 a C# importer, 3 a 200-word story, 4 the Python importer, 5 the translation into German, French and Farsi. The image turn earlier in the history becomes the text note "[image omitted]" in both arms. The bodies hold Hamed's conversation and stay in `benchmarks/results/g5-glm-b1/` (gitignored).

n = 4 samples a body an arm: 48 replies. Grading: `quality_blind_ab.py sheet` writes a shuffled, arm-free sheet and the mechanical checks (tool-call validity on body 0, 5-gram repetition, truncation, C# `dotnet build`); one grader (this session) grades every reply ok / flawed with a flaw class before `key.json` is opened; `score` joins them and gives Fisher's exact p per body and pooled. Replies cut at 700 tokens are graded on what they show, not on being unfinished.

Classes: W5 (agent tool turn, bodies 0-1), W1 (prose, bodies 3 and 5, including Farsi), W4 (code, bodies 2 and 4).

## 3. Predictions (written before the run)

| id | prediction | falsifier |
|---|---|---|
| P-1 | Exact decode 0.33-0.45 s a token (about 63 ms + 3.1 ms x 88-107 reads, the simulator's misses at 46 GiB) | outside the range |
| P-2 | Budget 1 decode 0.15-0.22 s a token (at most one read a MoE layer; 37-41 reads a token in arm 1) | outside the range |
| P-3 | Budget 1 / exact token ratio 0.40-0.55 | outside the range |
| P-4 | Body 0: every reply in both arms is a valid tool call (4/4 and 4/4) | budget 1 has 2 or more invalid calls where exact has none |
| P-5 | Pooled over the 24 replies an arm, budget 1 has 0 to 3 more flawed replies than exact, and the excess, if any, sits in the long prose (bodies 3 and 5) | budget 1 has 5 or more more flawed replies, or a failure class exact never shows (a loop, a wrong-language passage, a derailed or invented answer) in 2 or more replies |
| P-6 | C# (body 2): both arms flawed in most replies (GLM's base rate); no budget signal | |
| P-7 | Nothing in either arm loops (5-gram repetition share under 0.3) | any reply over 0.3 |

Hypothesis: the lightest experts of a layer carry little router weight and the rescale restores the layer's magnitude, so even one read a layer keeps free-running agent output usable; budget 1's larger KL on prose (0.139, about 9x the noise) shows as an occasional odd word, not as broken replies. Alternative: at 1 read a layer the dropped mass is large enough on prose to derail long replies (the 500-token translations most exposed).

Decision rule, applied after scoring: an excess of at most 2 flawed of 24 and no new failure class makes budget 1 a candidate `serve-glm.sh` default, which stays Hamed's call; an excess of 3-4 keeps budget 2; 5 or more closes G5. This panel cannot see a rate difference below about 30 points (the limit of arm 2).

## 3a. Amendment before the run (2026-11-17, nothing sampled yet)

The first start stopped at `settle.sh`: 46 GiB needs 73 GiB available, and with Hamed's applications open the machine had about 47, then about 53 after `sudo purge` (applications hold about 35 GiB; GLM adds about budget + 21 GiB wired, G4-RECORD.md section 1). Hamed chose a lower budget over closing applications. The run uses **24 GiB** (`bash docs/glm-speed/g5_run.sh 4 24`; settle needs 51 GiB). Everything else in sections 2 and 3 stands except the speed predictions, which are replaced:

| id | prediction at 24 GiB (derived) | falsifier |
|---|---|---|
| P-1' | Exact decode 0.40-0.62 s a token (about 63 ms + 3.1 ms x 110-180 reads: roughly half the slots of 46 GiB, the G4 table's slope extended, extrapolated) | outside the range |
| P-2' | Budget 1 decode 0.16-0.24 s a token (at most one read a MoE layer; with more misses nearly every layer has one) | outside the range |
| P-3' | Budget 1 / exact token ratio 0.30-0.50 | outside the range |

Consequence for quality: at 24 GiB budget 1 drops more experts a token than at 46 (most of 110-180 misses, against 68-92 in arm 1), so this is a harsher test of budget 1 than the shipped 46 GiB setting would be. A clean result here covers 46 GiB a fortiori; a dirty one does not by itself condemn budget 1 at 46. Time: about 2.5 h (exact about 96 min, budget 1 about 36 min, prefills about 15 min).

## 4. Limits known before the run

One conversation, one grader, n = 4, thinking off, 700-token cap (the C#, Python and translation replies will often be cut), no image turn. A null result does not show budget 1 is harmless on code or on long context beyond about 17k tokens.

## 5. Time (estimate)

Copy and byte-compare the bank, about 18 min. Replies average about 470 tokens: exact 24 x 470 x ~0.38 s, about 71 min; budget 1, about 34 min; prefills, one cold system block (about 205 s) and the per-body increments, about 10 min. About 2 h in all.

## 6. Result (measured, 2026-11-17)

Run: `bash docs/glm-speed/g5_run.sh 4 24`, 12:42-14:26 (6,200 s), one guarded process, 48 replies, no guardian action; footprint peak wired 57.1 GiB, peak compressor 14.6 GiB, minimum free 0.0 GiB, peak swap 1 MB, pressure 1. The bank had been copied back to `~/GLM-5.3-Flash-bank` first (249 s, all 44 files byte-identical to the X10Pro copy, 812 s, marker written). Samples, sheet, grades and key in `benchmarks/results/g5-glm-b1/` (gitignored; they hold Hamed's conversation).

Speed (decode, replies of 20 tokens or more):

| arm | ms a token, median (range) | expert reads | dropped experts a token | tokens |
|---|---:|---|---:|---:|
| exact | 463 (374-564) | about 163 misses at 51 % hits (a 700-token reply) | 0 | 7,756 |
| budget 1 | 205 (192-222) | about 40-42 at 77-83 % hits | 123 (median) | 7,085 |

Budget 1 / exact = 0.44.

Grading: one grader (this session) graded all 48 replies from the shuffled sheet into `grades.json` before `key.json` was opened.

| body | exact flawed | budget 1 flawed |
|---|---:|---:|
| 0 tool call | 4/4 | 3/4 |
| 1 tool result | 0/4 | 1/4 |
| 2 C# | 2/4 | 3/4 |
| 3 story | 1/4 | 2/4 |
| 4 Python | 2/4 | 1/4 |
| 5 translation | 4/4 | 4/4 |
| all | 13/24 | 14/24 (Fisher p = 1.0) |

Flaws by arm (the grader's labels): exact: malformed tool call 4, wrong action 6, wrong code 2, invented 1. Budget 1: malformed tool call 4, wrong action 4, garbled 3, wrong code 2, invented 1. Loops: none. The mechanical checks: 0 of 8 C# replies build (all cut or not code), no 5-gram repetition over 0.3.

What the flaws are (read after the key was opened):

- Body 0: 7 of 8 `search_files` calls in both arms pass `file_glob` without the schema's required `pattern` (the live DeepSeek call passed `pattern`). Same rate in both arms; GLM's habit, not the budget.
- Body 5: every reply in both arms fails to translate: empty `text_to_speech("Test")` or `execute_code` import calls, or a 700-token tool block cut at the cap. GLM at this 30k-token context (thinking off, a history written by DeepSeek) does not do the task at all; not the budget.
- Bodies 2 and 4: exact's code flaws are structural (twice the code went into a tool-call block and only an intro line shows; two broken helper functions).
- Corrupted tokens appear only under budget 1, in 6 replies of 24, against 0 of 24 exact: a misspelt home path (`/Users/hamacprooshani`, with a correct fallback), "couldn barely", an identifier `generic_placeholder_line`, a `for` loop with `i)` for `i++` and a broken JSON comment, a tool name that swallowed prose and `<tool_call>` text, a stray stub `def _coect(...)`. Three of these the grader labelled "garbled" blind; the other three sit under wrong code and malformed tool call. Counted post hoc as one class, 6/24 against 0/24 is Fisher p = 0.022.

Against the predictions:

| id | result | verdict |
|---|---|---|
| P-1' | exact 374-564 ms, median 463 | held |
| P-2' | budget 1 192-222 ms, median 205 | held |
| P-3' | 0.44 | held |
| P-4 | valid calls 0/4 exact, 1/4 budget 1 by the schema | missed for both arms (GLM omits `pattern`); its falsifier (budget 1 worse where exact is clean) did not fire |
| P-5 | excess +1 of 24, inside 0-3; but "garbled" 3 replies under budget 1, 0 under exact | **falsified by its second clause**: a failure class exact never shows, in 2 or more replies |
| P-6 | C# flawed 2/4 and 3/4, builds 0/4 and 0/4 | held |
| P-7 | no loops | held |

Explanation (hypothesis): at 24 GiB budget 1 drops about 123 of about 163 needed experts a token, and the rescale keeps a layer's magnitude but not its direction often enough that, every few hundred tokens, a low-probability token wins sampling: a word or identifier comes out misspelt. Arm 1's teacher-forced numbers point the same way (top-1 0.83 on prose at 46 GiB, against 0.94 at budget 2). Exact GLM is not free of the class: its C# garbled with stray tokens at every arm of 18.108 (exact 2/2) and in 18.44-18.49; in these 24 exact replies it showed none. The blind "garbled" label alone (3 against 0) is Fisher p = 0.23; the registered rule counts the class, not the p.

Decision by the registered rule: a new failure class, so budget 1 is not a candidate default. `serve-glm.sh` stays at budget 2. This was a harsher setting than Hamed's 46 GiB (123 drops a token against 68-92 in arm 1); a 46 GiB rerun could show fewer corrupted tokens, but the rule as written does not make budget 1 a default from this run.

Limits: one grader, one conversation, n = 4, 24 GiB, thinking off, 700-token cap; the corrupted-token class was formed after the key was opened from labels given blind; two of six bodies (0 and 5) fail in both arms for reasons unrelated to the budget, so the panel's effective n is smaller than 24.

Kind: research (predicted, measured, explained as hypothesis): 5 held, 1 missed for both arms, 1 falsified (P-5). No default changed.
