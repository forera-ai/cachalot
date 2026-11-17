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

## 4. Limits known before the run

One conversation, one grader, n = 4, thinking off, 700-token cap (the C#, Python and translation replies will often be cut), no image turn. A null result does not show budget 1 is harmless on code or on long context beyond about 17k tokens.

## 5. Time (estimate)

Copy and byte-compare the bank, about 18 min. Replies average about 470 tokens: exact 24 x 470 x ~0.38 s, about 71 min; budget 1, about 34 min; prefills, one cold system block (about 205 s) and the per-body increments, about 10 min. About 2 h in all.

## 6. Result

(filled in after the run, HANDOFF 18.279)
