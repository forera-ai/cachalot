# `CACHALOT_GLM_VISION_MIN_PIXELS`: the knob and its server check

Written 2026-11-06 against runtime 0.80.3 plus the uncommitted knob, before the run. Hamed: "Yes, build it" (HANDOFF 18.269, case (a)).
HANDOFF section 18.270 when scored.

## 1. The knob

`CACHALOT_GLM_VISION_MIN_PIXELS=N` (default off; unset, empty or 0 is the checkpoint's preprocessing): a still image under N pixels is scaled up
(bicubic, aspect kept) before GLM's tower, through `glm.vision._geometry`, under the 8,000-token ceiling. Video keeps the checkpoint's floor. An image's
digest adds N when the knob is set, so rows and prefix spans of two settings never mix; off keeps the old digests. A startup line names it.
`serve-glm.sh` documents it and does not set it. Disk snapshots are unaffected: system blocks hold no images in Hermes's use.

## 2. What the server check adds

18.269 padded the image in the client, with `ImageOps.pad`; the knob resizes inside the runtime to the same 924 x 896 canvas (content 924 x 887 against 922 x 886). The unit tests
pin the token count, not the tower's reading. This check runs the real path once, and the card, which the threshold now pads too.

## 3. Design and method

One fresh `serve-glm.sh --expert-budget-gib 40` with `CACHALOT_GLM_VISION_MIN_PIXELS=810000`, shipped otherwise (budget 2). `driver-knob.py`: A492 native bytes, five
replies, and C361 native bytes, three, in rotation while both remain; same request as `FORM-RECORD.md`. Scored with the two private truth files. No off arm: 18.268 and 18.269 are the
controls (A492 8 of 14 with an error, C361 0 of 3).

## 4. Predictions (registered before the run)

- **K-1.** The startup line appears; 8 replies, no error; A492's prompt is 1,083 tokens (as the padded copy of 18.269); C361's is 1,089 (1,062 image tokens, derived).
- **K-2.** A492: at most 1 of 5 with an error.
- **K-3.** C361: at most 1 of 3 with an error.
- **Falsifier.** A492 at 2 or more of 5: the runtime path does not reproduce the client padding, and the default stays off with the difference to be found.

Runtime estimate: about 30 minutes (server start, two cold prefills, 8 replies). `serve-glm.sh`'s default stays off whatever this shows; a default is Hamed's call.

## 5. Results (measured 2026-11-06, commit 1b64560)

**Run.** `arm-knob.sh` 09:02-09:34, 32 minutes, sysctl 88064, no other runtime, ChatGPT open; pressure level 1, swap at most 1,878 MB, peak wired 68.7 GiB. `knob-report.txt` holds `analyse-knob.py`'s output.

| image | replies | with an error | prompt tokens | reply tokens | control, knob off |
|---|---|---|---|---|---|
| A492 | 5 | **0** | 1,083 | 186-208 | 8 of 14 with an error, 369 tokens |
| C361 | 3 | **0** | 1,089 | 93-116 | 0 of 3, 79 tokens |

**Scored.** K-1 held (startup line, 8 replies, no error, 1,083 and 1,089 exactly). K-2 held (0 of 5). K-3 held (0 of 3). Falsifier not fired.

**Reading (derived).** The runtime path reproduces the client padding: pooled upscaled form 0 of 15 against native 8 of 14. The cold first prefill of the upscaled form took 276 s, inside the native form's 246-284 s, so
on a cold server the extra 714 tokens are not visible against the reads; a warm cost is unmeasured. Decode a token is unchanged (1.14 s, 70 misses).

**Not measured.** Other screenshots; photographs; several small images in one context (each adds up to about 1,000 tokens); a Hermes session; 46 or 52 GiB.

## 6. Kind

Research and a default-off knob: predicted, measured, explained (3 held). No default changed; a default is Hamed's call.
