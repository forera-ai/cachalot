# GLM-5.3-Flash on small screenshots: transcription record

Written 2026-11-05 against runtime 0.80.1, before the run. Hamed chose it from v313's first jobs ("GLM vision, ~30 min"): does GLM's
tower also misread small screenshots? HANDOFF section 18.268 when scored.

## 1. Baseline (recorded and derived)

Recorded, DeepSeek: the 492 x 512 address screenshot gets 198 image tokens and was misread in 2 of 6 text replies (18.263, 18.264); the
906 x 943 copy, 531 tokens, 0 of 7. With the upscale to 810,000 pixels, 1 of 10 (18.264). Recorded, GLM: one image in the live session
of 18.90, two character errors; GLM has no number on small screenshots.

Derived from `glm.vision._geometry` and the checkpoint's image config (no minimum beyond 16 tokens, 28-pixel tokens):

| image | GLM image tokens | DeepSeek shipped |
|---|---|---|
| A492, address form, 492 x 512 | 342 | 486 (198 before 0.80.0) |
| A906, the same form, 906 x 943 | 1,122 | 531 |
| C361, calendar card, 361 x 108 | 52 | 494 (202 before 0.80.0) |
| C902, the card padded to 902 x 270 | 330 | not used |

So GLM gives the address form more tokens than the pre-0.80.0 DeepSeek did, and the card far fewer. Only very small images are short of tokens.

## 2. Hypothesis

GLM's misreadings follow its image token count, as DeepSeek's did: the card at 52 tokens is the weak case, the address form at 342 is
about as good as at 1,122, and padding the card up to 330 tokens helps it. Alternative: the card's pixels are all there at native size
(GLM keeps them, one 28-pixel token over a roughly 15-pixel text line), so it reads as well without padding, and interpolation adds nothing.

## 3. Design and method

One fresh `./serve-glm.sh --expert-budget-gib 40` (amended from 46, section 4) on the shipped GLM defaults (decode miss budget 2, prefetch off, contiguous bank), under
`benchmarks/guarded_run.sh`. `driver.py 5`: five rounds over the four images, in rotation so that the warming expert cache falls on all
four alike, 20 replies. Each request carries Hamed's recorded words ("check this image and write me what is written inside the image"),
temperature 0.7, `reasoning_effort` none, `max_tokens` 400, no system prompt, no tools, non-streaming. The padded copy is made in the
driver with `ImageOps.pad`, as DeepSeek's knob does, so no runtime change is needed to read its effect.

Scored by `analyse.py` against the private truth files: the address form with the image-ocr scorer (`~/cachalot-private/ocr-truth.json`),
the card against five strings read from the image (`ocr-truth-card.json`). Images, replies and truth stay outside the repository; the
output is item ids and counts.

Limits, written first: no Hermes context, so this tests the tower and not Hermes's `vision_analyze` detour; five replies a cell; one
screenshot of each kind; decode runs at budget 2, so an error is not separated from the budget by this run; 40 GiB, below GLM's other runs at 46.

## 4. Predictions (registered before the run)

- **G-1.** 20 replies, no error, no tool call. Prompt-token differences match the table within 4: A906 - A492 = 780, A492 - C361 = 290,
  C902 - C361 = 278.
- **G-2.** A906: at most 1 of 5 replies with an error (DeepSeek 0 of 7 at half the tokens).
- **G-3.** A492: 1 to 3 of 5 with an error, at least as many as A906.
- **G-4.** C361: at least 2 of 5 with an error.
- **G-5.** C902: at least 2 fewer replies with an error than C361.

**Decision rule.** (a) A492 at most 1 of 5 and C361 at most 1 of 5: GLM does not misread small screenshots here; no knob, the question
closes. (b) Errors on C361 or A492 and G-5 held: a GLM upscale knob is a candidate, a build is Hamed's go. (c) Errors and G-5 missed:
padding does not help, no knob; an exact-decode control would separate the budget, and is Hamed's call.

Runtime estimate: about 35 minutes (server start, 20 replies at about 2 minutes on a cold cache at 1.1-1.6 s a token). Needs no other
runtime; the guard's arithmetic at 46 GiB asks for 75 GiB available.

**Amendment before any reply (2026-11-05).** The budget is 40 GiB, not 46: `sudo purge` and closing Chrome and Word left 63 GiB by the
guard's count, short of 75 at 46 GiB (about 68 by GLM's real 5.5 GiB trunk, derived). Hamed: "Use 40 GiB". The guard's sum at 40 is 69, so the
run uses `--force` with the guardian armed; about 62 GiB is the real need (derived). With fewer resident experts, budget 2 drops more, so
errors here may lean above a 46 or 52 GiB run. Predictions and decision rule unchanged.

## 5. Results (measured 2026-11-05, commit ad0ddd0)

**Run.** One fresh `serve-glm.sh --expert-budget-gib 40`, 21:59-22:50, sysctl 88064, no other runtime, Chrome and Word closed, ChatGPT and
Codex open. The guardian stopped it at its 3,000 s limit, not for memory: 13 of 20 replies landed (the other 7 rows are connection errors).
My 35-minute estimate missed: replies ran 91-202 tokens, not about 60, and the first round's cold prefills took 70-284 s each. Free
memory fell to 0.1 GiB, wired 63.8-65 GiB, swap 1,792 to 1,933 MB, pressure level 1 throughout. `report.txt` holds `analyse.py`'s output.

| image | GLM image tokens | replies | with an error | reply tokens |
|---|---|---|---|---|
| A492 | 342 | 4 | **3** (row5 each time) | 164-202 |
| A906 | 1,122 | 3 | 1 (row4) | 178-202 |
| C361 | 52 | 3 | **0** | 106-116 |
| C902 | 330 | 3 | 0 | 91-100 |

Error kinds, read from a private diff by character class, no text kept: the A492 errors are one or two wrong letters in one address line;
the A906 error is one extra digit. Every reply had all four city rows. Decode 0.86-0.94 tok/s, 65-71 misses a token, hit 75-78 %.

**Scored.**
- **G-1 split:** the prompt differences held exactly (780, 290, 278); no tool call; 13 replies, not 20 (the guard's limit).
- **G-2 held so far:** 1 of 3; at most 1 of 5 is not settled.
- **G-3 held so far:** 3 of 4, inside 1 to 3 of 5 only if a fifth reply were clean; a higher rate than A906.
- **G-4 not supported:** 0 of 3; it would need both unrun replies wrong.
- **G-5 falsified as measured:** the card had no errors for padding to remove.

**Decision rule.** Case (c): errors on A492, G-5 missed, so no GLM knob. Limit: padding ran only on the card, which GLM read cleanly, so
this does not show that padding fails on the address form. Whether budget 2 causes the errors is also open; an exact-decode control is
Hamed's call.

**Reading (derived, hypothesis).** The image token count alone does not order GLM's errors: 52 tokens read cleanly and 342 did not. The
card holds a few short strings in large type; the form holds long lines in small type. That is one screenshot of each kind at n = 3-4,
and 40 GiB leans towards more dropped experts than GLM's other runs. GLM's own rate on the form (4 of 7 across the two sizes) is no
better than DeepSeek's recorded 2 of 6 and 1 of 10.

## 6. Kind

Research: predicted, measured, explained as hypothesis (G-1 split, G-2 and G-3 held so far, G-4 not supported, G-5 falsified); cut short by the
guard's time limit. No code or default changed.
