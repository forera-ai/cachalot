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

One fresh `./serve-glm.sh --expert-budget-gib 46` on the shipped GLM defaults (decode miss budget 2, prefetch off, contiguous bank), under
`benchmarks/guarded_run.sh`. `driver.py 5`: five rounds over the four images, in rotation so that the warming expert cache falls on all
four alike, 20 replies. Each request carries Hamed's recorded words ("check this image and write me what is written inside the image"),
temperature 0.7, `reasoning_effort` none, `max_tokens` 400, no system prompt, no tools, non-streaming. The padded copy is made in the
driver with `ImageOps.pad`, as DeepSeek's knob does, so no runtime change is needed to read its effect.

Scored by `analyse.py` against the private truth files: the address form with the image-ocr scorer (`~/cachalot-private/ocr-truth.json`),
the card against five strings read from the image (`ocr-truth-card.json`). Images, replies and truth stay outside the repository; the
output is item ids and counts.

Limits, written first: no Hermes context, so this tests the tower and not Hermes's `vision_analyze` detour; five replies a cell; one
screenshot of each kind; decode runs at budget 2, so an error is not separated from the budget by this run; 46 GiB as in GLM's other runs.

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

## 5. Results

Not yet run.

## 6. Kind

Not yet scored.
