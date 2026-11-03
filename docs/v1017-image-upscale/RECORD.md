# Upscaling small screenshots before the vision tower: replay record

Written 2026-11-03 against runtime 0.78.4 plus the uncommitted knob, before the run. Hamed: "Build and test it" (lever V-1 of HANDOFF
18.263). HANDOFF section 18.264 when scored.

## 1. Baseline (recorded)

HANDOFF 18.263: on the two 492 x 512 copies of Hermes's address-form screenshot (C1, C2), 21 of 24 replies over both arms called
Hermes's `vision_analyze` tool instead of reading the image, and 2 of the 3 text replies under exact decode misread an address number;
the 906 x 943 copy (C3) was read cleanly in 7 of 7. The checkpoint's preprocessing upscales below 544 x 544 pixels only, so the
492-pixel screenshot becomes 198 image tokens; C3 becomes 531.

## 2. Hypothesis

At about the token count C3 gets, the small screenshot is read as well as C3 is: fewer misreadings, and the model calls the vision tool
less often because it can read the image itself. Alternative: upscaling adds no information (the pixels are interpolated), so neither
rate moves.

## 3. Design and method

`CACHALOT_VISION_MIN_PIXELS=N` (default off; at or under 544 x 544 is off) replaces the checkpoint's `vision_min_pixels` in
`image_processor_mlx` (`image_config_from_env`, read by `VisionEncoder`): an image under N pixels is scaled up (`ImageOps.pad`, as the
reference does) before the ViT; the 1,024-token cap still applies. The image's row and prefix-cache key (`image_digest`) includes N when
it is not the checkpoint's, so rows of one setting are never reused for the other; the official setting keeps the old digests. A
startup line names it. Tests `tests/test_vision_min_pixels.py`. At N = 810,000 (900 x 900) the 492 x 512 screenshot plans to 882 x 924
and 486 image tokens; C3 is planned exactly as before.

`run-all.sh`: two fresh `./serve.sh` servers on the shipped defaults (budget 0, tool-exact), arm A shipped, arm B with N = 810,000,
settle between. `driver.py 10`: C1 and C2, ten replies each, as Hermes sent them (temperature 0.7, tools, non-streaming,
`max_tokens` 600). Scored with `docs/v1017-image-ocr/analyse.py`'s scorer and the private truth file: tool calls, text replies, text
replies with a character error.

## 4. Predictions (registered before the run)

- **U-1.** The upscale line appears in arm B only; arm B's prompt is 288 tokens longer (486 against 198 image tokens); 40 replies, no error.
- **U-2.** Shipped: tool calls in 14-20 of 20 (18.263: 21 of 24 on these copies).
- **U-3.** Upscale: fewer tool calls than shipped, by at least 4 of 20.
- **U-4.** Upscale: text replies with a character error at most 1 in 10 (C3's rate, 0 of 7); shipped: at least as high a share as
  upscale's.
- **Falsifier.** Upscale's tool-call count within 2 of shipped's and its text error share no lower than shipped's: the upscale does
  not help, and the lever closes.

Runtime estimate: about 35 minutes (two conversation prefills an arm, 20 replies of about 20 s, a settle). Needs
`iogpu.wired_limit_mb` at 88064, no other runtime.

## 5. Results (measured 2026-11-03, commit 01ea2ea)

**Run.** `run-all.sh` 18:08-19:05, sysctl 88064, no other runtime; arms 1,610 s and 1,676 s (most C2 requests re-prefilled the
conversation, the replay's prefix eviction of 18.261), peak wired 76.4 / 81.0 GiB. `report.txt` holds `analyse.py`'s output (item ids
only).

| arm | prompt tokens (C2, C1) | tool call instead of a transcription | text replies | with a character error |
|---|---|---|---|---|
| A, shipped | 30,591, 30,752 | 17 of 20 (C1 9, C2 8) | 3 | 0 |
| B, `CACHALOT_VISION_MIN_PIXELS=810000` | 30,879, 31,040 | **10 of 20** (C1 8, C2 2) | 10 | 1 |

Tool calls 17/20 against 10/20, Fisher p = 0.041. Text replies with an error 0/3 against 1/10 (p = 1.0). The one upscaled error is
the same split as 18.263's exact-decode errors: the five-digit house number written as its first four digits, a space, and the last
digit with the next ones; it occurs at the boundary of the search-match highlight in the screenshot (the typed prefix is drawn bold),
which may be what the model splits on rather than the pixel size.

**Scored.**
- **U-1 held:** the upscale line in arm B only; prompts exactly 288 tokens longer (486 against 198 image tokens); 40 replies, no error.
- **U-2 held:** shipped tool calls 17 of 20.
- **U-3 held:** upscale 10 of 20, 7 fewer (predicted at least 4).
- **U-4 split:** upscale's text error share 1 of 10 (held, at most 1 in 10); shipped's 0 of 3 is not at least as high (missed). Pooled
  with 18.263's text replies on these two copies (exact decode 2 of 3 with an error), the shipped preprocessing has 2 of 6 against
  upscale's 1 of 10: suggestive, not significant.
- **Falsifier not fired** (tool calls differ by 7).

**Reading (derived, hypothesis).** With about 2.5 times the image tokens the model reads the small screenshot itself twice as often
instead of asking Hermes's vision tool for it, and its transcriptions are at least as accurate. Cost: 288 more prompt tokens an image
(about 2-3 s of prefill at the measured short-prefill rates) and the rows of a different preprocessing (keyed apart). C3-sized images and
larger are unchanged by construction. A default is Hamed's call; the evidence is one screenshot in three recorded copies.

## 6. Live Hermes session with the knob on (2026-11-03, 0.79.0)

Hamed ran his usual script against `CACHALOT_VISION_MIN_PIXELS=810000 CACHALOT_SERVER_DUMP=$HOME/cachalot-dump-upscale.jsonl ./serve.sh`
(the startup line confirmed the setting), plus one new screenshot. 15 requests, no error; the dump and Hermes's export stay outside the
repository. One sample, not an A/B.

- **The address-form screenshot (492 x 512, now 486 image tokens):** answered in text on the first try, no `vision_analyze` detour;
  scored against the private truth file: no error (label, typed value without punctuation, five address lines, cities, button). Prefill
  585 tokens in 12.66 s.
- **A new screenshot** (361 x 108 JPEG of a calendar card, given as `@file:` in the user message; 494 image tokens instead of the
  checkpoint's 202): the model called `vision_analyze` on the path (the message carried a path, not an image), Hermes's tool loaded the
  image into the context, and the next reply transcribed all four strings exactly (two times, an event title, a location).
- **The rest:** Desktop listing through `search_files` (whose `*` glob returned 1.5 MB; the model recovered with `terminal`), C# (builds
  as a library; no `Main`, as a "no project" snippet), TypeScript (syntax check passes), Python (compiles), both stories, the fish
  translations and the hummingbird rewrite with all three translations, no English word inside the Farsi. Decode 7.3-11.3 tok/s.

## 7. Kind

Research and a default-off knob: predicted, measured, explained as hypothesis (U-1 to U-3 held, U-4 split). No default changed.
