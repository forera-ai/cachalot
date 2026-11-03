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

## 5. Results

Not run yet.
