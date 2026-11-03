# Does the decode miss budget cause character errors in screenshot transcriptions? Replay record

Written 2026-11-03 against runtime 0.78.3, before the run. Hamed chose the vision lane ("Vision") and this test ("Run it"). HANDOFF
section 18.263 when scored.

## 1. Baseline (recorded, live, all under the shipped decode miss budget 0)

The usual Hermes script asks DeepSeek to transcribe a screenshot of an address form: a label, a typed value, five suggested addresses
with their cities, a disabled button. The text cursor sits right after the typed value. Live transcriptions: 2026-11-02 (0.77.0) read
the typed value with an added comma (the cursor), the rest exact; 2026-11-01 exact; an earlier session (HANDOFF 13328, 0.58.x) had two
character errors in five addresses; another (13647) exact. Every one was decoded under budget 0, which also turned integer tool
arguments into words (18.243, 18.255). No exact-decode transcription exists.

The screenshot and the ground truth hold addresses: the truth file lives at `~/cachalot-private/ocr-truth.json`, outside the repository;
the scorer prints item ids and counts only.

## 2. Hypothesis

Budget 0 drops experts while the digits and punctuation are decoded and so raises the character error rate; exact decode reads the
screenshot more faithfully. Alternative: the errors are perception errors fixed at prefill (the image rows and the prompt are the same
under both arms; the budget acts in decode only), so the error rate does not move.

## 3. Method

`run-all.sh`: two fresh `./serve.sh` servers on the 0.78.3 defaults, arm A budget 0 (shipped), arm B exact
(`CACHALOT_DECODE_MISS_BUDGET=`), settle between. `driver.py 6`: each of the three recorded requests (C1, C2: 492 x 512 PNG; C3: 906 x
943 PNG; pinned by dump line and messages sha256), six replies each, copy-major, as Hermes sent them (temperature 0.7, its tools,
reasoning effort none), non-streaming, `max_tokens` 600. `analyse.py` scores each reply on eight required strings (label, the five
address lines, the highlighted row's city, the button), at least four "Edmonton" lines, and no punctuation glued to the typed value
(markdown removed first); a tool call or an error counts as one error. Errors a reply, replies with any error, a two-sided permutation
test on errors a reply. On the two recorded live replies the scorer gives C1 `typed_punct` and C2 none, as read by eye.

## 4. Predictions (registered before the run)

- **O-1.** Both logs carry the expected budget state; 36 replies, no error.
- **O-2.** Budget 0: at least one error in 3-10 of 18 replies (live: 2 of 4 transcriptions had one or more).
- **O-3.** Exact: fewer replies with an error than budget 0, and at most half its errors a reply.
- **O-4.** `typed_punct` occurs on the 492-pixel copies (C1, C2) more often than on the 906-pixel C3 under both arms (the cursor is
  a few pixels wide at 492).
- **Falsifier.** Exact errors a reply at or above budget 0's: the budget does not cause the transcription errors.
- **Inconclusive by design.** Budget 0 with 0-2 erroneous replies of 18: no claim.

Runtime estimate: about 40 minutes (three conversation prefills of about 2-3 minutes and 18 short replies of about 25 s an arm, a
settle). Needs `iogpu.wired_limit_mb` at 88064 and no other runtime.

## 5. Results (measured 2026-11-03, runtime 0.78.3, harness commit dd90b8a)

**Run.** `run-all.sh` 16:18-17:11, sysctl 88064, no other runtime; arms 1,346 s and 1,337 s, peak wired 79.7 / 78.1 GiB, peak swap
about 100 MB. `report.txt` holds `analyse.py`'s output (item ids only); replies are in the gitignored `benchmarks/results/image-ocr/`.

| arm | tool call instead of a transcription | text replies | text replies with a character error |
|---|---|---|---|
| A, budget 0 | 15 of 18 (C1 6/6, C2 6/6, C3 3/6) | 3 (all C3) | 0 |
| B, exact | 11 of 18 (C1 5/6, C2 4/6, C3 2/6) | 7 (C1 1, C2 2, C3 4) | **2** (both C2) |

Every tool call is Hermes's `vision_analyze` on the attached file's path (the user message carries "Image attached at: <path>"), a
valid route in Hermes, not a malformed reply; live, both recorded turns were answered in text (the replay sends the same 30,752-token
prompt; live the image was prefilled in a 298-token call after the previous turn, here in a cold 11-15k-token prefill). Tool-call rate
15/18 against 11/18, Fisher p = 0.26.

The two exact-decode errors are on the 492-pixel copy and are the live kinds: a five-digit house number split or truncated (a space
inserted after its first four digits, or digits dropped), and letters invented on digits (a trailing "S", a "B" for a "8"). No reply in either arm glued punctuation to the
typed value. The 906-pixel C3 was read without an error in all 7 of its text replies (3 budget 0, 4 exact).

**Scored.**
- **O-1 held:** budget states as expected; 36 replies, no error.
- **O-2 missed:** budget 0 had 15 erroneous replies of 18 under the registered metric, all of them tool calls; its 3 text replies were
  clean.
- **O-3 falsified:** exact errors a reply 0.89 against budget 0's 0.83 (permutation p = 1.0); among text replies only exact erred.
  The registered falsifier fires: the budget does not cause the transcription errors.
- **O-4 not testable:** no `typed_punct` in either arm.

**Explanation (derived, hypothesis).** The errors follow the image's resolution, not the decode arm. The official preprocessing
keeps native size above `vision_min_pixels` = 544 x 544: the 492 x 512 screenshot is upscaled about 8 % and becomes about 198 image
tokens, each covering about 42 x 42 source pixels, so a 10-pixel-high digit string sits inside one or two tokens; the 906 x 943 copy
gets about 530 tokens and read cleanly 7 of 7. The live 0.58.x errors were on the same small screenshot. The tool-call preference is
the model choosing Hermes's vision tool for an image it is given a path to; it is not affected by the arm at this sample size.

**Next lever (not built).** A knob that raises the minimum pixel count before the tower (upscale small screenshots to about 900 px
on the long side, about 500 image tokens, under the 1,024 cap): output-changing for images only, priced by this same replay on C1/C2
(errors and tool-call rate) against the shipped preprocessing.

**Kind.** Research: predicted, measured and explained as hypothesis (O-1 held, O-2 missed, O-3 falsified, O-4 not testable). No
default changed.
