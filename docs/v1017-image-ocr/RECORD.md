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

## 5. Results

Not run yet.
