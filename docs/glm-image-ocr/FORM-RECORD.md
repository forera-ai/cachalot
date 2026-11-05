# GLM on the address form: padded against native, budget 2 against exact

Written 2026-11-05 against runtime 0.80.2, before the run. Hamed: "Yes, ~2 hours" (v314's first option). HANDOFF section 18.269 when scored.

## 1. Baseline (recorded)

`RECORD.md`, 18.268, same 40 GiB and budget 2: the 492 x 512 form had an error in 3 of 4 replies (one or two wrong letters in the same
address line), the 906 x 943 copy in 1 of 3. Padding ran only on the card, which had no errors, and exact decode did not run.

## 2. Hypothesis

More image tokens help GLM on the form, and budget 2 does not cause the errors. Alternative one: padding adds no information (GLM already keeps
the native pixels at 342 tokens), so neither image differs. Alternative two: budget 2's dropped experts cause the letter errors.

## 3. Design and method

Two fresh `./serve-glm.sh --expert-budget-gib 40` servers under `guarded_run.sh --force --max-seconds 5400`: arm A shipped (budget 2), arm B
exact (`CACHALOT_GLM_DECODE_MISS_BUDGET=off`), settle between. `driver-form.py 5`: five rounds over A492 (342 image tokens) and A886 (padded to 886 x
922, 1,056 tokens, derived), 20 replies, otherwise as `RECORD.md`. Scored with the image-ocr scorer and private truth; `analyse-form.py`
prints ids and counts.

Limits, written first: one screenshot; five replies a cell; the arms run in sequence, not interleaved; prefill of the two images is exact in both arms;
no Hermes context.

## 4. Predictions (registered before the run)

- **H-1.** 20 replies, no error. Prompt difference A886 - A492 is 714 tokens within 4, in both arms; nothing stops at the guard's limit.
- **H-2.** Budget 2, A492: 2 to 4 of 5 with an error.
- **H-3.** Budget 2, A886: at least 2 fewer than A492.
- **H-4.** Exact, A492: within 1 of budget 2's A492.
- **H-5.** Exact, A886: at most 1 of 5.

**Decision rule, on pooled counts.** (a) A886 at least 4 of 10 fewer than A492: a GLM upscale knob is a candidate, a build is Hamed's go. (b) Exact
at least 4 of 10 fewer than budget 2: budget 2 is the cause on images; exact image turns on GLM are a candidate, Hamed's call. (c) Neither: no
knob, and the GLM screenshot question closes as the model's own at this checkpoint.

Runtime estimate: about 2 hours (arm A about 45 minutes, a settle, arm B about 65 minutes at 190-200 tokens a reply, 0.9 and about 1.6 s a token).

## 5. Results

Not yet run.

## 6. Kind

Not yet scored.
