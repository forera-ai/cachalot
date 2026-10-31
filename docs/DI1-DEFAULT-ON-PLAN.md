# DI-1 default-on in `serve.sh`: plan and predictions (before any build or run)

Written 2026-10-31 against runtime 0.71.3, **before any code is changed and before any run**. Hamed chose "Yes, plan default-on" after the server A/B (`docs/DI1-SERVER-AB-RESULTS.md`, §18.241). Nothing is flipped until he approves this plan and its snapshot choice (section 2). Results will go in `docs/DI1-DEFAULT-ON-RESULTS.md`.

## 1. Evidence so far (recorded)

* Speed: bench -39 % of a 22k cold prefill (§18.238); live session 102 s for a cold 16,040-token block (§18.239); server A/B, same build, ABBA, 0.668 of the knob-off time over three large prefills, reuse unchanged (§18.241).
* Quality: long-context gate inside noise, KL 0.0014 against re-chunking's 0.0101 (§18.238); live answers fine (§18.239); first greedy token equal on three of four replayed requests, the fourth split by run order (§18.241).
* Scope of the knob (`generation.prepare_prompt`): text prompts of more than one prefill call only (more than 4,096 new tokens); image prompts and single-call prompts take today's path unchanged. DeepSeek only (`finish_decoder`); GLM and MiniMax are untouched.

## 2. The snapshot question (Hamed's choice)

With the knob on, `prepare_prompt` snapshots only at the requested boundaries (the system block's end) and at the prompt's end. Each of those runs the decoder half once, so the snapshot is complete. Two things follow.

**2a. Mixing knob-on and knob-off snapshots.** The disk snapshots are keyed on `runtime_identity(..., NUMERICS_VERSION)`. Today a knob-on snapshot and a knob-off one for the same tokens share the key, so either mode restores the other's. The two states differ in rounding only (layers 21 to 39 windows), but the repository's rule (`snapshot_store.py`: the tag covers "prefill's KV bits: kernels, quantization, attention, Engram, the tokenizer path, chunk sizes") keys every change that moves prefill KV bits, so a restored snapshot always equals what this configuration would compute cold. GLM already keys its chunk size this way.

| option | what it does | cost | recommendation |
|---|---|---|---|
| **M1: key the mode** | the identity's version string becomes `NUMERICS_VERSION + "-di1"` when the knob is on, so the two modes keep separate snapshot sets on disk | the first request after the flip prefills each system block once more, cold, about 108 s for Hermes's 16k block; switching back finds the old set intact | **recommended**: keeps the rule that a restored snapshot equals a cold prefill of the same configuration; no numerics bump for the knob-off path |
| M0: no marking | as today | none | breaks that rule silently; a later quality question could not tell which mode a reused state came from |

**2b. The in-block chunk pins (§18.65) are not taken with the knob on.** With the knob off, every 4,096-token boundary inside the system block is pinned (and persisted, up to 8,192 tokens), so a block changed after token 4,096 (a new model name or provider) still reuses 4,096 tokens. With the knob on, such a change re-prefills the whole block. Priced on the A/B's rates (derived): a changed 16,040-token block costs about 108 s with the knob on and no pin, against about 163 - 41 = 122 s with the knob off and the 4,096 pin. The knob still wins without the pin. Taking pins in DI-1 mode would cost a decoder pass of up to 2,541 rows each (about 12 s, derived) on every cold block, to save about 28 s only when the block changes. **Recommendation: leave the pins off in DI-1 mode** (today's DI-1 behaviour); not built.

## 3. Build (after approval)

1. `serve.sh`: `export CACHALOT_PREFILL_ENCODER_ONLY=${CACHALOT_PREFILL_ENCODER_ONLY-1}` with a comment pointing at §18.238, §18.241 and this plan; `=0` is the old path. **`chat.sh` stays off**, as it does for the decode miss budget: it is the exact configuration handed to others. The code default in `generation.py` stays `0`, so benchmarks and in-process runs are unchanged.
2. `cli.py` serve: one startup line, like the miss budget's: `DI-1 encoder-only prefill on: ... (changes outputs in rounding; CACHALOT_PREFILL_ENCODER_ONLY=0 is the full prefill)`.
3. M1 (if chosen): `_attach_snapshot_store` passes `NUMERICS_VERSION + "-di1"` when `encoder_only_enabled()`; tests that the two identities differ and the knob-off one is unchanged.
4. Tests: the `serve.sh` header resolves `CACHALOT_PREFILL_ENCODER_ONLY=1` by default and keeps an explicit `0` (by sourcing the header, as 0.71.2 checked its bank defaults); the identity tests.
5. Lab brief for the default change; CHANGELOG, README, HANDOFF, prompt, memory; version **0.72.0** (a default change).

Estimate: build and tests about 25 minutes.

## 4. Verification run (after the build; estimate and confirm first)

One `serve.sh` process with no knob in the environment and a new empty snapshot directory, then a restart:

1. Request 1 of `~/cachalot-di1-live.jsonl` with `max_tokens` 1: a cold prefill in DI-1 mode.
2. Stop and restart the server on the same directory; request 1 again: it should reuse the system block's boundary snapshot from disk.
3. Stop and restart with `CACHALOT_PREFILL_ENCODER_ONLY=0` on the same directory; request 1 again: under M1 it should find no snapshot of its mode and prefill cold.

Estimate (derived): 108 + 163 s of cold prefill, three loads of about 20 s, a few seconds of reuse; **about 7 minutes**.

## 5. Predictions

All **predicted**; probabilities mine.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| D-1 | startup without the knob in the environment | the DI-1 startup line is printed and request 1 prefills cold in 100 to 115 s (0.80) | over 140 s (the knob did not take effect) |
| D-2 | restart, same directory | request 1 reuses at least 16,000 of 16,040 tokens and prefills in under 5 s (0.85) | it prefills more than 4,096 tokens (the DI-1 boundary snapshot did not persist or did not load) |
| D-3 | restart with `=0` (M1 only) | request 1 reuses 0 tokens and prefills in 150 to 175 s (0.85) | it reuses the DI-1 snapshot (the key did not separate the modes) |
| D-4 | the one greedy token | equal on all three runs ("Hey", as in every A/B arm) (0.85) | it differs |
| D-5 | tests | every test in `tests/` that passed before passes after (0.95) | any regression |

## 6. What is not covered

The compression re-prefill in a live session (it is a long new prompt, so the knob applies, but no live session has had one with the knob on); the knob-off path's in-block pin behaviour stays as it is.
