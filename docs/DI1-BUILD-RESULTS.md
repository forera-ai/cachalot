# DI-1 build: results of the first equality runs (an encoder-only prefill with an exact decoder tail ends in a state as close to the full prefill as a re-chunking does)

Run 2026-10-21 under `docs/DI1-BUILD-RECORD.md` (eight predictions E-1 to E-8 committed in e1b19c3 before any run, control N-1 to N-3 committed in dfb7615 before it ran). Runtime 0.66.63 plus this build (`CACHALOT_PREFILL_ENCODER_ONLY`, default off). Prompt: the 6,100-token `A_code_review` prompt, two calls (4,096 + 2,004), then 16 greedy tokens. One process per arm under `guarded_run.sh` (34 GiB budget, the server's environment, the internal 2-bit bank; swap 3.5 to 3.6 GB was already in use and did not grow). Raw: `docs/di1-eq-D-vs-F-2026-10-21.json`, `docs/di1-eq-N-vs-F-2026-10-21.json`. Every number is **measured** unless marked derived.

## 1. Arms

| arm | code | mode | process wall | peak wired |
|---|---|---|---|---|
| B | the pre-change code (a worktree of 7dba490) | full | 88 s | 60.7 GiB |
| F | the changed code | full | 89 s | 60.5 GiB |
| D | the changed code | deferred (encoder-only, `finish_decoder`) | 75 s | 60.1 GiB |
| N | the changed code | full, calls of 3,050 + 3,050 (the noise control) | 94 s | 59.7 GiB |

## 2. Scores

| id | result | score |
|---|---|---|
| E-1 | F against B: logits, every window ring, the layer-20 caches and the 16 decoded tokens bit-identical | **held** (the refactor, which moves the layer loop body verbatim, is a no-op) |
| E-2 | D against F: windows of layers 0 to 19 and the compressed caches of layers 2, 8, 14 bit-identical | **held** |
| E-3 | D against F: same top-1; KL(F to D) 0.0359 (logits mean absolute difference 0.49) | **missed** (interval below 0.001), not past the falsifier (0.05) |
| E-4 | decoder windows of layers 21 to 39: relative max 0.10 to 0.45 (layer 20's window is bit-identical) | **falsified** as registered (0.2) |
| E-5 | layer 20's compressed KV and index keys, and the candidates and top-k, **bit-identical** | **held, beaten** (predicted within 0.01) |
| E-6 | the 16 greedy tokens equal; per-token decode KL max 0.0079 | **held** |
| E-7 | the prefill wall | **not measured as registered** (the script did not time the prefill): the whole process took 75 s against 89 s, which with the same load and decode implies prefill + decode of about 0.80 (derived) |
| E-8 | peak wired 60.1 GiB against 60.5 | **held** |
| N-1 | the control N against F: decoder-window relative max 0.125 to 0.757 (at least 0.1) and logits KL 0.0203 (0.003 to 0.1) | **held** |
| N-2 | D against N per decoder layer: D's relative max is 0.31 to 0.82 of N's, below N's at all 19 layers (3 N at 19 of 19; within a factor of 3 either way at 17 of 19) | **held** |
| N-3 | N's 16 greedy tokens equal F's; N's decode KL max 0.0462 | **held** |

## 3. What it says

**Measured.** (1) The refactor is bit-identical (B = F). (2) Everything the encoder produces, and everything layer 20 publishes for later rows (compressed KV, index keys, candidates, top-k), is bit-identical between the full and the deferred prefill, so the tail pass sees exactly the global state a full prefill leaves. (3) The decoder windows and the final logits differ, by more than my intervals assumed (E-3 and E-4 missed or fell). (4) **A plain re-chunking of the same full prefill differs from F by more**: window relative max up to 0.76 (D: 0.45), logits KL 0.020 (D: 0.036, one position), decode KL max 0.046 (D: 0.0079), and D's difference is smaller than N's at every decoder layer. Both arms keep the same top-1 and the same 16 tokens.

**Reading (hypothesis).** The decoder-window differences are the rounding of regrouping the MoE and the batched attention (the lane's Q-NOISE band, KL mean 0.0144 for a chunking change), amplified by routing near-ties; the tail pass is not noisier than the change of chunk size the runtime already treats as noise. My E-3 and E-4 intervals assumed an "exact in exact arithmetic" shortcut would land close to bit-identity; the control shows the right reference is the band, and I had registered no band. Not shown: that no *systematic* error hides inside the same noise (one prompt, 16 tokens, one re-chunking as the reference); that is what the quality arms are for.

## 4. Limits and what is not done

One prompt of 6,100 tokens (the saving at 22k tokens, the case that matters, was not run: derived 39 to 42 % of layer time, `docs/DI1-SPLIT-RESULTS.md`); the quality gate (`pareto.py`: paired NLL with a block bootstrap, the 21-task battery) was not run; snapshots made with the knob on differ in rounding from knob-off ones and are not marked (a snapshot is reused across modes); text only, DeepSeek only, no image or MTP capture; the saved tail is tied to the 4,096-token call and the 34 GiB budget used here. **The default stays off.**

## 5. Next (needs Hamed's confirmation of each long run)

(1) A 22,000-token timing run, full against deferred, one process each (about 8 to 10 minutes, `di1_tail_equality.py` now records the prefill time); (2) the `pareto.py` gate against the noise arm for the knob (output-changing: Hamed's call on any default); (3) a live Hermes session with the knob on (`CACHALOT_PREFILL_ENCODER_ONLY=1 ./serve.sh`) reading `[request]` prefill times.
