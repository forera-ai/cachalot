# Lever F: Apple's system model as a speculative-decoding drafter for DeepSeek or GLM (feasibility report)

Written 2026-10-29 against runtime 0.69.0. **Recommendation: no-go.** Nothing was implemented; this report prices
the idea from measured constants and says what would have to change for it to reopen. Labels follow the charter:
measured, derived, hypothesis.

## 1. The four things that are not interchangeable

| technique | what it needs | output guarantee |
|---|---|---|
| Lossless speculative sampling (Leviathan et al., Chen et al.) | the drafter's probability q(x) for every drafted token, the target's p(x), an accept/reject with probability min(1, p/q) and a residual resample | exactly the target's sampling distribution, at any temperature |
| Greedy candidate verification | drafted tokens in the target's vocabulary, one multi-position target forward, accept the longest prefix equal to the target's argmax | identical to the target's greedy output (temperature 0 only) |
| Text-level speculative assistance | drafted *text*, re-tokenized with the target tokenizer, then greedy verification as above | identical to greedy output, if verification is done right; the re-tokenization boundary must be handled |
| Approximate decoding / prompt-level lookahead | none of the above | changes the output; not speculative decoding |

## 2. What the target runtime has (measured or read in the source)

* **Target tokenizer, incremental decode, full-vocabulary logits:** yes. `cachalot.model.generation.sample_logits` takes
  1-D vocabulary logits; decode is token by token.
* **A multi-position forward:** only through the prefill path. `benchmarks/verify_forward_cost.py` on the 2-bit bank
  measured **243 ms + 27 ms per extra position** (HANDOFF section 9.20, 2026-09-21; the constants are dated and may have
  moved with the 0.6x releases, but no faster multi-position path has been built since).
* **Verify reads more experts:** routing replays give 515 MiB per accepted token at width 1, 600 at width 2, 876 at
  width 5 (HANDOFF section 9.20, `speculation_bytes_q2_44.json`): a verify of K positions routes each position to its
  own experts, so on an SSD-streamed MoE the drive cost grows with K.
* **KV checkpoint and rollback:** prefix snapshots exist (in memory and on disk, chunk-granular); a per-token rollback of
  a rejected suffix in the decode caches is **not established** by this report (not read in the source).
* **The model's own drafter was already closed:** DeepSeek's DSpark head (a drafter trained against the target, five
  tokens a forward) measured **1.03x against its own 1.15x bar** on FP4 constants (HANDOFF section 9.1), and the
  re-replay at 9.49 MiB experts did not reopen it (section 9.20). GLM's MTP closed on arithmetic (a two-token verify
  reads ~1.75x the experts, section 18.40). MiniMax: misses scale exactly with the verify width.

## 3. What the system model exposes (apple-fm-sdk 0.2.1, checked against the installed package)

* Text in, text out; JSON-schema guided generation; `token_count` and `context_size` (8,192).
* **No logits, no token probabilities, no tokenizer, no token ids, no KV cache, no session state you can fork.**
  `Capabilities.logits`, `tokenizer_access` and `kv_access` are all false. Apple's internal draft asset (if any) is not
  reachable through the public API, and this project does not use private frameworks.
* Measured on this Mac (docs/APPLE-FM-RESULTS.md): 0.33 s for a short reply, first streamed delta 0.89 s, 51.7 tok/s
  generation, 1,019 tok/s prompt processing. Two requests at once are served one after the other.

## 4. Consequences

1. **Lossless speculative sampling is impossible** with this drafter at any temperature above 0: the acceptance test
   needs q(x), and the API does not return it. Claiming lossless sampling here would be wrong.
2. **Greedy text-level verification is possible in principle** (temperature 0 only): draft text, re-tokenize it with
   DeepSeek's tokenizer, verify K positions in one forward, accept the matching prefix. The output would equal the
   target's greedy output. It cannot serve Hermes, whose requests sample (DeepSeek's chat default, MiniMax at 0.7).
3. **Context mismatch.** The drafter holds 8,192 tokens; Cachalot's agent contexts are 22-30k. The drafter would see a
   truncated context and re-process it on every draft call (no session fork), which lowers acceptance and costs time.

## 5. The arithmetic (derived)

DeepSeek's decode token is 80 ms all-resident through the server (0.60.10) and about 100 ms in a live 22-27k session
(0.61.2). One draft-and-verify round of K drafted tokens costs at least

    verify(K) = 243 + 27 (K - 1) ms         (measured constant, section 9.20)
    draft(K)  = t_first + K / 51.7 s        (t_first 0.33-0.89 s measured; the drafter re-reads its context each call)

and yields a + 1 tokens (a accepted plus the target's own next token). Against plain decode it wins only if
draft + verify < (a + 1) x t_token.

| K | verify | draft (t_first 0.33 s) | needed a at 100 ms a token | needed a at 80 ms |
|---:|---:|---:|---:|---:|
| 4 | 324 ms | 407 ms | 6.3 (> K) | 8.1 (> K) |
| 8 | 432 ms | 485 ms | 8.2 (> K) | 10.5 (> K) |
| 16 | 648 ms | 640 ms | 11.9 of 16 | 15.1 of 16 |

Even with the draft fully hidden on the Neural Engine while the GPU decodes (the best case of lever E), the verify alone
needs a + 1 >= verify / t_token, so a >= 2.2 of 4, 3.3 of 8 and 5.5 of 16 drafted tokens accepted as a matching
*prefix* at 100 ms a token (a prefix: the first mismatch ends the round). A 3B model of another family, reading a
truncated context, matching DeepSeek's greedy continuation for that long on average is not plausible (hypothesis;
DeepSeek's own trained drafter did not reach a paying acceptance), and every verify width also multiplies the expert
reads (section 2), which this table does not even charge.

## 6. What would reopen it

* A multi-position forward that costs close to one decode token (not the prefill path's 243 ms), measured; and
* a drafter that exposes token probabilities in the target's vocabulary (a local MLX model, not the system model); or
* a published API in a later SDK exposing per-token log-probabilities, which would still leave the tokenizer mismatch.

**Proposed experiment if those change:** greedy, 64 prompts of W1 and W4, draft K = 4 and 8 tokens with the drafter,
re-tokenize, score the per-position agreement with the target's recorded greedy continuation offline (no runtime run),
then price with the measured verify cost. Until then: **no-go, nothing built, disabled by construction** (the lever
registry entry `F` has status `no-go` and no implementation).
