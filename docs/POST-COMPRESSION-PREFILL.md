# What a Hermes context compression costs on the server (measured on the 142-request session of 2026-10-21) and what can be done about it

From the recording `benchmarks/results/hermes-live-di4/requests.jsonl` (request and reply timestamps, token counts) and the read-only parts of Hamed's `~/.hermes/config.yaml` (nothing was edited). Every number is **measured** unless marked derived.

## 1. Cost

| | requests | time on the server | share of the session's 10,759 s of request time |
|---|---|---|---|
| auxiliary summary calls (a 1-message request, 4.4k to 17.0k prompt tokens, 1.8k to 4.3k decoded) | 6 (244, 411, 300, 523, 600, 600 s) | 2,678 s | **24.9 %** |
| the first main request after each compression (a 34k to 44k token prompt, 17.2k reused) | 4 (187, 303, 217, 301 s) | 1,007 s | 9.4 % |
| the other 131 requests | 131 (median 32 s) | 7,073 s | 65.7 % |

Compression therefore takes **34 % of the server's time** in this session; the summary calls are decode-bound (two hit 600 s), the follow-up requests prefill-bound. The summaries were used: each is a 9k to 17k character message in the next history.

## 2. Why the server cannot reuse anything

After a compression Hermes rewrites the history: the shared prefix with the previous request is the system block (17,217 tokens) and nothing after it. By the char share of each rewritten prompt (derived, rough), the re-prefilled part is the summary (about 1.8k to 6.5k tokens) plus the **protected tail of 24 to 61 messages (about 19k to 31k tokens)**, which Hermes keeps verbatim but which sits at new positions, so its old KV cannot be reused. DI-4 showed a finer reuse grid does not reach any of this (`docs/DI4-GRAIN-RESULTS.md`).

## 3. Options, all Hermes-side except the last (Hamed's calls; nothing was changed)

1. **A hosted `auxiliary.compression` model.** His config has `auxiliary.compression: provider: auto, timeout: 120`, so the summary runs on this server (`auto` is the main model). A hosted model removes the six summary calls (about 2,678 s, 24.9 % of this session's server time, derived as removed outright) and the queueing behind them; it sends the session's text to a third party, which is his decision. The earlier advice is the same (`docs/manual-tests/hermes-desktop.md` section 5).
2. **A smaller `protect_last_n` (his value is 30).** The protected tail is the bulk of the post-compression prefill; ten messages instead of thirty would cut that re-prefill by roughly two thirds (derived from the tail share above) at the cost of less verbatim recent context. A quality trade, not measured.
3. **DI-1's exact-tail prefill (built, off, `docs/DI1-BUILD-RESULTS.md`)** shortens exactly these re-prefills: each is a 5 to 7 call prefill, so the decoder half runs once instead of 5 to 7 times: about 37 to 40 % of their layer time (derived from the 46.5 % decoder share), about 300 s of the 819 s the four spent prefilling at 95 tok/s (derived). It does not touch the summary calls, which are decode-bound.
4. A server policy for the rewritten history (for example, holding the summary and tail in a snapshot) has no safe form: the tail's positions change, so its KV cannot be moved.

## 4. Limits

One session and one user's task mix; the char-based token split is rough; latencies are server timestamps (a request's reply time minus its request time), so they include queueing; the hosted-model option is unmeasured here.
