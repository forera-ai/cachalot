# DI-4 priced: 128-token-grain prefix reuse on a real Hermes session with four compressions (a small lever)

Run 2026-10-21 with `benchmarks/prefix_grain_replay.py` (0.66.62) on Hamed's recording `benchmarks/results/hermes-live-di4/requests.jsonl` (57 MB, local and gitignored; 142 requests, 141 replies; DeepSeek V4.1 Flash via `./serve.sh`, internal bank, `CACHALOT_SERVER_DUMP`; prompts from 468 to 64,821 tokens; four compression events). Replay through the real `prepare_prompt` and `PrefixCache` with no model, 1.5 GiB cache. Scored output: `docs/di4-grain-2026-10-21.json`. Every number is **measured** by that replay unless marked **derived**. The prediction was the lane file's draft (section DI-4): "at least a quarter of mid-session requests diverge inside a chunk, recovering a median of 1,000+ tokens; falsified if 80 % or more diverge exactly at a snapshot point or at the end of the prompt". It was written before any data existed that could test it.

## 1. Result

| | tokens | |
|---|---|---|
| prefilled by the cache as it ran | 279,056 | |
| prefilled with a 128-grain reuse (largest multiple of 128 in the shared prefix, minus one 128 replay window) | 259,772 | **-19,284 (-6.9 %)** |
| requests that diverged inside a chunk (grain hit above the cache's reuse) | 10 of 142 (7 %) | median saving 256 tokens |
| requests where the cache already reused the whole shared prefix (within one token) | 102 of 142 (72 %) | |
| the rest | 30 | shared prefix shorter than the cache's snapshot grid allows, or a cold prompt |

The ten requests: four tiny side requests (468 and 319 tokens, 128 to 256 tokens each), one small one (1,217 of 1,615), a 3,456-token saving on a 7,009-token summary call (#65), 468 on a 53,647-token turn (#98), 191 on #107, and **#106: 12,928 tokens of a 17,049-token summary call** (shared 13,156 tokens with an earlier prompt; the cache reused 0). Request #106 alone is 67 % of the saving; without it the saving is **6,356 tokens (2.3 %)**.

## 2. Scores and what it means

* Registered prediction: **missed** (7 % diverged against a quarter; median 256 against 1,000+). The falsifier (80 % or more diverging exactly at a snapshot point or the prompt's end) did not fire (72 % already reused the shared prefix) but the lever's expected size is about a fifteenth of the draft's.
* **Derived:** at the recorded 100 tok/s a 19,284-token saving is about 190 s over a session that spent about 2,800 s prefilling; concentrated in one event, and most of that event is a summary call.
* **Where the prefill goes (measured, the four compression re-prefills and the seven summary calls):** after a compression the history is rewritten, so the shared prefix is only the system block (17,217 tokens) and the next turn re-prefills 26,650, 17,735, 16,737 and 16,642 tokens (**77,764 tokens, 27.9 % of all prefill**); the auxiliary summary calls prefill another **79,403 (28.5 %)**, because the summary prompt is a different prompt with nothing to share. A finer reuse grid cannot reach either. These two together are 56 % of the session's prefill.
* **DI-4 as a standalone lever: not worth building** at 2.3 to 6.9 % with one event dominating; its machinery (store at multiples of 128, replay the last window) is the same as DI-3's snapshot policy, so it stays a design option there, not a lever. Closed as standalone; the numbers are above.
* **For DI-3 (the snapshot policy):** reuse stopped strictly inside the 17,217-token system block in 25 requests, all at 191 tokens (a shared preamble of side requests); reuse at exactly the system block 4 times. So **in-block intermediate snapshots of the 4,096 grid were not what served this session**; the pinned end-of-block snapshot was. That supports dropping unpinned intermediate in-block snapshots in a cold prefill (DI-1's policy (a)) on this evidence, to be confirmed on a second session.

## 3. Limits

One session, one user, one task mix (a story-writing session with large tool results); the cache budget was 1.5 GiB; the replay credits a grain hit by token equality and the 128 replay window and does not model the exact SWA rebuild or its quality; images are rendered without their rows as on the server; the shared prefix is searched among earlier prompts followed by their own replies. The lever's output risk (DI-2's bounded replay) was not priced.

## 4. Open

(1) What would shrink the post-compression re-prefill (56 % with the summary calls): a Hermes-side matter (summary placement), or a server-side policy for the history rewrite, Hamed's call; (2) the DI-1 exact tail helps every cold block but not these re-prefills of rewritten histories beyond the encoder-only saving per chunk (the tail rule needs the same chunks, so it applies to them too: derived -40 % of their layer time); (3) a second session to confirm the in-block snapshot finding.
