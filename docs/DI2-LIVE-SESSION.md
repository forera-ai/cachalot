# DI-2 keep 140 on by default: live Hermes sessions (2026-11-02, 0.75.0)

HANDOFF section 18.254. The open item after 18.253 was a live Hermes session with the new `serve.sh` default (DI-1
encoder-only prefill plus DI-2 keep 140, decode miss budget 0, the 1017 checkpoint with its hotlist, 48 GiB). Hamed ran
Hermes Desktop against

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && CACHALOT_SERVER_DUMP=$HOME/cachalot-dump-di2.jsonl ./serve.sh
```

three times on the same day. The script is the one used for the old checkpoint (18.239) and 1017's first session (18.243,
`docs/V41-1017-LIVE-SESSION.md`): greeting, list the Desktop with a tool, C# JSON/CSV importer, 200-word story, read an
image, the importer in TypeScript and Python, 500-word story, translate it to German, French and Farsi, then a hummingbird
rewrite with the three translations.

## Sources

- The server's `[request]` lines (terminal) and the request dump `~/cachalot-dump-di2.jsonl`, replies decoded with the
  checkpoint's tokenizer. The dump and the session export hold personal file names and are not in the repository.
- Hermes's export of run 3 (`list-files-on-desktop-7-local-92.json`, 26 messages) and `~/.hermes/logs/errors.log`.
- The 2026-11-01 dump `~/cachalot-dump-1017.jsonl` (0.72.x, DI-1 on, DI-2 not built) as the comparison for run 1's defect.

## Run 1 (server 13878): cold start, a defect that is not DI-2's

Hermes had its MiniMax entry selected while pointed at port 8011: requests carried `model: minimax-m3`, temperature 0.7,
and the system prompt said `Model: minimax-m3`. The answers came from DeepSeek.

- Cold 15,861-token system block: **91.20 s** (the 0.74.1 A/B measured 90.06 s on the 16,040-token block). The snapshots
  on disk were keyed for the exact tail, so this was the expected one cold prefill after the re-key.
- Follow-up prefills on a reused prefix: 11,326 tokens in 53.07 s, 6,620 in 33.65 s (about 200 tokens a second).
  Decode 9.8-11.6 tok/s, 84-99 % hits.
- **Non-JSON tool arguments, three times.** `search_files`'s integer `limit` was written with `string="false"` as
  `strongest`, `COPIOUS`, `collapsed`; Hermes logged "Unrepairable tool_call arguments ... replaced with empty object" and
  resent the same request with `max_tokens` unset, 8192, 32768 and 65536. The server refused the last
  (`Prompt + requested generation exceeds runtime max_seq_len=65536: prompt=15889, max_new_tokens=65536`); the next
  attempt without `max_tokens` wrote `200`. About 30 s and one server error lost.
- **Not caused by DI-2.** The 2026-11-01 dump, before DI-2 existed, has the same pattern: 3 of 5 `limit` values non-JSON
  (`collapsed`, `collapsed`, `FB_MAX`) and the same escalating retries (18.243). It is the decode miss budget 0 defect
  measured there (non-JSON arguments 2/10 at budget 0, 0/10 exact, p = 0.047), which Hamed kept.

Mid-session Hamed switched Hermes to `deepseek-v4.1-flash`. Hermes then rewrote the system prompt (the `Model:` line and an
added "Tool-use enforcement" section), so the whole ~34k-token conversation had to prefill from the start; Hamed stopped
the server with Ctrl-C during that prefill. A Ctrl-C during a prefill prints a traceback ending in
`RuntimeError: cannot schedule new futures after shutdown` (the read-ahead pool is shut down under the running request);
it is shutdown noise, not a crash. The same happened once more on a second short start (server 21259). The C# reply that
appears in Hermes's history for that turn was not produced by this server (the dump has no completed C# request).

## Run 3 (server 25576): the clean session

`model: deepseek-v4.1-flash`, temperature 0.7, `reasoning_effort: none`. 13 requests, 26 messages, no retries, no server
error.

| | this session (0.75.0, keep 140) | 1017 first session (0.72.0, 18.243) |
|---|---|---|
| system block | 16,036 tokens reused from disk, prefill 0.87 s | cold 16,040 tokens, 104.10 s |
| follow-up prefills | 6,624 tokens 33.35 s; 2,292 24.19 s; image turn 647 13.15 s; others 4-242 tokens 0.87-9.05 s | 6,417 44.2 s; 2,276 25.6 s |
| decode, replies of 66-3,022 tokens at 16-37k context | 9.43-11.12 tok/s (the first, 9-token reply 6.92 with a cold cache) | 10.07-11.21 tok/s |
| hits on those replies | 91-98 %, 4.8-20.4 misses a token (first reply 77 %, 47.7) | 93-98 %, 4.9-14.6 |

Measured in one session each, sampled, not an A/B. The 16,036-token snapshot was saved by run 1's interrupted prefill
when it crossed the system-block boundary, so this start reused the DeepSeek system block. The follow-up prefills are
shorter than 18.243's at the same sizes (33.4 against 44.2 s for about 6.5k tokens, 24.2 against 25.6 s for about 2.3k),
in line with DI-2's fixed saving on multi-call prefills (18.252), but the two sessions differ in context and cache
state. The wired governor shrank the expert slots to 4,924-5,178 when system wired memory reached 73.0-74.1 GiB with
Hermes Desktop open, and grew them back.

Answers:

- **Tool calls:** all well formed. `search_files` with `limit` 50 (one integer argument: not evidence that the budget 0
  defect is gone), `terminal` `ls -la` after the model noticed the search had recursed into subfolders, `vision_analyze`
  on the attached image (Hermes answers that the image is already attached; one wasted request of about 18 s, Hermes's
  behavior), and Hermes's background `memory` call with a JSON array in a `string="false"` parameter.
- **Image:** transcription exact (label, input value with the cursor, five suggestions with subtitles, the highlighted
  row, the disabled button).
- **C#:** builds in a .NET 8 class library with implicit usings off (it brings its own `using` lines). Two comments are
  wrong: `StreamJsonArray` says it avoids holding the whole file, but `JsonDocument.Parse` does; `LoadCsv` says it maps
  by column name, but its mapper takes the row by position.
- **TypeScript:** passes Node 22's type-stripping syntax check (`tsc` not installed). **Python:** parses.
- **Stories:** 237 and 453 words for 200 and 500 asked. The hummingbird rewrite keeps the fish story's structure
  line for line. German, French and Farsi translations complete on both turns; the Farsi is fluent with a few literal
  phrases and no English left in it (18.243 had one English word mid-sentence).
- Ctrl-C at the end waited for the request in flight (the `memory` call) and shut down without a traceback; Hermes's
  background review thread then retried against the stopped server, as expected.

## What this settles and what it does not

- DI-2 keep 140 ran a full live Hermes session with no defect attributable to it; the one cold block matched the A/B,
  and reused-prefix prefills and decode are where they were.
- The budget 0 non-JSON integer argument is still live on 1017 (3 of 4 `limit` values in run 1, 0 of 1 in run 3) and costs
  Hermes up to four retries and a refused 65,536-token request each time. Two fixes were offered to Hamed and are not
  built: exact decode inside DSML tool-call blocks (output-changing, needs a replay A/B), and fitting an explicit
  `max_tokens` to the room left, as the server already does when the client sends none.
- Not exercised: DI-2's continuation case in isolation, a cold block with the DeepSeek system prompt on 0.75.0 (only the
  MiniMax-labelled one was cold), thinking mode.
