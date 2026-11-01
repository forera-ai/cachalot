# DeepSeek-V4.1-Flash-1017: first live Hermes session (2026-11-01)

Runbook step 6 (`docs/V41-1017-RUNBOOK.md`). Hamed ran one Hermes Desktop session on 0.72.0 with

```bash
CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-1017 CACHALOT_SNAPSHOT_DIR=$HOME/.cache/cachalot/prefix-snapshots-1017 CACHALOT_SERVER_DUMP=$HOME/cachalot-dump-1017.jsonl ./serve.sh
```

The script was the one used in the old checkpoint's live session of 2026-10-31 (HANDOFF 18.239): greeting, list the Desktop
with a tool, C# JSON/CSV importer, 200-word story, read an image, the importer in TypeScript and Python, 500-word story,
translate it to German, French and Farsi, then a hummingbird rewrite with the three translations.

## Sources

- The server's `[request]` lines (terminal) and the request dump `~/cachalot-dump-1017.jsonl` (19 requests, 18 replies),
  replies decoded with the checkpoint's tokenizer.
- Hermes's `~/.hermes/logs/errors.log` for the client side of the failures.
- The export Hamed attached, `new-list-files-on-desktop-folder-3-local-de.json`, is **not** this session: its messages
  are timestamped 2026-10-31 11:51-12:17 and its image is `dashboard_20261031_120007`, while this session's dump has
  `dashboard_20261101_022159`. It is the old checkpoint's session (18.239) and is used here only as the comparison.

## Setup facts

- Bank: `/Users/hamedprooshani/DeepSeek-V4.1-Flash-1017-q2g128`, built 2026-11-01 in 35 min (12 sampled experts byte
  for byte), internal SSD. No hotlist. Decode miss budget 0, DI-1 on (the 0.72.0 `serve.sh` defaults). Hermes sent
  temperature 0.7, `reasoning_effort: none`.
- The old internal bank was deleted after a byte-for-byte compare with its X10Pro copy; the old model now reads its bank
  from the X10Pro.

## Speed (measured, one session, sampled: not an A/B)

| | 1017 (this session) | old checkpoint (18.239) |
|---|---|---|
| cold 16,040-token system block | 104.10 s (6.49 ms a token) | 102.45 s (6.39) |
| follow-up prefills | 6,417 tokens 44.2 s; 2,276 25.6 s; others 13-302 tokens 1.6-9.0 s | 11,478 63.4 s; 6,480 44.6 s |
| decode, replies of 154-3,893 tokens at 25-31k context | 10.07-11.21 tok/s | 7.7-10.9 tok/s |
| hit rate on those replies | 93-98 %, 4.9-14.6 misses a token | not recorded here |

The 1017 routing, with no hotlist, kept as many experts resident as the old one with its hotlist. The wired governor
shrank the expert slots 16 times (to 4,901 of 5,178 at the lowest, at shutdown) when system wired memory reached
73.0-74.2 GiB with Hermes Desktop open, and grew them back 7 times when it fell.

## Answers

Good:

- Image transcription exact (label, input value, five suggestions with subtitles, the highlighted row, the disabled
  button), via a well-formed `vision_analyze` call.
- C#: builds and runs in a default `dotnet new console` project (implicit usings on); the CSV parser handles quoted
  commas and doubled quotes. With implicit usings off it fails (`Where` without `using System.Linq`). The usage
  example's `columnNameMapper` (`col.Replace("_", "").ToLowerInvariant() is var _ ? col : col`) returns the column
  unchanged while its comment promises `first_name` to `FirstName`. The old checkpoint's C# had one build error.
- TypeScript: `tsc --strict` clean (TypeScript 7.0.2), runs under Node 22, correct on the quoted-field test.
- Python: runs, correct on the same test.
- Stories 217 and 560 words for 200 and 500 asked (old: 178 and 469). The hummingbird story and its three translations
  are complete (3,893 tokens); German and French read clean; the Farsi leaves one English word mid-sentence
  ("... از یک پرستوی کوچ‌رو heard — ... شنیده بود").

Defects:

1. **Invalid `search_files` arguments, three times.** The tool's `limit` is an integer (default 50). The model wrote it
   with `string="false"` and the values `collapsed`, `collapsed`, `FB_MAX`; the official parser
   (`encoding.py`, `decode_dsml_to_arguments`) puts a `string="false"` value into the JSON as is, so the arguments were
   not JSON. Hermes logged "Unrepairable tool_call arguments for search_files — replaced with empty object" and retried
   with `max_tokens` 8192, then 32768, then 65536; the server refused the last (`prompt=16068, max_new_tokens=65536`
   exceeds `max_seq_len`), and the retry without `max_tokens` wrote `limit` 50. About 28 s lost. The old checkpoint
   wrote `limit` 200 the first time.
2. **The first translation turn returned raw markup.** "translate it to German, French and Farsi" got a 48-token
   `execute_code` call (`from hermes_tools import write_file` / `print("ok")`) with a malformed parameter tag
   (`<｜DSML｜ name="code" ...>`, the word `parameter` missing). The parser rejected it, the server surfaced the block
   as text (by design, `server/engine.py`), and Hermes showed `<｜DSML｜ calls> ...` as the answer. No translation was
   made on that turn; the next turn's request produced one. The old checkpoint translated on the first try.
3. **The Desktop listing's counts are wrong.** "102 entries" is the recursive `search_files` count, not the top level
   (143 visible items); "~80 screenshots" against 108; "252 items" in the receipts folder against 250. The folder names
   are right. The old checkpoint also approximated ("~90 PNG screenshots").
4. **Server error on the first request.** Hermes's first "Hi" failed 90 s into the cold prefill with
   `openai.APIError: 'w1.weight'` (02:08:51) and was retried; the retry prefilled the whole block again (104 s), so the
   greeting took 198 s. The server printed no traceback: `server/app.py`'s worker passes the exception to the client
   only. First occurrence in Hermes's error log.

## Explanation

- Defects 1 and 2 are tool-call format slips in sampled output (T 0.7) under the decode miss budget 0, which drops the
  5-15 non-resident experts a token. This session alone cannot separate the 1017 weights from the drop; the replay
  below does.
  The attribution test is the one used for MiniMax in 18.10: replay the dumped requests (dump lines 3 and 30) several
  times per arm, budget 0 and the exact path (`CACHALOT_DECODE_MISS_BUDGET=off`), at T 0.7, counting malformed calls.
- Defect 4 (derived, not reproduced): an expert record's arrays dict is emptied only when its slot is parked
  (`cache/slots.py`, `park` and `park_slab`), and a `KeyError` on `w1.weight` means a record with an empty dict reached
  an expert forward. The wired governor parked slots during this request (5,178 to 5,032 at 74.2 GiB wired, before
  the first `[request]` line). `ResidentExpertStore.set_capacity` assumes no prefill in flight and every read evaluated;
  something held a resident record across the `_wired_fit(force=True)` that runs before each 4,096-token prefill call.
  Which holder it was is open; the first fix is to print the traceback in the server.

## Attribution replay (2026-11-01, measured)

Hamed asked for it. The two dumped requests (dump lines 3 and 30) posted as Hermes sent them (temperature 0.7, top_p
1.0, `reasoning_effort: none`) to `serve.sh` on 1017, one guarded server process per arm: `CACHALOT_DECODE_MISS_BUDGET=0`
(shipped) and `=off` (exact). `max_tokens` capped at 400 (Desktop) and 500 (translate): a call is under 100 tokens and
the opening decides the class. Replies taken from each arm's request dump and classified with the checkpoint's own
`encoding.parse_message_from_completion_text`. The guard preflight wanted 77 GiB and found 73.5 (Hermes Desktop, Codex
and Claude open), so it ran with `--force`; the guardian never fired. Driver and classifier in the session scratchpad
(`replay1017.py`, `classify1017.py`), about 22 minutes.

| request | arm | n | well-formed call | call with non-JSON arguments | malformed DSML | text | call cut at the cap |
|---|---|---|---|---|---|---|---|
| Desktop (`search_files`) | budget 0 | 10 | 8 | **2** (`limit`: `SETT`, `ible`) | 0 | 0 | 0 |
| Desktop | exact | 10 | **10** (`limit` 100/200 or left out) | 0 | 0 | 0 | 0 |
| translate | budget 0 | 8 | 0 | 0 | 0 | 5 | 3 (`execute_code` writing the translations) |
| translate | exact | 8 | 4 | 0 | 0 | 4 | 0 |

The four exact-arm translate calls: `execute_code` with `print("ok")` ("no tool work needed"), `text_to_speech` of
`noop`, `terminal` `date`, and `execute_code` writing the translations to a scratch folder.

Decode over the 18 replies an arm: budget 0 11.09 tok/s (98 % hits, 4.7 experts dropped a token), exact 9.30 tok/s
(95 % hits, 12.6 misses read a token): the exact path is 16 % slower here.

What it shows:

- **Defect 1 follows the miss budget.** Non-JSON `limit` values: budget 0 had 2 of 10 in the replay and 3 of 4 live
  (5 of 14); the exact path had 0 of 10. A one-sided Fisher exact test on 5/14 against 0/10 gives p = 0.047 (derived,
  small n). The drop leaves `string="false"` integer arguments exposed; the tag structure stays right.
- **Defect 2 is mostly the model, not the drop.** Pointless tool calls before translating (the live `print("ok")`)
  came back on the **exact** path, 3 of 8, once with the same `print("ok")`; budget 0 never did it in 8. The broken tag
  itself (`<｜DSML｜ name=`) did not reproduce in 16 replays on either arm, so its cause is open; one live sample.
- Not covered: the old checkpoint on the same two requests (the live comparison was a single sample), and the full
  length of the cut calls.

## Second live session: exact decode (2026-11-01 12:36-13:04)

Hamed ran the same script again on the same build with `CACHALOT_DECODE_MISS_BUDGET=off` added to the command above. Sources: the
server's `[request]` lines, the request dump (lines 36-61 of `~/cachalot-dump-1017.jsonl`, 13 requests, 13 replies), the Hermes
export `list-desktop-folder-contents-3-local-db.json` (24 messages, 12:36-13:03 local, matches the dump), Hermes's `errors.log`
and `agent.log` (no entries in the window). One sampled session, not an A/B: the tool results and so the later prompts differ
from the first session.

Speed (measured):

| | budget 0 (first session) | exact (this session) |
|---|---|---|
| system block | cold, 104.10 s | snapshot reused (16,036 tokens), 1.06 s |
| decode, the nine replies of 100+ tokens | 9,399 tokens in 868.6 s, 10.82 tok/s (92 ms a token) | 11,203 tokens in 1,307.1 s, 8.57 tok/s (117 ms) |
| per reply | 10.07-11.21 tok/s | 7.68-9.87 tok/s |
| misses a token on those replies | 4.9-14.6 dropped | 10.2-27.3 read |

Exact decode is 21 % slower a token here (the replay gave 16 %). 117 ms at about 19 reads a token fits the measured
token = 80 + 2.0 ms x misses. The first reply after the restart ran at 4.39 tok/s (63 % hits, empty expert cache). The wired
governor shrank slots 12 times (lowest 4,901) and grew them back 8 times, at 73.0-74.1 GiB system wired, as before.

Answers (checked here):

- **Tool calls: 2 of 2 well-formed**, `search_files` with `limit` 100 as an integer, then `terminal` `ls -la`. No Hermes
  repair, retry or error. The first session had 3 non-JSON `limit` values and three retries.
- **Translation on the first try**, no tool call (the first session returned a malformed DSML block on that turn). German,
  French and Farsi are complete for both stories; no Latin word in either Farsi text (the first session had one).
- Image: transcription exact, read natively (the first session went through `vision_analyze`; both are correct).
- C#: builds with 0 warnings with implicit usings on **and off** (the first session failed with them off), and maps the
  quoted-field test (`"Doe, Jane"`, `"Bob ""B"""`) correctly into the typed `Customer`.
- TypeScript: leaves out `import { readFileSync } from 'node:fs'` and says so in its notes; with it added, `tsc --strict`
  (TypeScript 7.0.2) is clean and Node 22 gives the right rows.
- Python: runs; correct on the quoted test and on a BOM, CRLF, embedded-newline and blank-line file. Its usage example's
  `import_csv_typed` coercion leaves `balance` a string.
- Stories 202 and 488 words for 200 and 500 asked; the hummingbird rewrite 485.
- Desktop listing: the 17 folders and one application symlink are right; "147 top-level items" against 145 (143
  visible), "roughly 90" PNGs against 109, "~252" receipts against 250: the same approximations as the first session.
- The `'w1.weight'` server error did not recur. That is not a test: the system block came from the snapshot, so there was no
  cold prefill for the governor to shrink under.

Reading: everything the replay put on the miss budget (non-JSON integer arguments) was absent, as the replay predicts for the
exact path; the C# usings difference and the clean first translation are single samples. Hamed's budget-0 decision above
stands unless he changes it; this session is the price on the other side, about 25 ms a token.

## Open

- **Decided (Hamed, 2026-11-01): keep budget 0.** The Desktop result is the first sign since 0.60.0 that the decode drop
  costs a tool call (non-JSON integer arguments, which Hermes repairs with a retry); the exact path is about 16 % slower
  decode here. `serve.sh` is unchanged: 1017 serves with `CACHALOT_DECODE_MISS_BUDGET=0` like the old checkpoint, and
  `=off` stays the exact path for a session that needs it.
- Server: log the traceback of a failed request; then find the holder of the parked slot (defect 4).
- Runbook steps 5 (pareto gate against the old checkpoint, about 75-95 minutes), 7 (routing trace and hotlist) and 8
  (default switch) are not done.
