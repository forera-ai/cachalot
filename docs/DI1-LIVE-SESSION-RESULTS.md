# DI-1 in a live Hermes session (`CACHALOT_PREFILL_ENCODER_ONLY=1 ./serve.sh`, 2026-10-31)

Hamed ran the server with the knob on, a fresh snapshot directory (so the first request prefilled the system block cold) and the request dump, and worked one Hermes Desktop session (26 messages, 13 server requests). Sources: the server log (`docs/v270-results/di1-live-server.log`), the request dump (`~/cachalot-di1-live.jsonl`, not in the repository: it holds a listing of his Desktop) and his Hermes export (`~/Downloads/list-files-on-desktop-folder-3-local-de.json`, not in the repository). Runtime 0.71.0, the internal 2-bit bank, `serve.sh` defaults (48 GiB budget, decode miss budget 0, temperature 0.6), `iogpu.wired_limit_mb` 88064. Every number is **measured** unless marked derived. No prediction was registered for this session (a check Hamed asked for, not an experiment); it is a measurement.

## 1. Prefill

| request | prompt | reused | prefilled | prefill | ms a prefilled token |
|---|---|---|---|---|---|
| 1 (cold system block) | 16,040 | 0 | 16,040 | 102.45 s | 6.39 |
| 3 (a large tool result) | 27,639 | 16,161 | 11,478 | 63.38 s | 5.52 |
| 4 (a second tool result) | 34,184 | 27,704 | 6,480 | 44.62 s | 6.89 |
| 7 (an image turn) | 36,563 | 36,265 | 298 | 9.35 s | (image encode included) |
| the other 9 | 16,068 to 42,356 | all but 13 to 69 | 13 to 69 | 1.56 to 3.85 s | |

**Against the knob off (derived, not a same-session pair):** the same-day bench on this build measured a cold 22,000-token prefill at 10.21 ms a token without the knob and 6.20 with it (`docs/V270-MAC-JOBS-RESULTS.md` section 1); earlier live sessions without the knob ran at 10.2 to 10.4 ms a token on older builds (18,318 tokens in 186.1 s on 0.61.1; 10,346 in 107.2 s on 0.61.5). The live cold block ran at 6.39 ms a token, where the bench's deferred arm sits; a knob-off cold block of 16,040 tokens would take about 164 s at the bench's rate, so the live saving is about 38 % (derived). The three large prefills together took 210.5 s for 34,000 tokens; at 10.2 ms a token they would take about 347 s (derived).

**Snapshot reuse works with the knob on:** request 2 reused 16,049 tokens of request 1 (the snapshot at the system block's end is taken in DI-1 mode), and every later request reused all but its new tail. `spliced=1` on requests 3 to 13 (the runtime's own replies spliced back) as usual.

## 2. Decode and the rest of the server

Decode 7.7 to 10.9 tokens a second (a 3,043-token reply at 10.65), hit rate 82 to 98 %. The wired governor gave back expert slots when the system passed 73 GiB and took them back below it, as designed. No error, traceback or warning in the log beyond the usual `transformers` model-type notice.

## 3. Answers (read from the export; checked where a check exists)

* **Tool calls:** `search_files` on the Desktop (402 paths, nested), then on its own initiative `terminal` with `ls -la ~/Desktop` for the top level, summarised correctly; `vision_analyze` on the attached screenshot, which the tool answered as already attached. All well-formed.
* **Image:** the transcription of an address-form screenshot is exact (the field label, the typed `1227`, five suggestions with their subtitles, the highlighted row, the disabled `Continue` button), compared with the image.
* **C#** (a JSON and CSV importer, 146 lines plus a usage block): `dotnet build` fails on **one** error, `JsonSerializerOptions.AllowTrailingComma` (the property is `AllowTrailingCommas`); with that one word fixed it builds with 0 errors and 0 warnings. A wrong API name, not garbled text.
* **TypeScript:** passes a syntax check (`node --check`, which strips types; a broken control fails it); no type check (`tsc` is not installed). **Python:** compiles.
* **Prose:** a "200-word" story of 178 words and a "500-word" story of 469; German, French and Farsi translations of the second (1,615 Arabic-script characters, no replacement characters) and a rewritten hummingbird version in all three languages; fluent on a read.

No context compression happened (the prompt grew to 42,356 tokens without one), so the post-compression re-prefill that DI-1 should also shorten was not exercised.

## 4. What it means and what it does not

**Derived:** the knob gives in live use the saving the bench measured (about 38 % of a cold system block, and of the large tool-result prefills), without a visible change in the answers or the tool use, and with snapshot reuse intact. **Not shown:** a same-session, same-build knob-off pair on these prompts (the comparison above uses the bench's rate); that the C# API-name slip is unrelated to the knob (one sampled answer at temperature 0.6 cannot say either way; the long-context gate's quality numbers are the evidence on that); a compression's re-prefill. The default and the snapshot policy remain Hamed's decisions.
