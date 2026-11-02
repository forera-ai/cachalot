# Experiment record: exact decode inside tool-call blocks under the decode miss budget 0 (replay A/B)

Written 2026-11-02 before the first arm, runtime 0.75.1 plus the default-off knob `CACHALOT_DECODE_TOOL_EXACT`, committed together with
this record. Hamed: "Do A/B Test now" (after 18.254 offered exact decode inside tool-call blocks).

## 1. Question

Under `serve.sh`'s decode miss budget 0, 1017 sometimes writes `search_files`'s integer `limit` as a word (`collapsed`, `FB_MAX`,
`strongest`, `COPIOUS`), which is not JSON; Hermes then resends the turn up to four times. Does lifting the budget from a reply's
tool-call opener to its end remove those non-JSON arguments, and what does it cost in decode time?

## 2. Intervention

`CACHALOT_DECODE_TOOL_EXACT=1` (`server/engine.py`, default off): once the reply's text contains the DSML tool-call opener, the engine sets
the expert store's `decode_miss_budget` to `None` (exact: every non-resident expert is read) for the rest of the reply and restores it when
the request ends, however it ends. Text before the opener, and replies without a tool call, decode under the budget as before. Prefill is
not affected (the budget is a decode setting). Tests: `tests/test_tool_exact_decode.py`.

## 3. Baseline (recorded)

- 18.243 replay (0.72.1, 1017, budget 0 against `=off`, request 3 of the 2026-11-01 dump posted 10 times an arm): non-JSON `limit` 2 of 10
  at budget 0, 0 of 10 exact (p = 0.047 with the live calls); exact decode 9.30 against 11.09 tok/s.
- Live: 3 of 4 `limit` values non-JSON on 2026-11-01 (one more call in the session made it 3 of 5), 3 of 4 in run 1 and 0 of 1 in run 3 on
  2026-11-02 (18.254).
- Token cost: token = 77 + 1.62 ms a miss at 48 GiB on 1017 (LEDGER DS1017-MISS-48); budget 0 drops the misses instead of reading them.

## 4. Predictions (fixed before the run)

- T-1. Budget 0 arms (A): non-JSON `limit` in at least 2 of 24 replies. Falsified if 0 or 1 of 24 (the replay would then be underpowered
  for the question; reported, not re-run without Hamed).
- T-2. Tool-exact arms (B): 0 of 24 non-JSON. Falsified by 2 or more of 24; 1 of 24 is reported as a miss.
- T-3. If T-1 holds, B has fewer non-JSON replies than A with one-sided Fisher p < 0.10 (p < 0.05 needs A at 5 or more of 24).
- T-4. Every B reply is a well-formed tool call (24 of 24 parse; `search_files` or another tool).
- T-5. Cost: B's median decode time a reply token is 3-25 % above A's on these replies (about 85 of 93 tokens fall inside the block; warm
  cache, 1.2-15 misses a token at budget 0 in run 1). Falsified above +35 % or below 0.
- T-6. Run time: each arm 3-5 minutes, the four arms with settles 15-25 minutes.

Stop rules: the guard's own; one attempt per arm; a crash or a guard kill is reported, not retried, without Hamed.

## 5. Method

`docs/v1017-tool-exact/run-all.sh`: ABBA (A1 budget 0, B1 tool-exact, B2 tool-exact, A2 budget 0); quiet gate (no other runtime, 5-minute
load under 3.5) before each arm, `settle.sh --budget-gib 40 --timeout 300` between. Each arm (`arm.sh`): a manifest, a fresh `./serve.sh`
with the shipped defaults (1017, internal bank, DI-1 + DI-2 keep 140, budget 0, wired 80, hotlist, the shipped snapshot directory so the
system blocks load warm) under `guarded_run.sh --budget-gib 48 --max-seconds 1500 --force`; B arms add `CACHALOT_DECODE_TOOL_EXACT=1`.
`driver.py` posts 12 requests, rotating three recorded versions of the Desktop-listing turn (P1, P2, P3: the two live turns that produced
non-JSON `limit` values and the clean one; pinned in `driver.py` by dump line and messages sha256), as Hermes sent them (temperature 0.7,
21 tools, `reasoning_effort` none), non-streaming, `max_tokens` 400. 24 replies an arm, 8 per prompt. Scored by `analyse.py`: a reply is
non-JSON when a tool call's arguments do not parse or `search_files`'s `limit` is not an integer; decode speed from the servers' `[request]`
lines. Raw output in `benchmarks/results/tool-exact/` (gitignored). The dumps hold personal file names and stay out of the repository.

## 6. Limits known before the run

The replay repeats one turn; the expert cache warms over 12 requests, so later samples drop fewer experts than a live turn would (run 1's
retries still failed at 1.6-2.8 misses a token). Sampling at temperature 0.7 makes each reply a draw; 24 an arm detects a large effect
only. The knob is only priced on tool-call replies; it costs nothing elsewhere by construction. Any default change is Hamed's.

## 7. Results (2026-11-02, 18:54-19:13; `report.txt`)

Four arms ran, no guard kill, every request answered (48 replies).

| | budget 0 (A1 + A2) | tool-exact (B1 + B2) |
|---|---|---|
| non-JSON `limit` | **2 of 24** (A1 P3 `COP...`, A2 P1 `grep`) | **0 of 24** |
| well-formed tool calls | 22 of 24 | 24 of 24 |
| `limit` values written | 50 x6, 100 x6, 200 x5, 30, 2, and the two words | 200 x13, 100 x8 (others without a `limit`) |
| decode, steady requests (4th to 12th of an arm) | A1 10.65-11.36 tok/s, A2 12.43-12.80 | B1 11.85-12.86, B2 11.66-12.67 |
| decode, the first three requests (10-27 misses a token) | A1 10.05-10.38, A2 10.38-12.32 | B1 7.23-9.78, B2 7.92-10.43 |

- Fisher one-sided, tool-exact fewer than budget 0: p = 0.245. Pooled with 18.243's replay of the same turn (budget 0 2 of 10, exact 0 of
  10): 4 of 34 against 0 of 34, p = 0.057 (derived; 18.243's exact arm lifted the budget for the whole reply, not only the block).
- A1 ran slower throughout (10.05-11.36 tok/s, including steady requests at 0.3-1.1 misses a token) and prefilled P2's system block cold
  (90.0 s; the later arms found it on disk); the 5-minute load was 3.4-3.7 with `mediaanalysisd` busy before it. A2, run under the same
  quiet gate after the B arms, is the clean budget 0 comparison.
- Cost: on steady requests (0-4 misses a token) tool-exact equals budget 0 (B 11.66-12.86 against A2 12.43-12.80 tok/s); on the first
  requests of a fresh server, where 10-27 experts a token are missing, the exact block runs about 15-30 % slower (B1 7.23 against A2 10.38
  on the first request). The pooled median (-8.0 % a token for tool-exact) is A1's drift, not a speed-up.

Predictions: T-1 held at its edge (2 of 24). T-2 held (0 of 24). T-3 missed: with 2 against 0 of 24 the test cannot reach p < 0.10 (it
needed 3 or more in A; the arithmetic in the prediction was wrong). T-4 held (24 of 24). T-5 falsified as written (pooled median -8.0 %,
below 0) because A1 drifted; read against A2 the cost is 0 on warm requests and 15-30 % on the first cold ones, inside the predicted band
only for the latter. T-6 half held: 19 minutes in all (inside 15-25), but arms of 2.0-3.7 minutes, three of them under the predicted 3.

## 8. What it means

The replay keeps the defect rare (8 % at budget 0 against 6 of 9 in the live turns) because the cache warms over 12 identical requests:
budget 0 drops almost nothing once the turn's experts are resident. In the regime where the defect lives (a turn's first draws, many
misses) the knob does what it is meant to: budget 0 wrote words for an integer, tool-exact wrote 100 or 200 every time. It never made a call
worse, and its cost is confined to tool-call tokens: zero when the experts are resident, about 15-30 % on those tokens when they are not
(the worst measured: a 77-token call 10.6 s against 7.4 s, about 3 s more). Whether it becomes `serve.sh`'s default is Hamed's call.

## 9. Live Hermes session with the knob on (2026-11-02, 0.76.0) and the default

Hamed ran the usual Hermes script (greeting, Desktop listing with a tool, C# importer, 200-word story, image, TypeScript and Python
importers, 500-word story, translations to German, French and Farsi, hummingbird rewrite with translations) against
`CACHALOT_DECODE_TOOL_EXACT=1 CACHALOT_SERVER_DUMP=$HOME/cachalot-dump-toolexact.jsonl ./serve.sh`: 13 requests, one sample, not an A/B.
The dump and Hermes's export hold personal file names and stay out of the repository.

- **Tool calls:** `search_files` with `limit` 100 and `terminal` with `timeout` 30, both valid JSON on the first try; no Hermes retry, no
  entry in Hermes's error log. At budget 0 the same turn wrote a non-JSON `limit` in 3 of 4 calls on 2026-11-01 and in 3 of 4 earlier on
  2026-11-02 (18.243, 18.254).
- **Cost:** the two tool-call replies (93 and 92 tokens) decoded at 7.80 and 7.35 tok/s with 26.3 and 31.3 misses a token, against
  10.24-10.39 tok/s at about 15 misses a token for the same calls at budget 0 in 18.254's run 3: about 3 s more a call on a cold cache, as
  the replay's first requests showed. The other replies decoded at 10.17-11.04 tok/s (4.8-14.9 misses a token).
- **Prefill:** Hermes's system prompt had changed again (16,156 tokens, no saved block matched): cold 92.82 s; follow-ups 5,224 tokens
  28.06 s, 6,624 34.89 s.
- **Answers:** C# builds on .NET 8, TypeScript passes a syntax check, Python parses; stories 193 and 534 words; the fish story's German,
  French and Farsi translations complete. Three defects, all outside tool-call blocks (decoded under budget 0, which the knob does not
  touch): the hummingbird turn wrote the English rewrite only, without the three translations it asked for; the Farsi left one English
  word ("mattered") mid-sentence, as 18.243's had ("heard"); the image transcription read the input as `1227,` (an added comma), the rest
  exact.

Hamed, 2026-11-02, after the session: "Yes, make default". `serve.sh` exports `CACHALOT_DECODE_TOOL_EXACT=${CACHALOT_DECODE_TOOL_EXACT-1}`
since 0.77.0; `=0` or an explicit empty value keeps budget 0 inside tool calls; `chat.sh` and the code default are unchanged.
