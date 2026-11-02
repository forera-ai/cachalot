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
