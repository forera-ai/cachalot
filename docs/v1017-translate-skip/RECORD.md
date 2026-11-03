# Does the decode miss budget make Hermes skip the translations? Replay record

Written 2026-11-03 against runtime 0.78.1, before the run. Hamed chose the Hermes lane and this test ("Hermes: budget-0 defects", then
"Server replay A/B"). HANDOFF section 18.261 when scored.

## 1. Baseline (recorded, live, all under the shipped decode miss budget 0)

The usual Hermes script ends with two translation turns: "translate it to German, French and Farsi languages" (the fish story) and
"So try to change the story to a small humming bird tries to reach the far forests and translate that to these 3 languages too". The
dumps of four sessions hold both turns, and each turn's reply is visible in the next request's messages:

- **Hummingbird turn, translations skipped: 1 of 4.** The 2026-11-02 0.76.0 session (`~/cachalot-dump-toolexact.jsonl` line 22) opened
  with "**English: The Long Flight**", wrote the English story (2,811 characters) and ended; the other three continued into German,
  French and Farsi (10-12.7k characters).
- **Farsi with a stray English word: 2 of 6** complete Farsi replies ("mattered", 2026-11-02; "heard", 2026-11-01). Not tested here
  (it needs full 3-4k-token replies; a later run if this one points at the budget).
- No exact-decode (`CACHALOT_DECODE_MISS_BUDGET` off) sample of either turn exists. The budget drops 15.9 experts a token on W1
  (DS1017-TOKEN-48) and is known to change tool-call arguments (18.243, 18.255).

## 2. Hypothesis

Under budget 0 the dropped experts move the probability of ending the reply at the end of the English section enough to make a skip
likely in a fraction of replies; exact decode makes it rare. Alternative: the skip is the model's own behaviour at temperature 0.7 and the
budget does not change its rate.

## 3. Method

`run-all.sh`: two fresh `./serve.sh` servers on the shipped 0.78.1 defaults (1017, DI-1 + keep 140, 48 GiB, hotlist, tool-exact, the
shipped snapshot directory), arm A budget 0 (shipped), arm B exact (`CACHALOT_DECODE_MISS_BUDGET=`), settle between. `driver.py 4` posts
each of the four recorded hummingbird requests (pinned by dump line and messages sha256) four times, copy-major, as Hermes sent them
(temperature 0.7, 21 tools, reasoning effort none), streamed, `max_tokens` 1,500. The client disconnects as soon as `started()` sees a
translation begin (a Persian character, a German/French/Farsi heading line, or six German or eight French function words), so a
reply costs only its English part. Classes (`analyse.py`): translated, skip (the stream finished without a marker and without a tool
call), cap (1,500 tokens without a marker), tool, error. The classifier marks the three recorded continuing replies at their German
heading (characters 2,420-3,240) and leaves the recorded skip unmarked. 16 replies an arm.

Not an ABBA: the measure is a rate, not a time, and each arm starts from a fresh server with the same request order, so both meet the
same cache history. Two arms of 16 can show only a large effect: 4 of 16 against 0 of 16 is p = 0.10 (two-sided Fisher), 5 of 16
against 0 is p = 0.04.

## 4. Predictions (registered before the run)

- **S-1.** Both server logs carry the expected budget state (A prints the budget line, B does not); 32 replies, no error.
- **S-2.** Budget 0: 1 to 6 skips of 16 (live 1 of 4, i.e. about 25 %; the interval is the binomial spread around it).
- **S-3.** Exact: 0 or 1 skip of 16.
- **S-4.** Skips cluster on H1 (the copy that skipped live) under budget 0 at least as much as on any other copy.
- **S-5.** No tool call and at most one cap reply in either arm (the English part is 600-800 tokens).
- **Falsifier.** Exact skips at least as often as budget 0, or exact skips 3 or more of 16: the budget is not the cause, and the defect
  is the model's (or the prompt's) at temperature 0.7.
- **Inconclusive by design.** Budget 0 with 0-2 skips and exact with 0: no claim either way.

Runtime estimate: about 70 minutes (two arms of about 30 minutes: four conversation prefills and 16 English parts of about 85 s, plus a
settle). Needs `iogpu.wired_limit_mb` at 88064, no other runtime.

## 5. Results

Not run yet.
