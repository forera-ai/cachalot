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

## 5. Results (measured 2026-11-03, runtime 0.78.1, harness commit 569b8d1)

**Run.** `run-all.sh` 12:29-13:52, sysctl 88064, no other runtime; arm A 2,454 s, arm B 2,424 s, peak swap 128 MB in each (the guardian
allows growth up to 512 MB), pressure 1. `report.txt` holds `analyse.py`'s output; the replies are in the gitignored
`benchmarks/results/translate-skip/`.

| arm | replies | translated | skip | cap | tool call | error |
|---|---|---|---|---|---|---|
| A, budget 0 (shipped) | 16 | 15 | **0** | 0 | 1 (H2) | 0 |
| B, exact | 16 | 16 | **0** | 0 | 0 | 0 |

Every translated reply was marked by a German heading after the English story (`**Deutsch**`, `## German`, ...), at 2,400-3,200
characters, except one exact reply (H2, k = 2) that wrote "Here are the three translations of the hummingbird version" and went straight
to `## German` with no English rewrite (195 characters): a different omission, under exact decode. The budget 0 tool call (H2, k = 2,
758 tokens, 16.3 misses a token) is a tool call where the user asked for a translation; which tool is not recorded (no server dump in this
run). Decode under budget 0: 9.6-11.4 tok/s at 4.6-16.3 misses a token.

**Scored.**
- **S-1 held:** arm A printed the budget line, arm B did not; 32 replies, no error.
- **S-2 missed:** 0 skips of 16 under budget 0 (predicted 1 to 6).
- **S-3 held:** 0 skips of 16 exact.
- **S-4 not testable:** no skips. H1, the copy that skipped live, translated 4 of 4 under budget 0 and 4 of 4 exact.
- **S-5 split:** no cap reply in either arm (held); one tool call under budget 0 (predicted none, missed).
- **Falsifier and the inconclusive clause.** The written falsifier ("exact skips at least as often as budget 0") is met by its letter
  (0 against 0), but this outcome is the one section 4 called inconclusive by design (budget 0 with 0-2 skips, exact 0). Read: no claim
  that the budget causes the skip, and none that it does not.

**What it does say (derived).** The live skip did not come back in 16 budget-0 replays of the same four turns. With 0 of 16 the skip
rate on this replay is below about 17 % under budget 0 (one-sided 95 % bound, 1 - 0.05^(1/16) = 0.171), and also below it exact; the live 1 of
4 is consistent with a rare event of a few percent in either mode. The replay differs from the live turn in one known way: a fresh
server's expert cache instead of a session's, so budget 0 dropped what a fresh cache drops.

**Side checks (measured offline, same session).** Arm A re-prefilled 18-20k tokens on every H4 request and on H3's last (177-201 s
each) while only the 16k system block was reused. The cause is the replay shape: four requests share one prompt, each reply's snapshot
makes the prompt's own snapshot "a prefix of another entry", which the 1.5 GiB prefix cache evicts first once full. Replaying the three
live session dumps through `PrefixCache` with that tier changed (a snapshot with two or more extensions ranked with the leaves) prefills
exactly the same tokens as today's order (134,162; 164,826; 34,785, no request different), so live Hermes sessions do not pay it; not a
lever.

**Kind.** Research: predicted and measured; the result falls in the registered inconclusive band (S-1, S-3 held; S-2 missed; S-4 not
testable; S-5 split). No default changed.
