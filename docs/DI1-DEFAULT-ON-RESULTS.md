# DI-1 default-on in `serve.sh`: results (5 of 5 held; snapshots keyed on the mode)

Built and checked 2026-10-31 against runtime 0.71.3 plus the build (c322004). Plan and predictions: `docs/DI1-DEFAULT-ON-PLAN.md` (D-1 to D-5, committed first in 2d9489e). Hamed chose M1, mode-keyed snapshots, and the pins left off. Raw outputs: `docs/di1-default-on/`. Every number is **measured** unless marked derived.

## 1. What changed

* `serve.sh` exports `CACHALOT_PREFILL_ENCODER_ONLY=${CACHALOT_PREFILL_ENCODER_ONLY-1}`; `=0` is the full prefill. `chat.sh`, the GLM and MiniMax scripts and the code default (off) are unchanged.
* A DeepSeek server with the knob on prints `DI-1 encoder-only prefill on: ...` after the decode miss budget's line.
* `snapshot_store.numerics_version(encoder_only)` gives `0.68.0-di1` with the knob on and `0.68.0` without; `cli._attach_snapshot_store` keys the disk store on it. The knob-off identity is the one every snapshot before 0.72.0 was written under.
* `tests/test_di1_default_on.py` (5 tests: the `serve.sh` default and an explicit `0`, `chat.sh` unset, the code default, the tag, the store's identity per mode). `benchmarks/di1_default_check.sh`; `di1_server_ab_driver.py --requests`.

## 2. The check (three `serve.sh` starts on one new snapshot directory, the live session's request 1 with `max_tokens` 1)

| run | environment | startup line | snapshots loaded | reused | prefill | token |
|---|---|---|---|---|---|---|
| 1 | no knob set | printed | 0 | 0 | 104.14 s | Hey |
| 2 | no knob set, restart | printed | 1 (16,036 tokens) | 16,036 | 0.96 s | Hey |
| 3 | `CACHALOT_PREFILL_ENCODER_ONLY=0`, restart | absent | 0 ("0 more on disk") | 0 | 161.12 s | Hey |

After run 1 the directory held 3 files, after run 3 six: the two modes wrote separate sets side by side, and run 3's store did not see run 1's. The watchdog never tripped (pressure 1 throughout, swap +25 MB).

**D-1 held** (104.1 s in 100 to 115; the line printed). **D-2 held** (16,036 of 16,040 reused from disk, 0.96 s). **D-3 held** (0 reused, 161.1 s in 150 to 175: the key separates the modes). **D-4 held** ("Hey" in all three). **D-5 held**: the suite gives 1,204 passed and 7 skipped against 1,196 passed and 10 skipped on 2d9489e (the 5 new tests; the 3 extra skips there are tests that need untracked result files a clean checkout lacks).

## 3. Deviation from the plan, stated

**The check ran at a 40 GiB expert budget, not 48** (Hamed's choice). Hamed's download of DeepSeek-V4.1-Flash-1017 to the X10Pro (started 14:35) held enough file cache that only about 61 to 65 GiB stayed available; the first launch at 48 timed out in the settle gate and nothing ran. The checks D-2 to D-4 do not depend on the budget; D-1 and D-3's timings could, and both fell inside their windows (104.1 s against the A/B's 106.7 and 110.0 at 48 GiB; 161.1 against 163.0 and 163.2). The settle gate ran at 32 GiB.

**Erratum for the A/B (`docs/DI1-SERVER-AB-RESULTS.md`, 0.71.3):** that download started at 14:35, during arm 1, and ran through arms 2 to 4. It is not in that write-up. Its effect looks small: the two off arms, one starting before the download and one well inside it, agree within 0.1 % on request 1, 0.5 % on request 3 and 1.2 % on request 4.

## 4. What it means

`./serve.sh` now prefills long DeepSeek prompts with DI-1. The first request after updating prefills each saved system block cold once, about 104 to 110 s for Hermes's block (the old snapshots stay on disk under their own key and serve `CACHALOT_PREFILL_ENCODER_ONLY=0`). A block that changes partway through re-prefills in full, with no in-block pin (derived: about 108 s with the knob, against about 122 s without it and with the pin; plan section 2b).

## 5. Not covered

A compression's re-prefill in a live session with the default on; the knob-off path's in-block pins are unchanged and were not re-checked.
