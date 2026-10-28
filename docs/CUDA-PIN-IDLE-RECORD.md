# CUDA lane follow-up Q: does the host's continuous idle clear the dirty lines, and do a deeper or two-thread staging pipeline help? (experiment record; predictions before any build)

Written 2026-10-28 against runtime 0.68.4, **after follow-up P and before any code for this session is written or run**. Charter section 7.2 fields 1 to 3 (baseline, hypothesis with a predicted interval, falsifier) are fixed here; this file is committed first and results go in `docs/CUDA-PIN-IDLE-RESULTS.md` afterwards. A prediction edited after its run is void. Hamed asked for it on 2026-10-28 ("the same instance is up, continue with what remains"): the four next cells of `docs/CUDA-PIN-PIPELINE-RESULTS.md` section 6, on `pricey-voice-4663` (RTX 5090, AMD Ryzen 7 7800X3D host), the instance of O and P.

## 1. Baseline (follow-ups O and P on this instance, 2026-10-28)

* O: a just-written 9.95 MB pinned buffer copies at 0.359 ms, at 0.182 ms once its dirty lines are flushed, written back or bypassed. Left alone with the host thread sleeping, it clears in one step 10 to 43 ms after the write. Copy trains with `time.sleep(1 ms)` after each copy (period 1.44 ms) held it slow; with `time.sleep(2 ms)` (2.34 ms) it settled.
* P: copy trains on a fixed schedule, sleeping all but the last 0.2 ms of each wait and then spinning, held it slow at every period from 1.4 to 2.4 ms (18 of 18 trains). P's longest continuous host idle per period was about 1.8 ms (derived), O's settling trains about 2.0 ms.
* P's staging pipeline (two buffers, cold sources, overlapped): non-temporal fill 0.420 ms a record (fill 0.398, copy 0.262), `memcpy` 0.598 (fill 0.573, copy 0.359), copy-only 0.182. Every fill arm host-bound.

**Hypotheses.**

* **H-idle:** the dirty lines are cleared when the host CPU stays idle for a continuous stretch of roughly 2 ms or more (for example, a core complex entering a deep idle state that writes its L3 back); a busy host thread, spinning or copying, prevents it however long it lasts.
* **H-freq:** the device's reads of the buffer (their frequency) are what hold it slow; host idle does not matter.
* **Pipeline:** the overlapped pipeline is bound by the single filling thread; more staging buffers do not help, a second filling thread does (for the non-temporal fill, up to the DRAM bandwidth it shares with the copy).

## 2. Cells (each its own fresh process; `benchmarks/cuda/pin_route.py`, groups `q1` to `q3` new, `k4` and `l1` reused as controls)

| id | cell | what it does |
|---|---|---|
| k4 | control, interleaved | as before (start and end of the session) |
| l1 | control, batch-first | as before |
| q1 | `idle_stretch` | 18 trains of 300 ms at a **fixed period of 3.0 ms**, three for each sleep length L in {1.0, 1.5, 1.8, 2.0, 2.2, 2.5} ms in a fixed shuffled order: allocate a 9.95 MB buffer; every 3.0 ms one single timed copy (as in O and P), then `time.sleep(L)`, then spin until the next period starts. Every copy's ms, its start, the host time its call returned and the measured sleep kept; round trip after the train |
| q2 | `busy_wait` | 18 buffers, six for each mode in {`sleep30`, `spin30`, `spin60`}, cycled: allocate (the CPU writes it), then sleep 30 ms, or spin the host thread for 30 or 60 ms with no copy, then time at once |
| q3 | `pipe_variants` | P's pipeline machinery (16 cold sources of 9.95 MB, 64-record passes, overlapped only, a copy stream, four device slots) for eight arms: (`nt`, 2 buffers, 1 thread), (`nt`, 3, 1), (`nt`, 4, 1), (`nt`, 2, 2 threads), (`memcpy`, 2, 1), (`memcpy`, 3, 1), (`memcpy`, 2, 2), (`aged`, 2, 1). With two threads each record's fill is split in halves, one filled on a worker thread while the calling thread fills the other (the helper's ctypes calls release the GIL), then the copy is issued. Three passes per arm in the order arms 1 to 8, 8 to 1, 1 to 8. Per record: fill ms and copy ms; per pass: wall ms; the last device slots checked against their sources |

Session order (`session.sh pinroute7`, output directory `benchmarks/results/cuda-idle`): `k4`, `l1`, `q1`, `q2`, `q3`, `k4`, six fresh processes. Gates as in P (round trip, slot content, sanity, no failed cell, not truncated).

**Differences from the earlier cells, listed (the v274 rule):** q1 fixes the period at 3.0 ms (longer than any O or P train) and varies only the sleep; its copy is O's and P's `one_copy_ms`; the spin fills the rest of each period, so the host is never idle outside the sleep. q2 has no copy at all before its timing, unlike every train. q3's `nt` 2-buffer 1-thread arm is P's arm; the arm order and pass count differ from P.

## 3. Predictions (registered before any code)

All **predicted**; probabilities mine. A copy is fast at or below 0.25 ms and slow at or above 0.28 ms; a train **holds** if none of its copies is fast; a buffer's class in q2 is its p50 of 20 copies after 5 warm-up.

| id | quantity | prediction | falsifier |
|---|---|---|---|
| Q-1 | the controls separate | both `k4` runs slow share at least 0.9 and `l1` at most 0.1 (0.90) | either fails: nothing below is scored |
| Q-2 | q1 ends | the L = 1.0 and 1.5 ms trains hold (6 of 6) and the L = 2.5 ms trains settle (3 of 3) (H-idle; 0.60) | every L = 1.0 ms train settles (a 3.0 ms period settles whatever the host does between copies: H-freq) |
| Q-3 | q1 threshold | L50, the sleep at which the hold share first falls to 0.5 or below (linear between the swept values), is between 1.8 and 2.2 ms (0.45) | L50 at or below 1.5 ms or above 2.5 ms, or no train settles |
| Q-4 | q2 a busy host | `spin30` slow share at least 0.8 and `sleep30` at most 0.2 (H-idle; 0.60) | `spin30` slow share at most 0.2 (a spinning host thread does not hold the state) |
| Q-5 | q2 a longer busy host | `spin60` slow share at least 0.8 (0.55) | `spin60` slow share at most 0.2 |
| Q-6 | q3 replicate | the (`nt`, 2, 1) arm per record within 5 % of P's 0.420 ms (0.70) | more than 15 % off |
| Q-7 | q3 depth | the (`nt`, 3, 1) and (`nt`, 4, 1) arms each within 10 % of (`nt`, 2, 1) per record (0.65) | either at least 20 % lower (depth pays) |
| Q-8 | q3 two threads | (`nt`, 2, 2) per record between 0.25 and 0.36 ms (0.45) | at or above 0.40 ms (a second thread does not help) |
| Q-9 | q3 which fill wins with two threads | (`memcpy`, 2, 2) per record above (`nt`, 2, 2) (0.60) | (`memcpy`, 2, 2) at or below 0.9 times (`nt`, 2, 2) |
| Q-10 | completion | every gate true, no failed cell, every mode available, six processes under 60 s of run time together (0.75) | any gate false, a failed cell or truncation |

## 4. What each outcome means

* **Q-2, Q-4, Q-5 held:** the clearing needs the host to be idle in a continuous stretch near 2 ms (Q-3 gives it); a CUDA host tier on such a VM stays slow while any host thread is busy, so the non-temporal fill (or a flush) is required, not a staging delay.
* **Q-2 falsified (L = 1.0 settles at a 3.0 ms period):** the device copy period is the variable after all (between 2.4 and 3.0 ms), and P's 2.4 ms trains held because 2.4 is below it; **Q-4 falsified** says the same from the other side.
* **Q-7 held, Q-8 held:** the pipeline is bound by one filling thread, and two threads are worth their cost; **Q-8 falsified:** the fill is bound by memory bandwidth shared with the copy, not by one core.
* Nothing here is a statement about another card, host CPU, instance population, a real model or a default.

## 5. Stop rules and cost

Stop and bring the files home on a failed cell, a false gate, or truncation; a defect found on the instance is fixed on the Mac and counted (Q-10). The instance has been up since about 19:39 UTC; cap for this follow-up **30 minutes of build-to-terminate**. Estimated run time: `q1` about 6 s, `q2` about 1.5 s, `q3` about 1 s plus 159 MB of sources, controls about 1 s each: under 15 s.

## 6. Needs Hamed

Nothing new: the instance is up and he said go. He terminates it after the files are home.
