# CUDA lane follow-up P: results (no period from 1.4 to 2.4 ms lets a just-written buffer settle on this schedule; in a cold-source staging pipeline a non-temporal fill is fastest, at 0.42 ms a record)

Written 2026-10-28 against runtime 0.68.3. Record: `docs/CUDA-PIN-PIPELINE-RECORD.md` (P-1 to P-11, committed first in 9ad8518 before any code; the build in 2b4550a, before any run). Scored by `benchmarks/cuda/pin_route_pipeline_score.py`; its output is `docs/cuda-results/pin-pipeline-2026-10-28.json`. Raw run directories are kept outside the repository under `~/cachalot-private/cuda-results/p1/` (archive sha256 `23dec09f…6205f8`, the same on the instance and on the Mac). Labels follow plan section 0.

## 1. What ran

**Instance.** `pricey-voice-4663` again (follow-up O's instance, still up: RTX 5090 32 GB, AMD Ryzen 7 7800X3D host, driver 580.159.03, module loaded, PCIe Gen 5 x16, `/usr/bin/cc`), at Hamed's word ("RTX 5090 is up. Go: run pinroute6 per the runbook, score it, write it up and release."). Over SSH with the private config of O, no sudo.

**Code.** Pack `cuda-scaffold-9d3b317.tar.gz` (REVISION `sha=9d3b317 dirty=0`; `benchmarks/cuda` identical to the build commit 2b4550a), sha256 `7c3037c0…6c4eb7`, the same on both ends, unpacked over O's checkout (the venv kept). `check`, `setup`, `doctor`, `smoke` (109 rows, every gate true) clean.

**Run.** `session.sh pinroute6` from 20:07:31 to 20:08:10 UTC: five fresh processes (`k4`, `l1`, `p1`, `p2`, `k4`), each status complete with every gate true (round trip, every pass's device slots equal to their sources, sanity, no failed cell), 10.7 s of run time. The helper reported all four CPUID features; no mode was unavailable. One attempt.

## 2. Scores of P-1 to P-11

All numbers measured; the explanation is mine and a hypothesis unless stated. 9.95 MB records; a copy is "fast" at or below 0.25 ms and "slow" at or above 0.28 ms by its own CUDA-event time.

| id | result | score | note |
|---|---|---|---|
| P-1 | both `k4` runs 24 of 24 slow; `l1` 0 of 24 (0.182 ms) | in interval | |
| P-2 | every train at every period held: 18 of 18, 125 to 215 copies each, every copy 0.361 to 0.402 ms; realised periods within 0.000005 ms of target | **past falsifier** | the 2.4 ms trains held, unlike O's 2.34 ms trains (section 3) |
| P-3 | hold share 1.0 at every swept period: no P50 inside 1.4 to 2.4 ms | **past falsifier** | |
| P-4 | no period with mixed outcomes | in interval | vacuous: there was no transition to be sharp |
| P-5 | serial copies: `memcpy` 0.99 slow (0.361 ms); `flush`, `nt` 1.0 fast (0.185 ms); `aged` 0.995 fast (0.182 ms) | in interval | O's flush and non-temporal results hold with a cold source |
| P-6 | serial `memcpy` fill 0.248 ms (interval 0.25 to 0.60); flush pass 0.280 ms (interval 0.30 to 0.50); `nt` fill 0.375 ms, above `memcpy`'s | outside interval | all three narrowly or in direction; no falsifier fired |
| P-7 | `aged` pipelined 0.182 ms a record | in interval | the pipeline machinery costs nothing above the copy |
| P-8 | pipelined per record: `nt` 0.420, `memcpy` 0.598, `flush` 0.776 ms | in interval | `nt` fastest, `flush` slowest |
| P-9 | pipelined wall against max(serial host, serial copy): `flush` +12 %, `nt` +12 %, `memcpy` **+66 %** (0.598 against 0.361) | **past falsifier** | the overlapped `memcpy` fill takes 0.573 ms against 0.248 ms serial |
| P-10 | `memcpy` over `aged` pipelined: 3.28 | outside interval (1.5 to 2.5) | the penalty is larger than predicted, not hidden |
| P-11 | every gate true, one attempt, every mode available, 10.7 s | in interval | |

**Six held (P-1, P-4 vacuously, P-5, P-7, P-8, P-11), two outside their intervals (P-6, P-10), three falsified (P-2, P-3, P-9).**

## 3. What it shows

Measured on this instance:

1. **p1: on a fixed schedule with a 0.2 ms spin before each copy, no period from 1.4 to 2.4 ms let a buffer settle**, over 300 ms (every one of 2,949 copies slow). O's trains slept a whole `time.sleep(2 ms)` after each copy (period 2.34 ms) and settled at 17 to 43 ms. The two differ in how the host waits: p1 sleeps for all but the last 0.2 ms and then spins, so its longest continuous host idle per period is about P minus 0.6 ms (copy, synchronisation, spin; derived, not recorded), about 1.8 ms at 2.4 ms, against about 2.0 ms in O's 2 ms trains. **Derived reading:** what holds the buffer slow is not how often the device reads it but whether the host thread stays idle long enough at a stretch; the threshold of continuous idle is then between about 1.8 and 2.0 ms on this host. This reading is not measured (no cell varied idle with the period fixed).
2. **p2, serial, cold source:** a `memcpy` fill takes 0.248 ms and leaves the copy slow (0.361 ms); a `memcpy` plus `clflushopt` (0.412 + 0.280 ms) and a non-temporal fill (0.375 ms) leave it fast (0.185 ms). The non-temporal fill is slower than `memcpy` from a cold source on this CPU (the opposite of P-6's guess) but replaces a fill plus a flush.
3. **p2, overlapped:** per record `aged` 0.182 ms (the copy bound), **`nt` 0.420 ms**, `memcpy` 0.598 ms, `flush` 0.776 ms. Every fill arm is host-bound. Overlap changes the host work: the `memcpy` fill rises from 0.248 to 0.573 ms (2.3 times) while the device reads the other, dirty staging buffer; the flush arm's fill rises 0.412 to 0.481 ms; the non-temporal fill barely moves (0.375 to 0.398 ms). The device copy beside a non-temporal fill slows from 0.185 to 0.262 ms (between the modes), beside a flushed fill to 0.207 ms, beside a `memcpy` fill it stays at the slow 0.359 ms.
4. Every pass repeated within 0.003 ms a record across its three passes; content checks true throughout.

**Hypotheses (not tested here).** (a) For 1: a deep idle state of the host's cores (on Zen, a core complex that enters CC6 writes its L3 back) is what clears the dirty lines; it needs a continuous idle of about 2 ms or more, and only some of those idle periods reach it, so the clear comes 10 to 40 ms after the write. This also fits O's one-step drop and its timing. (b) For 3: the device reading a staging buffer whose lines the CPU holds dirty generates coherence traffic that slows the CPU's concurrent ordinary stores into the other buffer, while non-temporal stores, which bypass the caches, are hardly affected; the non-temporal fill and the device copy then share DRAM bandwidth, which is why the copy slows to 0.262 ms.

**For a CUDA expert-streaming design (derived; one host CPU model, a 2-buffer pipeline, 9.95 MB records).** Fill the host staging buffers with non-temporal stores: 0.42 ms a record, 1.42 times faster than `memcpy` (0.60) and 1.85 times faster than `memcpy` plus a flush (0.78). The aged bound (0.18 ms) is out of reach for any fill on this host: the fill, not the PCIe copy, is the limit, at about 24 GB/s. A host thread that busy-waits or is never idle for about 2 ms keeps CPU-written buffers slow indefinitely.

## 4. Corrections to earlier sections

* O (`docs/CUDA-PIN-FLUSH-RESULTS.md` section 3, point 3) read "copies every ~1.4 ms hold the state, every ~2.4 ms do not" as a property of the device copy frequency. On P's schedule a 2.4 ms period holds too; O's settling trains slept a whole 2 ms at a stretch. The variable that O's G = 1 and G = 2 trains separated is more likely the host's continuous idle per period than the copy frequency (derived; section 3, point 1).

## 5. Limits

One instance (O's), one host CPU model, one card, 9.95 MB records, two staging buffers, 16 sources. p1 did not record host busy time per period, so the idle figures are derived. Whether the source was cold was not measured (159 MB cycled against a 96 MB L3 shared with other tenants). The idle-state hypothesis is untested; nothing here reads C-state residency.

## 6. Next cells (not registered)

The host-idle threshold directly: copy trains with a fixed period and the wait split between sleep and spin (idle stretch 1.6, 1.8, 2.0, 2.2, 2.5 ms at one period), and a train that spins for 20 ms after a write with no copies at all (does a busy host thread alone keep a never-copied buffer slow?); a pipeline depth of 3 and 4 staging buffers for the non-temporal fill; the non-temporal fill on two host threads.
