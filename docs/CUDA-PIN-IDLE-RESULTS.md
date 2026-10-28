# CUDA lane follow-up Q: results (a busy host thread alone holds a just-written buffer slow; the clear needs the host mostly idle; a deeper or two-thread staging pipeline does not help)

Written 2026-10-28 against runtime 0.68.4. Record: `docs/CUDA-PIN-IDLE-RECORD.md` (Q-1 to Q-10, committed first in a668cc4 before any code; the build in 2346580, before any run). Scored by `benchmarks/cuda/pin_route_idle_score.py`; its output is `docs/cuda-results/pin-idle-2026-10-28.json`. Raw run directories are kept outside the repository under `~/cachalot-private/cuda-results/q1/` (archive sha256 `82a3dec0…aa6ace`, the same on the instance and on the Mac). Labels follow plan section 0.

## 1. What ran

**Instance.** `pricey-voice-4663` (RTX 5090, AMD Ryzen 7 7800X3D host), the instance of O and P, still up, at Hamed's word ("the same instance is up, continue with what remains"). Over SSH with the private config of O, no sudo.

**Code.** Pack `cuda-scaffold-2346580.tar.gz` (REVISION `sha=2346580 dirty=0`), sha256 `b20baece…7d6c8d`, the same on both ends, unpacked over the checkout; `doctor` and `smoke` (106 rows, every gate true) clean.

**Run.** `session.sh pinroute7` from 20:17:07 to 20:17:49 UTC: six fresh processes (`k4`, `l1`, `q1`, `q2`, `q3`, `k4`), each status complete with every gate true, 12.2 s of run time; no mode unavailable; one attempt.

## 2. Scores of Q-1 to Q-10

All numbers measured; the explanation is mine and a hypothesis unless stated.

| id | result | score | note |
|---|---|---|---|
| Q-1 | both `k4` runs 24 of 24 slow; `l1` 0 of 24 | in interval | |
| Q-2 | q1 at a 3.0 ms period: L = 1.0 and 1.5 ms held 6 of 6; L = 2.5 ms settled 2 of 3 (at 20.0 and 80.4 ms), one held | outside interval | not past the falsifier: no L = 1.0 ms train settled |
| Q-3 | hold share by L: 1.0, 1.5, 1.8, 2.0 ms 1.0; 2.2 ms 0.67; 2.5 ms 0.33. L50 = 2.35 ms (measured sleeps 0.06 to 0.09 ms longer than L) | outside interval (1.8 to 2.2) | |
| Q-4 | q2 `spin30` 6 of 6 slow (0.359 ms), `sleep30` 0 of 6 (0.182 ms) | in interval | no copy ran before the timing |
| Q-5 | q2 `spin60` 6 of 6 slow (0.359 ms) | in interval | |
| Q-6 | (`nt`, 2, 1) 0.426 ms a record, P 0.420 | in interval (+1.4 %) | |
| Q-7 | (`nt`, 3, 1) and (`nt`, 4, 1) 0.422 ms each | in interval (-0.9 %) | |
| Q-8 | (`nt`, 2, 2) **0.455 ms**: the two-thread fill takes the same 0.401 ms, and the copy beside it slows to 0.299 ms (81 % of copies slow) | **past falsifier** | a second thread does not speed the non-temporal fill |
| Q-9 | (`memcpy`, 2, 2) 0.540 ms against (`nt`, 2, 2) 0.455 ms | in interval | the two-thread `memcpy` fill is faster (0.482 against 0.564 ms) but its copies stay slow |
| Q-10 | every gate true, one attempt, every mode available, 12.2 s | in interval | |

**Seven held (Q-1, Q-4 to Q-7, Q-9, Q-10), two outside their intervals (Q-2, Q-3), one falsified (Q-8).**

## 3. What it shows

Measured on this instance:

1. **A busy host thread alone keeps a just-written buffer slow.** With no copy at all, a buffer whose host thread spins for 30 or 60 ms after the write copies at 0.359 ms; one whose thread sleeps 30 ms copies at 0.182 ms. The device's reads are not what holds the state (H-freq falsified as stated); the host CPU's activity is.
2. **At a fixed 3.0 ms period, the clear needs a long sleep in every period.** Sleeping 1.0 to 2.07 ms (measured) per period held 12 of 12 trains; 2.29 ms settled 1 of 3; 2.59 ms settled 2 of 3. The settle, when it comes, is again 20 to 80 ms after the write.
3. **Continuous idle alone does not fit O and Q together; idle fraction does (derived).** On this same instance O's `time.sleep(2 ms)` trains slept about 2.0 to 2.06 ms in a 2.34 ms period (O did not record its sleeps; derived from the period and Linux's measured overshoot here, 0.06 to 0.09 ms) and settled (2 of 2), while q1's slept 2.07 ms (measured) in a 3.0 ms period and held (3 of 3). Taking each train's sleep over its period (O's and P's derived): held at 0.69 (q1, 2.0 ms), 0.72 to 0.74 (O, 1 ms), about 0.75 (P, 2.4 ms), mixed at 0.76 (q1, 2.2 ms) and 0.86 (q1, 2.5 ms), settled at 0.88 and above (O, 2 to 20 ms). The fraction of time the host is idle, near 0.8 to 0.9 and not sharply, is the variable that orders every train of O, P and Q. This is a fit to four cells, not a registered test.
4. **The staging pipeline does not go faster with depth or a second filling thread.** Non-temporal fill, two buffers: 0.426 ms a record; three or four buffers: 0.422 ms; two filling threads: 0.455 ms (the fill stays at 0.401 ms, about 25 GB/s, and the copy beside it slows). `memcpy` gains 8 % from a second thread (0.589 to 0.540 ms) but remains above non-temporal. The non-temporal fill is bound by memory bandwidth, not by one core (hypothesis: the guest's DRAM write bandwidth is about 25 GB/s and the copy's reads share it).

**Hypothesis (not tested here).** The host's deep idle state (on Zen, a core complex in CC6 writes its L3 back) is what clears the dirty lines; a power-management governor enters it only when the cores have been mostly idle for a while, so the clear depends on the recent idle fraction and comes as one step, 20 to 80 ms after the write. Another tenant's activity on the same core complex would then also hold buffers slow, which fits the "slow population" instances of earlier sessions (hypothesis, not tested).

**For a CUDA expert-streaming design (derived; one host CPU model, 9.95 MB records).** Waiting for a filled buffer to age is not a design: any busy host thread (a decoder, a polling loop, another tenant) holds it slow indefinitely. Fill the staging buffers with non-temporal stores on one thread, two buffers deep: 0.42 ms a record on this VM, where neither depth nor a second thread helps.

## 4. Corrections to earlier sections

* P's results (section 3, point 1) read the settle variable as "a continuous host idle of about 2 ms". q1 holds slow at a 2.07 ms sleep in a 3.0 ms period while O settled at about the same sleep in a 2.34 ms period: the variable that orders O, P and Q is the idle fraction, not the stretch (derived).

## 5. Limits

One instance (the same as O and P), one host CPU model, one card, 9.95 MB records. The idle-fraction reading is a post-hoc fit across three sessions' cells, not a registered prediction; nothing here reads the host's C-state residency (a guest cannot). The hypothesis about other tenants is untested. q3's two-thread fill is two Python threads calling the helper, not a native thread pool.

## 6. Next cells (not registered)

A registered idle-fraction sweep (period and sleep varied together so the stretch and the fraction separate: for example 2.0 ms sleep at periods 2.2, 2.4, 2.7, 3.0 ms, and 1.0 ms sleep at 1.1, 1.2, 1.4 ms); the same cells on an instance of another listing (an Intel host, or a whole-machine rental where no other tenant shares the core complex); the non-temporal fill with 4 MiB and 64 MiB records.
