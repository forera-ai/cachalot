# CUDA lane follow-up O: results (the half-rate pinned copy is the CPU's dirty cache lines; they clear in one step at 10 to 22 ms; an O_DIRECT fill does not avoid it on this VM)

Written 2026-10-28 against runtime 0.68.1. Record: `docs/CUDA-PIN-FLUSH-RECORD.md` (O-1 to O-15, committed first in 8f8b9ff before any code; the build in d6f9f68, before any run). Scored by `benchmarks/cuda/pin_route_flush_score.py`; its output is `docs/cuda-results/pin-flush-2026-10-28.json`. Raw run directories are kept outside the repository under `~/cachalot-private/cuda-results/o1/` (archive sha256 `6a466fc7…cf56c6`, the same on the instance and on the Mac). Labels follow plan section 0.

## 1. What ran

**Instance.** `pricey-voice-4663` (id 3ff4b372), one RTX 5090 32 GB of the same listing as `calm-tongue-6106` (7 vCPUs of an **AMD Ryzen 7 7800X3D**, 23 GB RAM), driver 580.159.03, the NVIDIA module loaded (3 modules), PCIe Gen 5 x16, Linux 6.8.0-134, `/usr/bin/cc` present. Ready at about 19:39 UTC on 2026-10-28. The agent ran over SSH at Hamed's word ("instance is ready"), with a private ssh config outside the repository, and used no sudo.

**Code.** Pack `cuda-scaffold-d6f9f68.tar.gz` (REVISION `sha=d6f9f68 dirty=0`, sha256 `5a1cb354…9161e`, the same on both ends). `check`, `setup` (torch 2.7.1+cu128, arch gate true for compute capability 12.0), `doctor` and `smoke` (109 rows, 2.0 s, every gate true) were clean.

**Run.** `session.sh pinroute5` from 19:43:14 to 19:44:06 UTC: seven fresh processes (`k4`, `l1`, `o1`, `o2`, `o3`, `o4`, `k4`), each status complete with every gate true (round trip, the new-bytes content checks, sanity, no failed cell), 14.1 s of run time in all. The helper compiled on the instance and reported CPUID features 15 (clflush, clflushopt, clwb, sse2); `O_DIRECT` was accepted. Nothing was unavailable. One attempt.

## 2. Scores of O-1 to O-15

All numbers measured; the explanation is mine and a hypothesis unless stated. 9.95 MB buffers; "slow" at or above 0.28 ms, "fast" at or below 0.25 ms; per-buffer p50 of 20 copies after 5 warm-up unless stated.

| id | result | score | note |
|---|---|---|---|
| O-1 | both `k4` runs 24 of 24 slow; `l1` 0 of 24 (median 0.182 ms) | in interval | the new instance is the fast population |
| O-2 | o1: T = 0, 5 ms slow share 1.0; T = 20, 50 ms 0.0 | in interval | N's ends reproduce on a second instance |
| O-3 | o1 slow share by T: 0, 5, 7.5, 10 ms 1.0; 12.5 ms 0.75; 15 ms 0.5; 17.5 ms 0.25; 20 ms 0.0. T50 = 15.0 ms | in interval (at its upper edge) | |
| O-4 | none of the 36 o1 buffers between 0.20 and 0.28 ms; each is 0.358 to 0.359 or 0.182 to 0.183 | **past falsifier** | see section 3, point 2: o2's per-copy times show a step too |
| O-5 | G = 1 ms: both trains slow to the end (277 and 278 copies, 0.361 to 0.411 ms); G = 2 ms: both settle (first fast copy at 20.7 and 42.7 ms) | **past falsifier** | reads every ~1.4 ms hold the state; reads every ~2.4 ms do not |
| O-6 | G = 20 ms: copies 2 to the end fast in both trains (20 of 20 after the first) | in interval | |
| O-7 | `none` and `memcpy` 5 of 5 slow each (0.359 ms) | in interval | |
| O-8 | `clflushopt` 0 of 5 slow (0.182 ms; the flush took 0.38 ms) | in interval | |
| O-9 | `nt` 0 of 5 slow (0.182 ms; the non-temporal rewrite took 0.37 to 0.44 ms) | in interval | |
| O-10 | `clwb` 0 of 5 slow (0.182 ms) | in interval | not discriminating on this CPU, see section 3 |
| O-11 | o4 `none` 0 of 6 slow (0.182 ms), `direct_wait` 0 of 6 (0.182 ms), `buffered` 6 of 6 (0.346 ms) | in interval | |
| O-12 | `direct` (an `O_DIRECT` read copied at once) 6 of 6 slow (0.358 ms); the read took 0.88 ms for 9.95 MB | **past falsifier** | N's design rule "fill by `O_DIRECT`" fails on this VM |
| O-13 | slow median over the fast median: `k4` 1.97, o1 1.97, o2 1.99, o3 1.97, o4 1.96 | in interval | |
| O-14 | `clflushopt` and `nt` both fast | in interval | both on the H-dirty side |
| O-15 | every gate true, one attempt, every mode available, 14.1 s | in interval | |

**Twelve held, three falsified (O-4, O-5, O-12).**

## 3. What it shows

Measured on this instance:

1. **The cause is the CPU's dirty cache lines.** A buffer the CPU has just written copies at half rate; the same buffer copies at full rate when its lines are flushed (`clflushopt`, 0.38 ms for 9.95 MB) or written back (`clwb`) right before the copy, or when its new bytes are written with non-temporal stores, which bypass the caches. A non-temporal store is still a write, so H-page (a state keyed on a recent write, outside the caches) is falsified as registered. `clwb` being fast does not separate "clean but resident" from "evicted": whether Zen 4's `clwb` keeps the line is not established here. N's n4 (a CPU read of an aged buffer stays fast) already points at dirtiness, not residency.
2. **The lines clear in one step, not gradually.** In o1 every buffer is either 0.359 or 0.182 ms; in o2 a train's per-copy time drops from 0.365 to 0.225 ms between two consecutive copies (0.225, not 0.182, because a single copy after an idle gap carries about 0.04 ms of wake-up: the G = 2 ms trains, with less idle, settle at 0.193 to 0.197 ms). The step comes 10.8 to 42.7 ms after the write, mostly at 17 to 22 ms (o1: none by 10 ms, one in four by 12.5 ms, all by 20 ms). A gradual line-by-line eviction (O-4's premise) would show intermediate copies; there are none. (N's n3, where 256 MiB of other writes gave p50s of 0.184 to 0.230 ms, is the one reading with intermediate values; it was not re-measured.)
3. **Device reads hold the state only if they are frequent.** Copies every ~1.4 ms (G = 1 ms) and back to back (N's n1) keep the buffer slow for as long as they run; copies every ~2.4 ms (G = 2 ms) or sparser do not, and the step comes at the usual time. Neither "one idle window of 5 to 20 ms" (H-reset, which predicted G = 2, 5 and 10 ms slow throughout) nor "idle time accumulates" (which predicted G = 1 ms settling after about 28 ms) fits.
4. **`O_DIRECT` does not avoid it here.** An `O_DIRECT` read into an aged pinned buffer, copied at once, is slow (0.358 ms); with 50 ms of idle after the read it is fast. The read moved 9.95 MB in 0.88 ms (11.3 GB/s), far above a disk read: the bytes came from the host's cache and reached guest memory through a host CPU copy (hypothesis: the hypervisor's emulated disk), which leaves them dirty in the shared L3 like a guest write. An ordinary read (2.7 ms, the guest kernel's copy) is slow too.

**Note added 2026-10-28 (0.68.4):** follow-up P (`docs/CUDA-PIN-PIPELINE-RESULTS.md`) ran copy trains at 1.4 to 2.4 ms with a 0.2 ms spin before each copy and every one held slow, including 2.4 ms; point 3's "copy frequency" is more likely the host's continuous idle per period (about 2 ms, derived).

**Hypothesis (H-step, not tested here).** Something on the host periodically writes back or evicts a whole block of dirty lines at once (for example another tenant's streaming traffic through the 96 MB L3, or a host-side flush), roughly every 10 to 20 ms; frequent device reads of the lines keep them from being chosen. The one-step drop and its timing are measured; the mechanism is not.

**For a CUDA expert-streaming design (derived; not measured in a streaming loop, one host CPU model).** A host tier the CPU fills (decompression, a page-cache read, a buffered or, on this VM, an `O_DIRECT` read) and copies at once moves at half rate. A `clflushopt` pass over the filled buffer (about 0.38 ms per 9.95 MB on this CPU, about 26 GB/s) or non-temporal stores in the fill (no extra pass) restore the full rate immediately; waiting 20 ms also works but costs latency. On bare metal with a real DMA disk path `O_DIRECT` may behave differently; this VM cannot say. Re-copying a just-filled buffer in a tight loop never warms it up.

## 4. Corrections to earlier sections

* N's results (section 3, and the 0.68.1 design note) proposed "fill by `O_DIRECT`" as a way round the slow copy. On this VM it is not (O-12); the `clflushopt` or non-temporal fill is.
* N read its settle window as 5 to 20 ms "of idle"; o2 shows sparse copies do not delay it, so it is a time after the write that frequent copies (every ~1.4 ms or faster) prevent, not a required idle window.

## 5. Limits

Two instances of one listing, both AMD Ryzen 7 7800X3D hosts with a shared 96 MB L3; one card model; 9.95 MB buffers only. The step's cause is not identified. G between 1 and 2 ms (the frequency threshold) is not resolved. Whether Zen 4's `clwb` retains lines is not established. Nothing here is about an Intel host, bare metal, an L40S, a model, a default or the Mac runtime.

## 6. Next cells (not registered)

The frequency threshold between 1 and 2 ms; the same `clflushopt` and non-temporal cells at 64 MiB and at the expert-record sizes of other models; a fill-then-flush pipeline timed end to end (does the 0.38 ms flush hide behind the previous copy?); `O_DIRECT` on bare metal or a host with a passthrough NVMe; an Intel host (DDIO puts inbound DMA into the L3 in the modified state, so the result could differ).
