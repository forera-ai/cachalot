# Power accounting for SSD reads and DRAM transport (charter L5b)

Written 2026-10-10 against runtime 0.62.30, at Hamed's request: "while our main asset is SSD read and RAM transportations, without a proper power consumption record of these two all our numbers will be rough and inaccurate; find a way to retrieve and index the power usage of those too; this power consumption per token should be monitored in Cachalot Lab too; introduce a proper brief for Cachalot Lab". This session only prepared the plan; the work is the next session's (`docs/NEXT-SESSION-PROMPT.md`, "First jobs").

How to read this file. Every statement is tagged **verified** (checked on this machine, with the command), **recorded** (a measurement already in `docs/LEDGER.md` or HANDOFF) or **to verify** (believed from how other tools behave; not checked here). Nothing below is a new measurement except the two recon facts marked verified.

## 1. The problem

The energy numbers of 0.62.28-0.62.29 (`ENERGY-IDLE`, `ENERGY-DECODE`) are `powermetrics` CPU + GPU + ANE package power. The runtime's two main costs are not in them:

- **SSD reads.** Every miss reads an expert (9.95 MB DeepSeek, 13.5 MiB GLM, 21.1 MiB MiniMax) from the internal NVMe (`APPLE SSD AP1024Z`, 1 TB; verified with `system_profiler SPNVMeDataType`) or the X10Pro over USB. A read-bound DeepSeek token reads ~453 MB (recorded, `ST-BW-CURVE`); a GLM token reads ~100 experts, 1.4 GB (recorded, `GLM-MISS-COST`).
- **DRAM transport.** A token streams ~7.5 GB through unified memory (recorded, `DS-BYTES-TOKEN`, an estimate) plus every expert that was read lands in DRAM before the GPU uses it.

So the recorded 1.73 J (all-resident) and 2.52 J (read-bound) per DeepSeek token are lower bounds, and the 1.4-1.5 times ratio between them is unreliable in size (it ignores exactly the part that differs between the two regimes). Package power also cannot say whether a lever that removes reads saves more or less energy than it appears to.

## 2. What is known about the sources

| source | what it could give | status |
|---|---|---|
| `powermetrics` CPU/GPU/ANE | package power by component, 1 s granularity, needs sudo, stated by Apple to be "estimated and may be inaccurate" | **verified** (output of 0.62.28-29; the text of `powermetrics -h`); no DRAM or SSD line in the output we recorded |
| `powermetrics --show-extra-power-info` | "unsupported power info (may change between releases)" | **verified** that the flag exists (`powermetrics -h`); **to verify** what it prints on this Mac (Hamed runs it with sudo) |
| `powermetrics --show-all` | every sampler the platform offers, so whether any DRAM/interconnect/disk power line exists here | **to verify** (needs sudo; one 5-sample run settles it) |
| `powermetrics -s disk` or `--show-process-io` | IO byte and operation rates, not power | **to verify**; useful for the bytes side of joules per byte |
| IOReport (`/usr/lib/libIOReport.dylib`, private) | the "Energy Model" channel group that third-party monitors (macmon, mactop, socpowerbud) read: cumulative energy counters for CPU clusters, GPU, ANE and, as separate channels, DRAM and the memory fabric; sampled in-process, millisecond resolution, probably without root | library **verified** present; channel list and root requirement **to verify** by enumerating channels with a small C/ctypes/Swift probe |
| SMC (`AppleSMC` via IOKit) | total system power (keys such as `PSTR` for system total and `PDTR` for DC input on desktop Macs) and temperatures; the closing check "sum of components against total" | **to verify** which keys exist on a Mac Studio (Mac15,14, verified model) |
| the internal SSD | no known power sensor; NAND temperature can be read (`benchmarks/nand_temp.c`, recorded) | **to verify**; a differential method (section 3, E2) is the fallback |
| the X10Pro over USB | no sensor inside; an inline USB-C power meter on its cable would measure it exactly | hardware: Hamed's decision (a meter is cheap and is not the halted M19 drive purchase) |
| DRAM by a model | energy per byte (pJ per bit) times bytes moved, from the counters we already keep | **estimate**; must be labelled so (charter section 8.1 rule 11) |

Traps already met: the `tasks` sampler aborts `powermetrics` on this machine (recorded, 18.116); background bursts of 5-10 W every ~10 s and an idle baseline that wobbles by ~2 W (recorded, 18.115-18.116); `powermetrics` reports at the sampler's interval and cannot resolve a single token.

## 3. Method, in order (each step estimated and confirmed with Hamed before a long run)

- **E0, inventory (no model, minutes).** (a) Hamed runs once: `sudo powermetrics --show-all -i 1000 -n 5 -o benchmarks/results/energy/show-all.txt` and `sudo powermetrics --samplers cpu_power,gpu_power,ane_power --show-extra-power-info -i 1000 -n 5 -o benchmarks/results/energy/extra-info.txt` (do not add `tasks`). (b) A small read-only probe, `benchmarks/power_sources.py` plus a C or ctypes helper, enumerates IOReport groups and channels (`Energy Model` and anything with DRAM, memory controller, fabric or disk in its name) and the SMC keys whose names suggest power. Output: a table of every channel found, its unit, whether it needed root. Stop rule: if neither IOReport nor SMC nor `--show-all` yields a DRAM figure, DRAM stays a modelled estimate (section 3, E1b) and this is reported as such.
- **E1, DRAM calibration.** Hold the SoC otherwise quiet and stream a buffer at fixed rates (`benchmarks/micro_read_size_roofline.py` reads at 47-87 % of 819 GB/s; a rate-limited variant is needed). Record power against GB/s: the slope is joules per byte. E1a uses the DRAM channel if E0 found one; E1b, if not, uses the difference between package power and the system total (SMC) at several rates and labels the slope *derived*, not measured.
- **E2, SSD differential.** With the GPU idle, read experts back to back at 0, 25, 50, 100 % of the drive's rate (`benchmarks/expert_read_speed.py`, `expert_read_qd.py`, `CACHALOT_READ_THROTTLE_GBPS`) on the internal drive and on the X10Pro. Power added = system total (SMC) minus the SoC and DRAM components (from E0/E1). The residual per GB/s is the SSD (plus its controller and the PCIe/USB link) and is tagged *derived*. The X10Pro with an inline meter is *measured*. NAND temperature (`nand_temp.c`) is logged alongside as a sanity check on activity.
- **E3, complete joules a token.** Re-run the DeepSeek arms of 0.62.29 (`benchmarks/energy_arms.sh`) with the extended sources; report for each phase the SoC, DRAM, SSD and residual joules a token, and a coverage figure (components divided by the system total). Acceptance for calling a total "complete": coverage within 10 % of the SMC total, or the residual reported as a named component of unknown cause. Then GLM (91 % of its token is waiting) and MiniMax.
- **E4, index.** Section 4.

## 4. Indexing: how the numbers stay findable

- `benchmarks/power_sources.py`: one small module with a uniform reader (`read_energy()` returns cumulative joules by component and the source of each), so benchmarks, the energy driver and, later, the runtime read power the same way; a source that is unavailable returns `None`, never zero (charter section 7).
- Scorecard fields (JSONL, through `benchmarks/run_manifest.py`): `joules_per_token_soc`, `_dram`, `_ssd`, `_other`, `_total`, `watts_*`, `bytes_read_ssd_per_token`, `bytes_moved_dram_per_token`, `power_coverage`; every field carries the tag `measured`, `derived` or `estimated` and the name of its source channel.
- `docs/LEDGER.md`: a new section "Energy" collecting `ENERGY-*` rows (the existing `ENERGY-IDLE`, `ENERGY-DECODE` move there) plus one row per component constant: `ENERGY-DRAM-PJ-BYTE`, `ENERGY-SSD-INT-W-GBPS`, `ENERGY-SSD-X10-W-GBPS`, with source, regime and status like every other row.
- `benchmarks/whatif.py`: a joules column, power times predicted token time plus bytes times the per-byte constants.
- `docs/EXPERIMENTS.md` regenerated at each release as before.

## 5. Showing it in Cachalot Lab

The runtime has to expose the numbers before Lab can show them. Two routes, chosen in the next session by what E0 finds:

1. **In-process** (if IOReport/SMC energy is readable without root): the runtime samples cumulative energy by component at request boundaries (and optionally at 1 Hz), and `GET /v1/stats` gains an `energy` object (cumulative joules by component since start, the source and tag of each, `available: false` when there is none), the `[request]` log line gains `J/tok=`, and each response's `usage.cachalot` gains `energy_joules` by component. Off by default behind a knob (working name `CACHALOT_ENERGY`); a read must never add a GPU round trip or a syscall to the decode hot path (use the sampling thread, not the token loop; `docs/SEAMS.md` section 2 item 2 and the MLX rules in the skill apply).
2. **Offline** (if root is needed): the runtime does nothing; Lab ingests a `powermetrics` recording plus the server's `[request]` timestamps (or a trace) and attributes joules to requests itself. Lab already owns comparison and interpretation; the runtime owns authoritative instrumentation.

Either route triggers a Lab brief (`RELEASE.md` item 7: new `/v1/stats` fields, log line, knob). The next session writes it as `docs/lab/briefs/YYYY-MM-DD-runtime-X.Y.Z-energy.md` after the fields exist, following the format of the existing briefs in that folder, and tells Hamed it is there. Its content, fixed now so the runtime work targets it:

- **Contract first**: the exact JSON of `/v1/stats.energy`, `usage.cachalot.energy_joules`, the `[request]` field, the knob and its default, with an example of each and what `unavailable` looks like.
- **Views Hamed wants** (his words: "power consumption per token should be monitored in Cachalot Lab"): per-request and per-session joules a token and watts by component (SoC, DRAM, SSD, other); a live power strip next to tokens per second; joules a token against token time (the read-bound regime costs energy through time, 18.116); comparison of two runs or arms with the baseline subtracted; the tag (measured, derived, estimated) shown on every figure; a coverage indicator (components over system total); the manifest of each run one click away.
- **Rules for Lab** (Lab's `AGENTS.md` and `docs/PRODUCT_DIRECTION.md` govern): a missing source shows as unavailable, never zero; an estimate is never styled like a measurement; planned features are not implemented capabilities; aggregate durations do not establish overlap or critical paths (the L2 trace, when it exists, is the source for that).
- **What Lab does not do**: collect power itself with sudo, or edit runtime policy.

## 6. What needs Hamed

- The two `sudo powermetrics` runs of E0 (commands above), and later sampler runs for E3 (`-n 900`, never `tasks`).
- Whether to buy an inline USB-C power meter for the X10Pro (the only way to *measure* that drive's power).
- Permission to read the SMC from a user-space helper, and whether a privileged helper for the runtime is acceptable (route 1) or Lab should ingest recordings (route 2).
- Review of the Lab brief before he implements it.

## 7. Limits of this plan

Written from the recon facts marked verified and from how other tools are known to behave; no IOReport channel, SMC key or `--show-all` output has been looked at. If E0 finds that this Mac exposes no DRAM or SSD power at all, the honest result is a labelled model (energy per byte and per read, calibrated by the differential method) and a coverage figure that says how much of the system total it explains. The answer to "how much does a token cost in joules" never becomes a single number without its tag.

## 8. Status after E0 (2026-10-10, 0.62.31; HANDOFF 18.118)

E0 is done and its results replace the "to verify" tags in section 2 for these rows. `powermetrics --show-all` and `--show-extra-power-info`: **measured**, CPU, GPU and ANE power only, no DRAM, fabric or SSD line. IOReport: **measured**, "Energy Model" holds `DRAM0_n`, `DCS0_n`, `AMCC0_n` in mJ; they are frozen unless a root `powermetrics` is sampling at the same time, and a non-root process reads them live then (root alone does not unfreeze them). SMC: **measured**, `PSTR` is whole-system power without root (22-40 W idle-ish), noisy. SSD: no sensor found. Consequences: route 1 (section 5) is possible if Hamed's sampler is up during a reading; DRAM has a measured source whose link to bytes moved is still E1's job; the SSD stays *derived* (E2) unless an inline meter is added. Whether DCS and AMCC lie inside or beside DRAM is open. The instrument is `benchmarks/power_sources.py`.

**E1 (2026-10-10, 0.62.32, HANDOFF 18.119).** DRAM energy is linear in bytes: 49.9 pJ per byte (DCS 12.0, AMCC 32.6; 94.5 together, possibly overlapping). The residual between PSTR and the IOReport components falls from 18 W at idle to 3 W at full stream and is unexplained, which limits E2 until bounded.
