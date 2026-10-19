# CUDA session B2 runbook: the huge-page retest with a private mapping, and allocation context (RTX 5090)

Predictions and stop rules: `docs/CUDA-HUGEPAGE-RETEST-RECORD.md` (committed in 4340c15 before any instrument code). Instrument: `benchmarks/cuda/backing_probe.py`, cells `h1` to `h7`. The S1, S2, V1, N2 and B1 runbooks apply for the provider, key, private ssh config, cost log and teardown; this file lists what differs. **Nothing here starts an instance.** Hamed creates it when the agent says it is ready (about seven minutes to initialize) and says go for the named live instance; the agent runs over SSH only for that one instance and never runs sudo. Cap: **1 GPU-hour** from Ready; the instrument itself runs 40 to 100 s in all.

## Differences from B1

1. Same listing as B1 (RTX 5090, 32 GB, 24 GB RAM, 800 GB disk) and the same key; user `riftuser`; the port is on the console.
2. Pack from a clean commit on the Mac: `cd /Users/hamedprooshani/Projects/deepseek-v41-mac && bash benchmarks/cuda/pack.sh` (REVISION must say `dirty=0`); no trace file is needed.
3. On the host: `check`, `setup`, `doctor`, `smoke` as before, then **`bash benchmarks/cuda/session.sh hugepage`**: seven fresh processes in order (`h7`, `h6`, `h4`, `h1`, `h2`, `h3`, `h5`), stopping at the first failure. Each cell is its own process because the allocation context is one of the variables.
4. Read before scoring: each run's `raw-samples.json` has `host_thp` (the host's THP policy and counters, read not set) and, per buffer, three smaps points (`make`, `register`, `copy`). B1 read `shmem_enabled=never`; B2's private mapping follows `enabled`. R-P1 must hold before R-P5 says anything about page size; R-P7 says which population this instance has; R-P8 is only testable if `h7` shows mostly slow buffers.
5. `h5` reports `populate_unavailable` when the kernel refuses `MADV_POPULATE_WRITE`; that is a result, not a failure.
6. Copy back with `results`; score by **field name**, never by column position. Key fields: `huge_verified_count`, `huge_verified_count_post_register`, `huge_fraction_max_drop_after_register`, `shmem_pmd_verified_count`, `huge_verified_slow_count`, `huge_verified_ms_max`, `small_page_slow_share`, `slow_share`, `prefix_h2d_gbps_median`.

## Stop rules

Stop on a failed cell, a false gate, truncation, or 80 % of the cap before the first process. A defect found on the instance is fixed on the Mac. Housekeeping stands (fill the B1 hours, delete the S1, S2, V1, N2 and B1 ssh configs, known_hosts files and the key).
