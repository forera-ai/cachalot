# CUDA session N3 stage C runbook: a router-weight gate on the recorded predictor (RTX 5090)

Predictions, design and stop rules: `docs/CUDA-N3C-EXPERIMENT-RECORD.md` (committed in c72cea2 before any stage C code). Harness: `benchmarks/cuda/overlap_arena.py` (the stage B file with the gated arms `G10` and `G20`: `P0L` with the recorded predicted set filtered to experts whose recorded router weight is at least 0.10 or 0.20). The gated bookkeeping equals the independent reference and `prefetch_coverage.py --gates` on the real trace (`benchmarks/cuda/n3c-expected.json`, gate (g)). The S1, S2, V1, N2, B1, B2, N3 and N3-B runbooks apply for the provider, key, private ssh config, cost log and teardown; `SESSION-RUNBOOK-N3.md` has the trace copy; this file lists what differs. **Nothing here starts an instance.** Hamed creates it when the agent says it is ready (about seven minutes to initialize) and says go for the named live instance; the agent runs over SSH only for that one instance and never runs sudo. Cap: **1 GPU-hour** from Ready, and every run passes an explicit `--max-seconds` below what is left of it.

## Steps

1. Pack from a clean commit on the Mac (`bash benchmarks/cuda/pack.sh`, REVISION `dirty=0`); upload and unpack as in N3; copy the 11 MB trace to `~/cachalot-cuda/n3-trace.npz`. The host refuses to run if the trace sha256 differs from the one in `n3c-expected.json`.
2. **First check the instance**: `nvidia-smi` must show the GPU and driver (a GPU whose open-driver probe says "no GSP" is replaced from the console, not debugged), and `getent hosts pypi.org` must resolve. Then `check`, `setup`, `doctor`, `smoke` (all clean).
3. **`bash benchmarks/cuda/session.sh n3csmoke`**: group `p17`, 40 decode tokens, `D`, `P0L`, `G20` and `O` at C = 1.0 ms level L5, `D` and `G10` at C = 0.3 ms level L4 (10 cells; each group has a `D` arm so `H` is defined). This is the first GPU run of the gated arms. Read every gate: `event_hash`, `determinism`, `counts`, `chain2`, `coverage` (gate g), `readback`, `calibration`, `drift`, `closure`, `sanity` all true, no failed cell, not truncated. `L5` reporting `graph_unavailable` is a valid outcome, not a failure (then the C = 1.0 cells are reported at no level and the stage is re-planned at L4 on the Mac).
4. Only if the smoke is clean: `MAXS=900 bash benchmarks/cuda/session.sh n3cfull` (19 cells; the estimate is 6 to 9 minutes of cells plus qualification and calibration; `MAXS` is the explicit cap in seconds, default 1200). A truncated run is reported as truncated.
5. Copy back with `results`; score by **field name**. Key fields per (level, C): `E_D`, `E_P0L`, `E_G10`, `E_G20`, `E_O`, `H_*`, `drift`, `floor_ms`, `compute_busy_ratio_*`, `per_copy_ms_c`, per cell `wall_ms_token`, `host_issue_ms_token`, `demand_misses_per_token`, `prefetch_loads_per_token`, `covered_per_token`, `wasted_per_token`, `bookkeeping_chain2`; and the first cell's `source_copy_ms_*` and `source_records_slow_share`.

## What the gates mean here

As N3 stage B, plus gate (g): each gated cell's demand misses, prefetch loads and covered counts a token equal the offline coverage model with the same threshold on the same tokens (stored in `n3c-expected.json` and checked on the Mac when it is written); a mismatch is a harness bug, fixed on the Mac.

## Stop rules

As the record, section 6. Housekeeping stands (delete the S1, S2, V1, N2, B1, B2, N3 and N3-B ssh configs, known_hosts files and the key).
