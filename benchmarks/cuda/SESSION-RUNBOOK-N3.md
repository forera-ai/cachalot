# CUDA session N3 stage A runbook: layer-ahead prefetch against demand fetch with a stand-in compute (RTX 5090)

Predictions, design, erratum and stop rules: `docs/CUDA-N3-EXPERIMENT-RECORD.md` (committed in c2305a6; erratum 2db00e5, both before any harness run). Harness: `benchmarks/cuda/overlap_arena.py`; its bookkeeping equals an independent reference and `prefetch_coverage.py` on the Mac (`benchmarks/cuda/n3-expected.json`, hashes on the real trace). The S1, S2, V1, N2, B1 and B2 runbooks apply for the provider, key, private ssh config, cost log and teardown; this file lists what differs. **Nothing here starts an instance.** Hamed creates it when the agent says it is ready (about seven minutes to initialize) and says go for the named live instance; the agent runs over SSH only for that one instance and never runs sudo. Cap: **1 GPU-hour** from Ready.

## Steps

1. Pack from a clean commit on the Mac: `cd /Users/hamedprooshani/Projects/deepseek-v41-mac && bash benchmarks/cuda/pack.sh` (REVISION must say `dirty=0`); upload and unpack as in B1.
2. Copy the trace (11 MB, not in the repository) to the host: `scp -F ~/cachalot-private/ssh_config_NAME benchmarks/results/routing-trace-deepseek-long/trace.npz NAME:~/cachalot-cuda/n3-trace.npz`. The host refuses to run if its sha256 differs from the one in `n3-expected.json` (it equals N2's).
3. `check`, `setup`, `doctor`, `smoke` as before (all clean), then **`bash benchmarks/cuda/session.sh n3smoke`**: group `p17`, 40 decode tokens, C = 1.0 ms, arms `N D P0 O N` (about a minute plus the 24 GiB arena allocation). **This is the first GPU run of the stream code** (events, `wait_event`, the copy stream, the stand-in kernels, the calibration): read the gates before anything else: `event_hash`, `determinism`, `counts`, `readback`, `calibration`, `drift`, `closure`, `sanity` all true, no failed cell, not truncated.
4. Only if the smoke is clean: `bash benchmarks/cuda/session.sh n3full` (22 cells in two groups; the estimate is 12 to 20 minutes, the cap in the script 2,400 s).
5. Copy back with `results`; score by **field name**, never by column position. Key fields: per arm `wall_ms_token`, `compute_busy_ms_token`, `compute_stall_ms_token`, `host_gap_ms_token`, `copy_busy_demand_ms_token`, `copy_busy_prefetch_ms_token`, `host_issue_ms_token`, `closure_share`, `demand_misses_per_token`, `prefetch_loads_per_token`, `covered_per_token`, `wasted_per_token`, `calibration_error`; per C `E_D`, `E_P0`, `E_P40`, `E_O`, `H_P0`, `H_P40`, `H_O`, `drift`, `compute_busy_ratio_*`, `per_copy_ms_c`; and the `N0` cell's `source_copy_ms_*` and `source_records_slow_share` (which population this instance has).

## What the gates mean here

`closure` fails a cell when fewer than 80 % of tokens have their host wall time within 5 % of the GPU timeline span (the cell's overlap claims are then withdrawn); `drift` fails a C group when the closing `N` control differs from the opening one by more than 3 %; `calibration` fails a C when the stand-in pair alone is not within 5 % of C (the cell group is skipped and said). Any false gate stops `n3smoke` (and `n3full` refuses to start after an unclean smoke).

## Stop rules

Stop on a failed cell, a false gate, truncation, an event-hash mismatch (a harness bug: fix on the Mac), a drift flag on the first `N` pair (an unstable instance), or 80 % of the cap before the first scored cell. A defect found on the instance is fixed on the Mac. Housekeeping stands (delete the S1, S2, V1, N2, B1 and B2 ssh configs, known_hosts files and the key).
