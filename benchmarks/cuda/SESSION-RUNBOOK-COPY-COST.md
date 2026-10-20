# CUDA copy-cost sweep runbook: the slope against copy time over layer time (NVIDIA L40S 48 GB)

Predictions, design and stop rules: `docs/CUDA-COPY-COST-RECORD.md` (ten predictions SW-1 to SW-10, committed first in 3e88a1b). The harness is the low-C file with two new profiles: `i-smoke` (12 cells, 40 tokens: copy fractions 1.0 and 0.25 at C = 0.3 ms) and `i-full` (48 cells a run: copy fractions 1.0, 0.5, 0.25 as three cell groups, each with levels 3 (per-row records) and 4 at C = 0.3 and 0.6 ms, arms `N`, `D`, `P0L`, `N`, 400 decode tokens). A group copies only the first `fraction x 9.95 MB` of each expert record; slot assignment and every count are unchanged. The earlier runbooks apply for the provider, key, private ssh config, cost log and teardown. **Nothing here starts an instance.** Hamed creates it and says go for the named live instance; the agent runs over SSH only for that one instance and never runs sudo. Cap: **1 GPU-hour** from Ready; every run passes an explicit `--max-seconds` (900 a run by default). Estimate: 8 to 12 minutes a run plus per-group arena allocation (24 GiB) and source qualification; 25 to 35 minutes of GPU time for the smoke and both runs.

Which card: an **NVIDIA L40S 48 GB** (Ada, sm_89, PCIe 4.0 x16). Only the **new multi-lead trace** is needed (`n3d-trace.npz`, 39.8 MB). The 17 % arena is 24.2 GiB, which fits 48 GB with the pinned sources and the stand-in weights.

## Steps

1. Pack from a clean commit on the Mac (`bash benchmarks/cuda/pack.sh`, REVISION `dirty=0`); upload and unpack as before. Copy the trace: `scp -F ~/cachalot-private/ssh_config_NAME benchmarks/results/routing-trace-deepseek-leads/trace.npz NAME:~/cachalot-cuda/n3d-trace.npz`. The run refuses a trace whose sha256 differs from `n3d-expected.json`.
2. **First check the instance**: `nvidia-smi` must show the GPU and driver, and `getent hosts pypi.org` must resolve. A GPU that does not initialize is replaced, and the replacement is created **before** the faulty instance is terminated. Then `check`, `setup` (the arch gate must say sm_89 is supported), `doctor`, `smoke` (all clean).
3. **`bash benchmarks/cuda/session.sh n3ismoke`**. Read every gate of the run manifest: `event_hash`, `determinism`, `counts`, `chain2`, `readback`, `calibration`, `drift`, `closure`, `sanity`, `floor`, `rowsum`, `copyevents` (true), no failed cell, not truncated. Note the `source_copy_ms_median` row of both groups (SW-1: f = 1.0 expected 0.36 to 0.50 ms; SW-2: f = 0.25 about 0.26 to 0.40 of it). If the copy times of the two groups are not ordered (f = 0.25 not clearly cheaper), stop, copy the smoke home and say so before spending the full runs.
4. Only if the smoke is clean: `MAXS=900 bash benchmarks/cuda/session.sh n3ifull` (two runs, A then B, 48 cells each). A truncated run is reported as truncated.
5. Copy back with `results` (compress the raw samples first: the instance link is about 30 KB/s), then on the Mac, with the two run directories (each holds `raw-samples.json`, `rows.jsonl` and `run-manifest.json`):
   `~/venvs/deepseek-v41/bin/python benchmarks/cuda/copy_cost_score.py RUN_A_DIR RUN_B_DIR --json docs/cuda-results/copy-cost-DATE.json`
   The script prints SW-1 to SW-10 beside the registered intervals, the six points and the fitted line. Score by hand in `docs/CUDA-COPY-COST-RESULTS.md`; SW-4 to SW-7 use run B, SW-1, SW-2 and SW-9 run A, SW-3 and SW-10 both, as registered.

## Stop rules

As the record, sections 5 and 6. Housekeeping stands (delete the ssh configs, known_hosts files and key of every earlier instance; enter the billed hours).
