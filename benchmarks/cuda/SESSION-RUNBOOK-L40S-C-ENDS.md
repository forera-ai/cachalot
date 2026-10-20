# CUDA L40S layer-time ends runbook: the slope at C = 1.0 and 0.15 ms (NVIDIA L40S 48 GB)

Predictions, design and stop rules: `docs/CUDA-L40S-C-ENDS-RECORD.md` (ten predictions J-1 to J-10, committed first in 04a172c; the J-10 count is corrected in its section 7 before any run). The harness is the copy-cost file with two new profiles: `j-smoke` (12 cells, 40 tokens: level 3 at C = 1.0 and 0.15 ms, level 4 at 0.15 ms) and `j-full` (32 cells a run: levels 3 and 4 at C = 1.0, 0.6, 0.3 and 0.15 ms, arms `N`, `D`, `P0L`, `N`, 400 decode tokens, full copy). The earlier runbooks apply for the provider, key, private ssh config, cost log and teardown. **Nothing here starts an instance.** Hamed creates it and says go for the named live instance; the agent runs over SSH only for that one instance and never runs sudo. Cap: **1 GPU-hour** from Ready; every run passes an explicit `--max-seconds` (720 a run by default). Estimate: 6 to 8 minutes a run plus arena allocation (24 GiB) and source qualification; about 20 minutes of GPU time for the smoke and both runs.

Which card: an **NVIDIA L40S 48 GB** (Ada, sm_89, PCIe 4.0 x16). Only the **new multi-lead trace** is needed (`n3d-trace.npz`, 39.8 MB).

## Steps

1. Pack from a clean commit on the Mac (`bash benchmarks/cuda/pack.sh`, REVISION `dirty=0`); upload and unpack as before. Copy the trace: `scp -F ~/cachalot-private/ssh_config_NAME benchmarks/results/routing-trace-deepseek-leads/trace.npz NAME:~/cachalot-cuda/n3d-trace.npz`. The run refuses a trace whose sha256 differs from `n3d-expected.json`.
2. **First check the instance**: `nvidia-smi` must show the GPU and driver, and `getent hosts pypi.org` must resolve. A GPU that does not initialize is replaced, and the replacement is created **before** the faulty instance is terminated. Then `check`, `setup` (the arch gate accepts sm_86 on an L40S since f9f7f12), `doctor`, `smoke` (all clean).
3. **`bash benchmarks/cuda/session.sh n3jsmoke`**. Read every gate of the run manifest: `event_hash`, `determinism`, `counts`, `chain2`, `readback`, `calibration`, `drift`, `closure`, `sanity`, `floor`, `rowsum`, `copyevents` (true), no failed cell, not truncated. Note the `source_copy_ms_median` row (J-5: 0.34 to 0.42 ms). The `calibration` and `floor` gates at C = 0.15 are prediction J-3: if they fail there, stop and say so before the full runs, because the full profile's 0.15 ms cells would then not be scorable and the session should be reshaped (the ratio is then taken against the shortest layer time that holds).
4. Only if the smoke is clean (or its 0.15 ms failure is understood and accepted by Hamed): `MAXS=720 bash benchmarks/cuda/session.sh n3jfull` (two runs, A then B, 32 cells each). A truncated run is reported as truncated.
5. Copy back with `results` (compress the raw samples first: the instance link is about 30 KB/s), then on the Mac, with the two run directories (each holds `raw-samples.json`, `rows.jsonl` and `run-manifest.json`):
   `~/venvs/deepseek-v41/bin/python benchmarks/cuda/l40s_c_ends_score.py RUN_A_DIR RUN_B_DIR --json docs/cuda-results/l40s-c-ends-DATE.json`
   The script prints J-1 to J-10 beside the registered intervals, the four points per run, the s(0.15) / s(1.0) ratio, the affine-in-1/C fit and the 5090 form errors. Score by hand in `docs/CUDA-L40S-C-ENDS-RESULTS.md`.

**Waiting on a run:** never `pgrep -f` inside the ssh command (it matches its own shell). Use `ps aux | grep '[o]verlap_arena'` or read the run's log and result directory.

## Stop rules

As the record, sections 5 and 6. Housekeeping stands (delete the ssh configs, known_hosts files and key of every earlier instance; enter the billed hours).
