# CUDA follow-up P runbook: the copy period that holds a pinned buffer slow, and a fill-then-flush staging pipeline (RTX 5090)

Predictions and stop rules: `docs/CUDA-PIN-PIPELINE-RECORD.md` (eleven predictions P-1 to P-11, committed first in 9ad8518). Results it builds on: `docs/CUDA-PIN-FLUSH-RESULTS.md` (follow-up O). Instruments: `benchmarks/cuda/pin_route.py` groups `p1` and `p2` (and the reused controls `k4` and `l1`), `cache_touch.c` and `cache_touch.py` (O's helper: `ct_copy`, `ct_stream_copy` and `clflushopt`, compiled on the instance with the system `cc`), `session.sh pinroute6`, `pin_route_pipeline_score.py`. The session K and follow-up L runbooks apply for the provider, key, private ssh config, cost log and teardown. **Nothing here starts an instance.** Hamed creates it and says go for the named live instance; the agent runs over SSH only for that one instance and never runs sudo. Cap: **1 GPU-hour** from Ready; GPU run time is under 15 s (five processes).

Which card: an **RTX 5090 32 GB** of the fast population (the Korean listing). No trace is needed. If the host CPU is not an AMD Ryzen 7 7800X3D (as for M, N and O), say so before the run: p2's sources are sized against its 96 MB L3, and every number becomes a different population.

## Steps

1. Pack from a clean commit on the Mac: `cd /Users/hamedprooshani/Projects/deepseek-v41-mac && bash benchmarks/cuda/pack.sh` (REVISION `dirty=0`); upload and unpack as before (`ssh -F "$C"` with the private config written for this instance).
2. **First check the instance**: `nvidia-smi` must show the GPU and driver **and `lsmod | grep nvidia` must show the module**; `lscpu | grep "Model name"`. Then `check` (it prints the C compiler line: with none, p2 runs only its `aged` bound and P-5, P-6, P-8 to P-10 are not testable; say so before going on), `setup`, `doctor`, `smoke` (all clean).
3. **`bash benchmarks/cuda/session.sh pinroute6`**: `k4`, `l1`, `p1`, `p2`, `k4`, each its own process, stopping at the first failure. Look at `k4` and `l1` first: P-1 needs the interleaved control slow and the batch-first control fast; if they do not separate, say so (P-2 to P-10 are then not scored) before spending anything else. Then read p2's `cpu_features` row and any `wall_per_record_ms_*` row tagged `unavailable` (its note is the reason).
4. Copy back (`tar -czf ~/pipeline.tar.gz -C benchmarks/results cuda-pipeline`), verify the hashes on both ends, then on the Mac:
   `~/venvs/deepseek-v41/bin/python benchmarks/cuda/pin_route_pipeline_score.py PIPELINE_DIR [--failed-attempts N] --json docs/cuda-results/pin-pipeline-DATE.json`
   Score by hand in `docs/CUDA-PIN-PIPELINE-RESULTS.md`. Tell Hamed the instance can be terminated as soon as the archive's hash matches on the Mac.

**Waiting on a run:** never `pgrep -f` inside the ssh command; use `ps aux | grep '[p]in_route'` or read the run directory.

## Stop rules

As the record, section 5. Housekeeping stands.
