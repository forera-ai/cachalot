# CUDA follow-up O runbook: the CPU's dirty cache lines, the settle time, idle-gap trains and an O_DIRECT fill (RTX 5090)

Predictions and stop rules: `docs/CUDA-PIN-FLUSH-RECORD.md` (fifteen predictions O-1 to O-15, committed first in 8f8b9ff). Results it builds on: `docs/CUDA-PIN-SETTLE-RESULTS.md` (follow-up N). Instruments: `benchmarks/cuda/pin_route.py` groups `o1`, `o2`, `o3`, `o4` (and the reused controls `k4` and `l1`), `cache_touch.c` and `cache_touch.py` (the o3 helper, compiled on the instance with the system `cc`), `session.sh pinroute5`, `pin_route_flush_score.py`. The session K and follow-up L runbooks apply for the provider, key, private ssh config, cost log and teardown. **Nothing here starts an instance.** Hamed creates it and says go for the named live instance; the agent runs over SSH only for that one instance and never runs sudo. Cap: **1 GPU-hour** from Ready; GPU run time is under 15 s (seven processes).

Which card: an **RTX 5090 32 GB** of the fast population (the Korean listing). `calm-tongue-6106` (M and N) can be reused if it is still up. No trace is needed.

## Steps

1. Pack from a clean commit on the Mac: `cd /Users/hamedprooshani/Projects/deepseek-v41-mac && bash benchmarks/cuda/pack.sh` (REVISION `dirty=0`); upload and unpack as before.
2. **First check the instance**: `nvidia-smi` must show the GPU and driver **and `lsmod | grep nvidia` must show the module**. Then `check` (it now prints the C compiler line: with none, o3 will run only its `none` control and O-7 to O-10 and O-14 are not testable; say so before going on), `setup`, `doctor`, `smoke` (all clean). Read the CPU model from `doctor` (N's instance was an AMD Ryzen 7 7800X3D; another CPU is a different population for o3 and is recorded).
3. **`bash benchmarks/cuda/session.sh pinroute5`**: `k4`, `l1`, `o1`, `o2`, `o3`, `o4`, `k4`, each its own process, stopping at the first failure. Look at `k4` and `l1` first: O-1 needs the interleaved control slow and the batch-first control fast; if they do not separate, say so (O-2 to O-14 are then not scored) before spending anything else. Then read o3's `cpu_features` row (1 clflush, 2 clflushopt, 4 clwb, 8 sse2) and any `slow_share_mode_*` row tagged `unavailable` in o3 and o4 (its note is the reason).
4. Copy back with `results` and the follow-up directory (`tar -czf ~/flush.tar.gz -C benchmarks/results cuda-flush`), verify the hashes, then on the Mac:
   `~/venvs/deepseek-v41/bin/python benchmarks/cuda/pin_route_flush_score.py FLUSH_DIR [--failed-attempts N] --json docs/cuda-results/pin-flush-DATE.json`
   Score by hand in `docs/CUDA-PIN-FLUSH-RESULTS.md`.

o4 writes a 179 MB record file into its own run directory and removes it at the end of the cell; if a run stops inside o4, delete `o4-records-*.bin` before copying the directory home.

**Waiting on a run:** never `pgrep -f` inside the ssh command; use `ps aux | grep '[p]in_route'` or read the run directory. In zsh, pass the private config as `ssh -F "$C"` (a variable holding `-F path` is one word).

## Stop rules

As the record, section 5. Housekeeping stands.
