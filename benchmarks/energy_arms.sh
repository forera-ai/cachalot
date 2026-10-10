#!/bin/bash
# Energy arms (charter L5, HANDOFF 18.116): start the DeepSeek server (exact decode), wait until it is ready, THEN run the driver
# (the one-runtime guard in serve.sh refuses to start while a `benchmarks/...` python process exists, so the driver must come second).
# Hamed starts `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 900 -o benchmarks/results/energy/<file>.txt` first
# (do NOT add the `tasks` sampler: it aborted powermetrics on 2026-10-09). The driver logs the clock time of each phase to
# arms-driver.jsonl; cut the recording with benchmarks/powermetrics_parse.py --from --to --tokens.
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/energy
mkdir -p $D
# E3d (0.62.39): E_TAG=name prefixes every output file so a tagged run never overwrites an earlier one; extra env (for example CACHALOT_PREDICT_TOPK=0) passes through to serve.sh.
T=${E_TAG:+$E_TAG-}
echo "start $(date +%H:%M:%S)"
CACHALOT_DECODE_MISS_BUDGET=off ./serve.sh > $D/${T}arms-server.log 2>&1 &
SP=$!
until curl -sf http://127.0.0.1:8011/v1/models >/dev/null 2>&1; do
  kill -0 $SP 2>/dev/null || { echo "SERVER DIED"; tail -5 $D/${T}arms-server.log; exit 1; }
  sleep 3
done
echo "server up $(date +%H:%M:%S)"
# E3 (0.62.35): a user-level logger of IOReport energy, PSTR and /v1/stats beside the arms (needs the root powermetrics sampler up)
LOGOUT=$D/${T}e3-log.jsonl; LOGFLAGS=""
if [ -n "${E_FULL:-}" ]; then LOGOUT=$D/${T}e3b-arms.jsonl; LOGFLAGS="--full"; fi   # E3b: every channel + SMC keys
"$HOME/venvs/deepseek-v41/bin/python" benchmarks/energy_logger.py --out $LOGOUT $LOGFLAGS > $D/${T}e3-logger.log 2>&1 &
LP=$!
/usr/bin/env python3 benchmarks/energy_arms_driver.py > $D/${T}arms-driver.jsonl 2>&1
kill -INT $LP 2>/dev/null; wait $LP 2>/dev/null
pkill -INT -f "cachalot.cli serve"; wait $SP 2>/dev/null
echo "stopped $(date +%H:%M:%S)"
