#!/bin/bash
# Does the IOReport Energy Model advance for CPU/DRAM while powermetrics is sampling at the same time? (docs/POWER-ACCOUNTING-PLAN.md, E0)
# Run with sudo from the repo root:  sudo ./benchmarks/ioreport_concurrent_test.sh
# powermetrics runs for ~14 s in the background (output discarded); the read-only probe samples 5 s of it, starting 3 s in.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=benchmarks/results/energy/probe-concurrent.txt
/usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 14 -o /dev/null &
PM=$!
sleep 3
"$HOME/venvs/deepseek-v41/bin/python" benchmarks/power_sources.py --watch 5 > "$OUT" 2>&1
wait "$PM"
grep -E '^euid|^Energy Model components|^  (amcc|ane|cpu|dcs|dram|gpu) +[0-9]' "$OUT"
