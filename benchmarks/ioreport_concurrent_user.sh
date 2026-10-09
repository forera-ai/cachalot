#!/bin/bash
# Same test as ioreport_concurrent_test.sh, but the reader runs as YOU (no root); only powermetrics runs under sudo.
# Run WITHOUT sudo from the repo root:  ./benchmarks/ioreport_concurrent_user.sh   (it asks for your password once, for powermetrics)
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=benchmarks/results/energy/probe-concurrent-user.txt
sudo /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 14 -o /dev/null &
PM=$!
sleep 5
"$HOME/venvs/deepseek-v41/bin/python" benchmarks/power_sources.py --watch 5 > "$OUT" 2>&1
wait "$PM"
grep -E '^euid|^Energy Model components|^  (amcc|ane|cpu|dcs|dram|gpu|display_media|pcie) +[0-9]' "$OUT"
