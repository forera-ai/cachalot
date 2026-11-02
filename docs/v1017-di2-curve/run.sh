#!/bin/bash
# The DI-2 curve: seven pareto.py arms, one guarded process each, settle between (docs/v1017-di2-curve/RECORD.md). Raw .npz in the gitignored results folder.
cd /Users/hamedprooshani/Projects/deepseek-v41-mac || exit 1
A=docs/v1017-di2-curve/arms.json; O=benchmarks/results/pareto-di2; P=$HOME/venvs/deepseek-v41/bin/python
run() { arm=$1; shift; echo "=== $arm start $(date +%T)"; benchmarks/settle.sh --budget-gib 34 && benchmarks/guarded_run.sh --budget-gib 34 --max-seconds 2400 --tag di2_$arm -- env PYTHONPATH=src $P benchmarks/pareto.py run $arm --arms $A --out $O "$@"; rc=$?; echo "=== $arm end $(date +%T) rc=$rc"; return $rc; }
run ref || exit 1
for arm in noise-chunk3000 di1 keep1024 keep512 keep256 keep140; do run $arm --ref $O/ref.npz || exit 1; done
env PYTHONPATH=src $P benchmarks/pareto.py report $O --arms $A
echo "=== all done $(date +%T)"
