#!/bin/bash
# The rare-skip quality gate: keep140 (same-day control), rare1, rare2 against the 0.74.0 reference and noise arm
# (docs/prefill-rare-skip/RECORD.md). One guarded process an arm, settle between. About 20 minutes.
cd /Users/hamedprooshani/Projects/deepseek-v41-mac || exit 1
P=$HOME/venvs/deepseek-v41/bin/python; O=benchmarks/results/pareto-rare
T=$(mktemp -d /tmp/rare-gate-texts.XXXX)
for f in HANDOFF.md:docs/HANDOFF.md text_decode_runtime.py:src/cachalot/model/text_decode_runtime.py CHANGELOG.md:CHANGELOG.md; do
  git show "c1831dd:${f#*:}" > "$T/${f%%:*}" || exit 1
done
A=$T/arms.json; sed "s#TEXTS/#$T/#g" docs/prefill-rare-skip/arms.json > "$A"
mkdir -p $O; cp benchmarks/results/pareto-di2/ref.* benchmarks/results/pareto-di2/noise-chunk3000.* $O/
run() { arm=$1; shift; echo "=== $arm start $(date +%T)"; benchmarks/settle.sh --budget-gib 34 && benchmarks/guarded_run.sh --budget-gib 34 --max-seconds 2400 --tag rare_$arm -- env PYTHONPATH=src $P benchmarks/pareto.py run $arm --arms $A --out $O "$@"; rc=$?; echo "=== $arm end $(date +%T) rc=$rc"; return $rc; }
for arm in keep140 rare1 rare2; do run $arm --ref $O/ref.npz || exit 1; done
env PYTHONPATH=src $P benchmarks/pareto.py report $O --arms $A
echo "=== all done $(date +%T)"
