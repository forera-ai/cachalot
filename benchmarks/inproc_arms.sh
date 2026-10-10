#!/bin/bash
# E3e (docs/E3E-INPROC-KERNELS-RECORD.md): the in-process counterpart of energy_arms.sh. Environment mirrors serve.sh (48 GiB expert budget, wired limit 80,
# page cache bypass, fast synch) with CACHALOT_PREDICT_TOPK=0 as in E3d, so phase B is comparable with E3d's. Hamed starts
# `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 700 -o benchmarks/results/energy/<tag>-sampler.txt` first, closes Cachalot Lab, and runs
#   cd /Users/hamedprooshani/Projects/deepseek-v41-mac && E_TAG=e3e ./benchmarks/inproc_arms.sh > benchmarks/results/energy/e3e-run.log 2>&1
# SMOKE=1 runs the short precheck without the energy logger.
cd /Users/hamedprooshani/Projects/deepseek-v41-mac
D=benchmarks/results/energy
mkdir -p $D
T=${E_TAG:+$E_TAG-}
if pgrep -fl "deepseek-v41/bin/python|cachalot\.cli|Python\.app/Contents/MacOS/Python .*(benchmarks/|cachalot)" >/dev/null 2>&1; then
  echo "a runtime is already running; not starting a second one" >&2; exit 1
fi
export CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash
export CACHALOT_EXPERT_BANK=${CACHALOT_EXPERT_BANK:-$HOME/DeepSeek-V4.1-Flash-q2g128}
export CACHALOT_PAGE_CACHE=1
export CACHALOT_MLX_WIRED_LIMIT_GIB=80
export CACHALOT_EXPERT_CACHE_BUDGET_GIB=48
export CACHALOT_PREDICT_TOPK=${CACHALOT_PREDICT_TOPK-0}
export MLX_METAL_FAST_SYNCH=1
export PYTHONPATH=src
echo "start $(date +%H:%M:%S)"
if [ -n "${SMOKE:-}" ]; then
  ~/venvs/deepseek-v41/bin/python benchmarks/decode_power_inproc.py --smoke
else
  ~/venvs/deepseek-v41/bin/python benchmarks/decode_power_inproc.py --logger-out $D/${T}e3b-arms.jsonl > $D/${T}arms-driver.jsonl 2>$D/${T}arms-stderr.log
fi
echo "stopped $(date +%H:%M:%S)"
