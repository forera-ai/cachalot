# DI-1 long gate, re-run after the 0.72.2 stale-slot fix (2026-11-01, HANDOFF section 18.245)

`benchmarks/pareto.py run` once per arm on DeepSeek-V4.1-Flash-1017 (internal bank, no hotlist, 34 GiB, guarded), git e56b291 clean.
The texts are the 0.71.0 gate's (`docs/v270-results/pareto-long-*.json`), taken from git c1831dd: `docs/HANDOFF.md` from character 1,057,561,
`src/cachalot/model/text_decode_runtime.py` and `CHANGELOG.md` from 0, 60,000 characters each; texts hash `05a1148ccc1ff338` equals the
0.71.0 run's. `arms.json` points at the scratch copies of those files (`git show c1831dd:<path>` recreates them).

`di1-nofix` is `di1` with `ResidentExpertStore.holds` forced True (`nofix_run.py`), i.e. the 0.71.0-0.72.1 behaviour. The `.npz` arrays
were not kept; `report.txt` is `pareto.py report` over the four.
