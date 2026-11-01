# The 1017 quality gate (runbook step 5, 0.72.5)

`RECORD.md`: the predictions (committed in c900b74 before any arm) and the erratum (d3a1201, before any arm: 40 GiB,
not 48). `arms.json`: the arms file as run (a copy of `benchmarks/pareto_arms.v1017.json` with the base budget at 40 GiB).
`v1017.json`, `noise-chunk96.json`, `v1017-actq.json`, `old.json`: each arm's manifest (git d3a1201 clean, texts hash
`3efb4242d9e31da5`, tasks with their answers). `report.txt`: `pareto.py report` and the per-text breakdown. The
per-position `.npz` files stay in `benchmarks/results/pareto-v1017/` (not in the repository). Results and scoring:
`docs/HANDOFF.md` section 18.247.
