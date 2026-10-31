# DeepSeek-V4.1-Flash-1017: from download to a decision

Written 2026-10-31 at 0.71.2, before the download. Nothing here has run yet except the file comparisons in section 1. Every
step that deletes data or changes what `serve.sh` serves by default is marked **Hamed's call**; the rest can run in order.

## 1. What 1017 is (checked from the Hub's files, 2026-10-31)

- Revision `1bae0e1088993b9b82de08f59de5be5a34e42170` (2026-10-17), MIT licence, called "a minor update" of V4.1-Flash.
- `config.json` equals the local checkpoint's (`/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash`, revision `dba1be0`) in
  every field. Same 96,085 tensors, same shard map, same `total_size` (510,286,023,000 bytes); `tokenizer.json` and
  `tokenizer_config.json` are the same files.
- All 48 safetensors shards have different LFS hashes: new weights in the same shapes.
- `inference/model.py` differs only by the reference's 2026-10-12 change (`act_quant(o)` before `wo_a`, this runtime's
  `CACHALOT_ATTN_ACT_QUANT_O`, and the indexer publish fix this runtime already has). `encoding/encoding.py` is the
  2026-10-16 revision, which renders all 142 recorded Hermes requests identically (0.71.0).
- The model card's own numbers against V4.1-Flash: Terminal-Bench 4.0 31.2 to 42.3, ProgramBench 20.3 to 29.1, DeepSWE
  74.2 to 77.0, Automation-Bench 54.8 to 60.5, HLE 36.8 to 42.6, with shorter outputs. Not reproduced here.

What follows from the same shapes: the bank, the expert record (9.95 MB at 2-bit g128), the trunk and every kernel are
unchanged, so a token costs what it costs today, up to whatever the new routing does to the hit rate. Engram tables are
read straight from the checkpoint shards. Disk snapshots need no change: their identity
(`src/cachalot/model/snapshot_store.py`, `runtime_identity`) already includes the checkpoint's `config.json` and index
(path, size, mtime) and the bank's files, so a 1017 snapshot can never be loaded into the old model or the reverse.

## 2. Disk plan

| | free now | after the download | after step 3 | after step 4 |
|---|---|---|---|---|
| X10Pro | 599 GiB | ~124 GiB | ~124 GiB | ~124 GiB |
| internal | 102 GiB | 102 GiB | ~244 GiB | ~101 GiB |

Neither drive can hold the new 142.4 GiB bank beside everything that is there today. The plan deletes the internal copy
of the old bank (its byte-identical twin stays on the X10Pro, and `serve.sh` falls back to it on its own, at the
X10Pro's 1 GB/s) and builds the 1017 bank on the internal SSD, where it will be served from. The old checkpoint stays,
so going back costs nothing but the slower old bank until it is copied back.

## 3. Steps

Estimates are wall time on this machine. One runtime at a time; Hermes and other GPU-heavy apps closed for steps 5 and 6.

### Step 0 — download (Hamed, already handed over)

```bash
caffeinate -i hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-1017
```

475 GiB; time depends on the connection. Re-running the same command resumes.

### Step 1 — verify the download (~10 minutes with hashes)

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ~/venvs/deepseek-v41/bin/python -I scripts/verify_hf_download.py deepseek-ai/DeepSeek-V4.1-Flash-1017 /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --sha256
```

Pass: last line `... 0 bad ...`, exit status 0. Then the runtime's own layout check:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python -m cachalot.cli doctor --model /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-1017
```

### Step 2 — free the internal SSD (**Hamed's call**, ~7 minutes for the compare)

First prove the X10Pro copy is still byte-identical to the internal one (it was on 2026-10-03, HANDOFF section 18.51):

```bash
for f in /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128/*.safetensors /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128/config.json; do cmp "$f" "/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q2g128/$(basename "$f")" || echo "DIFFERS: $f"; done; echo compare-done
```

Only if it prints `compare-done` and no `DIFFERS` line, delete the internal copy. **This cannot be undone**; restoring it
is a 142 GiB copy back from the X10Pro (~4 minutes plus a 7-minute compare):

```bash
rm -rf /Users/hamedprooshani/DeepSeek-V4.1-Flash-q2g128
```

From here until step 7, `./serve.sh` and `./chat.sh` still serve the old model, from the X10Pro bank (slower misses).

### Step 3 — build the 1017 bank on the internal SSD (~35-50 minutes)

The shipped bank's exact recipe (2-bit, group 128, the searched fit), from the new checkpoint:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/guarded_run.sh --budget-gib 8 --max-seconds 14400 --tag bank1017 -- env PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/build_affine_bank.py --model-path /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-1017 --out /Users/hamedprooshani/DeepSeek-V4.1-Flash-1017-q2g128 --bits 2 --group 128 --fit search --verify 12
```

The old bank took 33 minutes (HANDOFF-2026-09-17). An interrupted build resumes with the same command. When it exits 0
with 40 shards, mark it trusted so the launch scripts pick it up:

```bash
[ "$(ls /Users/hamedprooshani/DeepSeek-V4.1-Flash-1017-q2g128/*.safetensors | wc -l | tr -d ' ')" = 40 ] && [ -f /Users/hamedprooshani/DeepSeek-V4.1-Flash-1017-q2g128/config.json ] && touch /Users/hamedprooshani/DeepSeek-V4.1-Flash-1017-q2g128/.cachalot-verified && echo marked
```

### Step 4 — smoke test (~3 minutes)

The launch scripts derive the bank and hotlist from the checkpoint directory's name (0.71.2), so only the model path
changes. A separate snapshot directory keeps the old model's saved system blocks from being pruned:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-1017 CACHALOT_SNAPSHOT_DIR=$HOME/.cache/cachalot/prefix-snapshots-1017 ./chat.sh "Write a C# method that returns the n-th Fibonacci number using BigInteger."
```

Pass: the startup line reads `expert bank: /Users/hamedprooshani/DeepSeek-V4.1-Flash-1017-q2g128 (..., 2-bit, 9.49 MiB/expert, 15360 experts)`, there is no `hotlist:` line (1017 has no hotlist yet), and the answer is coherent. The bank the scripts resolve for a model path was checked when 0.71.2 was made: the 1017 path gives `/Users/hamedprooshani/DeepSeek-V4.1-Flash-1017-q2g128` once it carries the marker, else the same name beside the checkpoint on the X10Pro, which does not exist, so the run fails rather than mixing revisions.

### Step 5 — quality gate against the old checkpoint (~75-95 minutes)

`benchmarks/pareto_arms.v1017.json`: the 1017 reference, its rounding arm, 1017 with `act_quant(o)`, and the old
checkpoint. Each arm is its own guarded process; the old arm reads its bank from the X10Pro, so its timing column is not
comparable and it takes longest.

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/settle.sh --budget-gib 48 && benchmarks/guarded_run.sh --budget-gib 48 --max-seconds 2400 --tag p1017_ref -- env PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/pareto.py run v1017 --arms benchmarks/pareto_arms.v1017.json --out benchmarks/results/pareto-v1017
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && for arm in noise-chunk96 v1017-actq; do benchmarks/settle.sh --budget-gib 48 && benchmarks/guarded_run.sh --budget-gib 48 --max-seconds 2400 --tag p1017_$arm -- env PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/pareto.py run $arm --arms benchmarks/pareto_arms.v1017.json --out benchmarks/results/pareto-v1017 --ref benchmarks/results/pareto-v1017/v1017.npz || break; done
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && benchmarks/settle.sh --budget-gib 48 && benchmarks/guarded_run.sh --budget-gib 48 --max-seconds 5400 --tag p1017_old -- env PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/pareto.py run old --arms benchmarks/pareto_arms.v1017.json --out benchmarks/results/pareto-v1017 --ref benchmarks/results/pareto-v1017/v1017.npz
```

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/pareto.py report benchmarks/results/pareto-v1017 --arms benchmarks/pareto_arms.v1017.json
```

How to read it: against a different checkpoint, KL and top-1 measure difference, not damage. The two numbers that
compare the models are the old arm's paired NLL difference (positive: the old weights predict the fixed prose, code and
markdown worse than 1017) and the 21-task pass rates (C# that must `dotnet build`, Python against asserts, JSON, exact
answers). `noise-chunk96` must sit inside the band as usual, which confirms the 1017 bank behaves like the old one under
the runtime. Speed is the reference arm's ms a step against earlier 48 GiB references on the internal bank (~140 ms).

### Step 6 — a live Hermes session (Hamed drives it)

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-1017 CACHALOT_SNAPSHOT_DIR=$HOME/.cache/cachalot/prefix-snapshots-1017 CACHALOT_SERVER_DUMP=$HOME/cachalot-dump-1017.jsonl ./serve.sh
```

The first request prefills the system block cold (about 100-170 s at 16k tokens; DI-1 off is the default). Read the dump
for tool-call format, thinking and image turns as in earlier live checks.

### Step 7 — a routing trace and the 1017 hotlist (one session, not a timing run)

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && CACHALOT_MODEL_PATH=/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-1017 CACHALOT_SNAPSHOT_DIR=$HOME/.cache/cachalot/prefix-snapshots-1017 CACHALOT_ROUTING_TRACE=$HOME/cachalot-routing-1017-1.trace.npz ./serve.sh
```

After a real working session, stop the server (the trace is saved at exit) and build the hotlist under the name the
launch scripts look for:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/build_hotlist.py $HOME/cachalot-routing-1017-1.trace.npz --out /Users/hamedprooshani/cachalot-hotlist-DeepSeek-V4.1-Flash-1017.json
```

### Step 8 — make 1017 the default (**Hamed's call**)

If steps 5 and 6 favour 1017: change the `CACHALOT_MODEL_PATH` default in `serve.sh`, `chat.sh` and `benchmarks/pareto.py`
(`serve_defaults`) to the 1017 directory, re-record the DeepSeek floor and hit rate at 48 GiB (the routing changed), and
decide whether the old checkpoint (475 GiB) and its X10Pro bank (142 GiB) stay as a fallback.
