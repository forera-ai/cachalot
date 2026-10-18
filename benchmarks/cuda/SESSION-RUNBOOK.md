# CUDA session S1 runbook: RTX 5090, inventory, smoke, baseline

**How S1 was actually run (2026-10-18):** after the instance existed, Hamed asked the agent to do the work over SSH with the session's disposable key. The agent used a private ssh config outside the repository (`~/cachalot-private/ssh_config`, host alias `cuda-s1` with the provider's address, mapped port, the provider's default user, key `~/.ssh/cachalot_cloudrift`, a private known-hosts file) so that the commands below work with `-F ~/cachalot-private/ssh_config cuda-s1` in place of `-i "$CUDA_KEY" "$CUDA_HOST"`. The provider maps SSH to a non-standard external port, shown beside the address in the console; the first host key is trusted on first use. No sudo was used. The manual steps below remain the way to run a session without the agent.

Operator: Hamed, from his own terminal over SSH. The agent holds no key and no token and sees only what is pasted back. Decisions D1 to D5: `docs/CUDA-RESEARCH-PLAN.md` section 17.1. Predictions and stop rules: `docs/CUDA-S1-EXPERIMENT-RECORD.md` (read its section 2 first). Cap: **2 GPU-hours** of billed time, setup included. Nothing in this runbook starts an instance; you do that in the provider console.

Variables you set once per terminal. Hostnames, user names and addresses stay in your terminal and are never pasted into chat or files:

```bash
export CUDA_HOST='REPLACE_WITH_USER@REPLACE_WITH_HOST'
export CUDA_KEY="$HOME/.ssh/cachalot_cloudrift"
export CUDA_REGION='REPLACE_WITH_COUNTRY'
```

## 0. Before the clock starts (nothing billed yet)

1. Fill the private cost file `~/cachalot-private/cuda-cost-log.md`: hourly rate of the RTX 5090 type, billing unit and minimum, stop-versus-terminate disk billing, credit balance, expiry date, cumulative cap. Work out the session cost cap (rate x 2 h) and write it there. Do not tell the agent the amounts.
2. SSH key: the private key you generated must not sit in `Downloads`. Move it, lock it, and check that it matches the public key you registered with the provider (this prints only the public half):

```bash
mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
install -m 600 "$HOME/Downloads/cachalot-ssh.txt" "$HOME/.ssh/cachalot_cloudrift"
ssh-keygen -y -f "$HOME/.ssh/cachalot_cloudrift"
```

The line printed must be the same `ssh-ed25519 ...` public key you gave the provider. If it is, delete the copy in `Downloads` yourself (`rm "$HOME/Downloads/cachalot-ssh.txt"`). If the private key has a passphrase, `ssh-add "$HOME/.ssh/cachalot_cloudrift"` once.
3. Rotate the CloudRift API key that was pasted into the first chat, in the provider console. This runbook does not use it.
4. Pack the scaffold from a clean commit (the agent commits first and tells you when). It writes a `REVISION` file into the package, never into the repository:

```bash
cd /Users/hamedprooshani/Projects/deepseek-v41-mac && bash benchmarks/cuda/pack.sh
```

Note the archive path and the sha256 it prints.

## 1. Provision (console, billed from here)

Start the RTX 5090 instance type in the provider console. Choose an image that already has an NVIDIA driver of the 570 series or newer (the pinned wheel is CUDA 12.8). Enable auto-shutdown if the provider offers it. Write the **start time (UTC)** into the cost file now. Set `CUDA_HOST` from the console's SSH details.

Connect and look around (the first connection asks you to trust the host key; compare its fingerprint with the console if the console shows one):

```bash
ssh -i "$CUDA_KEY" -o IdentitiesOnly=yes "$CUDA_HOST" 'echo connected'
```

## 2. Upload and check (about 5 minutes)

Replace the archive file name with the one `pack.sh` printed:

```bash
cd "$HOME/cachalot-private/cuda-packs" && ARCHIVE="$(ls -t cuda-scaffold-*.tar.gz | head -1)" && shasum -a 256 "$ARCHIVE" && scp -i "$CUDA_KEY" -o IdentitiesOnly=yes "$ARCHIVE" "$CUDA_HOST:cuda-scaffold.tar.gz"
ssh -i "$CUDA_KEY" -o IdentitiesOnly=yes "$CUDA_HOST" 'sha256sum cuda-scaffold.tar.gz && mkdir -p cachalot-cuda && tar -xzf cuda-scaffold.tar.gz -C cachalot-cuda && cd cachalot-cuda && bash benchmarks/cuda/session.sh check'
```

The two sha256 values must be equal. Paste the `check` output to the agent (it contains no addresses and no UUIDs). **Stop here if** `nvidia-smi` fails, the driver is older than 570, or the `venv` module is missing; do not install anything with `sudo`, tell the agent.

## 3. Setup (about 10 to 15 minutes, installs pinned wheels)

Open an interactive session so the install survives a dropped connection, then run setup:

```bash
ssh -i "$CUDA_KEY" -o IdentitiesOnly=yes "$CUDA_HOST"
```

On the instance:

```bash
cd ~/cachalot-cuda && bash benchmarks/cuda/session.sh setup
```

The script prints the two `pip install` lines it will run before it runs them: `torch==2.7.1` from the official PyTorch `cu128` index and `numpy==2.2.6`. These pins are expectations from general knowledge, not facts about the machine; the line `arch list` at the end is the real answer. **Stop if** it does not contain `sm_120` (the toolchain gate, prediction S1-P1): end the session, copy the output home, and the agent revises the pin on the Mac.

## 4. Inventory (1 minute)

```bash
cd ~/cachalot-cuda && bash benchmarks/cuda/session.sh doctor
```

Paste the printed manifest to the agent (environment variables are allow-listed and the file is checked for addresses, UUIDs, e-mails and key-shaped strings before it is written). Check against the record: capability 12.0, free VRAM, the negotiated PCIe generation and width (`interconnect`).

## 5. Smoke run (45 seconds of wall clock)

```bash
cd ~/cachalot-cuda && bash benchmarks/cuda/session.sh smoke
```

It prints one line: status, row count, seconds of the 45 s cap. Then look at the run manifest gates:

```bash
cd ~/cachalot-cuda && python3 -c "import json,glob; p=sorted(glob.glob('benchmarks/results/cuda/*-smoke-*/run-manifest.json'))[-1]; m=json.load(open(p)); print(m['status'], m['gates'], m['skipped'])"
```

Paste both to the agent. Exit code 4 or any `failed_cells` means the first-run defect the record gave a 0.4 chance: **stop**, go to section 7, and the agent fixes it on the Mac.

## 6. Baseline run (up to 300 seconds)

Only if the smoke run is clean. The script refuses otherwise:

```bash
cd ~/cachalot-cuda && bash benchmarks/cuda/session.sh baseline
```

Then, still on the instance:

```bash
cd ~/cachalot-cuda && bash benchmarks/cuda/session.sh results
```

It prints the results archive path and its sha256. Leave the interactive session (`exit`).

## 7. Copy back, verify, and the agent reads it (on the Mac)

The results arrive as untrusted data: unpack them into a new empty directory, run Python with `-I`, and let the agent read only the validated files.

```bash
cd "$HOME/cachalot-private" && mkdir -p cuda-results && cd cuda-results && D="s1-$(date -u +%Y%m%dT%H%M%SZ)" && mkdir "$D" && scp -i "$CUDA_KEY" -o IdentitiesOnly=yes "$CUDA_HOST:cachalot-cuda/cuda-s1-results-*.tar.gz" "$D/" && shasum -a 256 "$D"/*.tar.gz && tar -tzf "$D"/*.tar.gz | head -40
```

The sha256 must equal the one the instance printed, and the listing must show only relative `results/cuda/...` paths. Then:

```bash
cd "$HOME/cachalot-private/cuda-results" && D="$(ls -d s1-* | tail -1)" && tar -xzf "$D"/*.tar.gz -C "$D" --no-same-owner && /Users/hamedprooshani/venvs/deepseek-v41/bin/python -I /Users/hamedprooshani/Projects/deepseek-v41-mac/benchmarks/cuda/cuda_bench.py validate "$D"/results/cuda/*/*.json "$D"/results/cuda/*/*.jsonl
```

Tell the agent the directory name; it reads the files from there.

## 8. Teardown (before anything else, even if the session failed)

1. Results copied off the instance and checksum-verified (section 7).
2. `nvidia-smi` on the instance shows no process using the GPU.
3. Instance stopped or terminated **in the console**, and the console shows "stopped" or "terminated", not "stopping". Check what a stopped instance still bills for (disk) and decide on purpose.
4. Any attached volume or snapshot kept or deleted on purpose.
5. **End time (UTC)** and hours billed written into the cost file; cap checked against the cumulative cap.
6. The SSH key used for the session removed from the provider account if it was issued only for this session. Credentials rotated.
7. Tell the agent the number of GPU-hours (a count); the agent never needs the amount.

## Time and cost budget (estimates, to be replaced by the first measured session)

| step | minutes |
|---|---:|
| provision and connect | 5 to 10 |
| upload and check | 5 |
| setup (wheel download and install) | 10 to 15 |
| inventory | 1 |
| smoke | 1, plus reading |
| baseline | 5 to 8 |
| copy back and teardown | 10 |
| reading and pasting between steps | 10 to 20 |

Total about 50 to 70 minutes, cap 120. Cost is the hourly rate times the hours the console reports; the agent does not know the rate.

## If something goes wrong

| symptom | action |
|---|---|
| `nvidia-smi` fails or the driver is older than 570 | stop, tell the agent; no driver changes from this runbook |
| `python3 -m venv` fails | stop; installing the OS package needs `sudo` and your decision |
| `arch list` lacks `sm_120` | stop, copy the output, terminate; the agent revises the pin on the Mac |
| smoke `FAIL` or `failed_cells` | stop, go to section 7, terminate; fixes happen on the Mac, not on the billed clock |
| smoke `truncated` | stop; do not run the baseline; report the elapsed seconds |
| session reaches 80 % of its cap (96 minutes) before the baseline | skip the baseline, do section 7 and 8 |
| connection drops | reconnect and continue from the last step; `session.sh` steps are idempotent except `setup` (safe to rerun) |
