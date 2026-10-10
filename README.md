<div align="center">

# 🐋 Cachalot

**Run DeepSeek V4.1 Flash (552B MoE) on a single Apple Silicon Mac, streaming experts from SSD.**

[![License: MIT](https://img.shields.io/badge/License-MIT-f5de53.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![MLX](https://img.shields.io/badge/MLX-0.32%2B-black.svg)](https://github.com/ml-explore/mlx)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Apple%20Silicon-lightgrey.svg)](#hardware)
[![CI](https://github.com/forera-ai/cachalot/actions/workflows/ci.yml/badge.svg)](https://github.com/forera-ai/cachalot/actions/workflows/ci.yml)

Maintained by [forera-ai](https://github.com/forera-ai).

*A cachalot is a sperm whale: it dives deeper than anything else its size and comes back up with what it went for. This runtime does the same with a 475 GB checkpoint on a 96 GB machine.*

</div>

---

## What it is

Cachalot is an inference runtime, written from scratch in Python + [MLX](https://github.com/ml-explore/mlx) + hand-written Metal kernels, for the official
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) checkpoint. The model has 552B parameters,
289 GB of which are FP4 routed experts that never fit in unified memory. Cachalot keeps the dense trunk resident,
holds a bounded working set of experts in GPU-visible memory, and streams the rest from SSD on demand,
prefetching ahead of the compute that needs them.

It exposes the model as an **OpenAI-compatible HTTP server**, so agent harnesses such as OpenCode, Hermes, Continue, aider,
or any `openai` SDK client can use it as a drop-in local model.

The configuration that ships serves the routed experts from a **2-bit affine g128 bank built here from the FP4
checkpoint** — 9.49 MiB per expert against FP4's 17.93, which halves the bytes a token reads. On the 40-case
coding gate it is equal to a hosted FP4 and a hosted FP8 reference arm on every column.

It is **not** a port of the PyTorch reference and it is **not** a generic MLX model loader. Every component was
implemented against the released `inference/model.py` semantics and validated token-for-token, including the parts
that make V4.1 Flash unusual:

| V4.1 Flash component | Cachalot implementation |
|---|---|
| Single-pass mHC hyper-connections (Sinkhorn mixing) | Fused single-dispatch Metal kernel |
| Compressed Sparse Attention 2: Full / Reindex / Reuse layer modes | Exact source / index-only / reuse block topology, shared cross-layer KV and top-k |
| Hierarchical sparse indexer with candidate blocks (layer 20 → 24/28/32/36) | Implemented, per-token candidate replay in layer-major prefill |
| FP4 E2M1 expert weights with E8M0 block scales | Custom Metal GEMV, bit-exact dequantization tables |
| FP8 E4M3 trunk with dynamic activation quantization | Custom Metal GEMV, official `act_quant` semantics |
| Engram conditional memory (196B params, 2 × 384M-row tables) | Sparse `mmap` row reads, exact n-gram hashing and token normalization |
| Sliding-window attention over a bounded 128-position window, FP4 KV cache (E2M1 + E4M3 scale per 16) | Implemented |
| Official chat protocol (`encoding.py`), thinking mode, reasoning effort | Loaded from the checkpoint, not reimplemented |

Image input works end to end through the server (OpenAI `image_url` content parts). The OpenAI-compatible
server (`./serve.sh`) is tested against Hermes Agent: parallel tool calls, file writes, resumed sessions and
images. DSpark speculative decoding is not implemented (see [Roadmap](#roadmap)).

**Prefill runs the full prompt through all 40 layers.** The row above used to read "SWA bounded replay",
which a 2026-09-19 review reasonably read as the decoder replay shortcut in §3.2.2 of DeepSeek's technical
report -- process the whole prompt through the encoder, build the decoder's global KV from its output, and
replay only the final 128 positions through the decoder. **That is not implemented**, it is approximate by
DeepSeek's own account because sliding-window dependencies accumulate across layers, and it would be an
opt-in research branch rather than a drop-in speedup. The bounded window the row refers to is the attention
window itself.

## Why this exists

A 552B MoE activates only 8B parameters per token during prefill and 16B during decode, so the *compute* fits a Mac
Studio comfortably. The *bytes* do not: each token touches 240 routed experts (40 layers × top-6), and the
15,360 of them total **289 GB as FP4** — 145 GB re-quantized to the 2-bit bank that ships, which is still
more than the machine has. Any runtime for this class of machine is therefore an exercise in **caching and
I/O scheduling** before it is an exercise in kernels. Cachalot is built around that fact:

```
                       ┌──────────────────────────────────────────────┐
  HTTP (OpenAI API) ─▶ │  cachalot serve   ── request queue ──▶ V41Model │
                       └──────────────────────────────────────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     │                 TextDecodeRuntime                    │
                     │  layer-major prefill (expert-major MoE)  ·  decode   │
                     └──────────────┬───────────────────────────┬──────────┘
                                    │                           │
              ┌─────────────────────▼──────────┐   ┌────────────▼─────────────────┐
              │ Resident trunk (~12 GiB, MLX)  │   │ ResidentExpertStore           │
              │ attention · router · shared    │   │ wired slot pool (auto-sized)  │
              │ expert · norms · heads · RoPE  │   │ per-layer quotas ·            │
              └────────────────────────────────┘   │ deterministic admission ·     │
                                                   │ cross-turn reuse              │
                                                   └────────────┬─────────────────┘
                                                                │ miss
                                                   ┌────────────▼─────────────────┐
                                                   │ Loader workers (8 threads)    │
                                                   │ preadv() straight into the    │
                                                   │ slot's unified memory         │
                                                   └────────────┬─────────────────┘
                                                                │
                                                   ┌────────────▼─────────────────┐
                                                   │ safetensors shards on SSD     │
                                                   │ 289 GB routed experts         │
                                                   │ Engram tables (mmap)          │
                                                   └──────────────────────────────┘
```

## Status

Cachalot is **alpha**. It produces reference-quality output and runs multi-turn sessions at 9.4–9.6 tok/s
on the configuration in [Performance](#performance) (0.9.x sessions; 7.9 tok/s through `./serve.sh` on 0.54.0, same section). Decode is no longer bound by SSD bandwidth — the drive is
idle 45 % of the time — and is now limited by the share of experts that are already resident. Read
[Performance](#performance) before deciding whether it fits your use.

| Area | State |
|---|---|
| Text generation, official chat protocol, thinking mode | ✅ working |
| Second model: GLM-5.3-Flash (MLX 4-bit), experts streamed from SSD (`./serve-glm.sh`, `./chat-glm.sh`) | ✅ 0.50.0 (internal SSD again; decode reads the next layer's likely experts early, -9 to -13 % a token with the same tokens, 2.5 -> 3.1 tok/s at a 52 GiB budget; a 64 GiB budget pages and is slower): 0.18.0: text, tools, thinking, prefix cache in memory and on disk across restarts; prefill ~90 tok/s, decode 3.3-3.6 tok/s, 14.3 all-resident (from the internal SSD; the copy lives on the X10Pro since 0.52.3, where decode measures 0.66 tok/s, 1.52 s a token, drive-bound, 0.61.1); vision, MTP not yet Caveat (0.51.6): its code output is corrupted in most of the C# replays; not the decode path (HANDOFF 18.43), not the 4-bit attention and shared experts either (an 8-bit hybrid garbled 12 of 12, 18.49) |
| Third model: MiniMax-M3 (MLX 3-bit), experts streamed from SSD (`./serve-minimax.sh`, `./chat-minimax.sh`) | ✅ 0.48.0: text, tools, thinking, the checkpoint's top_p 0.95 for chat and for requests that omit it, temperature 0.7 for server requests that send none (the terminal chat samples at 0.7 too since 0.48.0), prefix cache in memory and on disk, expert cache kept across restarts; prefill ~240 tok/s at 16k (a 17k agent block in 101 s), 148 tok/s at 64k, a 150-token tool result in ~8.5-9 s, 1,000 tokens in ~13 s; decode 4.7-7.1 tok/s after a 2k prefill (4.8-7.5 through the server, 3.5-6.3 at a 25k agent context with the display on, measured on 0.25.0); the expert cache shrinks as the KV cache grows, so long contexts stay below the memory ceiling; experts read from a bias-free bank (6 % fewer bytes, byte-identical outputs; prefill reads take a compressed copy of each record's scales, 6 % fewer again) with direct reads, a second copy on the X10Pro adds read bandwidth; hit experts computed while misses load; prefill reads ahead the next layer's predicted experts; the expert cache is warmed between requests; a 68 GiB expert cache (62 without `sysctl iogpu.wired_limit_mb=88064`; decode -11 % at 68) under a memory governor that gives slots back before long prefills, under memory pressure and when other apps' GPU memory would push the GPU into paging (agent turns decode -15 % against 56 GiB, byte-identical); experts stored on disk exactly as they sit in memory (5 % fewer bytes a read, agent turns decode -5 %, byte-identical); the first turn after a restart starts at once instead of waiting for the saved expert set; decode reads the next two layers' missing experts as soon as a layer misses, from the routing the GPU already ran without them (agent turns decode -15 %, byte-identical); expert slots keep each weight group's scale and offset as one byte into a small per-projection table (142 more cached experts than 0.31.0 in the same memory, agent turns decode -9 %, byte-identical); decode picks its experts on the GPU and the host checks each layer one step behind (agent turns -9.5 %, prose -21 %, byte-identical); short follow-ups at an agent's 30k context in ~6 s instead of ~17 (decode keeps MLX's buffer cache small, so the GPU stays under its working set) and saved system blocks kept on disk rather than in GPU memory between conversations; miss substitution on by default since 0.43.0, which changes outputs slightly (quality measured at or near rounding noise; `CACHALOT_MINIMAX_MISS_DROP=0 CACHALOT_MINIMAX_PREFILL_MISS_DROP=0` for the exact path): decode skips its lightest missing experts (+26 % decode on a checkable-task battery) and prefill does the same where it saves a read (short follow-ups -20 %); the first request after a restart reuses an agent's saved system block in 2-10 s (0.44.0; 0.43.x prefilled it cold, 100-320 s); the memory governor watches macOS memory pressure between its fits, so other apps' memory no longer turns short follow-ups into 30-40 s stalls; a GQA decode kernel from 4k context; replies stop when they loop or count up an invented list (0.40.0, 0.46.0); a lower temperature for agents measured (1.0 derails a long tool result 5 times in 12, 0.7 once) and made the server's default (0.46.1, 0.47.0); the terminal chat restores the expert cache and the prompt head after a restart and has `/stats`, `/clear`, `/help` |
| Layer-major prefill with expert-major MoE scheduling | ✅ working |
| Auto-sized, wired expert slot pool with zero-copy SSD loads | ✅ shipped |
| Cross-turn expert residency | ✅ working, validated |
| MLX allocator tuning (2 GiB free-buffer cap) | ✅ shipped, removed 100–380 ms allocation stalls |
| Routing trace + offline cache-policy analysis | ✅ `benchmarks/` |
| Unit tests without checkpoint | ✅ `pytest -q` |
| OpenAI-compatible server (`/v1/chat/completions` SSE, tools, thinking, images, `/v1/completions`) | ✅ working, tested with Hermes Agent (0.10.0) |
| Prefix cache (only new tokens are prefilled per turn) | ✅ working; snapshots where the system prompt ends, so a new agent session reuses all of it (206 s → 1.1 s), and keeps that snapshot on disk across server restarts (163 s → 3.3 s, 0.11.0); an agent's re-serialized tool calls reuse the model's own reply tokens (0.12.0); the chunk snapshots inside a system block are pinned, so a mid-block change such as Hermes compression reuses up to the last chunk before it (0.13.0); snapshots on disk survive upgrades (0.14.0); parallel agents keep their own latest turn, with a 1.5 GiB byte budget and eviction by tier (0.15.0); the disk keeps 32 system blocks by last use and loads the ones not preloaded when a request needs them, so a batch of subagents no longer pushes the main agent's block off it (0.16.0) |
| Chunked prefill (4096 tokens per call) | ✅ shipped 0.10.0; a 13.5k-token agent prompt no longer runs the Metal heap out |
| `cachalot serve / chat / doctor / bench` CLI | ✅ working |
| Parallel loading of a decode layer's expert misses | ✅ shipped |
| Fused top-k expert Metal kernels, bf16 head GEMV | ✅ shipped |
| Fused decode path: router top-k, sparse attention, hyper-connection mixes, RoPE/RMSNorm, FP8 quantization + vectorized FP8 GEMV | ✅ shipped, all-resident token 0.10 → 0.068 s |
| FP4 expert GEMM on simdgroup matrix units for prefill | ✅ shipped, 2.5× less GPU time per expert |
| Speculative next-layer expert loads + background Engram rows in prefill | ✅ shipped, SSD busy 59 % → 91 % of a 2048-token prefill |
| Auto budget capped by memory available at start | ✅ shipped |
| Kernel warm-up at load (first token 1 s → 0.35 s) | ✅ shipped |
| Second checkpoint copy on another drive, byte-striped expert reads (`CACHALOT_MIRROR_PATH`) | ⚠️ shipped, **off on the 2-bit bank**: an 18 % loss at 9.49 MiB per expert where it was a gain at 17.93 |
| 2-bit affine g128 expert bank, half the bytes of FP4 at equal gate quality | ✅ shipped, `benchmarks/build_affine_bank.py` |
| Traced decode MoE block and memoised kernel parameters | ✅ shipped, all-resident token 85 → 76.4 ms |
| Coding-quality gate against a hosted reference arm (40 cases) | ✅ 20/20 C++ compile, 0/101 malformed includes |
| `./chat.sh` launcher for the shipped configuration | ✅ shipped |
| Batched prefill (attention for all 40 layers, HC, router, routed + shared experts, Engram) | ✅ shipped |
| DSpark / MTP speculative decoding | ⛔ measured and closed twice; needs a decode-shaped multi-position forward first |
| Vision (ViT + aligner, image spans in prefill, `image_url` in the server) | ✅ DeepSeek: working 0.10.0; reads a real chart's every value correctly. Tower loads on first image (~1 s, 0.9 GiB). ✅ GLM-5.3-Flash since 0.51.0 (its own 24-block ViT, bit-identical to mlx-vlm's in 0.51.5, images keyed by content hash in the prefix cache; two charts read exactly, a resent image reuses its prefix; an image in a system message reads correctly too, 0.51.10; video since 0.52.0: a `video_url` part is sampled at 2 fps through ffmpeg, matches mlx-vlm's processor bit for bit, a 1.2k-token clip prefills in 25 s) |

## Hardware

Tested on a **Mac Studio M3 Ultra, 96 GB unified memory, 60-core GPU**, with the checkpoint on the internal SSD
and, earlier, on a Crucial X10 Pro over USB 3.2 Gen 2.

Requirements:

- Apple Silicon Mac with **≥ 64 GB unified memory**. The expert budget auto-sizes to the machine:
  ~14 GiB of experts on 64 GB, ~50 GiB on 96 GB, ~73 GiB on 128 GB, the full 270 GiB on 512 GB
  (at which point the SSD is only touched at load time).
- macOS 14+ with Metal 3 or newer.
- **~480 GB** of storage for the checkpoint. Storage speed is the single largest performance factor;
  `cachalot doctor` measures yours:

| Storage path | Sequential read | Measured effect |
|---|---:|---|
| USB 3.2 Gen 2 SSD | ~1.0 GB/s | FP4 decode 1.3 s/token, cold 512-token prefill 8–9 min |
| Thunderbolt 4/5 NVMe enclosure | 3–6 GB/s | Proportionally faster misses, no code change |
| Internal Mac SSD (tested) | 5.2 GB/s | **2-bit bank: 0.128 s/token interactive, 16.4 s cold 512-token prefill**; FP4 0.33 s/token |

The loader saturates a USB SSD at queue depth 1 (17.7 ms per expert read) and reads at 5.7 GB/s from the internal
disk, so more threads do not help; faster storage does.

## Install

```bash
git clone https://github.com/forera-ai/cachalot.git
cd cachalot
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[server]"
```

Download the checkpoint (≈475 GB) to a fast disk:

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash --local-dir /Volumes/FastSSD/DeepSeek-V4.1-Flash
```

## Quick start

```bash
# Check hardware, memory budget, storage speed and checkpoint layout
cachalot doctor --model /Volumes/FastSSD/DeepSeek-V4.1-Flash

# One-shot prompt
cachalot chat --model /Volumes/FastSSD/DeepSeek-V4.1-Flash "Explain unified memory in two sentences."

# Interactive session (/clear, /exit)
cachalot chat --model /Volumes/FastSSD/DeepSeek-V4.1-Flash

# OpenAI-compatible server on http://127.0.0.1:8000
cachalot serve --model /Volumes/FastSSD/DeepSeek-V4.1-Flash --port 8000
```

Then from any OpenAI client:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "Write a haiku about sperm whales."}],
    "stream": true
  }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="cachalot")
for chunk in client.chat.completions.create(
    model="deepseek-v4.1-flash",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Python API

```python
from cachalot import V41Model

with V41Model.from_pretrained("/Volumes/FastSSD/DeepSeek-V4.1-Flash") as model:
    reply = model.chat(
        [{"role": "user", "content": "What is a Sinkhorn iteration?"}],
        max_new_tokens=256,
        temperature=0.6,
        thinking_mode="thinking",
    )
    print(reply.content)
```

`V41Model` is the stable public boundary. Everything below it (`TextDecodeRuntime`, caches, kernels) may change
between minor versions.

Harness setup (OpenCode, Hermes, aider, Continue, OpenAI SDK): [docs/integrations.md](docs/integrations.md).

## Configuration

All knobs live in `cachalot.config.RuntimeConfig` and can be overridden on the CLI or via `CACHALOT_*` environment variables
(`CACHALOT_EXPERT_CACHE_BUDGET_GIB=48`, `CACHALOT_MODEL_PATH=...`, `CACHALOT_MAX_SEQ_LEN=...`, `CACHALOT_PORT=...`).
If you keep a second identical copy of the checkpoint on another drive, `CACHALOT_MIRROR_PATH=/Volumes/.../DeepSeek-V4.1-Flash`
makes every expert read fetch its tail from that drive concurrently (`CACHALOT_MIRROR_FRACTION`, default 0.10 = the
share of bytes for the second drive; use its bandwidth divided by the total). **Whether this helps is a property of
the expert size, not of the drives**: with 17.93 MiB FP4 experts it is worth ~5 % on decode and ~12 % on short
prefills, and with the 9.49 MiB 2-bit experts that ship it is an 18 % *loss*, because the per-read latency it adds
stops being amortized. It is off in the shipped configuration. See `docs/HANDOFF.md` section 9.11.1.

| Setting | Default | Meaning |
|---|---:|---|
| `expert_cache_budget_bytes` | auto | Routed experts kept resident. Auto = unified memory − 12 GiB trunk − 2 GiB MLX cache − 32 GiB system reserve, capped at the Metal recommended working set. `--expert-budget-gib 40` for a fixed value. |
| `mlx_wired_limit_bytes` | auto | Trunk + experts + cache are wired via Metal residency sets so macOS cannot compress them under pressure (without this decode was 3× slower, see [docs/performance.md](docs/performance.md)). |
| `system_reserve_bytes` | 32 GiB | Memory left for macOS, page cache and other applications when auto-sizing. Lower it on a dedicated machine. |
| `mlx_cache_limit_bytes` | 2 GiB | Cap on MLX's free-buffer cache. Larger values recreated allocator stalls under concurrent materialization. |
| `io_workers` | 8 | Loader threads. Bandwidth-bound; more threads do not raise throughput on USB SSDs. |
| `CACHALOT_ENGRAM_PARALLEL_MIN` | 8 | Engram row batches at or above this size are read through the reader's 16-worker pool instead of by the calling thread. A decode token asks for 24 rows twice; at the old threshold of 64 those 96 `pread`s went out one at a time from the decode thread and cost 28 ms per token behind a busy drive. |
| `CACHALOT_DECODE_ENGRAM_PREFETCH` | 1 | Issue both Engram layers' row reads at the top of the token rather than at the layer that consumes them. The row ids depend only on the token being decoded, so layer 14's read hides behind thirteen layers of compute. |
| `CACHALOT_PREFILL_KEEPALIVE` | 0.5 | Seconds between one-element GPU evals while a prefill chunk waits for its Engram rows. Without it the GPU queue idles for seconds, macOS un-wires the model and pages it back over 10-15 s per chunk; with it a 12k prefill is 15 % faster, bit-identical. `0` disables. |
| `prefix_cache_bytes` | 1.5 GiB | Memory for prefix-cache snapshots (~5 MB + ~3 KB per token each; `prefix_cache_entries`, 64, is only a ceiling). Evicted by tier: earlier turns already contained in a later snapshot first, then chunk snapshots inside a system block, then each conversation's latest state, system blocks last. |
| `--default-max-tokens` | 8192 (`serve.sh`) | New tokens for a request that sends no `max_tokens`, shortened to what fits in `max_seq_len`. At 2,000, agent context summaries and long tool calls were cut off. |
| `max_seq_len` | 32768 | Sequence capacity for KV and compressed caches (a few hundred MB; CSA2 keeps KV tiny). |
| `CACHALOT_DARWIN_ROLE` | 1 | `serve` and `chat` ask macOS to schedule them like the focused app (Darwin role UI_FOCAL). With the display on, window compositing sometimes doubles the non-read part of a decode token; this won every such pair measured by 6-40 % and is a null otherwise. `0` leaves the default role. |
| `MLX_METAL_FAST_SYNCH` | 1 (`serve`, `chat`) | MLX waits on a shared-memory counter instead of an `MTLSharedEvent` for GPU completion. Bit-identical output; +8 to +25 % decode when window compositing slows the machine, a null otherwise. `0` restores MLX's default. |
| `CACHALOT_VISION_ABLATE` | empty | Debug only: `delims`, `engram_mask` and/or `bias_vl` undo one of the vision fixes, for `benchmarks/vision_ablation.sh`. Answers are not the shipped model's while set. |

**Memory budget guidance.** More resident experts is the only software lever that materially cuts SSD bytes:
in the routing trace, 40 → 50 GiB removed 6 % of bytes and 40 → 64 GiB removed 13 %, while smarter eviction
policies were worth 1–2 % (see [docs/performance.md](docs/performance.md)). The auto budget takes what the machine
has minus a 32 GiB reserve; shrink the reserve with `CACHALOT_SYSTEM_RESERVE_GIB` on a dedicated box, or pin a budget
with `--expert-budget-gib` if you run other memory-hungry software alongside.

## Performance

Two numbers matter and they are measured differently. **A live interactive session** is what a user sees; an
**all-resident token** is the compute floor the runtime reaches when every expert it needs is already in
memory. Everything below is the hardware above, checkpoint and expert bank on the internal SSD, 512-token
context, official chat protocol.

### The shipped configuration

2-bit affine g128 expert bank (9.49 MiB per expert, built from the FP4 checkpoint by
`benchmarks/build_affine_bank.py`), 44 GiB expert budget, startup hotlist, 72 GiB wired, no mirror, no
frequency penalty. Launch it with `./chat.sh`.

| what | result |
|---|---|
| Interactive decode, prose | **9.4–9.6 tok/s** at a 52 GiB budget, two sessions (7.6–7.9 at 44) |
| Interactive decode, ~1,500 tokens of Objective-C | **8.2–8.5 tok/s** at a 52 GiB budget, two sessions (5.6–6.9 at 44) |
| Session expert hit rate | **92.3–92.4 %** at 52 GiB, two sessions; 89.9–90.2 % at 44, repeated across four sessions and three runtime versions |
| Resident experts, MLX peak | 5,276–5,314 experts, 67.7 GiB at 52 GiB (4,480–4,495 and 55.1–55.6 at 44) |
| Follow-up prefill (prefix cache) | 90–116 ms per prompt token |
| Cold 512-token prefill | 16.4 s |
| Quality, 40-case coding corpus | 20/20 C++ blocks compile, 18/18 Python blocks parse, **0 of 101 malformed `#include` lines** — every column equal to a hosted FP4 and a hosted FP8 reference arm. That corpus was C++ and Python only; 0.9.5 adds six Objective-C tasks, compiled and run against expected output, that no bank has been scored on yet |

The same configuration as a benchmark, with a colder working set than a conversation builds: **170 ms per
token (5.90 tok/s)** at an 83.5 % hit rate, reading 627 MiB per token, drive busy 55 % of decode,
reproducible to ±0.3 %.

**0.60.7 (2026-10-06): review fixes.** The run manifest no longer records credentials such as `CACHALOT_API_KEY` by value, and its Hermes Desktop check needs one process line. HANDOFF section 18.73.

**0.62.40 (2026-10-11): nor is it the server.** Driving the DeepSeek decode kernels in one process without the server, the all-resident decode still draws 11.5 W above the component fit (10.9 W through the server), at 79.3 ms and 5.79 J a token; prefill alone sits 7.8 W above it. The unexplained power follows the model's real GPU work, and the server, the drive, the prediction and the read pattern are excluded; the kernels against the resident memory is the remaining split. HANDOFF section 18.127, `docs/E3E-INPROC-KERNELS-RECORD.md`.

**0.62.39 (2026-10-10): the unexplained decode power is not storage.** With the prediction and its speculative loads switched off, an all-resident DeepSeek decode reads nothing from the drive and still draws 10.9 W above the component fit (12.6 W with the prefetch on); the read-bound phase 14.0 W. About 11 W is left that no drive, read pattern or prediction effect explains: the model's kernels, the server's CPU work or the resident memory are the candidates. HANDOFF section 18.126, `docs/E3D-NO-PREFETCH-RECORD.md`.

**0.62.38 (2026-10-10): what the internal SSD costs.** Reading the internal bank, the drive adds about 1.1 W per GB/s to the system (1.7 W at 1 GB/s, 7.6 W at 6.4 GB/s), so in a real decode it accounts for about 3 W (all-resident, prefetch reads) to 5 W (read-bound) of the 13-14 W excess over the component fit, not for it all: about 10 W remain unexplained in both phases. HANDOFF section 18.125, `docs/E2I-INTERNAL-SSD-RECORD.md`.

**0.62.37 (2026-10-10): the decode power excess is not a read pattern.** A synthetic GPU read of scattered or synchronized blocks adds at most 5 W over a sequential stream, against the 13 W a real DeepSeek decode draws beyond the component fit; the earlier attribution to the supply rails is withdrawn as not established. The next suspect is the internal SSD, which reads 1.8 GB/s even when every expert is resident. HANDOFF section 18.124, `docs/E3C-GATHER-RECORD.md`.

**0.62.36 (2026-10-10): the missing decode power is on the supply rails.** Logging every energy channel and the SMC rails through the synthetic loads and the decode arms, the 13 W that a real DeepSeek decode draws beyond the component fit reproduced (12.6 and 13.9 W), is not a channel the fit left out and not the fans, and shows up on the SoC supply rails and the memory rail (11, 8 and 5 W). A streaming calibration is a lower bound for irregular access; the system rail is the total. Thermal state moves a token's system energy 6-11 % between sessions. HANDOFF section 18.123, `docs/E3B-REFIT-RECORD.md`.

**0.62.35 (2026-10-10): joules a token at the system rail.** With the sampler beside and every component logged, a DeepSeek token costs 6.42 J of system energy when every expert is resident (74.5 W, 86 ms) and 10.73 J when it waits for the drive (76.2 W, 141 ms), 1.67 times as much; the CPU+GPU+ANE figures of 0.62.29 were about a quarter of it. The system-power fit made on synthetic loads under-predicts real decode by 13-14 W, cause unknown, so only the measured total stands. HANDOFF section 18.122, `docs/E3-JOULES-TOKEN-RECORD.md`.

**0.62.34 (2026-10-10): what the external drive costs.** Reading at its 1 GB/s wall the X10Pro adds about 2.7 W to the system (derived from system power after removing the reader's own activity; +-1 W), (1.4 W at half the rate). Per byte that is about 50 times DRAM; whether the drive's power is fixed or proportional to the rate is not separable at this precision. The internal drive is not measured yet. HANDOFF section 18.121, `docs/E2-SSD-POWER-RECORD.md`.

**0.62.33 (2026-10-10): the system-power residual explained as a fit.** Loading the CPU, the GPU compute and a memory copy separately, system power (SMC `PSTR`) is a linear sum of the IOReport components (R^2 0.999, 1.3 W rms): about 1.25 W per CPU or GPU watt, 1.04 per memory-controller watt and only 0.34 per DRAM-channel watt, over 18 W of baseline. A total joule figure must therefore be a fit to the system rail, not a sum of the counters. HANDOFF section 18.120, `docs/E1B-RESIDUAL-RECORD.md`.

**0.62.32 (2026-10-10): what a byte of DRAM traffic costs.** Streaming 0 to 735 GB/s through the GPU with the IOReport energy counters read beside a root `powermetrics`, DRAM energy is linear in bytes at 49.9 pJ per byte (R^2 0.999); the memory-controller channels add 12 and 33 pJ/B (94.5 pJ/B all three). The prediction on file (30-50 pJ/B) held; its rate and watts figures missed by 2x. HANDOFF section 18.119, `docs/E1-DRAM-CALIBRATION-RECORD.md`.

**0.62.31 (2026-10-10): DRAM energy is measurable, with a sampler running.** The first step of the power-accounting plan found that `powermetrics` itself reports no DRAM or SSD power, but IOReport exposes DRAM, DRAM-controller and memory-cache-controller energy counters that advance only while a root `powermetrics` is sampling; a plain user process can read them then. The SMC reports whole-system power without root. No SSD sensor exists. New `benchmarks/power_sources.py`. HANDOFF section 18.118.

**0.62.30 (2026-10-10): the energy lane gets a plan.** The joules a token measured so far are CPU+GPU+ANE package power and leave out SSD reads and DRAM transport; `docs/POWER-ACCOUNTING-PLAN.md` sets out how to measure or model those two, index them, and show energy per token in Cachalot Lab. Planning only; nothing run.

**0.62.29 (2026-10-09): joules a token.** A DeepSeek token costs 1.73 J of CPU+GPU+ANE package energy when every expert is resident and 2.52 J when it waits for the drive (1.4-1.5 times: power falls 13 %, the token takes 1.7 times as long). Package power only, no DRAM or drive. `benchmarks/energy_arms.sh`. No runtime change.

**0.62.28 (2026-10-09): the first energy reading.** On the working machine at rest the CPU+GPU+ANE package draws 3.7 W on average (1.4 W in quiet moments); `benchmarks/powermetrics_parse.py` turns a `powermetrics` recording into watts and joules a token for the decode arms that follow. Package power only, no DRAM or drive. No runtime change.

**0.62.27 (2026-10-09): the per-token trace priced.** A tracer with ~340 events a token costs 0.17 ms on a simulated DeepSeek token (0.2 % of the 80 ms floor; stop rule 1 ms), so building the critical-path trace (charter L2) is cleared; the real-server price is next. `benchmarks/trace_overhead.py`; no runtime change.

**0.62.26 (2026-10-09): a validated what-if calculator and the research status.** `benchmarks/whatif.py` predicts decode token time for hypotheticals (faster storage, fewer misses, a miss budget) from three small models; GLM's token fits `92 + 14.7 ms x reads` and predicts four held-out budget arms within 2.5 %, DeepSeek's storage curve misses its one held-out real drive by 20 %. `docs/EXPERIMENTS.md` indexes the experiment records and `docs/RESEARCH-DIRECTION.md` section 13 scores every track and orders what is left (the per-token critical-path trace is next). No runtime change.

**0.62.25 (2026-10-09): GLM's stray-word break traced.** A teacher-forced probe shows the place where most compile-panel replies broke is GLM's own prediction: its top candidate there is the stray word (38-65 %) in prefill mode and in exact decode, and budget 2 does not cause it; the decode path is not at fault either. The cause is the checkpoint, so use MiniMax or DeepSeek for code. New `benchmarks/glm_site_probe.py`.

**0.62.24 (2026-10-09): GLM miss budget, compile-level panel.** A small C# task built with `dotnet build` compiled 0 of 12 times with exact decode and 0 of 12 with budget 2 (a token 35 % faster), so it cannot separate them; 15 of 24 replies break at the same place, which points at GLM's own code corruption. Use MiniMax or DeepSeek for code. Measurement only.

**0.62.23 (2026-10-09): GLM miss budget, C# panel.** A blind panel of 12 C# replies per arm found GLM writes flawed C# in 9 of 12 replies with exact decode and in 9 of 12 with budget 2, so the budget shows no extra damage at this size; the token is 34 % faster (1.794 to 1.184 s). GLM garbles most C# either way: use MiniMax or DeepSeek for code. Measurement only.

**0.62.22 (2026-10-09): documentation only.** The second-pass re-prefill seen in the budget arms of the last run comes from GLM's in-memory prefix list evicting a prompt-end snapshot when a sampled reply re-renders identically (chance, not the budget; decode timings unaffected).

**0.62.21 (2026-10-09): GLM miss budget, second run.** A repeat of the free-running replay (24 replies, 3 hours) reproduced the speeds: a GLM token 1.585 s exact, 1.467 with budget 4 (-7 %), 1.090 with budget 2 (-31 %), with no visible change in tool calls, stories or C# at this sample size. Measurement only; `serve-glm.sh` stays at budget 2.

**0.62.20 (2026-10-09): `serve-glm.sh` defaults to decode miss budget 2** (Hamed's call; changes outputs, a GLM token -32 %). `CACHALOT_GLM_DECODE_MISS_BUDGET=off` restores exact decode.

**0.62.19 (2026-10-09): GLM miss budget, free-running.** On dumped Hermes turns GLM with a decode miss budget of 4 decoded 7 % faster and with 2 32 % faster than exact, with no visible loss on tool calls and stories (24 replies graded blind; too few to see a small difference). No default changed; it is your call. HANDOFF section 18.108.

**0.62.18 (2026-10-08): the Cachalot Lab link points to forera-ai/cachalot-lab.**

**0.62.17 (2026-10-08): the `LICENSE` copyright holder is forera.ai** (MIT text unchanged).

**0.62.16 (2026-10-08): the repository moved to [forera-ai/cachalot](https://github.com/forera-ai/cachalot).** Owner and maintainer are now the forera-ai organisation; no code changed.

**0.62.15 (2026-10-08): GLM miss budget, teacher-forced quality.** On three texts a GLM decode miss budget of 4 stays inside GLM's own prefill-against-decode spread (KL 0.010, token -8 to -12 %) and a budget of 2 costs KL 0.024 for -33 % of the token; budget 0 is clearly worse (dNLL +0.12). Free-running quality is not measured and no default changed. HANDOFF section 18.107.

**0.62.14 (2026-10-08): GLM prefetch off by default; a GLM miss budget knob.** `serve-glm.sh` now starts with `CACHALOT_GLM_PREDICT_TOPK=0` (Hamed's call; outputs unchanged, two live pairs both favoured off). New and off by default: `CACHALOT_GLM_DECODE_MISS_BUDGET=N` reads at most N non-resident experts per decode layer and drops the lightest (changes outputs; unmeasured, quality arms are next). HANDOFF section 18.106.

**0.62.13 (2026-10-08): GLM prefetch off against K = 5, live.** On the same Hermes replay GLM decoded 1.615 s a token with prediction off and 1.727 with the shipped K = 5 (-6.5 %), same misses a token; with 0.61.1's pair, both favour off by 3-9 %, each inside machine drift, so no default changes. The 0.62.12 simulator's predicted loads match the live counters (59 against 61 a token, 68 % used). HANDOFF section 18.105.

**0.62.12 (2026-10-08): `cache_sim.py` models decode prefetch.** `--prefetch-gates` replays a trace that carries predicted sets and reports loads, use, waste and reads per token for any model. On GLM the shipped prefetch reads 126 experts a token against 107 with it off (68 % of its loads used); a router-weight gate of 0.5 keeps 88 % precision at 109 reads, and by the bytes-over-rate bound no gate beats prediction off at 1 GB/s (exact outputs), as for DeepSeek. Also fixes a position-grouping bug that merged two requests sharing a decode segment (one recorded DeepSeek trace; no other published number moved). HANDOFF section 18.104.

**0.62.11 (2026-10-08): the first GLM trace with weights and predicted sets.** GLM's next-layer predictor (top 5) overlaps the layer's real routing 80 % of the time, and a gate on its weight can keep 41 % of the predicted loads at 93.5 % precision. An offline drop-threshold what-if on the same trace says a tau of 0.10 would cut misses a token from 107 to 43 (and drop 13 % of a layer's routing mass); nothing about quality is known and it would change outputs. HANDOFF section 18.103.

**0.62.10 (2026-10-08): GLM traces carry router weights and predicted sets.** Built as priced in 0.62.9 (a tapped gate hands the weights to the sync that already reads the indices; the predictor's set is already a host array). Off unless `CACHALOT_ROUTING_TRACE` is set; unit-tested including the real `__call__` path; not yet run on a model. HANDOFF section 18.102.

**0.62.9 (2026-10-08): pricing weights and predicted sets for GLM and MiniMax.** Recording the predicted next-layer sets is free for both (the hosts already read them); router weights are free for MiniMax under its shipped default and cost ~0.1 % of a GLM token. Priced, not built; the case for building is GLM only. New `benchmarks/router_price.py`. HANDOFF section 18.101.

**0.62.8 (2026-10-08): a longer GLM trace.** Raising the decode cap on the GLM Hermes replay (748 decode tokens, 3.3 times the last): GLM's routing-concentration excess over uniform routing is 17.3 points (its three traces span 17.3-20.0). On the one conversation all three models share, the long traces read DeepSeek 36.1 > MiniMax 21.0 > GLM 17.3. `cache_sim.py` is within 1 % of the live GLM run. HANDOFF section 18.100.

**0.62.7 (2026-10-08): a longer DeepSeek trace.** The same Hermes conversation the MiniMax traces used, through `serve.sh` (2,028 decode tokens): DeepSeek's routing-concentration excess over uniform routing is 36.1 points (its four traces span 34.7-40.3) against MiniMax's 21.0 on the same conversation, a 15-point gap with the workload held fixed. `cache_sim.py` is within 1 % of the live DeepSeek run. HANDOFF section 18.99.

**0.62.6 (2026-10-08): a longer MiniMax trace.** The same Hermes conversation with a 600-token decode cap (1,904 decode tokens, 2.5 times the first trace): MiniMax's routing-concentration excess over uniform routing is 21.0 points, the first trace's 21.1, so the number is stable to about a point; the cold-prompt value (25.5) is outside that, a real workload effect for MiniMax. `cache_sim.py` is within 3 % of the live run on the longer trace. HANDOFF section 18.98.

**0.62.5 (2026-10-08): GLM on an agent conversation.** Replaying six dumped Hermes requests (21-24k context) through `serve-glm.sh`: GLM's routing concentration is the same as on cold prompts (excess over uniform routing 19.8 against 19.2 points), so the ordering DeepSeek > MiniMax > GLM does not depend on the workload for GLM; `cache_sim.py` is within 3 % of the live run. HANDOFF section 18.97.

**0.62.4 (2026-10-08): MiniMax on the same cold prompts.** On the eight cold prompts all three models got, the excess of the busiest 10 % of experts' routing share over uniform routing is DeepSeek 34.6 points, MiniMax 25.6, GLM 19.2: three levels, in that order. (0.62.2 had called MiniMax and GLM alike; that compared different workloads.) HANDOFF section 18.96.

**0.62.3 (2026-10-08): DeepSeek on the same cold prompts.** On the eight short cold prompts that GLM got, DeepSeek's busiest 10 % of experts carry 50.6 % of a layer's decode routes, GLM's 35.2 %; against uniform routing at each length (16 %), the excess is 34.6 against 19.2 points. So the gap in routing concentration is not an effect of Hermes conversations against toy prompts. HANDOFF section 18.95.

**0.62.2 (2026-10-08): the first GLM routing trace.** Eight short prompts through `serve-glm.sh` (347 decode tokens): the busiest 10 % of GLM's experts carry 35 % of a layer's decode routes, the same as MiniMax and well under DeepSeek's 50-53 %; against the value uniform routing would give at each trace's length (13-16 %), DeepSeek is about twice as concentrated as the other two. `cache_sim.py` is within 5 % of the live GLM run at 46 GiB. HANDOFF section 18.94.

**0.62.1 (2026-10-08): the first MiniMax routing trace.** Replaying a dumped Hermes conversation through `serve-minimax.sh` with the tracer on (768 decode tokens, router's own picks): the busiest 10 % of MiniMax's experts carry 35 % of a layer's decode routes, against 50-53 % on DeepSeek's Hermes traces, so MiniMax's routing is flatter; live decode hit 84 % at 35.7 misses a token. `cache_sim.py` no longer assumes 40 layers (it skipped MiniMax's layers 40-59); its MiniMax numbers are still ~20 % optimistic. HANDOFF section 18.93.

**0.62.0 (2026-10-08): routing trace for GLM and MiniMax.** `CACHALOT_ROUTING_TRACE=<file>` now records the routed experts of a GLM or MiniMax server (unweighted, no device read, off by default), so `cache_sim.py` can replay them and routing skew can be compared across the three models. Not yet run on a live model. HANDOFF section 18.92.

**0.61.14 (2026-10-08): instrument fix.** `CACHALOT_READ_THROTTLE_GBPS` (the emulated slower expert drive, off by default) now also applies to MiniMax's coded-bank reads, which it skipped; nothing else changed. HANDOFF section 18.91.

**0.61.13 (2026-10-08): measurement only.** The first live Hermes session on GLM-5.3-Flash with the contiguous bank (X10Pro, 46 GiB): a cold 21k-token Hermes block takes 39.7 minutes once, later requests reuse it (25-49 s for an 18-34 token follow-up, 6 minutes for a 2.7k tool result, 3 minutes for an image turn); decode 0.53-0.77 tok/s. Prose and tool calls were good, the C# snippet was garbled, the image transcription missed two characters. HANDOFF section 18.90.

**0.61.12 (2026-10-07): measurement only.** The ~1.5 s a short prefill costs after a long context (DS-PREFILL-SHORT) is split by a per-chunk trace: with the experts resident a 19-token chunk costs 0.55 s (about 0.26 s fixed plus 15 ms a token) at 512 and at 22,000 tokens of context; the rest is expert reads (240-1,023 a chunk). Context length adds at most ~0.2 s, on one chunk size. HANDOFF section 18.89.

**0.61.11 (2026-10-07): documentation only.** `docs/ARCHITECTURE-COMPARISON.md` compares DeepSeek, MiniMax and GLM by architecture (expert size, top-k, resident share, floor, miss cost, what binds) with every figure cited to `docs/LEDGER.md`, and `docs/SEAMS.md` maps the policy, execution and instrumentation seams of the runtime. HANDOFF section 18.88.

**0.61.10 (2026-10-07): instrument only.** `benchmarks/cache_sim.py --context-tokens N` folds the measured context term of the decode floor (LEDGER DS-FLOOR-CTX: +4 ms by 1.8k tokens, then 0.09 ms per 1k) into its milliseconds-a-token column, so a Hermes-sized context (25k: 84-85 ms floor, not 79) is priced without a hand correction. HANDOFF section 18.87.

**0.61.9 (2026-10-07): the contiguous GLM bank is on by default.** A four-arm interleaved A/B through the server (46 GiB budget) puts a GLM-5.3-Flash token on the X10Pro at 1,531 ms with the contiguous bank against 1,631 ms with the shipped layout (-6.1 %); generated text and expert counts are identical, so it is bit-identical, and `serve-glm.sh` now uses the bank (163 GB at `/Volumes/X10Pro/models/GLM-5.3-Flash-bank`) when it exists. HANDOFF section 18.86.

**0.61.8 (2026-10-07): the real contiguous GLM bank on the X10Pro, and a correction.** With the layers 3-5 bank written to the X10Pro, a paired interleaved run reads the contiguous layout 6.6 % faster than the shipped one (about -5.6 % of a GLM token, bit-identical). The raw-block rates behind 0.61.6's "the USB link does 1.14-1.24 GB/s" and 0.61.7's "-16 % of a GLM token" were page-cache hits and are withdrawn (the same instrument now reads 0.97-1.0 GB/s); the queue-depth results stand. No default or speed changed. HANDOFF section 18.85.

**0.61.7 (2026-10-07): GLM's record layout on the X10Pro, priced (corrected in 0.61.8: ~-5.6 % of a token, not ~-16 %).** GLM's shipped expert layout reads from the USB drive at 0.94 GB/s from two in flight; contiguous 13.5 MiB blocks on the same drive read at 1.14 GB/s, which would take about 18 % off GLM's read time and ~16 % off its 1.5 s token (derived; a contiguous GLM bank exists since 0.49.0 and is bit-identical). The real check needs the bank copied to the X10Pro. No default or speed changed. HANDOFF section 18.84.

**0.61.6 (2026-10-07): queue depth on the X10Pro (the raw-block part is withdrawn in 0.61.8).** Reads of 9.95 MB experts from the USB drive plateau at 0.97 GB/s with two in flight and stay there to sixteen, with latency growing as the number in flight, so queue depth is not a speed lever; the same drive reads contiguous 1-13 MiB blocks at 1.14-1.24 GB/s, so the expert record's nine scattered pieces leave about a fifth of the link unused. On the internal SSD two in flight are needed to reach the plateau (5.0 against 7.0-7.3 GB/s). No default or speed changed. HANDOFF section 18.83.

**0.61.5 (2026-10-07): the decode floor against context length.** DeepSeek's all-resident token through the server is about 79 ms at a short prompt, 83 ms at 2k tokens and 85 ms at 25k: a small step over the first ~2k tokens and then almost flat (about 0.09 ms per 1k), so context length is not a speed lever at agent sizes. The earlier 95.6 ms reading at 22k was drift (an ascending run slowed 16 % over 11 minutes). No default or speed changed. HANDOFF section 18.82.

**0.61.4 (2026-10-07): a weight gate on decode prefetch, priced offline.** `benchmarks/pred_gate_price.py` replays the recorded predicted sets through the expert store's residency rule and the storage-bandwidth curve. It reproduces the prediction-on token (435 against a measured 445.7 ms at 1 GB/s) but not the prediction-off one (-14 %), so it is used for ordering. A gate is not built: with exact outputs no gate beats turning prediction off, and with the shipped miss budget 0 a gate only trades dropped experts for bytes (weight >= 0.20: -43 % at 1 GB/s for 12.6 more dropped experts a token, -7 % on the internal drive). No default or speed changed. HANDOFF section 18.81.

**0.61.3 (2026-10-07): predicted sets in routing traces.** A routing trace can now record the next-layer prediction and its router weights, so a precision-gated prefetch can be priced offline before anything is built. On one 2,407-token server run the predicted top-6 matches the next layer's routed set 72 % of the time, and predictions with a router weight of 0.20 or more are 82 % precise while keeping 66 % of the loads. Recording only; no default or speed changed. HANDOFF section 18.80.

**0.61.2 (2026-10-07): the first live Hermes session on 0.61.1.** Decode 9.8-10.7 tok/s at 22-27k context. A new day's first request reuses the saved system block (22,407 of 22,411 tokens, 0.95 s against a 212 s cold prefill) and a changed provider string reuses the 4,096-token pin, both checked on a server replay of the dumped session. A prefill of under ~70 tokens costs 1.8-3.8 s (a fixed ~1.5 s a chunk). The image transcription was exact; the C# snippet had one compile error (an invented `ClassMap.Map(Type, string)` overload), unattributed to budget 0 from one sample. HANDOFF section 18.79.

**0.61.1 (2026-10-06): prefetch on a slow drive.** GLM-5.3-Flash on its USB drive is measured: 1.52 s a token (0.66 tok/s), 91 % of decode waiting on reads. Its prefetch roughly breaks even there (off is 4 % faster, inside drift), unlike DeepSeek's, because 72 % of its predicted loads are used against DeepSeek's 35-40 %: on a saturated drive speculative reads pay only at high precision. DeepSeek at an emulated 1 GB/s under the shipped budget 0 decodes at 281 ms a token (exact 446). Quantized kernels on the read-size curve: at a routed expert's ~3 MiB a launch the 2-bit kernel reaches 56 % of the chip's bandwidth and the FP8 GEMV 37 % (bf16 77 %), so the decode kernels are not bandwidth-bound. GLM's and MiniMax's `/v1/stats` now show the store's counters. HANDOFF section 18.78.

**0.61.0 (2026-10-06): storage bandwidth against the token.** A new off-by-default knob, `CACHALOT_READ_THROTTLE_GBPS`, emulates a slower drive; validated against the real USB drive (within 13 %). On mixed prompts (exact path, 48 GiB) a DeepSeek token is 119 ms on the internal SSD, 139 at 4 GB/s, 227 at 2, 446 at 1 and 390 on the USB drive: below about 2.6 GB/s the token is the drive's time to move ~450 MB (23.5 misses plus ~29 speculative loads, two thirds of them unused), not the per-miss cost. Turning decode prediction off on a 1 GB/s drive is 11-18 % faster (USB 346 against 390 ms), where on the internal SSD it is a ~1 ms win. HANDOFF section 18.77.

**0.60.10 (2026-10-06): the server floor is 80 ms.** Measured through `serve.sh` the same day as the in-process profile, an all-resident DeepSeek token is 79-80 ms; the 70 ms the docs carried was the intercept of a fit, and the same workload fits 73 or 93 ms depending on the run. A token is `80 ms + 2.0 ms x misses`. Prediction costs ~5 ms of an all-resident token, GPU-side expert selection under budget 0 prices at ~7 ms (held), and a new read-size curve shows a batch-1 GEMV needs 16 MiB a launch to reach ~87 % of the chip's bandwidth. HANDOFF section 18.76.

**0.60.9 (2026-10-06): the floor re-profiled.** An all-resident DeepSeek token in-process is 75-77 ms: 55.5 ms inside `mx.eval` and 17.9 ms of CPU building the next graph between 44 host syncs (the GPU idles about a quarter of the token); GPU work is 52 ms, led by attention (25.6) and the routed experts (10.1), whose kernels run at 29-45 % of the chip's memory bandwidth. The one-runtime guard now sees benchmarks started from the venv. HANDOFF section 18.75.

**0.60.8 (2026-10-06): a second C# task shows no cost of budget 0.** A short CsvHelper + System.Text.Json task, graded blind: exact 2/48 flawed, decode miss budget 0 2/48 at -22 % a token. Across three C# results the budget-0 code cost showed once (0.60.2, long free-form code in an agent context), so the default stays; `CACHALOT_DECODE_MISS_BUDGET=off` is the exact path. HANDOFF section 18.74.

**0.60.6 (2026-10-06): budget 1 on C#, and the first lab tracks.** On a C# request rebuilt from Hermes's store, three arms graded blind: exact 10/48 flawed, decode miss budget 1 10/48 at -2.7 % a token, budget 0 13/48 at -25 %; 0.60.2's budget-0 code cost did not reproduce on this body, so the code evidence is split and budget 1 (3 %) is not a useful middle setting. New: `docs/LEDGER.md` (every recorded constant with its regime and status), `benchmarks/run_manifest.py` (run manifests and scorecard rows), `benchmarks/batch_union.py` (batched decode priced offline: more misses a token, 1.27-1.42x aggregate at four streams, not built). HANDOFF section 18.72.

**0.60.2 (2026-10-05): the code-only check found a cost.** With the decode miss budget 0 (the `serve.sh` default), C# replies were flawed 26/48 against 11/48 exact (p = 0.003), mostly invented API members (15 against 1); TypeScript showed none (8/48 against 9/48). Decode -25 %. Graded blind by four model graders. The default is Hamed's call; HANDOFF section 18.68 lists the options.

**0.60.1 (2026-10-05): the model-graded quality check of the new default.** 192 replies on the six bodies of the 0.58.1 Hermes dump, graded blind by six model graders: flawed 26/96 exact against 28/96 with the decode miss budget 0 (Fisher p = 0.87), decode -18 % a token; the C# body leans worse with the budget (9 against 4 flawed, 5 invented API members against none, p = 0.15) and the `ls` summary leans better (7 against 12): no significant difference at n = 16. HANDOFF section 18.67.

**0.60.0 (2026-10-05): DeepSeek decode drops misses by default, and a new day reuses the system block.** `serve.sh` now sets the decode miss budget to 0 (about -22 to -27 % a token; `CACHALOT_DECODE_MISS_BUDGET=off` is the exact path), and an agent's date line in the system prompt shows a date up to 7 days old so the first message of a day reuses the saved block instead of prefilling it for ~227 s (`CACHALOT_SYSTEM_DATE_REUSE=0` turns that off). Both change what the model sees or computes, at Hamed's decision. HANDOFF section 18.66.

**0.59.0 (2026-10-05): the system block's first chunk survives a restart.** The server now writes the 4,096-token chunk snapshot inside an agent's system block to disk (it lived in memory only), so after a restart a Hermes session whose block changed in its first 6.3k tokens (the date, the model name or the provider string) reuses ~4k tokens, about 41 s of the 227 s first request, instead of prefilling it all cold. Outputs are unchanged; not yet measured live. HANDOFF section 18.65.

**0.58.2 (2026-10-05): why a Hermes session start is a 227 s prefill.** Hermes renders the date, the model name and the provider string into the first 6.3k tokens of its 22.5k-token system block and the tool schemas follow, so any change between sessions (the model name decides whether a 178-token tool-use section is added) re-prefills the whole block at 99 tok/s; with an unchanged configuration the saved block is reused and the first message takes ~2 s. On 0.58.1 the image turn of a Hermes conversation took 41 s (113 s before the splice fix) and the decode ran 9.0-9.7 tok/s with the miss budget at 0 (HANDOFF section 18.64).

**0.58.1 (2026-10-05): image turns keep the conversation's saved prefix.** With an image in the request the server did not put its own reply tokens back into the client's re-rendered history, so a Hermes conversation re-prefilled everything after its first tool call on the image turn (7,531 tokens, 80 s at 22k context); it now splices before the image and moves the image spans with it. Output unchanged (HANDOFF section 18.63).

**0.58.0 (2026-10-05): router-share substitution, closed.** A missing expert with a small router share replaced by the best resident expert of the next four ranks (MiniMax's rule) was run against the decode miss budget on the topic-shift stream: at the same speed it costs about ten times the log-likelihood of dropping the expert (substitution then budget 0: 92.5 ms a token, +0.134 nats; budget 0 alone: 98.3 ms, +0.013; exact 131.3 ms), so DeepSeek keeps the drop as its only approximate lever, still off by default (HANDOFF section 18.62).

**0.57.2 (2026-10-04): the decode miss budget at a topic change.** Teacher-forced through Python, prose and JSON, dropping every missing expert costs about 0.11 nats a token in the 100 tokens after a shift, fading within about 200, and about 0.02 averaged over the stream; six generated long replies that change topic twice graded the same as exact; the token is 24-27 % faster throughout. A milder cap of one is about 6 % faster at no measurable cost. Still off by default (HANDOFF section 18.61).

**0.57.1 (2026-10-04): the decode miss budget, checked on agent turns.** Six dumped Hermes requests (22-25k tokens of context, 25 tools) sampled 8 times per arm at temperature 0.7, 96 replies graded blind: 6 of 48 flawed with the budget at 0 against 14 of 48 exact (not significant), valid tool calls 8 of 8 in both, and long replies 18 % faster. It is still off by default and the default is a decision for the owner (HANDOFF section 18.60).

**0.57.0 (2026-10-04): an opt-in decode miss budget for DeepSeek.** `CACHALOT_DECODE_MISS_BUDGET=0 ./serve.sh` drops every expert that is not resident during decode and rescales the layer's router weights: through the server path at a 22k-token agent context the token goes from 140 to 109 ms (-22.6 %, paired and controlled). It changes outputs (the harness puts the log-likelihood change inside rounding noise but the tail heavier), it is off by default, and its quality at agent scale is not yet measured (HANDOFF section 18.59).

**0.56.1 (2026-10-04): the first full-size Pareto sweep.** Dropping every missing expert in decode (`miss_budget` 0, reachable only through `V41Model`) cuts the DeepSeek decode step 27 % (131 to 95 ms in the harness) with a log-likelihood change inside rounding noise but a heavier tail (KL max 1.3 nats): a candidate, not a default (HANDOFF section 18.58).

**0.56.0 (2026-10-03): a Pareto harness for output-changing levers.** `benchmarks/pareto.py` compares configurations on paired NLL, KL, a checkable task battery (including C# that must build) and decode speed, with a noise band from a numerically equivalent arm. A first quick run shows dropping every decode miss cuts the step 20 % at a KL outside the noise band; the verdict needs the full-size sweep (HANDOFF section 18.57).

**0.55.2 (2026-10-03): D2 priced and held.** A speculative verify of K tokens reads K tokens' worth of experts (28 misses a token at every K), so it only amortises the 70 ms floor: 0.92-1.12x of a 126 ms token depending on the verify's cost per position (HANDOFF section 18.56).

**0.55.1 (2026-10-03): D3 priced and held.** Only 53 % of DeepSeek's decode layers have all six experts resident at 48 GiB, so GPU-side expert selection would net about 0 ms (13 ms with free rewinds) of a 126 ms token (HANDOFF section 18.55).

**0.55.0 (2026-10-03): measurements, no speed change.** MiniMax's all-hit token is 44-49 ms through the server (the research's 77 ms was a fit intercept) and a 4 GiB holder costs it nothing at 68 GiB. DeepSeek's first weighted routing trace (`CACHALOT_ROUTING_TRACE`, 1,958 decode tokens at 22k context) reproduces the live 88 % hit and 7.9 tok/s in the simulator; substituting cheap misses would remove only 16 % of misses at a 1.6 % routing-mass cost. HANDOFF section 18.54.

**0.54.0 (2026-10-03): `./serve.sh` runs DeepSeek at 48 GiB, and a wired-memory governor keeps it there when the machine changes.** Measured on a settled
Mac Studio with the bank on the internal SSD: once the whole system's wired memory passes about 74.5 GiB the GPU pages and the all-resident token costs 160+ ms instead of 70. The old 52 GiB budget crossed it (system wired 75.1 GiB): a mixed 12-request
set decoded at 4.76 tok/s against 8.31 at 48 GiB (7.72 / 8.14 / 8.31 / 8.00 / 4.76 at 36 / 44 / 48 / 50 / 52). It is the system's total, not the runtime's setting: a wired limit of 84 GiB changed nothing, and another process wiring 4 GiB took the default
48 GiB to 4.85 tok/s. The runtime now reads `vm.page_wired_count` between tokens and gives expert slots back while it is above a ceiling (76 % of RAM, 73.0 GiB; `CACHALOT_WIRED_CEILING_GIB`, 0 off): the same 4 GiB holder then costs nothing (7.92 tok/s), and
52 GiB under the governor matches 48 (8.13). The token is `80 ms + 2.0 ms x misses` (0.60.10 measured the 80 ms floor directly; the 70 ms these runs fitted was an intercept). The 9.4-9.6 tok/s rows below are 0.9.x sessions and have not been reproduced since. HANDOFF sections 18.52-18.53 have the tables.

**The 52 GiB row is two sessions now, on 0.9.0 and 0.9.2, and they replicate**: 9.42 and 9.59 tok/s on
prose, 8.53 and 8.15 on Objective-C, 92.37 % and 92.31 % hit rate, and an MLX peak identical to the byte.
Run it with `CACHALOT_MLX_WIRED_LIMIT_GIB=80 ./chat.sh --expert-budget-gib 52`. It is 22–27 ms per token
faster than 0.7.0 at 44 GiB, of which the larger budget explains about 6 ms by the miss arithmetic and the
Engram change the rest; the two have still not been separated by an A/B. The budget's effect is what
`benchmarks/simulate_policies.py` predicted offline — +2.6 points of hit rate, +2.37 measured — and MLX
peaked at **67.7 GiB against the 77.8 GiB the flag wires**, 10.1 GiB of headroom, with no memory-pressure
event. `chat.sh` still defaults to 44 GiB.

### Where a token's time goes

An all-resident token is **76.4 ms** (13.1 tok/s): about 56 ms inside `mx.eval` and 21.5 ms of CPU building
the next graph. A streaming token adds what it blocks on.

| block | ms/token |
|---|---:|
| **The miss**, ~1.7 ms each — 1.41 blocked, 0.31 inside `mx.eval`, ~0.05 CPU | 55–67 at an 83.4 % hit rate, ~19 at a session's 92.4 % |
| The FP8 GEMV family: four attention projections (fused `wq_a`/`wkv` since 0.9.9) and both shared-expert GEMVs (fused `w1`/`w3` since 0.9.8), 5.14 GB/token, re-measured post-fusion in 0.9.11 | ~14.0 |
| Routing prediction, computed and submitted | 11 |
| The 44 `mx.eval` round trips, ~0.20 ms each | 9–12 |
| Routed experts, six per layer, traced | 6.9 |
| Compressor, indexer and compressed-KV write, eight source layers | 5.5 |
| `wo_a`, a BF16 grouped matmul at 625 GB/s | 4.3 |
| Hyper-connection glue | 4.2 |
| Head, Engram forwards and the layer's own router | 3.2 |
| Sparse attention over the KV itself, all forty layers | ~2.6 |
| Engram row reads on the decode thread | 1.9, and 28.4 before 0.9.0 |

**The ceiling is the expert hit rate, not the kernels**, and 0.9.2 measured that rather than assuming it.
The same continuation decoded a second time, with everything it needs already resident, costs **79.6 ms a
token — the all-resident floor to a tenth of a millisecond on every column**, so every millisecond between
the floor and a live token is a miss and nothing else. Read concurrency from 2 to 16 workers moves nothing;
bypassing the page cache costs 16 ms. And the miss itself is at the drive's rated wall: at `io_workers=8`
experts read at 1.46–1.47 ms each whether 43 GiB is wired or nothing, so there is no memory-pressure tax
hiding inside it either (0.9.4).

**And the GPU side is at the machine.** Attention's 22.5 ms turned out to be 86 % weight streaming: a reuse
layer spends 0.377 ms of its 0.441 reading the five projections that build Q and project the output, and
0.064 on the sparse attention whose shapes had been the standing suggestion. Underneath four of those five
is one kernel, `fp8_gemv_decoded`, which moves 5.14 GB per token — more than twice the routed experts — at
**79 % of what `mx.sum` gets over the same bytes**. The shared expert is at 80 % of the same ceiling,
`wo_a` in BF16 is faster than any FP8 form of itself, and the lanes-per-row policy nobody ever tuned is
within 0.3 % of the tuned one. The remaining lever is the hit rate: a larger budget is worth 2.6 points of
decode hit from 44 to 52 GiB by offline replay, and the live session moved 2.37.

**What 0.9.0 took, and how it was hiding.** A streaming token used to spend 33 ms more outside the store's
blocking calls than an all-resident one, with no mechanism for it in three successive analyses. It was the
Engram row reads: a token asks for 24 rows twice, each row is two `pread`s, and the reader only used its
worker pool for batches of 64 or more — so 96 reads went out one at a time from the decode thread, behind
a queue the expert stream was filling. They now go through the pool and are issued at the top of the token
rather than at the layer that needs them. Same bytes, same order, an identical 16-token greedy
fingerprint, and 15 % off the benchmark token.

### The FP4 bank the checkpoint ships with

For reference, and because it is what runs without `CACHALOT_EXPERT_BANK`: 17.93 MiB per expert, 325–341 ms
per token (2.9–3.1 tok/s) at a 36 GiB budget with a 70–71 % hit rate, reading about 1,860 MiB per token, the
drive busy 80 % of decode. The 2-bit bank is faster because it reads half the bytes, and after the
hyper-connection fix in 0.6.0 it is not worse: both are equal to the hosted reference on the coding gate.

Method, instruments and the measurement behind every line above are in [docs/HANDOFF.md](docs/HANDOFF.md);
the older FP4-era analysis is in [docs/performance.md](docs/performance.md).

## How it works

**Layer-major, chunk-batched prefill.** The prompt is processed one layer at a time. Attention runs as one batched pass per
layer (windowed causal attention over a concatenated key pool plus the source layer's per-token compressed top-k),
the MoE phase is regrouped *expert-major* (each routed expert is dequantized once and applied to all of its tokens
with a GEMM), and hyper-connections, router, shared expert, Engram, compressor and indexer are row-batched with
per-token visibility masks. Per-token top-k accumulation order is preserved.

**Deterministic admission.** Before a layer's prefetch starts, `ResidentExpertStore.prepare_prefill_layer()` decides
the layer's resident set from logical work order and per-layer quotas. Asynchronous SSD completion order therefore
cannot change what ends up cached, which makes runs reproducible and cache behaviour debuggable.

**Cross-turn residency.** `reset()` clears sequence state but keeps experts. Returning to a previously seen task after
two unrelated ones saved 31 GiB of SSD reads in the validation benchmark.

**Zero-copy, zero-allocation expert loads.** Every routed expert has the same six tensor sizes, so the resident cache
is a pool of slots allocated once at start-up and wired with `mx.set_wired_limit`. A miss is a `preadv()` from the
shard straight into a writable view of the slot's unified memory; the GPU reads it in place. There is no per-expert
`mx.array`, no memcpy, and no allocator or Metal residency churn on the hot path.

**Memory pressure is the enemy.** Before wiring, macOS compressed cold expert buffers and every GPU access paid a
decompression fault: decode was 3× slower than its SSD bytes implied. Expert reads also bypass the page cache
(`F_NOCACHE`) so a 300 GB stream cannot push the rest of the system out. Details and the measurements that led here
are in [docs/performance.md](docs/performance.md).

**Allocator hygiene.** MLX's free-buffer cache is capped at 2 GiB; larger values produced hundreds of 100–380 ms
allocation stalls under concurrent materialization.

**Exactness.** Every kernel has a NumPy reference (`fp8_ref.py`, `fp4.py`) and the routing, sampling
(Gumbel-max equivalent of the official exponential-race sampler), prompt encoding and completion parsing use the
checkpoint's own code paths.

## Benchmarks

```bash
# Everything below auto-discovers ~/DeepSeek-V4.1-Flash or CACHALOT_MODEL_PATH

# Multi-turn locality benchmark (A → B → C → A, 256 tokens each)
PYTHONPATH=src python benchmarks/bench_multiturn.py

# Routing trace on four realistic prompts + offline policy analysis
PYTHONPATH=src python benchmarks/trace_routing.py --prompt-tokens 512 --decode-tokens 32
PYTHONPATH=src python benchmarks/analyze_trace.py benchmarks/results/trace_routing.trace.npz \
    --out benchmarks/results/trace_routing.md
```

`analyze_trace.py` reports the static coverage upper bound per memory budget, per-layer routing skew, simulated
hit rates for LRU / frequency-aware / static-hot policies, and adjacent-token expert overlap. Use it before
changing the cache policy or budget.

## Roadmap

Ordered by measured size in a token, not by expected difficulty. The full ranking, with what closed each
line, is `docs/HANDOFF.md` section 9.25.

**Research direction (2026-10-05).** Cachalot is now also run as a measurement-driven inference systems laboratory: every non-trivial lever gets a prediction written before the build, a measurement after it, and an explanation of any gap; the per-token critical path, curves with their knees, a validated predictive model, energy and a cross-architecture comparison are the standing goals. The charter, the inventory of what the repository already measures and the programme (tracks L0 to L7) are in `docs/RESEARCH-DIRECTION.md`; it changes no default and no priority. Done so far (0.60.6): L1, the constants and bottleneck ledger (`docs/LEDGER.md`), and L0, the run manifest and scorecard schema (`benchmarks/run_manifest.py`); batched decode was priced offline and not built. 0.61.0: the first L3 curve, storage bandwidth against the decode token (an emulated-drive knob validated against the real USB drive); the knee is near 2.6 GB/s and, below it, speculative prefetch costs more than it saves. 0.61.1: how much it costs depends on its precision (DeepSeek 35-40 % used: -11 to -18 %; GLM 72 %: about even), and the quantized decode kernels sit at 37-72 % of the chip's bandwidth.

**Current speed plan (2026-10-03).** Model priority: DeepSeek-V4.1-Flash, then MiniMax-M3, then GLM-5.3-Flash. DeepSeek's 2-bit bank is on the internal SSD again (GLM moved to the external drive). The plan, in that order: re-measure DeepSeek on a settled
machine; give it the GPU-side expert selection MiniMax already has; miss substitution; DSpark speculation with a decode-shaped verify; expert pruning as a router mask; then the same ideas for MiniMax and GLM, each measured against quality first.
See `docs/SPEED-RESEARCH-2026-10-03.md`.

1. ~~Prefix cache, OpenAI-compatible server, batched prefill, memory auto-sizing~~ shipped.
2. ~~A smaller expert bank~~ shipped in 0.4.0: 2-bit affine g128, 9.49 MiB per expert, half the bytes of FP4
   and equal quality once the hyper-connection defect was fixed.
3. ~~The hyper-connection residual mix~~ fixed in 0.6.0 — it was transposed, and it was the cause of every
   quality artefact this project had blamed on quantization.
4. ~~Graph construction on the decode thread~~ largely shipped in 0.6.0: the MoE block is traced once per
   process instead of rebuilt forty times a token, and the fused kernels' scalar parameters are memoised.
5. ~~DSpark / MTP speculative decoding~~ measured and closed twice, most recently on current constants: a
   K-position forward costs 242.6 ms + 26.9 ms per extra position because the only multi-position path is
   the prefill path. It needs a decode-shaped batched forward before the economics change.
6. ~~Mirror striping across two drives~~ shipped in 0.5.0 and withdrawn in 0.6.0: it is a property of the
   expert size, an 18 % loss at 9.49 MiB where it was a gain at 17.93.
7. **The expert hit rate**, which is the only lever a live session resolves. Offline replay at the shipped
   expert size says 44 → 52 GiB is worth 2.6 points of decode hit and 52 → 60 another 2.3; an interactive
   session peaks at 55.6 GiB against a 72 GiB wired limit and 67.7 at 52 against 77.8, so the headroom exists
   and 60 GiB projects to about 75.2 GiB. **Confirmed the only lever in 0.9.4**: the drive delivers the same
   6.8 GB/s whether the machine has 43 GiB wired or nothing, so there is no memory-pressure tax hiding in
   the miss to remove first.
8. ~~The 33 ms a streaming token spends above the all-resident floor~~ closed across 0.9.0 and 0.9.2: 27 of
   it was the Engram row reads, serialised on the decode thread behind the expert stream, and the rest is
   the miss. A streaming token that does not miss is the all-resident floor on every column.
9. ~~Attention's shapes~~ closed on the arithmetic in 0.9.2: a reuse layer is 0.377 ms of weight streaming
   and 0.064 ms of everything a fixed-capacity `attention_kv` would touch, and the `mx.compile` such a shape
   would unlock is closed three ways already. What is left is the FP8 GEMV kernel under it, which is already
   at 79 % of `mx.sum` over its own bytes — 3.5 ms per token, with `uint4` loads, pre-decoded activations
   and a tuned lane split all in place.
10. ~~The compressor and the indexer on the eight source layers~~ decomposed in 0.9.0: about two thirds of
    the 5.5 ms is the indexer, a tenth the compressor and the rest the compressed-KV write, and `INDEX_TOPK`
    is not a lever — an eightfold change in the width is worth 0.066 ms per layer.
11. ~~Whether the miss itself is slowed by decode's own wired memory~~ closed in 0.9.4: `io_workers=8` reads
    experts at 1.46–1.47 ms each whether 42.9 GiB is wired (45 % of the machine) or nothing, matching the
    drive's cold rating, and the runtime's own 1.41 ms blocked-per-miss component sits in that same band.
    No store-side overhead and no memory-pressure tax to remove — the budget (item 7) is the only lever
    left on the miss.
12. The Objective-C gate exists (0.9.5) but has no model reading: `code_validity.py` compiles Objective-C and
    runs each block against the stdout its task states, after three live Objective-C turns failed a compiler
    or contradicted their own output. Scoring the 2-bit bank on the six new tasks is about two hours.
13. Budgets above 52 GiB are unexplained rather than closed (0.9.7): a 54 GiB chat's Objective-C turn decoded
    at 4.83 tok/s against 8.4-8.8 at 48-52, 60 GiB was unusable, and six budget arms sampled with
    `benchmarks/memwatch.sh` show no OS-level memory-pressure event at any of them, including the slow one —
    the physical-memory-cliff hypothesis is refuted for that run. The collapse was specific to one long turn,
    not the whole session, which points at decode duration or thermal state rather than the budget itself;
    HANDOFF section 7.2.10 has the readings and the next check. `CACHALOT_MLX_WIRED_LIMIT_GIB` above 77.8 is
    clamped to 77.8 and does nothing.
14. ~~The shared expert's `w1`/`w3` fusion~~ shipped in 0.9.8: they read the same quantized activation and
    neither depends on the other, so one `[2I, H]` GEMV replaces two `[I, H]` ones, bit-identical by
    construction, 0.4 ms per token. Identified and left unshipped in 0.9.2 because the win is below what a
    live session can confirm; the first attempt to ship it crashed a live session instead — it called
    `mx.eval` inside the shared expert's own function, which the shipped 2-bit bank's `mx.compile`d MoE
    block calls mid-trace, and MLX refuses that. Fixed by building the fusion in eager Python before the
    compiled call; a live session then ran it clean.
15. ~~`wq_a`/`wkv` fusion~~ shipped in 0.9.9: the two worst-throughput shapes in the FP8 GEMV family
    (171 and 87 GB/s, occupancy-bound — 512 and 1280 output rows do not launch enough simdgroups to fill
    the GPU) both read the same activation and neither depends on the other, same shape as the shared
    expert's `w1`/`w3` fusion, so one `[1792, 5120]` GEMV replaces two, bit-identical by construction,
    0.35 ms per token. No kill switch: attention is never called from inside an `mx.compile`d trace, so
    the shared expert's eval-inside-a-trace crash does not apply here.
16. Vision, phase 1 pieces 1-2 checked, piece 3 all three steps done and wired (0.9.11): the ViT, `Aligner`
    and image preprocessing are ported to MLX (`src/cachalot/model/vision_mlx.py`, `image_processor_mlx.py`)
    and numerically match the official PyTorch reference on a real image — bit-identical preprocessing,
    forward diff at FP32 machine precision. `merge_image_embeddings` (0.9.10) splices an image's aligner
    rows into the embedded sequence at `image_token_id` positions — the official input boundary is a bare
    broadcast, not a learned expansion, so this needed no new numerical path. `route_topk_rows` (0.9.11)
    selects each row's `bias_vl` instead of the uniform `bias` where `image_mask` is set — the one router
    function that actually broadcasts one bias to a whole batch; decode's own router already takes one bias
    per token and needed no change, since a decode token is never an image position. `image_mask` is now
    threaded through `TextDecodeRuntime.prefill_tokens` and all five prefill block variants, live-smoke-
    tested clean on the real bank with a synthetic image span. `resident_trunk.py` still filters every
    `vision.*`/`aligner.*` tensor out of the resident trunk (the routing `bias_vl` tensors were never part of
    that filter — they are plain per-layer MoE tensors, already resident). Only the server's image-content
    parsing (piece 4) is left, and it can now build against a real, working `prefill_tokens(image_rows=...)`.
17. `wq_b`/indexer `wq_b` fusion, screened and rejected (0.9.10): both read `qr`, the same shape as the two
    shipped fusions above, and the fused output is bit-identical — but the indexer only runs on 8 of 40
    layers and `wq_b` is not occupancy-bound to begin with, so the recovered cost is 0.038 ms/token, two
    orders of magnitude below what a live session can confirm. Correctly left unshipped; the FP8 GEMV
    family's remaining gap (`wq_b`, `wo_b`, shared `w1`/`w3`/`w2`) has no fusion candidate left to check.
18. The FP8 GEMV family's post-fusion gap, re-measured (0.9.11), the right way the second time: a first
    attempt summed one script's isolated raw-kernel-launch numbers against another script's real-function
    numbers for the same unfused shapes and found them ~20% apart — discarded before it was published, not
    reported. Diffing only within each script's own before/after arms and summing the non-overlapping shape
    groups gives a self-consistent **2.90 → 2.36 ms/token** gap reduction against `mx.sum`, measured the same
    way, the same session. The original 3.48 ms figure used a third methodology and should not be diffed
    against either number.
19. Hermes Agent against the server, vision end to end, chunked prefill (0.10.0). Hermes's first prompt is
    13.5k tokens. Four things broke and are fixed: the 32k context Hermes refuses (`serve.sh` now serves
    65,536), a Metal out-of-memory in one-shot prefill (now 4,096-token chunks, checked against
    token-by-token decode for quality), a 500 on `reasoning_effort: "none"`, and abandoned requests that
    still cost a full prefill. Prefix-cache snapshots now keep only written cache rows, 6.5× smaller and
    bit-identical on restore. With them, prefill snapshots at every chunk boundary, so a new Hermes session's
    first request takes 18.9 s instead of 206 s. Vision piece 4 wires `image_url` content through the ViT
    into prefill. Three things piece 3 had missed against the reference were fixed on the way: learned
    delimiter embeddings, Engram masking on image positions, and image identity in the prefix cache.
20. An agent's system prompt reused whole, and kept across restarts (0.11.0). The server snapshots exactly
    where the rendered system block ends instead of at the last 4,096-token chunk inside it, so a new Hermes
    session prefills 12 tokens in 1.1 s instead of 1,207 in 18.9 s. Those snapshots are also written to disk,
    keyed by the runtime, checkpoint and bank, and loaded at startup: the first Hermes request after a
    restart prefills in 3.3 s instead of 163 s. Restores from disk are bit-identical.
21. Agent turns reuse the model's own reply (0.12.0). Hermes sends every reply back re-serialized (tool
    arguments in another key order, an empty thinking block as a space), which broke the prefix match a few
    tokens into the reply. When the client's copy says the same thing, the server now uses the model's own
    tokens for it: 11 % fewer warm prefill tokens on a 10-turn Hermes session, and all of a long `write_file`
    body. Images in history are no longer re-encoded every turn. A 10-turn Hermes session up to 26k tokens
    of context and a two-image vision conversation both ran correctly end to end; decode speed was measured
    not to depend on context length (54 vs 16k tokens).
22. `serve.sh`/`chat.sh` start guard fixed (0.12.1): it matched any command line containing "cachalot", so a
    `tee /tmp/cachalot-serve.log` in the same pipeline kept the server from starting at all.
23. The slow-decode window located, one lever for it, and the vision ablation (0.13.0). Decode sometimes ran
    at half speed at identical expert reads; the extra time is all in the non-read part of the token, and it
    only ever appeared with the display on (three displays here). Running as the focused application wins
    6-40 % in that window and costs nothing outside it. Every expert read is now timed and labelled as a
    page-cache or drive read on the per-request log. The chunk snapshots inside an agent's system block are
    pinned, which saves ~100 s on the one cold re-prefill after a Hermes compression. Of the three vision
    fixes, the Engram image mask is the one the answers depend on: without it the model counts nine red
    circles where there are three.
24. The slow window's main trigger (0.13.1): a visible Hermes Desktop window. Hiding it during generation
    took decode from 4.2-5.2 to 5.5-6.7 tok/s at identical expert reads. If you drive Cachalot from a GUI
    agent, hide the window while it generates.
25. Shared-memory Metal fences and release-proof snapshots (0.14.0). `serve` and `chat` set
    `MLX_METAL_FAST_SYNCH=1`: bit-identical, +8 to +25 % decode in the slow window on top of the focused-app
    role, a null outside it. Saved system-prompt snapshots are now keyed on a numerics version instead of the
    package version, so upgrading no longer costs an agent one cold re-prefill of its system prompt. A 6x6
    grid showed the learned image delimiters and `bias_vl` are needed too, not only the Engram image mask.
26. The prefix cache under parallel agents (0.15.0). With six Hermes subagents in flight, each with its own
    ~20k-token system block, the snapshots pinned inside those blocks crowded out every agent's previous
    turn, and each subagent turn re-prefilled 15-31k tokens. Eviction now goes by tier under a byte budget;
    replayed on that session, prefill drops from 480k to 257k tokens (~46 minutes). `serve.sh` also defaults
    to 8,192 new tokens instead of 2,000, which had cut context summaries and one tool call short. And each
    prefill chunk kept the GPU idle for seconds while its Engram rows loaded, long enough for macOS to un-wire
    the model; a tiny keep-alive eval while waiting makes a 12k prefill 15 % faster, bit-identical.
27. The snapshot disk under subagents (0.16.0). Every subagent's system block went to disk, and the 8 newest
    were kept, so one batch of them had already removed the main agent's 22k-token block before a restart
    needed it. The store now keeps 32 blocks by last use, preloads 4, and loads the rest when a request starts
    with one (live: a subagent's 19.5k-token block came off disk and its turn prefilled 396 tokens in 13.5 s).
    Reading the next prefill chunk's Engram rows early is bit-identical but no faster on cold rows (it competes
    with expert reads), so it ships off. Hermes puts each subagent's task context in front of ~13.5k identical
    tool-schema tokens; moving it to the user turn would save ~15 minutes per six-subagent batch
    (`docs/hermes-subagent-context.md`).
28. GLM-5.3-Flash beside DeepSeek (0.17.0). `./serve-glm.sh` / `./chat-glm.sh` run Vontra's MLX 4-bit build
    of GLM-5.3-Flash from the internal SSD, on the same port and API. mlx-vlm's model code runs everything but
    the routed experts, which stream through Cachalot's wired expert store. First numbers: 3.3-3.6 tok/s
    decode with a 52 GiB cache, 14.3 tok/s when every expert is resident, so the DeepSeek levers (hotlist,
    prediction, bank layout) are the path up.
29. GLM prefill and restarts (0.18.0). A long GLM prefill chunk routes to nearly every expert of every layer,
    3.5x what the cache holds, so the LRU path evicted every resident before it was reused. Prefill now keeps the
    residents, streams the rest through transient slots and reads the next layer while this one computes:
    60 to 90 tok/s, bit-identical, at the internal SSD's wall. GLM had also left MLX's buffer cache uncapped,
    so it swapped under the extra wired slots; it is capped at 2 GiB like DeepSeek's. `serve-glm.sh` now keeps
    the agent's system block on disk: a restart reuses it (6 s instead of ~50 s for a 2.4k-token block,
    ~4 minutes for Hermes's 20k).
30. MiniMax-M3 as a third model (0.19.0). `./serve-minimax.sh` / `./chat-minimax.sh` run the 3-bit MLX
    conversion of MiniMax-M3 from the internal SSD. The conversion's model file runs everything but the routed
    experts; its stacked expert tensors are cut into per-expert byte ranges and stream through the same store
    and prefill path as GLM. Prefill 80-100 tok/s, decode 2.5-4.0 tok/s; tools and thinking work. The internal
    GLM copy made room for it, so GLM now runs from the X10Pro.
31. MiniMax-M3 faster (0.20.0). Decode reads were already at the drive's wall, so decode gained elsewhere: one
    GPU round trip per layer instead of three, and the prefill's idle transient slots serve as decode cache until
    the next prefill (2.89 → 3.51 tok/s on the same text, same tokens). Prefill runs 8,192-token chunks, since a
    2,048 chunk already read nearly every expert: 16k tokens at 240 tok/s instead of ~85, same NLL.
32. MiniMax-M3 turns and restarts (0.21.0). Fewer GPU kernels per decode token (stacked q/k/v and gate/up
    matmuls, a fused RMSNorm): 93 → 85 ms of non-read time per token. A short follow-up prefill no longer evicts
    all of decode's borrowed cache, and a restarted server reads its last expert set back at startup (a 22-token
    first turn: 8.7 → 4.8 s). Measured to 64k tokens: it fits, decode 2.65 tok/s; attention is what grows.
33. MiniMax-M3 reads from two drives, and a decode attention kernel of its own (0.22.0). The X10Pro keeps the
    original download, byte for byte, so each expert read now takes its four smallest pieces (~10 % of its bytes)
    from there while the rest comes from the internal SSD: decode 5 % faster, cold prefill 9 %, same bytes, same
    tokens. Long contexts get a Metal kernel that reads each KV head once for its 16 query heads instead of 16
    times: decode attention at 64k 44 → 23 ms per token, a token 8 % faster there and 5 % at 32k, with output
    inside the model's own rounding noise.
34. MiniMax-M3 reads fewer bytes for the same experts (0.23.0). Every bias in the checkpoint's expert groups turned
    out to be a whole multiple of its group's scale, rounded, so a new expert bank keeps 2-bit codes instead of the
    16-bit biases and stores each expert contiguously: 22.2 instead of 23.6 MiB per read, one record instead of nine
    pieces. The biases are rebuilt exactly while the weights are still arriving, so the model computes on the same
    bytes as before: read wait per token -9 %, a token -6 %, identical output.
35. MiniMax-M3 decode 5.5 % faster, identical output (0.24.0). Expert reads skip the page cache now that each
    expert is one contiguous record (4 % faster per read, and the GPU part of a token ~5 ms shorter), a layer's
    already-cached experts are computed while its missing ones are still being read, and the X10Pro serves 13 %
    of each read instead of 10 %.
36. MiniMax-M3 no longer stalls at an agent's context (0.25.0). At 25k tokens the attention cache, and a second
    copy of it the server kept by accident, pushed memory to the machine's ceiling, and with the screen on decode
    fell below 1 token per second. The expert cache now gives memory back as the attention cache grows, and the
    server keeps one copy: the same 25k conversation decodes at 3.5-5 tok/s, with identical output.
37. MiniMax-M3 tool results prefill up to 40 % faster (0.26.0). A prefill chunk of 128 tokens or more read the
    whole next layer ahead of time, which fits GLM but wasted half the reads on MiniMax; it now does so only from
    768 tokens, and prefill reads take each expert's scales from a compressed copy (6 % fewer bytes). A 150-token
    tool result prefills in ~9-10 s instead of ~15, with identical output.
38. MiniMax-M3 predicts what a prefill needs next, and warms up while it waits (0.27.0). A tool result of
    100-3,000 tokens now reads ahead only the next layer's experts its router predicts from the current layer,
    6-15 % faster; and between requests the server reloads the experts the conversation uses most, so after a
    short pause the next reply decodes ~12 % faster. Identical output.
39. MiniMax-M3 decode uses the drive's idle time (0.28.0). While the GPU finishes a layer, the drive used to sit
    idle; each layer now predicts which expert the next layer will need and, if it is not cached, starts reading
    it then. Decode ~3-6 % faster, identical output. The same session found that a benchmark without the
    server's heartbeat had made pauses look costly; with it, warming between requests is worth ~7 % on decode.
40. MiniMax-M3 caches more experts in the same memory (0.29.0). A cached expert kept each weight group's offset
    as a 16-bit number although it is always a small whole multiple of the group's scale; it now keeps that
    multiple in 4 bits and the matrix kernel rebuilds the offset as it multiplies. 128 more experts fit in the
    52 GiB cache, so fewer are read from the SSD: decode ~7.5 % faster, tool-result prefills ~5 % faster,
    byte-identical output.
41. MiniMax-M3 and GLM sample the way their checkpoints ask (0.30.0). Chat, and any server request that does not
    set `top_p` (Hermes never does), sampled from the whole vocabulary; both checkpoints ask for top_p 0.95. At
    top_p 1.0, 4-8 % of the tokens came from the least likely 5 % of the distribution, which is where the stray
    closing sentences ("Your feedback is appreciated.") came from; at 0.95, none. The GLM/MiniMax terminal chat
    now reads the expert cache back at startup and keeps the prompt head on disk (a restart's first "Hi": 14.3 s
    and 21 % cache hits before, a 2 s prefill and 88 % after), and answers `/stats`, `/clear` and `/help` itself
    instead of sending them to the model.
42. MiniMax-M3 decode picks its experts on the GPU (0.31.0). Every one of the 57 expert layers used to stop and
    wait for the Mac's CPU to read which experts it had chosen before running them, so the GPU idled ~0.4-0.6 ms a
    layer. The cached experts now sit in a few large blocks with a table of where each one is; the GPU looks its
    experts up itself and runs on, while the CPU checks each layer's choice one layer later and only steps in
    when an expert is missing and must be read from the SSD. Identical output; agent-style turns decode 9.5 %
    faster, prose (where almost every expert is cached) ~21 %, code ~5 %. The pattern comes from Splash
    (incoai/splash), a Metal engine for models that fit in memory; none of its code is used.
43. MiniMax-M3 caches 142 more experts in the same memory (0.32.0). Each group of 64 weights in an expert has a
    scale and an offset; across the whole model no projection of an expert uses more than 205 different
    (scale, offset) pairs, so a cached expert now keeps one byte per group pointing into a 256-entry table instead
    of a 2-byte scale and a 4-bit offset code. A cached expert shrinks from 22.4 to 21.1 MiB, the 52 GiB cache
    holds 2,523 experts instead of 2,381, fewer experts have to be read from the SSD, and agent-style turns decode
    ~9 % faster with byte-identical output. Two things measured and not kept: the old read-ahead of the next
    layer's likely experts inside the 0.31.0 decode loop (too little lead time, slower), and reading the compact
    form from a separate file on the SSD (fewer bytes but a slower read).

44. MiniMax-M3 decode reads ahead when a layer misses (0.33.0). When a layer needs an expert that is not cached,
    the GPU has already run the next layer's attention and expert choice on this layer's output. With the missing
    expert's contribution left out (zero), that choice is right about nine times in ten, so the next layer's missing
    experts are read at the same time as this layer's, and the layer after that is guessed the same way. The SSD no
    longer waits while the host recomputes a layer: agent-style turns decode ~15 % faster (187 → 158 ms a token),
    output byte-identical. Measured and not kept: storing the 3-bit weights entropy-coded (9 % fewer bytes, but
    decoding them on the GPU costs more than the reads it saves) and a larger share from the X10Pro (it is at its
    USB 10 Gb/s limit already).

45. MiniMax-M3 caches experts in 56 GiB instead of 52 (0.34.0). The larger cache is now faster: agent-style turns
    decode ~13 % faster (163 → 141 ms a token), same output. It used to be slower, and a long prompt still needs the
    memory: an 8,192-token prefill at 56 GiB pushed macOS into memory pressure and then out of GPU memory. So
    Cachalot now watches the machine: before a long prefill it hands a few gigabytes of cache back and takes them
    back once decoding starts, it only grows the cache while macOS reports normal pressure and at least 8 GiB
    available, and it gives memory back when another app needs it. The chat's `/stats` now shows what a cache
    miss really costs a token and the SSD's actual speed.

46. MiniMax-M3 reads 5 % fewer bytes per expert and caches 62 GiB of them (0.35.0). The expert bank on disk is
    rewritten so each record is exactly what the cache holds (a byte per weight group pointing into a small table),
    21.1 MiB instead of 22.2: agent-style turns decode 5 % faster, output unchanged. With that, the cache grows from
    56 to 62 GiB without memory pressure: another 15 % (120 → 102 ms a token). A restart's first reply no longer
    waits ~8 s for the saved cache to load (11.8 → 4.8 s for "Hi"). And a finding: the macOS screensaver draws on
    the GPU and slows decoding by ~13 %; let the display sleep instead.

47. MiniMax-M3 makes room when another app uses the GPU (0.36.0). The GPU's memory is shared by every app, and the
    62 GiB expert cache sits about 1 GiB below the point where macOS starts paging GPU memory. Another app holding
    2 GiB of GPU memory made agent-style turns ~50 % slower (short prefills 2.3x), and that app itself ran out of GPU
    memory. Cachalot now reads how much GPU memory all apps have allocated and hands cache back before that point:
    the same test runs 57.6 s instead of 79-86 s, and nothing changes when Cachalot runs alone. The same
    measurements show why the cache cannot grow past 62 GiB on this Mac: 63 GiB is already slower.

48. MiniMax-M3 caches 68 GiB of experts when macOS lets the GPU use more memory (0.37.0). macOS lets the GPU keep
    at most ~78 GiB of a 96 GiB Mac by default; `sudo sysctl iogpu.wired_limit_mb=88064` raises that to 86 GiB
    until the next restart. With it, the expert cache grows from 62 to 68 GiB and agent-style turns decode 11 %
    faster (104.7 → 93.3 ms a token) with the same outputs; without it Cachalot falls back to 62 GiB on its own.
    The model's word table (0.5 GiB, one row used per token) moved off the GPU, which leaves room for ~24 more
    cached experts, and with busy apps holding GPU memory that alone was worth ~5 %.
    A Hermes Desktop session on it then showed a long conversation being thrown out of memory whenever Hermes sent
    a short side request, costing 3-4 minutes each time; the server now keeps enough room (0.37.1), and the same
    turn takes 8 seconds. A second session showed the server holding more memory than it should while starting
    up; it now loads fewer saved snapshots and makes room before reading its expert cache back (0.37.2).

49. Where MiniMax-M3's remaining time goes, measured (0.38.0). A decode token at the 68 GiB cache is about two thirds
    waiting for the SSDs (23 missing experts of 21 MiB each at ~6.5 GiB/s) and one third GPU work. The GPU side was
    checked piece by piece: decode attention is limited by the GPU's matrix units, long-context prefill attention
    already runs near the GPU's peak (17 TFLOPS), and the weight matvecs are at MLX's best kernel. Two smaller GPU
    changes (the routing after the gate as one kernel, the activation as one compiled kernel) give identical
    outputs and save 1.9 ms a token in one process, but nothing measurable through the server, so they ship off
    (`CACHALOT_MINIMAX_FUSED_ROUTE=1`, `CACHALOT_MINIMAX_COMPILE_SWIGLU=1`). The remaining speed levers are a faster
    second drive (Thunderbolt) and the slowdown while Hermes Desktop's window is on screen.

50. MiniMax-M3's prefill is at the GPU's limit (0.38.1). An 8,192-token chunk takes 30 s (36 s from 8k context on)
    and waits on the SSDs for under a second: the reads hide behind the compute. MLX's 3-bit matrix multiply runs
    at 17.5 TFLOPS, the same as a bf16 one, so about 90 % of a chunk is the GPU at its peak. The rest: 4 % is
    rebuilding each expert's scales and offsets before its multiply, 0.5 s a chunk is experts that get only a few
    tokens. Starting the shared expert earlier, batching the rebuilds, and taking the saved expert set back faster
    after a restart were all measured and gave nothing. Only a new multiply kernel that reads the compact
    scale/offset index itself would remove the 4 %.

51. MiniMax-M3 decode can skip its lightest missing experts (0.39.0, opt-in). About 90 % of a token's routed
    experts are already in memory; the rest are read from the SSDs, and those reads are two thirds of a token.
    With `CACHALOT_MINIMAX_MISS_DROP=0.20 CACHALOT_MINIMAX_MISS_SUB=4`, a missing expert that carries under 20 % of
    its layer's routing weight is not read: the best-scored expert that is already in memory among the next four
    takes its place, with the router's own weights. Misses per token fall ~40 %, decode through the server is
    9-13.5 % faster. Outputs are no longer bit-identical, but the measured quality sits inside the noise of a
    rounding change: on three texts the per-token log-likelihood moves -0.005 / -0.002 / +0.017 nats (every 95 %
    interval includes zero; a prefill-chunk change moves it +0.009 / -0.007 / +0.013), and 24 checkable tasks
    (arithmetic, code run against tests, extraction) pass 24 of 24 either way. Off by default: turn it on per
    launch.

52. GLM and MiniMax replies stop when they loop (0.40.0). Sampled at temperature 1.0, MiniMax-M3 sometimes falls
    into repeating the same line for thousands of tokens; in Hermes sessions four replies did (two of them the
    compression summary, which then timed out). The server now ends a reply once the same block of 10-200 tokens
    has come back six times in a row (`CACHALOT_LOOP_GUARD_REPEATS`, 0 turns it off) and prints `[loop guard]`.
    On every reply recorded in the sessions it stopped exactly those four and nothing else.

53. Short follow-ups at an agent's context are about three times faster (0.41.0). At a 30k-token Hermes context a
    200-token follow-up sometimes took 15-32 s instead of 5-7: while a reply decoded, MLX kept 2 GiB of freed
    buffers the memory governor had not counted, and the GPU went past its working set. MiniMax now keeps 0.25 GiB
    while decoding (`CACHALOT_MINIMAX_DECODE_CACHE_GIB`): 0 slow follow-ups in 20 against 12 in 35, same tokens.
    The copy of an agent's system block that sat in GPU memory for the next conversation (2.4 GiB) now stays on
    disk and reloads in ~0.4 s (`CACHALOT_MINIMAX_SPILL_BLOCKS`).

54. Short follow-up prefills can be a fifth faster, opt-in (0.42.0). About 70 % of a 200-token follow-up's prefill
    waits on expert reads, and a third of the missing experts serve a single token. With
    `CACHALOT_MINIMAX_PREFILL_MISS_DROP=0.20` such an expert is not read when every token routed to it gives it
    under 20 % of its weight and a cached runner-up can stand in: follow-ups at 10-12k context 5.80 -> 4.61 s. It
    changes outputs slightly (at or just above the size of a rounding change on three texts), so it is off unless
    you set it. The decode switch (`CACHALOT_MINIMAX_MISS_DROP=0.20 CACHALOT_MINIMAX_MISS_SUB=4`) now also passes a
    thinking-on and tool-call battery.

55. Miss substitution is on by default (0.43.0). Both switches from items 54 and the decode switch before it now
    start on: decode skips its lightest missing experts, prefill does so where it saves a read. Outputs differ
    slightly from the exact path (measured at or near the size of a rounding change). To get the exact path back,
    start with `CACHALOT_MINIMAX_MISS_DROP=0 CACHALOT_MINIMAX_PREFILL_MISS_DROP=0`.

56. Restarts and busy machines (0.44.0). A saved system block is found again after a restart: snapshot files now
    carry the runtime configuration in their name, so the exact path's copy of a block no longer stops the default
    configuration from saving its own (every 0.43.x restart prefilled Hermes's 21k-token block, 100-320 s; now
    2-10 s). And the memory governor samples macOS memory pressure twice a second: with other apps holding memory it
    no longer grows the expert cache into swap after a long prefill, which had made short follow-ups at 30k context
    take 30-40 s. `CACHALOT_HOST_GROW_QUIET_S=0` turns the watcher off.

57. The second drive adapts (0.45.0). A slice of every MiniMax expert read comes from a copy on the external X10Pro.
    When something else uses that drive (Spotlight indexing, a file copy), a fixed slice made every read wait for
    it. The slice now follows both drives' measured speed, between 2 % and the configured 13 %: with the X10Pro
    busy, reads are 12-17 % shorter and decode 7 % faster; with it idle, nothing changes.
    `CACHALOT_MINIMAX_MIRROR_ADAPT=0` keeps the slice fixed.

58. Prefill's expert kernels, measured (0.45.1). Two kernel designs were built and measured against MLX's own
    MiniMax prefill path, neither shipped: copies of MLX's four matmul kernels that read each weight group's
    scale and offset from the slot's one-byte index (bit-identical for every row count, faster alone, 4-7 % slower
    inside a real layer), and one grouped launch per projection over all of a layer's experts (up to 13 % faster at
    1-2k tokens, slower below 400, not bit-identical). What is left in prefill compute is expert matmuls with 12-32
    rows, which need a new kernel design; the instruments are in `benchmarks/minimax_prefill_kernels/`.

59. Counting runaway lists and an agent temperature (0.46.0). Sampled at 1.0, MiniMax-M3 sometimes lists invented
    items that count up (`noto 1`, `noto 2`, ... `noto 495`); no block repeats exactly, so the loop guard only
    caught one at its very end, 4,829 tokens in. A reply that ends with 64 list items differing only in one integer
    counting up by one, none of them in the prompt, now stops (`CACHALOT_LOOP_GUARD_INCREMENTING`, 0 turns it off):
    that reply stops at token 1,258, and none of 1,279 other recorded replies and tool outputs trips it. The same
    session measured a lower temperature for agents, which send none: a long directory listing derailed the reply
    5 times in 12 at 1.0, once at 0.7 and never at 0.5, and tool calls under Hermes's own system prompt went from
    30/32 to 32/32. Since 0.47.0 the MiniMax server samples at 0.7 when a request sends no temperature
    (`--default-temperature 1.0` restores the checkpoint's 1.0).

60. The terminal chat samples at 0.7 too (0.48.0). `./chat-minimax.sh` now passes `--temperature 0.7` like the server and Hermes
    (`--temperature 1.0` after the script name restores the checkpoint's 1.0). GLM-5.3-Flash moved back to the internal SSD
    (`~/GLM-5.3-Flash-MLX-4bit-MTP`) and DeepSeek's expert bank is read from the X10Pro copy (`serve.sh`, `chat.sh`).
61. GLM decode reads the next layer's likely experts early (0.50.0, on by default). Each decode layer scores the next layer's router on its own
    MoE input in the same sync as its routing (59 % of the next layer's misses are found at the top 8) and reads the five best-ranked non-resident
    experts once its own reads are in (`CACHALOT_GLM_PREDICT_TOPK=5`, 0 turns it off). Same tokens; a token 8 to 14 % faster on swapped in-process
    alternation pairs, 8.4 % per turn through the server's `stream()`, 14.5 % more tokens a second in two live Hermes sessions (2.6 to 3.0 tok/s).
    Measured and not shipped: a bias-free bank (8.6 % of groups break the rule), a 64 GiB budget (the GPU pages), a contiguous expert bank
    (`CACHALOT_GLM_BANK`, byte-verified, effect unmeasured), 16-bit pair-index slots (about 2.5 % of a token at best), a GPU-select loop
    (only 16 % of decode layers have every expert resident) and MTP (a two-token verify reads 1.75 times the experts). GLM decode is bound by
    the drive: what is left is a faster second drive.
62. GLM-5.3-Flash reads images (0.51.0). The checkpoint's own 24-block vision tower loads on the first image; `image_url` parts work through
    `./serve-glm.sh`, each image's tokens carry its content hash in the prefix cache (a resent image reuses its prefix), and a real screenshot
    read through Hermes was described accurately. Hermes's own `vision_analyze` tool times out after 120 s by default, shorter than one GLM pass
    (3-6 minutes for a 6,000-token image): raise `auxiliary.vision.timeout` in its config. Two full-size (3,840 x 2,160) images checked directly (0.51.4): a photograph described correctly, an illustration recognised
    with one invented sign name; an ~8,000-token image prefills in 178-287 s and a resent image costs nothing.
63. GLM-5.3-Flash: the tower matches mlx-vlm, and its code output is corrupted (0.51.5, no code change). The vision tower and image preprocessing are
    bit-identical to mlx-vlm 0.7.4's on the real weights (seven image sizes up to 7,973 rows). Three Hermes sessions at expert budgets 44 / 48 / 50 GiB
    decoded at 2.79 / 3.00 / 3.10 tok/s (about 1.7 % per GiB; one session per arm) and read a screenshot correctly at all three. But GLM's C# snippets
    do not compile in 9 of 12 replays, with and without Hermes's system prompt, and temperature 0 gives the same wrong text twice: not the budget, the
    temperature or the context. Traced in 0.51.6 (HANDOFF 18.43): not the decode path (decode and prefill are equally good on neutral text, the prefetch changes no logit), but a numerically noisy model taking the wrong fork where its top choices are close; the same C# replay on MiniMax-M3 garbles 1 of 19 replies against GLM's 12 of 21 (0.51.7, HANDOFF 18.44), so use MiniMax for code; a higher-precision GLM pipeline (fp32 residual stream; the linear-attention state is already fp32) does not lower the prefill-against-decode spread (0.51.8, HANDOFF 18.45), so another GLM quantisation is the only open check on that side.

## Cachalot Lab

A desktop app for this runtime, [Cachalot Lab](https://github.com/forera-ai/cachalot-lab), is built from
the brief in [`docs/lab/CODEX-LAB-PROMPT.md`](docs/lab/CODEX-LAB-PROMPT.md); each runtime release that
changes a knob, a default, a stats field, a log line or an endpoint adds a brief under
[`docs/lab/briefs/`](docs/lab/briefs/).

## Project layout

```
src/cachalot/
  model/        transformer blocks, attention variants, MoE, kernels, generation, public API
  cache/        resident expert store (LRU, quotas, deterministic admission)
  io/           prefetch executors
  storage/      safetensors indexing, expert reader, Engram row reader
  metrics/      routing tracer, system snapshots
  cli.py        `cachalot` command
benchmarks/     reproducible benchmark and analysis scripts
tests/          checkpoint-free unit tests (`pytest -q`), model tests (`pytest --model`)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: one architectural change per PR, every performance
claim comes with its benchmark JSON, and nothing may change model output without saying so.

## License

Cachalot is owned and maintained by [forera-ai](https://github.com/forera-ai) (repository: [forera-ai/cachalot](https://github.com/forera-ai/cachalot)) and is released under the [MIT License](LICENSE). The DeepSeek-V4.1-Flash weights are licensed separately by
DeepSeek under their [MIT model license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE).

## Acknowledgements

- [DeepSeek](https://www.deepseek.com/) for releasing V4.1 Flash with a complete reference implementation and
  protocol code, which made exact reimplementation possible.
- [Apple MLX](https://github.com/ml-explore/mlx) for a lazy array framework with first-class custom Metal kernels.
