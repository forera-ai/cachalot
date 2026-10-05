# Cachalot Lab — Build Brief for Codex

> Paste this whole document into Codex as the first message of the project. It is the
> product brief, the design brief, the runtime contract and the engineering rules in one
> place. Follow it in order. Where it says **STOP AND ASK**, stop and ask the owner
> (Hamed) before continuing.


> **Renamed 2026-10-05: Cachalot Studio is now Cachalot Lab.** The owner decided the new name;
> this brief uses it throughout. The product name, window title, menus, docs and every
> user-visible string become "Cachalot Lab". Identifiers that still say "studio" in this brief
> (the Git remote `cachalot-studio`, the working directory `/Volumes/X10Pro/Cachalot Studio`,
> the `Application Support/Cachalot Studio` data directory, the `cachalot-studio/...` keychain
> service names) are left exactly as they were until you rename them: their migration is yours,
> including moving existing user data and keychain items so nothing is lost. Report the new
> remote URL and working directory to the owner so the runtime's docs can follow. In the
> runtime repository the briefs moved from `docs/studio/` to `docs/lab/` (this file is now
> `docs/lab/CODEX-LAB-PROMPT.md`). The runtime also uses "the lab" in lower case for its
> research direction (`docs/RESEARCH-DIRECTION.md`); Cachalot Lab is the app that will show
> that research's scorecards, not the research itself.
---

## 0. Your role and the first three things you do

You are the lead designer and lead engineer of **Cachalot Lab**, a native-feeling macOS
desktop application that wraps the **Cachalot** inference runtime. You own the Lab
repository end to end: brand, design system, application code, packaging, release pipeline
and in-app updates. You do **not** own the runtime; you consume it.

**Repository and workspace (supplied by the owner):**

- Git remote: `https://github.com/prooshani/cachalot-studio.git`
- Working directory: `/Volumes/X10Pro/Cachalot Studio` (an external drive — the path
  contains a space, so quote it in every shell command and script; check the volume is
  mounted before starting work and stop if it is not)

Before writing any application code, do these three things in order:

1. **Confirm the repository.** Echo the remote URL and working directory above back to the
   owner in one line and ask them to confirm before you touch anything. If the directory
   already contains files, list them and ask before initializing over them. Do not create
   any other repository on any hosting service.
2. **Initialize the repository** in that working directory (section 11): `git init`,
   default branch `main`, `origin` set to the URL above, the baseline files, version
   `0.1.0`, a first commit, and a push only after the owner confirms. If the remote
   already has commits, fetch and build on them instead of force-pushing.
3. **Design the brand and logo** (section 5) and present it for approval
   (**STOP AND ASK** at the end of section 5). Only after the owner picks a direction do you
   start on the design system and the application.

Then proceed through the milestones in section 12.

---

## 1. What Cachalot is, and why Lab is not "another LM Studio"

**Cachalot** (public repo: `https://github.com/prooshani/cachalot`, MIT, Python 3.12+,
MLX, hand-written Metal kernels) is an inference runtime for Apple Silicon that runs
Mixture-of-Experts models **that do not fit in memory**. It keeps the dense trunk of the
model resident, holds a bounded, wired working set of routed experts in GPU-visible memory,
and streams every other expert from SSD on demand, with prefetch, prediction and
speculation layered on top. The reference machine is a Mac Studio M3 Ultra with 96 GB of
unified memory; the models it serves are 200–550 GB on disk.

LM Studio, Ollama and Unsloth Studio assume the model fits: pick a quantization that fits
VRAM, load it, chat. Their UI is a model picker plus a chat window. **Cachalot is the
opposite case**, and the Lab must make that case visible and controllable:

- The memory is **tiered**: resident trunk, wired expert cache (the "budget"), prefix-cache
  snapshots in memory, snapshots on disk, OS page cache, and finally the SSD (internal and
  optionally a second "mirror" drive). Performance is decided by how many experts per token
  miss the cache and how fast the drive answers.
- The runtime has **dozens of tuning knobs**, per model family, with real trade-offs: some
  are bit-identical speedups, some change model output, some need an admin `sysctl`, some
  are debug-only.
- Startup is **not instant** (loading the trunk, warming the expert set, preloading
  snapshots), a long prompt's prefill can take **minutes**, and decode speed is a live,
  fluctuating number (tokens/s, expert misses per token, SSD read latency).
- The runtime serves an **OpenAI-compatible HTTP API** that agent harnesses (Hermes,
  OpenCode, Continue, aider, any `openai` SDK client) connect to. The Lab is also a
  first-class chat client for that same server.

The Lab's job: **make an out-of-memory runtime feel as controlled and legible as an
in-memory one**, without hiding what makes it special. The central visual idea is *depth*:
data moving between the surface (GPU-resident) and the abyss (SSD).

---

## 2. Product goals and non-goals

### Goals

1. One-click start, stop and restart of the runtime for any configured model, with a
   clear, honest boot sequence (phases, progress, elapsed time, what it is waiting on).
2. A **profile system**: named, versioned YAML files that fully describe a runtime launch
   (model, family, paths, budgets, server options, sampling defaults, advanced knobs).
   Create, duplicate, diff, import, export, validate, and apply them from the UI.
3. A **live telemetry cockpit**: memory ledger, expert-cache health, SSD throughput,
   prefill/decode speed, per-request history, snapshot reuse — updated in real time.
4. A **chat workspace** that talks to the running server over the OpenAI API: streaming,
   thinking/reasoning content, reasoning effort, tool-call rendering, image input where the
   model supports it, per-message performance footer (prefill s, reused prefix tokens,
   decode tok/s).
5. An **API panel**: endpoint URL, model id, API key management, copy-paste snippets for
   curl / Python `openai` / JS / Hermes / OpenCode / Continue, and a request log.
6. **In-app updates** for both the Lab and the Cachalot runtime, with release notes,
   channels (stable / beta), and rollback, like LM Studio and Unsloth Studio (section 10).
7. A **Doctor** view: hardware, storage speed, memory, model layout checks, and the
   environment prerequisites (section 7.6), with one-click fixes where a fix is safe.
8. An interface that stands out at first glance — see section 4. This is the most
   important non-functional requirement.

### Non-goals (v1)

- No model training or fine-tuning.
- No running two runtimes at once. The runtime refuses to start a second instance
  (both would fight over the same wired memory); the Lab enforces the same rule.
- No hosting of models for other machines beyond binding the server to a chosen host/port.
- No changes to the runtime repository (`prooshani/cachalot`); you may clone it read-only for reference, outside the Lab working directory. If the Lab needs something the runtime does not
  offer, record it in `docs/RUNTIME_REQUESTS.md` (section 9.4) and degrade gracefully.
- No telemetry leaving the machine. No analytics SDKs. No accounts.

---

## 3. Platform and technology

The runtime only runs on **Apple Silicon macOS (14+)**. Lab v1 targets the same.

Required stack (choose these unless you find a concrete blocker; if you do, **STOP AND ASK**):

- **Shell:** Tauri 2 (Rust core, system WebView). Small binary, native menus, native
  notifications, `tauri-plugin-updater` for signed updates from GitHub Releases,
  `tauri-plugin-shell` for managed child processes, `tauri-plugin-fs`, `tauri-plugin-store`.
- **Frontend:** React 19 + TypeScript (strict) + Vite.
- **Styling:** Tailwind CSS v4 with a token layer (CSS variables) generated from the design
  tokens in section 6. No component library look-alike defaults; shadcn/Radix primitives
  are allowed for accessibility, restyled completely.
- **Motion:** Motion (Framer Motion) for UI transitions; a small WebGL/`<canvas>` layer
  (plain WebGL2 or `ogl`/`three` if justified) for the signature visualizations. Must hold
  60 fps on an M-series GPU **while the runtime is decoding**, and must throttle itself
  (section 4.6) — the runtime's own GPU time is sacred.
- **Charts:** a lightweight library (uPlot or visx) or hand-rolled SVG/canvas. No heavy
  chart dashboards that re-render the DOM every tick.
- **State/data:** TanStack Query for HTTP, Zustand for UI state, an event bus from the Rust
  core for process events and log lines.
- **YAML:** `serde_yaml` in Rust as the source of truth for validation and compilation;
  JSON Schema generated from the Rust types and used by the frontend editor (Monaco or
  CodeMirror 6 with YAML schema validation).
- **Tests:** Vitest + Testing Library for the frontend, `cargo test` for the core,
  Playwright (via `tauri-driver` or the web build against a mock runtime) for end-to-end.
- **Tooling:** pnpm, ESLint, Prettier, `cargo clippy -D warnings`, `cargo fmt`,
  conventional commits, GitHub Actions CI (macOS runners, arm64).

The Rust core owns everything privileged: spawning and supervising the runtime process,
reading its stdout/stderr, filesystem access, profile compilation, update checks and
installs, the Doctor probes. The frontend never builds a shell command string.

---

## 4. Design direction — the most important section

The owner's requirement, verbatim in spirit: **exceptional, futuristic, professional, and
it stands out at first glance.** Not a dark-mode admin template. Not "glassmorphism with a
purple gradient". Something a person screenshots.

### 4.1 Concept: "Abyssal Instrumentation"

A cachalot (sperm whale) is the deepest-diving animal of its size; it hunts by sonar in
total darkness and returns to the surface with what it went for. The runtime does the same
with a 500 GB checkpoint on a 96 GB machine. The Lab is the **instrument panel of a
deep-sea vessel**: calm, precise, luminous data on deep water.

- **Depth as the organizing metaphor.** The memory tiers map to depth zones, and this
  mapping is used consistently across the whole app:
  - *Surface / Sunlight zone* — GPU-resident trunk and hot experts (bright, warm-white).
  - *Twilight zone* — wired expert cache and in-memory prefix snapshots (cyan/teal).
  - *Midnight zone* — OS page cache and on-disk snapshots (deep blue/indigo).
  - *Abyss* — the SSD and the mirror drive (near-black, with bioluminescent flashes when
    an expert is fetched).
- **Sonar as the motion language.** Pings, concentric rings, sweeps and echoes signal
  activity: a request arriving, a prefill chunk completing, an expert miss being fetched.
  Motion always means something happened in the runtime; nothing animates for decoration.
- **Bioluminescence as the accent system.** A near-black, blue-shifted canvas; accents
  that glow (not neon-saturated — think deep-sea organisms: cyan, aquamarine, a cool
  violet, and a single warm "anglerfish amber" reserved for warnings and the primary CTA).
- **Instrumentation typography.** A precise geometric sans for UI and a monospaced face
  with tabular figures for every number. Numbers never jitter: tabular-nums everywhere,
  fixed-width readouts, units in a quieter weight.

### 4.2 Signature surfaces (these make the first impression)

1. **The Dive (boot sequence).** Starting a profile plays a short, skippable, *truthful*
   descent: each startup phase the runtime reports (process up, trunk loaded, expert bank
   opened, warm set loading N/M, snapshots preloaded, server listening, first health OK) is
   a depth marker the view sinks past, with real elapsed times. When the server is ready,
   the view "surfaces" into the cockpit. If a phase fails, the descent stops at that depth
   with the error and the last 50 log lines.
2. **The Sonar Field (expert cache visualization).** A circular or hexagonal field with one
   cell per routed expert (DeepSeek V4.1 Flash: 15,360; GLM-5.3-Flash: 12,096; MiniMax-M3:
   7,296 — read the real number from the profile/family, never hard-code in the renderer).
   Cells are arranged by layer (rings) and expert index (angle). Resident experts glow;
   hits pulse faintly; misses flash as a ping travelling in from the abyss edge; evictions
   fade. This must be GPU-instanced and cheap. If the runtime does not expose per-expert
   residency (today it does not — see section 9.4), drive it from aggregate counters
   (resident count, hit/miss deltas) with a clearly labelled "approximate" state, and
   switch to exact data when the runtime offers it.
3. **The Depth Gauge (memory ledger).** A vertical, full-height gauge of the machine's
   unified memory (e.g. 96 GB), stacked by zone: trunk, expert budget (and how full it is),
   KV cache, prefix snapshots, MLX buffer cache, other apps, free. The configured budgets
   are drawn as pressure lines; the live values as liquid levels. A warning line marks the
   GPU wired ceiling (section 7.6). This is the single most useful chart in the app.
4. **The Telemetry Strip.** A persistent, thin bottom strip visible on every screen while a
   runtime runs: state, model id, port, decode tok/s (sparkline), expert hit rate,
   misses/token, SSD GB/s, memory used vs. ceiling. Click expands to the full cockpit.
5. **The Chat Hull.** The chat view is quiet and readable first, instrumented second: the
   message column is calm, and each assistant message has a collapsible instrument footer
   (prefill time, reused prefix tokens, decode tok/s, finish reason). Thinking/reasoning
   content streams into a separate, dimmer "sonar log" panel that can be collapsed.
   During a long prefill, show a live, honest progress indicator (elapsed, tokens known,
   reused prefix tokens once reported) — never a fake spinner percentage.

### 4.3 Layout

- macOS-native window chrome (traffic lights, unified title bar, vibrancy where it helps).
- Left rail (icons + labels on hover or expanded): **Cockpit**, **Chat**, **Profiles**,
  **Models**, **API**, **Logs**, **Doctor**, **Updates**, **Settings**.
- Command palette (⌘K) that can start/stop profiles, switch models, open any setting by
  name (fuzzy over the knob catalog), copy the API URL, and jump to any screen.
- The bottom Telemetry Strip (4.2.4) is always visible when a runtime is running.
- Responsive from 1180×760 (13" laptop) to 6K displays; the cockpit re-flows into more
  columns on wide screens rather than scaling up.

### 4.4 Themes

- **Abyss** (default dark) and **Surface** (light, "daylight over water": pale sea-glass
  backgrounds, the same accent hues darkened for contrast). Both are first-class, both meet
  WCAG 2.2 AA contrast for text and 3:1 for UI graphics. Follow the system setting by
  default; allow override.
- High-contrast mode (respects `prefers-contrast: more`).

### 4.5 Craft rules

- Every number: tabular figures, a fixed unit, a sensible precision (tok/s 2 decimals,
  GiB 1 decimal, ms 1 decimal), and a tooltip with the raw value and its source
  (`/v1/stats` field name or log line).
- Every setting shows: what it does in one sentence, its default for the current model
  family, whether it is **bit-identical** or **changes output**, whether it needs a
  **restart**, whether it needs **admin**, and a "why" link to the runtime doc section.
- Empty states are designed, not blank: e.g. no model found shows the model search roots
  and a "Locate checkpoint…" button.
- Errors are written for humans with the exact runtime message available on expand.
- No layout shift when numbers update. No toasts for routine events; use the strip.
- Keyboard: every action reachable by keyboard; visible focus rings in the accent color.

### 4.6 Performance discipline (the UI must never slow the runtime)

The runtime's decode speed is sensitive to other GPU work on the machine — the owner has
measured a visible chat window of another desktop app costing ~30 % decode, and the macOS
screensaver ~13 %. The Lab must therefore:

- Pause all non-essential animation (Sonar Field, sonar pings, background shaders) when the
  window is hidden, minimized, occluded, or unfocused for more than 10 s; render at most
  one static frame per second in that state.
- Provide a **"Silent running"** toggle (default **on** while a request is in flight): the
  UI drops to static rendering, updates numbers at 1 Hz, and hides the WebGL layer. Show
  a subtle indicator that it is on.
- Poll `/v1/stats` at 1 Hz when visible, 0.2 Hz when hidden, and never while a poll is in
  flight. Never call `/v1/stats` more often than that.
- Respect `prefers-reduced-motion`: replace all motion with instant state changes.
- Measure it: add a dev-only overlay showing the Lab's own frame time and GPU usage, and
  document in `docs/PERFORMANCE.md` the decode tok/s of a reference prompt with the Lab
  closed, open-idle, open-cockpit, and open-silent-running.
- Never put the display or the machine to sleep, never change power settings.

### 4.7 Design deliverables before code

Produce, in `design/`:

1. `design/brand/` — logo concepts, final marks, app icon set (section 5).
2. `design/tokens.json` — the full token set (section 6), W3C design-tokens format.
3. `design/screens/` — high-fidelity static mockups (HTML/SVG or images) of: the Dive,
   Cockpit, Chat (with thinking and a tool call), Profiles editor (form + YAML side by
   side, with a diff), Models, API, Doctor, Updates, Settings, and the command palette —
   each in both themes.
4. `design/motion.md` — every animation: trigger, duration, easing (springs preferred,
   interruptible), reduced-motion fallback.

**STOP AND ASK** after the mockups: show them (render to PNGs and list paths) and get the
owner's approval or corrections before implementing screens.

---

## 5. Brand and logo (do this first, before the design system)

Design a logo for **Cachalot** (the runtime) and a companion mark for **Cachalot Lab**.

### Brief

- Subject: a sperm whale (cachalot) — the square, blunt head profile is the most
  recognizable silhouette; a whale diving vertically is the core gesture.
- Ideas to explore (pick three distinct directions and push each to a finished mark):
  1. **Sonar whale** — the head silhouette built from concentric sonar arcs.
  2. **The dive** — a minimal whale descending through stacked horizontal bands (the
     memory tiers), the tail at the surface line.
  3. **Monogram** — a "C" whose inner counter is the whale's head / an echo ring, usable at
     16 px.
- The Lab mark is the runtime mark in a containing shape (e.g. a porthole/rounded square)
  or with an instrument element (gauge tick marks), so the family reads as one.
- Must work: monochrome, one-color on dark and light, at 16/32/64/128/256/512/1024 px, as a
  macOS app icon (follow Apple's current icon grid and shape, with depth and material in the
  large sizes and a simplified glyph in the small ones), as a menu-bar template image
  (monochrome, 18 pt), and as a favicon.
- Avoid: cartoon whales, water-drop clichés, gradients that fail in monochrome, anything
  that resembles an existing brand's whale (Docker's, DigitalOcean's, etc.). Check this
  explicitly and state that you did.

### Deliverables

- `design/brand/concepts.html` — one page showing the three directions side by side, each
  on dark and light, at large size and at 16 px, plus the app icon mock and a wordmark
  lockup ("Cachalot" and "Cachalot Lab") in the chosen typeface.
- Vector sources: `design/brand/*.svg` (clean, hand-tuned paths, no embedded rasters).
- After approval: `design/brand/final/` with the runtime mark, the Lab mark, wordmark
  lockups (horizontal and stacked), the `.icns` / `AppIcon` set, the menu-bar template, the
  favicon, and `design/brand/USAGE.md` (clear space, minimum size, color usage, don'ts).

**STOP AND ASK**: present the three directions with your recommendation and wait for the
owner's pick before anything else.

---

## 6. Design system

Build tokens in three layers (primitive → semantic → component), exported to CSS variables
and to a TypeScript module.

- **Color primitives:** an abyss neutral ramp (blue-shifted near-black to sea-glass white,
  12 steps), and accent ramps: `biolume-cyan`, `aqua`, `violet`, `amber` (warnings/primary
  CTA only), `coral` (errors). Tier colors for the four depth zones (section 4.1) are
  semantic tokens used everywhere memory is shown, in every chart and legend.
- **Semantic:** `bg/canvas`, `bg/raised`, `bg/sunken`, `border/subtle|strong`,
  `text/primary|secondary|tertiary|numeric`, `state/ok|warn|error|info|busy`,
  `tier/surface|twilight|midnight|abyss`, `focus`.
- **Type:** UI sans (e.g. Inter Display / Geist / a comparable open font you can bundle
  with a license that allows it), mono with tabular figures (e.g. JetBrains Mono / Geist
  Mono). Bundle fonts locally; no network font loading.
- **Space / radius / elevation:** 4 px base grid; radii 6/10/16/24; elevation as light
  (inner glow + hairline border) rather than drop shadows in the dark theme.
- **Motion tokens:** spring presets (`snappy`, `calm`, `sonar`), durations, and the
  reduced-motion mapping.
- **Components (own implementations, accessible):** Button (primary/secondary/ghost/
  destructive), IconButton, Toggle, Slider with numeric entry and unit, Stepper, Select,
  Combobox, SegmentedControl, Tabs, Tooltip, Popover, Dialog, Sheet, CommandPalette,
  Table (virtualized), Tag/Badge (bit-identical / changes-output / restart / admin /
  experimental / debug), Readout (big numeric with unit and delta), Sparkline, Gauge,
  StackedBar, LogView (virtualized, ANSI-aware, filterable), CodeBlock with copy, YAML
  editor, DiffView, Toast (rare), EmptyState, ErrorPanel.
- A living style guide route (`/dev/styleguide`, dev builds only).

---

## 7. The runtime, as the Lab must understand it

Everything in this section is the **runtime contract as of Cachalot 0.38.1 (2026-09-28)**.
Put it in `docs/RUNTIME_CONTRACT.md` in the Lab repo, and encode the machine-readable
parts in `runtime-contract/` (section 9). Read the runtime repository yourself to confirm
and extend it; the owner's team will send you contract updates later (section 13).

### 7.1 How the runtime is launched today

- Python package `cachalot` in the runtime repo (`src/cachalot`), run as
  `<venv>/bin/python -m cachalot.cli <command> …` with `PYTHONPATH=src` (from a checkout).
  The owner's venv today is `~/venvs/deepseek-v41`.
- Commands: `serve` (OpenAI server), `chat` (terminal chat), `doctor` (hardware, storage,
  memory and checkpoint checks), `bench` (routing-trace benchmark), `--version`.
- Configuration is **not** a file today. It is **CLI flags + `CACHALOT_*` environment
  variables** (plus `MLX_METAL_FAST_SYNCH`). The shipped launch recipes are shell scripts
  in the repo root: `serve.sh` (DeepSeek), `serve-glm.sh` (GLM), `serve-minimax.sh`
  (MiniMax), and `chat*.sh` counterparts. **Read all six scripts**: they are the reference
  profiles, with comments explaining why each value is what it is.
- A guard in every script refuses to start if a process matching
  `deepseek-v41/bin/python|cachalot\.cli` is running. The Lab must do the equivalent
  check (by PID of processes it did not start, too) and offer "Attach to running runtime"
  (monitor-only if it did not start it) or "Stop it" (with confirmation).

**YAML profiles are a Lab feature.** The Lab defines the YAML schema, stores the
files, and **compiles** a profile into `{ argv, env, cwd }` for the child process. Design
the compiler as a pluggable backend so that when the runtime later accepts a native
`--config profile.yaml`, the Lab can switch to passing the file directly (feature-gated
on runtime version, see 9.3).

### 7.2 CLI flags (`serve`; `chat` shares the runtime ones)

Runtime-wide: `--model PATH`, `--max-seq-len INT`, `--expert-budget-gib FLOAT|auto`,
`--io-workers INT` (default 8), `--family auto|deepseek|glm|minimax`, `--verbose`.
Serve: `--host` (127.0.0.1), `--port` (8000; shipped scripts use 8011), `--model-id`,
`--api-key` (or `CACHALOT_API_KEY`), `--default-max-tokens` (1024; scripts use 8192),
`--default-temperature` (0.6), `--default-top-p`, `--thinking`,
`--reasoning-effort (1-100|low|high|max)`, plus penalty defaults (read `cli.py`).
Chat: `--max-new-tokens`, `--temperature`, `--top-p`, `--snapshot-dir`, `--thinking`,
`--reasoning-effort`, `--seed`, `--frequency-penalty`, `--presence-penalty`,
`--no-repeat-ngram-size`, `--penalty-window`, `--system`, `--no-typing-prefill`.

The Lab launches `serve` only. Its own chat UI talks HTTP to that server. (Do not wrap
the terminal `chat` command.)

### 7.3 Model families and their reference profiles

| | DeepSeek V4.1 Flash | GLM-5.3-Flash | MiniMax-M3 |
|---|---|---|---|
| family / model-id | `deepseek` / `deepseek-v4.1-flash` | `glm` / `glm-5.3-flash` | `minimax` / `minimax-m3` |
| routed experts | 15,360 | 12,096 | 7,296 |
| expert size read | 9.49 MiB (2-bit bank) / 17.93 MiB (FP4) | 13.5 MiB | ~21–22 MiB (coded bank / slot image) |
| resident trunk | ~12 GiB | ~5.5 GiB | ~6.0 GiB |
| expert budget (serve) | 52 GiB | 52 GiB | 68 GiB (62 without the sysctl, 7.6) |
| wired limit env | 80 GiB | 80 GiB | 96 GiB |
| max-seq-len | 65,536 | 131,072 | 131,072 |
| default temperature / top-p | 0.6 / – | 0.6 / – | 1.0 / 0.95 |
| vision (image_url) | yes | no | no |
| thinking / reasoning_effort | yes | per template | yes |
| expert bank env | `CACHALOT_EXPERT_BANK` | – | `CACHALOT_MINIMAX_BANK` |
| mirror drive | `CACHALOT_MIRROR_PATH` (off by default) | – | `CACHALOT_MINIMAX_BANK_MIRROR` + `CACHALOT_MIRROR_FRACTION` 0.13 |
| page cache | `CACHALOT_PAGE_CACHE=1` | `1` | `0` (direct reads) |
| snapshot dir | `~/.cache/cachalot/prefix-snapshots` | `…-glm` | `…-minimax` |

All three serve on the same port (8011) with the same API; switching model = restart with
another profile. Model paths in the owner's scripts point to the internal SSD and to an
external drive (`/Volumes/X10Pro/…`); the Lab must detect unmounted volumes and explain
the consequence (e.g. "mirror drive not mounted — mirror striping off").

### 7.4 HTTP API (the server started by `serve`)

- `GET /health` → `{status, model, busy}` (no auth).
- `GET /v1/models` → OpenAI list; each item has `id`, `created`, `owned_by: "cachalot"`,
  and **`max_context_length`**.
- `GET /v1/stats` → a flat JSON object. Fields vary by family; treat every field as
  optional. DeepSeek today: `expert_hits`, `expert_misses`, `expert_hit_rate`,
  `skipped_experts`, `predicted_loads`, `predicted_used`, `decode_miss_budget`,
  `ssd_bytes_read`, `expert_reads`, `expert_fast_reads`, `expert_read_seconds`,
  `resident_experts`, `resident_bytes`, `prefix_cache_entries`, `prefix_cache_hits`,
  `prefix_cache_misses`, `prefix_cache_reused_tokens`, `prefix_cache_bytes`,
  `mlx_active_bytes`, `mlx_cache_bytes`, `mlx_peak_bytes`, plus server fields `model`,
  `uptime_seconds`, `requests_served`, `tokens_generated`, `busy`, `images_served`,
  `replies_spliced`, `vision_loaded`, `vision_rows_reused`. GLM/MiniMax today:
  `model`, `uptime_seconds`, `requests_served`, `tokens_generated`, `busy`,
  `expert_hit_rate`, `resident_experts`, `prefix_snapshots`. Derive rates (SSD GB/s,
  hits/s) from deltas between polls; never assume a field exists.
- `POST /v1/chat/completions` — OpenAI schema plus extensions: `thinking: bool`,
  `reasoning_effort: 1..100 | "low" | "high" | "max" | "none"`, `top_k`, `seed`,
  `no_repeat_ngram_size`, `frequency_penalty`, `presence_penalty`, `tools`, `tool_choice`,
  `response_format`, `stop`, `stream`, `stream_options.include_usage`. `n` must be 1.
  Responses carry `reasoning_content` in thinking mode. **`usage.cachalot`** carries
  `reused_prefix_tokens`, `prefill_seconds`, `decode_seconds`, `decode_tok_per_s` — show
  these on every message.
- Streaming is SSE. During a long prefill the server sends `: keep-alive` comments and
  periodic **empty delta chunks**; the client must tolerate minutes of silence and must
  not time out (use no read timeout, or ≥ 30 min, while the process is alive).
- `POST /v1/completions` exists but is not available for every model (400 with a message).
- Auth: when an API key is set, every `/v1/*` route requires `Authorization: Bearer <key>`.
- The engine is **single-flight**: one generation at a time; other requests queue. A
  client disconnect cancels its request. The Lab must show "queued behind N" when its
  chat request waits, and must cancel cleanly (abort the fetch) when the user stops.

### 7.5 Logs (stderr) — the richest telemetry source today

The server prints one line per request:

```
[request] prompt=<n> reused=<n> prefilled=<n> prefill=<s>s completion=<n> decode=<s>s (<tps> tok/s) images=<n> spliced=<n> miss/tok=<x> hit=<p>% read=<ms>ms fast=<p>% finish=<reason> mlx=<active>/<peak>/<cache>GiB
```

(some fields are optional). Parse it with a tolerant parser (named regex per field,
unknown fields kept as raw key/values), and feed the Requests table and cockpit history.
Keep all raw log lines in a ring buffer (e.g. 20k lines) and in a rotating log file under
the Lab's data directory. Startup phases are printed to stderr as well — collect the
distinct startup lines of all three families (run them, or read the code), turn them into
a phase table in `runtime-contract/startup-phases.yaml`, and drive the Dive (4.2.1) from
it, falling back to "raw log" mode for unknown lines.

### 7.6 Environment prerequisites the Doctor must check (never auto-fix silently)

- Apple Silicon, macOS version, unified memory size, GPU core count.
- Python venv present and `cachalot` importable; MLX version.
- Model directories present (`config.json`), banks present (`bank.json` where relevant),
  external volumes mounted.
- Free disk space for snapshots.
- **GPU wired memory ceiling.** MiniMax at a 68 GiB budget needs
  `sudo sysctl iogpu.wired_limit_mb=88064`, which **resets at every reboot**. Read the
  current value (`sysctl -n iogpu.wired_limit_mb`, no sudo needed), show it on the Depth
  Gauge, and if the profile needs more, offer: (a) run the command via a native macOS
  admin prompt **only after the user clicks**, with the exact command shown, or (b) drop
  the profile to its no-sysctl budget (62 GiB for MiniMax). Never store the admin password.
- Memory pressure (`memory_pressure` / `vm_stat`) and swap usage right now.
- Storage read speed (reuse `cachalot doctor`, which measures it).
- A running runtime that the Lab did not start.
- Known performance hazards, as **advice only**: the screensaver and other visible GPU-heavy
  windows slow decode. Do **not** change any system power, display or screensaver setting.

### 7.7 Tuning knobs (`CACHALOT_*`)

There are ~120 environment knobs. Group them in the UI as **Essentials** (visible by
default) and **Advanced** (per family, collapsed, searchable, each with badges).
Essentials:

- Paths: model path, expert bank, mirror path, snapshot dir, hotlist file.
- Memory: expert budget (GiB or auto), wired limit (GiB), system reserve (GiB), MLX cache
  limit (GiB), prefix cache bytes / GLM-MiniMax prefix GiB, snapshot keep, snapshot
  preload, KV allowance (MiniMax).
- I/O: page cache on/off, io workers, mirror fraction.
- Server: host, port, model id, API key, default max tokens, temperature, top-p, thinking,
  reasoning effort, max seq len.
- Scheduling: `CACHALOT_DARWIN_ROLE` (1), `MLX_METAL_FAST_SYNCH` (1),
  `CACHALOT_IDLE_HEARTBEAT_SECONDS` (0.5), `CACHALOT_PREFILL_KEEPALIVE` (0.5).

Advanced: everything else (prefetch, prediction, speculation, fused kernels, prefill
chunking, eviction policy, GPU selection, debug switches). Build the catalog by reading the
runtime source (`grep -rhoE 'CACHALOT_[A-Z0-9_]+' src`, then each variable's default and
comment) and the README "Configuration" table. For each knob record the fields in 9.2. Mark
anything named `*_ABLATE`, `*_DUMP` or described as debug as **debug** (hidden unless
"Show debug knobs" is on), and anything whose comment says it changes outputs as
**changes-output** with a confirmation when enabling it.

---

## 8. Features, screen by screen

### 8.1 Cockpit
Depth Gauge (memory ledger), Sonar Field, readouts (decode tok/s last/avg, prefill tok/s,
hit rate, misses/token, SSD GB/s, read latency ms, reused prefix tokens, requests served,
tokens generated, uptime), time-series charts (last 5/30/120 min) for tok/s, hit rate,
misses/token, SSD throughput, MLX active/peak memory, and a Requests table (from 7.5) with
per-request drill-down. "Busy / idle / queued" state prominent. Export a session report
(Markdown + CSV) for benchmarking.

### 8.2 Chat
Multiple conversations (stored locally, SQLite via the Rust core), system prompt per
conversation, per-conversation sampling overrides (temperature, top-p, top-k, max tokens,
seed, penalties, thinking, reasoning effort — with the profile defaults shown as
placeholders), streaming with a stop button, reasoning panel, Markdown + code highlighting
+ LaTeX, copy/regenerate/edit-and-resend, image attach (only when the model supports
vision), tool-call and tool-result rendering (for inspecting agent traffic), per-message
instrument footer, and the honest long-prefill indicator (4.2.5). Explain on hover that
editing an early message forfeits prefix reuse (a full re-prefill), and show the expected
cost from the last measured prefill rate.

### 8.3 Profiles
List with model family, last used, and a health badge (paths exist, budget fits the
machine, sysctl OK). Editor with **Form** and **YAML** tabs kept in sync, JSON-Schema
validation with inline errors, a **"Compiled"** tab showing the exact argv and env the
runtime will get, a **Diff** against the reference profile or any other profile, and
"Save", "Save as", "Apply & restart". Import the reference profiles from the runtime
repo's `serve*.sh` on first run (a one-time importer that parses the scripts' `export` and
flag lines; keep the importer tested against copies of those scripts). Profiles are stored
as `~/Library/Application Support/Cachalot Studio/profiles/<slug>.yaml` and are
plain-text, git-friendly, commented. Include `schema_version` in every file and migrate
older versions forward automatically, keeping a backup.

Profile YAML sketch (finalize the schema yourself, keep it this readable):

```yaml
schema_version: 1
name: MiniMax-M3 · agent
family: minimax
runtime:
  install: managed            # managed | external
  version: ">=0.38.0"         # semver range, checked before launch
  python: auto                # or an absolute path to the venv's python
model:
  path: ~/MiniMax-M3-MLX-3bit
  bank: ~/MiniMax-M3-coded-bank
  mirror:
    path: /Volumes/X10Pro/models/MiniMax-M3-coded-bank
    fraction: 0.13
    when_missing: disable     # disable | fail
memory:
  expert_budget_gib: 68
  wired_limit_gib: 96
  requires_sysctl_wired_limit_mb: 88064
  fallback_expert_budget_gib: 62
io:
  page_cache: false
snapshots:
  dir: ~/.cache/cachalot/prefix-snapshots-minimax
  keep: 8
  preload: 2
  prefix_gib: 8
server:
  host: 127.0.0.1
  port: 8011
  model_id: minimax-m3
  api_key: { keychain: cachalot-studio/minimax }   # never plain text
  max_seq_len: 131072
defaults:
  max_tokens: 8192
  temperature: 1.0
  top_p: 0.95
env:                          # advanced knobs, validated against the catalog
  MLX_METAL_FAST_SYNCH: "1"
  CACHALOT_MINIMAX_IDLE_WARM: "1"
```

API keys live in the macOS Keychain, never in YAML.

### 8.4 Models
Discovered checkpoints (the runtime's search roots: `~`, `~/models`, `/Volumes/*`) plus
user-added paths; for each: family (from `config.json`), size on disk, location (internal /
external, mounted or not), banks found, measured read speed of that volume, and which
profiles use it. No downloading of models in v1; a "Get models" panel links to the runtime
README's instructions.

### 8.5 API
Server URL, model id, key (reveal/copy/rotate — rotating needs a restart), live request log,
and snippets: curl, Python `openai`, JS `openai`, Hermes config, OpenCode config, Continue
config, aider flags. Note Hermes needs `max_context_length ≥ 64,000` (so `max_seq_len`
≥ 65,536), and warn when a profile is below that.

### 8.6 Logs
Virtualized, filterable (level, `[request]` only, search, time range), pause/follow, copy,
save, and "Open log folder". Highlights for known line types.

### 8.7 Doctor
Section 7.6 checks as a checklist with state, detail and a fix action where safe. "Run
full diagnostics" also runs `cachalot doctor` and shows its output.

### 8.8 Updates
Section 10.

### 8.9 Settings
Theme, silent running, polling rates, notifications (request finished while the window is
in the background, runtime crashed), launch at login (off by default), start profile on
launch (off by default), data directory, log retention, "Show debug knobs", update channel,
reset.

### 8.10 Menu-bar extra
Template icon showing state (idle / running / busy). Menu: current profile, tok/s, Start /
Stop / Restart, switch profile, copy API URL, open Lab. The runtime keeps running when the
main window is closed (ask once whether to stop it on quit).

### 8.11 Process supervision
Spawn in its own process group; capture stdout/stderr line-buffered; health-check `/health`
with backoff until ready or a startup timeout (configurable, default 15 min — MiniMax
startup preloads multi-GiB snapshots); graceful stop (SIGINT, wait, then SIGTERM, then
SIGKILL after a timeout, each step visible); detect crashes and show the last log lines and
exit code; never auto-restart in a loop (at most one automatic restart, then stop and
explain). Only one runtime at a time.

---

## 9. The runtime contract layer (so the Lab can follow the runtime as it changes)

The runtime changes weekly. Make the Lab **data-driven** so most runtime changes are a
contract-file edit, not a code change.

### 9.1 `runtime-contract/` directory

- `families.yaml` — per family: id, display name, detection rule (`config.json` keys),
  expert count, expert size, trunk size, default ports/ids/sampling, capabilities (vision,
  thinking, completions endpoint), reference profile.
- `knobs.yaml` — the knob catalog (9.2).
- `stats.yaml` — every known `/v1/stats` field: family availability, unit, meaning,
  derived metrics built from it, where it is displayed.
- `log-lines.yaml` — `[request]` fields and startup phase patterns with regexes.
- `compat.yaml` — which runtime versions the Lab supports and which features each
  runtime version enables (e.g. `native_yaml_config: ">=X.Y.Z"`).

All of these are loaded at startup, validated, and shipped with the app; an update to them
ships with a Lab patch release.

### 9.2 Knob entry fields

`key` (env var or flag), `kind` (env|flag), `families`, `type` (bool|int|float|gib|path|
enum|string), `default` per family, `range`/`enum`, `unit`, `group` (essentials/memory/io/
prefetch/prefill/decode/kernels/snapshots/scheduling/debug), `effect`
(bit-identical|changes-output|unknown), `restart_required`, `admin_required`,
`stability` (shipped|experimental|debug|deprecated), `since`, `removed_in`, `summary`
(one sentence), `doc` (runtime doc anchor, e.g. `HANDOFF 18.6`).

### 9.3 Version gating
Read the runtime version (`python -m cachalot.cli --version`) before launch. Refuse
profiles whose `runtime.version` range does not match, and hide/disable knobs whose
`since`/`removed_in` exclude the installed version, with an explanation.

### 9.4 `docs/RUNTIME_REQUESTS.md`
Anything the Lab would like from the runtime (examples you will likely hit:
per-expert residency map endpoint, structured JSON startup events, `/v1/stats` parity for
GLM/MiniMax, a `--config file.yaml` flag, a machine-readable knob list, an explicit
"prefill progress" event in the stream, a graceful `/admin/shutdown`). Each request:
motivation, proposed shape, the Lab's current fallback. The owner's runtime team reads
this file.

---

## 10. Versioning, releases and in-app updates

### 10.1 Lab versioning
- Semantic Versioning 2.0.0, starting at **0.1.0**. Pre-1.0: minor for features, patch for
  fixes. 1.0.0 when the milestones in section 12 are all done and the owner approves.
- The single source of truth is the version in `package.json`, mirrored into
  `src-tauri/tauri.conf.json` and `Cargo.toml` by a script (`pnpm version:set X.Y.Z`) that
  CI verifies are in sync.
- Conventional Commits; `CHANGELOG.md` in Keep-a-Changelog format, updated **in the same
  commit** as every version bump; README updated when behavior visible to users changes.
- **Every push to `main` is a release-worthy semantic bump** (owner's rule for public repos):
  bump, changelog entry, README if needed, tag `vX.Y.Z`.

### 10.2 Release pipeline (GitHub Actions)
- On tag `v*`: build universal-or-arm64 macOS bundle, sign with a Developer ID certificate
  and notarize (secrets: `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`,
  `APPLE_SIGNING_IDENTITY`, `APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID`), sign the
  updater artifact with the Tauri updater key (`TAURI_SIGNING_PRIVATE_KEY`,
  `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`), create a GitHub Release with the `.dmg`, the
  `.app.tar.gz` + `.sig`, and `latest.json` for the updater. Pre-release tags
  (`v0.4.0-beta.1`) go to the beta channel.
- Until signing secrets exist, the pipeline must still produce unsigned artifacts and say so
  loudly in the release notes; **STOP AND ASK** the owner for the secrets rather than
  skipping signing silently. Never commit keys.

### 10.3 In-app Lab updates
- `tauri-plugin-updater` against the GitHub Releases `latest.json` of the Lab repo.
- Check on launch (after 30 s) and every 6 h; channel stable/beta; show release notes
  (rendered Markdown from the release body); "Install on quit" or "Restart now"; never
  install while a generation is in flight; keep the previous version for **rollback**.

### 10.4 Runtime management and updates (the LM Studio "runtime extension packs" analogue)
Two runtime install modes, per profile:

- **External** (the owner's current setup): point at an existing checkout + venv
  (e.g. repo at `~/Projects/deepseek-v41-mac`, venv at `~/venvs/deepseek-v41`). The Lab
  reads its version and git state (branch, commit, dirty) and never modifies it; "update"
  in this mode shows the newer version available and the exact `git pull` / `pip install`
  commands, and can run them only when the user clicks.
- **Managed**: the Lab installs runtime versions side by side under
  `~/Library/Application Support/Cachalot Studio/runtimes/<version>/` — the source from the
  runtime repo's GitHub Release (or tag archive) plus a dedicated venv created with `uv`
  (install `uv` if missing, with consent) from the runtime's `pyproject.toml`. Multiple
  versions may coexist; a profile pins a version or a range; the Runtimes screen shows
  installed versions, the active one, release notes, size on disk, and "remove".
- Runtime update checks: GitHub Releases API of `prooshani/cachalot`
  (`/repos/prooshani/cachalot/releases`, fall back to `/tags` if a version has no Release),
  cached, with ETag, unauthenticated, at most every 6 h.
- Before switching a profile to a new runtime version, run a **smoke test**: start it,
  `/health`, `/v1/models`, one short chat completion, stop; keep the old version until the
  new one passes. Offer rollback in one click.
- Model/snapshot compatibility: snapshots on disk are keyed by the runtime's numerics
  version; a runtime update may invalidate them (they are rebuilt automatically, but the
  first long prompt will be slower). Say so in the update dialog when the release notes
  mention snapshots.

---

## 11. Repository initialization (`/Volumes/X10Pro/Cachalot Studio`, remote `https://github.com/prooshani/cachalot-studio.git`)

```
cachalot-studio/
  README.md            # what it is, screenshots, install, dev setup, architecture
  CHANGELOG.md         # Keep a Changelog, starts with 0.1.0
  LICENSE              # MIT unless the owner says otherwise (ask)
  CONTRIBUTING.md
  HANDOFF.md           # living state of the project for the next session (see 13)
  .editorconfig .gitignore .nvmrc rust-toolchain.toml
  .github/workflows/   # ci.yml (lint, typecheck, test, build), release.yml
  design/              # brand, tokens, screens, motion (sections 4-6)
  docs/                # RUNTIME_CONTRACT.md, RUNTIME_REQUESTS.md, ARCHITECTURE.md,
                       # PERFORMANCE.md, DECISIONS/ (ADRs, one per major choice)
  runtime-contract/    # families.yaml knobs.yaml stats.yaml log-lines.yaml compat.yaml
  src/                 # React app
  src-tauri/           # Rust core
  tests/               # e2e + fixtures (captured /v1/stats payloads, log transcripts)
  mock-runtime/        # a small Python or Rust server that imitates cachalot serve:
                       # /health, /v1/models, /v1/stats, streaming chat with prefill delay,
                       # keep-alives, [request] log lines, and scripted startup phases
```

The **mock runtime** is mandatory: all UI and e2e work must run without a 500 GB model.
It must reproduce the awkward cases: 3-minute prefill with keep-alives, slow startup, a
crash mid-stream, a 401, a busy queue, fields missing from `/v1/stats`.

Initial commit: `chore: initialize Cachalot Lab 0.1.0`. Ask before the first push.

---

## 12. Milestones (each ends with a tagged release and a HANDOFF.md update)

1. **0.1.0 — Foundation:** repo, CI, brand approved, tokens, style guide, mock runtime,
   empty shell with navigation, theming, command palette.
2. **0.2.0 — Supervise:** profiles (schema, YAML editor, compiler, importer), process
   supervision, the Dive, Logs, Doctor basics, single-instance guard.
3. **0.3.0 — Chat:** conversations, streaming, reasoning, tool calls, images, instrument
   footer, honest prefill indicator, cancel/queue handling.
4. **0.4.0 — Cockpit:** Depth Gauge, telemetry strip, readouts, charts, requests table,
   Sonar Field (approximate mode), silent running, performance doc with measurements.
5. **0.5.0 — API & Models:** API screen and snippets, models discovery, keychain keys.
6. **0.6.0 — Updates:** release pipeline, Lab updater, managed/external runtimes,
   runtime updates with smoke test and rollback.
7. **0.7.0 — Polish:** menu-bar extra, notifications, accessibility audit (VoiceOver,
   keyboard-only pass, contrast), empty/error states, onboarding (first-run: find the
   runtime, find models, import reference profiles, run Doctor).
8. **1.0.0 — Release candidate** after owner review.

Each milestone: tests for new logic, screenshots in the PR/commit description for UI, and
no regressions in the mock-runtime e2e suite.

---

## 13. Working agreements and handoffs

- **HANDOFF.md** is the living, authoritative state of the Lab: current version, what
  is done, what is in progress, known issues, the runtime version the contract was last
  checked against, and the next steps. Update it at the end of every session and every
  milestone. The owner will share the repository with another engineering agent (Claude)
  that maintains the runtime; that agent will send you **runtime change briefs**
  (new/removed/renamed knobs, new stats fields, new families, new endpoints, changed
  defaults). Apply each brief as: contract files first, then code, then a version bump
  and changelog entry that names the runtime version it tracks.
- Ask one clear question when you are blocked on a decision that is genuinely the owner's
  (visual direction, license, signing secrets, repo URL, anything that publishes).
  Otherwise choose, record the decision as an ADR in `docs/DECISIONS/`, and continue.
- Never run `sudo` on your own, never change macOS system settings, never put the display
  or machine to sleep, never delete model files or snapshot directories without the user's
  explicit click-through confirmation in the UI.
- Measure before claiming performance. Every claim in `docs/PERFORMANCE.md` has the
  command and the numbers.
- Keep dependencies few and justified; record each major one in an ADR.
- Code quality: strict TypeScript, no `any` without a comment, Rust without `unwrap()` in
  non-test code paths, errors typed and surfaced to the UI with context.

---

## 14. Definition of done for v1.0

- A new user on an Apple Silicon Mac can install the `.dmg`, let the Lab find or install
  the runtime, point it at a model, import or create a profile, pass Doctor, start the
  runtime, watch the Dive, chat with streaming and reasoning, copy an API snippet into an
  agent harness, and watch the cockpit — without opening a terminal.
- The owner's three reference profiles (DeepSeek, GLM, MiniMax) import and launch with
  arguments and environment identical to `serve.sh`, `serve-glm.sh`, `serve-minimax.sh`
  (a test proves the compiled argv/env match the scripts).
- Decode speed with the Lab open in silent running is within measurement noise of the
  Lab closed (documented in `docs/PERFORMANCE.md`).
- Lab and runtime updates work end to end, including rollback.
- Both themes, reduced motion, keyboard-only and VoiceOver pass.
- The first screenshot someone sees makes them ask what it is.
