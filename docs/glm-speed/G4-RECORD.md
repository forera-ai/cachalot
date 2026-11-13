# G4: a larger GLM expert budget

Written 2026-11-13 against runtime 0.85.1, before any run. HANDOFF section 18.278 when scored. Lane: `docs/GLM-SPEED-LANE.md` row G4.

## 1. What the lane said, and what fits (derived, 2026-11-13)

The lane priced G4 at 60-68 GiB from `cache_sim.py` (18.271) and noted "68 needs about 97 GiB available". That note came from MiniMax (18.267), not GLM. GLM's own footprint, from the guard's footprint lines of the 13 arms of 18.274-18.277 (`benchmarks/results/glm-speed/g6`, `g6b`; measured): at 46 GiB the machine's peak wired was 72.0-74.5 GiB with about 6 GiB wired before the server started, peak RSS 57 GiB, and minimum free memory 0.1 GiB with applications open. So a GLM server at budget B wires about B + 27 GiB of the machine's total (derived; B + 21 for the process itself).

| budget GiB | machine peak wired (derived) | left of 96 GiB for everything not wired |
|---:|---:|---:|
| 46 | 73-74.5 (measured) | about 22 |
| 52 | about 79 | about 17 |
| 56 | about 83 | about 13 |
| 60 | about 87 | about 9 |
| 64 | about 91 | about 5 |
| 68 | about 95 | about 1 |

64 and 68 do not fit this 96 GiB machine with a desktop running. 60 needs every application closed and leaves about 9 GiB for macOS. 56 is the realistic arm. The guard's preflight (`benchmarks/guarded_run.sh`) uses DeepSeek's constants (trunk 12 GiB, transient 3 GiB) and under-states GLM's wired set by 6-7 GiB; its "planned wired ~61 GiB" at 46 compares with the measured 72-74.5.

## 2. Price (derived, `cache_sim.py`, 14.16 MB an expert, LRU, exact decode)

Misses a decode token on the four GLM routing traces (229-793 decode tokens each; the simulator was within 1-5 % of live GLM misses in 18.100 and 18.103):

| budget GiB | slots | `routing-trace-glm` | `-agent` | `-long` | `-weights` | change against 46 |
|---:|---:|---:|---:|---:|---:|---:|
| 46 | 3,488 | 87.7 | 100.3 | 103.1 | 106.6 | |
| 52 | 3,943 | 77.5 | 91.3 | 92.6 | 96.1 | -9 to -12 % |
| 56 | 4,246 | 72.5 | 86.4 | 86.0 | 89.6 | -14 to -17 % |
| 60 | 4,549 | 68.1 | 82.1 | 79.6 | 82.8 | -18 to -23 % |
| 64 | 4,853 | 63.9 | 77.6 | 74.0 | 76.6 | -23 to -28 % |
| 68 | 5,156 | 60.5 | 73.5 | 67.9 | 70.7 | -27 to -34 % |

On the internal bank a Hermes decode token is 254-295 ms at 57-75 misses (18.272, measured), so about 63 ms of compute plus about 3.1 ms a miss (derived). A 14-17 % cut of about 66 misses saves about 29-35 ms a token, -10 to -13 % (derived).

## 3. Hypothesis

GLM decode on the internal SSD is bound by misses, and the misses follow the budget as the simulator says, so 56 GiB cuts a decode token by about a tenth. Prefill is unaffected: a 2,048-token chunk reads nearly the whole bank whatever the budget, and short prefills are about 20 s of reads that a larger cache shortens only by its extra hits.

Alternative: the 10 GiB of slots move the machine into memory pressure (compressor, swap) and the token gains nothing or loses.

## 4. Design

`docs/glm-speed/arm.sh` (one fresh `serve-glm.sh`, no disk snapshots, the five requests of the 2026-11-06 live session, then the server stops; about 7 minutes an arm, 18.272). Four arms in ABBA order: 46, 56, 56, 46, tags `G4A1`-`G4A4`, `benchmarks/settle.sh` between arms without `--timeout`. Applications closed first, sysctl 88064, no other runtime. Before the 56 GiB arms, the guard's preflight must count at least 85 GiB available (section 1) or the run waits; `--force` is not used to cover a shortfall at 56. Same knobs as `serve-glm.sh` ships (miss budget 2, threshold 512). Total about 35 minutes.

## 5. Predictions (registered before the run)

- **G4-1.** Peak wired at 56 GiB 81-86 GiB; swap flat; no guardian kill.
- **G4-2.** Decode misses a token, mean over the replies of over 50 tokens, 10-20 % fewer at 56 than at 46 (pair means).
- **G4-3.** Decode tok/s on those replies 6-15 % higher at 56 (pair means), with both 46 arms within 5 % of each other.
- **G4-4.** Cold block prefill within the night's noise: the two budgets' means differ by less than 8 % (18.277's +-8 %).
- **G4-5.** Replies identical between budgets where the prefilled token counts match (exact path except the shipped miss budget 2, whose drops depend on residency: a difference here is reported, not a failure).
- **Falsifier.** Decode tok/s at 56 within 3 % of 46, or worse, with both control arms agreeing: the budget is not what a GLM token waits on, and G4 closes.

**Amendment before any run (2026-11-13, measured).** After Hamed's "Apps closed, run G4" the guard's count was 65.5 GiB (free 44.0, reclaimable file cache 17.8) with ChatGPT Classic, Chrome, Toggl and Mail still open; after his `sudo purge` it was 62.9 GiB (free 58.2, file cache 1.6): purge moves file cache into free memory, which the guard already counted, so it adds nothing to the guard's figure. Anonymous memory was about 26 GiB, the four applications about 4 GiB of it, and the GPU reported 8.8 GiB of system memory allocated with no runtime running. The 56 GiB arm needs 85 GiB by the guard; even the 46 GiB control arm needs 75 (18.274-18.277's arms ran forced at 73-76). No arm ran.

Runner: `bash docs/glm-speed/g4_run_all.sh [B]` (default 56). Before each B arm `settle.sh --budget-gib B` must count the guard's full need and the arm runs without `--force` (`arm.sh` now takes `ARM_FORCE=` for that); the 46 arms wait for 73 GiB and keep `--force`, as the earlier arms did; a failed gate stops the run. If only 52 fits (79 GiB), the run is `g4_run_all.sh 52`, and G4-2 and G4-3 scale to the price: misses 7-14 % fewer, decode tok/s 4-10 % higher; G4-1 becomes 77-82 GiB peak wired. The other predictions and the falsifier are unchanged.

## 6. Results

Not run.
