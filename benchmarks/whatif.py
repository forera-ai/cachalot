"""
What-if calculator for decode token time (charter L4, HANDOFF 18.113): a small model per architecture, calibrated on recorded
measurements, validated on measurements it was not fitted to, and a table of hypotheticals with an interval and the
assumptions each one extrapolates beyond. Every constant cites a docs/LEDGER.md row or a HANDOFF section. Nothing here is a new
measurement; a number from this file is a *prediction* until a run reproduces it.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/whatif.py --validate
    ~/venvs/deepseek-v41/bin/python benchmarks/whatif.py --model glm
    ~/venvs/deepseek-v41/bin/python benchmarks/whatif.py --model deepseek --storage-x 4

The Mac is not a datacenter: unified memory, one drive over USB or internal NVMe and one Apple GPU. The model says what a
resource would buy *on this machine in the regime it was measured*; the transferable statement is the structure (a read-bound token
costs bytes over the drive's rate, a floor does not shrink with storage), not the numbers.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

# ---- GLM-5.3-Flash on the X10Pro (decode miss budget arms, HANDOFF 18.108-18.109) -------------------------------------
# t(ms) = FLOOR + m x (MISS_BYTES / B). FLOOR and the per-miss cost were fitted to the per-request points of the two exact arms
# (n = 16 requests, r = 0.998); the b4 and b2 arms were not used in the fit.
GLM_FLOOR_MS = 92.0
GLM_MISS_MS = 14.70  # ms per read miss, fitted
GLM_MISS_BYTES = 13.5 * 1024 * 1024  # LEDGER GL-* expert record, 13.5 MiB
GLM_MEASURED = [  # (label, reads per token, measured ms per token): means of 8 replies each, arm 2 and arm 3 (18.108, 18.109)
    ("exact arm 2", 101.8, 1588.0), ("exact arm 3", 101.4, 1585.0),
    ("budget 4 arm 2", 92.7, 1467.0), ("budget 4 arm 3", 93.3, 1467.0),
    ("budget 2 arm 2", 66.1, 1091.0), ("budget 2 arm 3", 66.7, 1090.0),
]
GLM_HELD_OUT = ("budget 4 arm 2", "budget 4 arm 3", "budget 2 arm 2", "budget 2 arm 3")

# ---- DeepSeek V4.1 Flash (LEDGER ST-BW-CURVE, DS-FLOOR-SRV, DS-MISS-48, DS-BUDGET-CURVE) ---------------------------------
DS_FLOOR_MS = 78.0
DS_MISS_CPU_MS = 1.73  # per miss on the internal drive less the transfer share; ST-BW-CURVE
DS_MISS_MB = 9.95
DS_INTERNAL_GBPS = 6.8
DS_BYTES_TOKEN_MB = 453.0  # ST-BW-CURVE: bytes read a token at 23.5 misses, prediction on
DS_MISSES_48 = 23.5
DS_BW_POINTS = [(6.8, 118.6), (4.0, 139.3), (2.0, 227.4), (1.0, 445.7)]  # ST-BW-CURVE, exact, 48 GiB, throttled reads
DS_X10_REAL = (0.967, 390.1)  # ST-X10-QD plateau rate, ST-THROTTLE-VALID real X10Pro token: NOT used in the calibration
DS_MISS_BY_BUDGET = {36: 31.8, 44: 27.1, 48: 25.3}  # live misses a token, 18.52 (DS-BUDGET-CURVE derived 34.1/29.4/28.0/52: 27.0)

# ---- MiniMax-M3 (LEDGER MM-FLOOR, MM-MISS, MM-TOKEN-SPLIT); no held-out set ----------------------------------------
MM_FLOOR_MS = 46.5
MM_MISS_MS = 3.6
MM_TOKEN_SPLIT = (23.0, 115.6)  # misses, ms at 68 GiB (0.38.0)

# ---- the DeepSeek floor split (LEDGER DS-FLOOR-SPLIT-0609) --------------------------------------------------------------
DS_FLOOR_GPU_SHARE = 52.2 / 75.4  # chained GPU kernel time over the in-process token
# bandwidth-bound share of those kernels is not measured: kernels stream at 29-45 % of 819 GB/s (MC-READ-SIZE); range, not point
BW_BOUND_SHARE_RANGE = (0.0, 1.0)


def glm_token(reads: float, storage_x: float = 1.0, floor_x: float = 1.0) -> float:
    """GLM token (ms) at `reads` read misses a token."""
    return GLM_FLOOR_MS / floor_x + reads * GLM_MISS_MS / storage_x


def deepseek_token(gbps: float, misses: float = DS_MISSES_48) -> float:
    """ST-BW-CURVE: max(floor + misses x (cpu part + transfer at the rate), bytes over the rate). Internal-drive misses cost
    1.73 + 9.95 MB over 6.8 GB/s; slower drives add the transfer difference; below the knee the token is bytes / rate."""
    first = DS_FLOOR_MS + misses * (DS_MISS_CPU_MS + DS_MISS_MB * (1.0 / gbps - 1.0 / DS_INTERNAL_GBPS))
    second = DS_BYTES_TOKEN_MB * (misses / DS_MISSES_48) / gbps
    return max(first, second)


def minimax_token(misses: float, storage_x: float = 1.0) -> float:
    return MM_FLOOR_MS + misses * MM_MISS_MS / storage_x


@dataclass
class Check:
    name: str
    predicted: float
    measured: float
    fitted: bool

    @property
    def err(self) -> float:
        return self.predicted / self.measured - 1.0


def validation() -> list[Check]:
    out = []
    for label, reads, ms in GLM_MEASURED:
        out.append(Check(f"GLM {label}", glm_token(reads), ms, label not in GLM_HELD_OUT))
    for gbps, ms in DS_BW_POINTS:
        out.append(Check(f"DeepSeek {gbps:g} GB/s (emulated)", deepseek_token(gbps), ms, True))
    out.append(Check("DeepSeek real X10Pro", deepseek_token(DS_X10_REAL[0]), DS_X10_REAL[1], False))
    out.append(Check("MiniMax 68 GiB (0.38.0)", minimax_token(MM_TOKEN_SPLIT[0]), MM_TOKEN_SPLIT[1], False))
    return out


def interval(model: str) -> tuple[float, str]:
    """Half-width of the band a prediction is quoted with: the largest held-out error of that model, rounded up."""
    if model == "glm":
        return 0.04, "held-out budget arms off by 0.2-2.5 % (4 arms); 4 % quoted"
    if model == "deepseek":
        return 0.20, "held-out real X10Pro 20 % over (the model reads the plateau rate; the live run read ~1.15 GB/s); 20 % quoted"
    return 0.15, "one unfitted point (+11 % on an older build); 15 % quoted, not validated"


def scenarios(model: str, storage_x: float | None, floor_x: float | None) -> list[tuple[str, float, str]]:
    rows = []
    if model == "glm":
        base_reads = 101.6
        base = glm_token(base_reads)
        rows.append(("baseline: exact decode, 46 GiB, X10Pro (0.963 GB/s), prefetch off", base, "measured 1,585-1,588 ms"))
        for k in (storage_x,) if storage_x else (2, 5, 20):
            rows.append((f"storage x{k:g}", glm_token(base_reads, storage_x=k),
                         "assumes every miss cost is bytes over the rate; the 92 ms part includes the cost of the GPU work, so the answer cannot go under it"))
        rows.append(("storage infinite", glm_token(0.0), "limit: the fitted floor; an extrapolation, no run has had GLM all-resident"))
        rows.append(("decode miss budget 4 (93.0 reads)", glm_token(93.0), "validated: measured 1,467"))
        rows.append(("decode miss budget 2 (66.4 reads)", glm_token(66.4), "validated: measured 1,090"))
        uses = 42 * 8  # routed expert uses a token (42 MoE layers, top-8)
        rows.append((f"hit rate {100 * (1 - base_reads / uses):.0f} -> 86 % (reads {base_reads:.0f} -> {0.14 * uses:.0f})", glm_token(0.14 * uses),
                     "assumes reads scale with misses at the same bytes each; the cache size that buys is not modelled (needs a trace and cache_sim)"))
        rows.append(("perfect prefetch (every read hidden)", glm_token(0.0),
                     "upper bound; GLM prefetch measured -6.5 % at best (18.105), 72 % precision"))
    elif model == "deepseek":
        base = deepseek_token(DS_INTERNAL_GBPS)
        rows.append(("baseline: internal drive, 48 GiB, 23.5 misses, exact", base, "measured 118.6-126.2 ms (ST-BW-CURVE, DS-TOKEN-48)"))
        rows.append(("same, on the X10Pro (0.967 GB/s plateau)", deepseek_token(0.967), "measured 390.1 ms; model reads the plateau and is 20 % over"))
        for k in (storage_x,) if storage_x else (2, 5, 20):
            rows.append((f"X10Pro storage x{k:g} (rate {0.967 * k:.2f} GB/s)", deepseek_token(0.967 * k),
                         "ST-BW-CURVE regime: below ~2.6 GB/s the token is bytes over rate, above it floor-dominated; rates above 6.8 GB/s are beyond the measured curve"))
        rows.append(("storage infinite", deepseek_token(1e9),
                     f"limit: floor + the non-transfer part of 23.5 misses ({DS_MISS_CPU_MS - DS_MISS_MB / DS_INTERNAL_GBPS / 1.0:.2f} ms each); beyond any measured rate (max 6.8 GB/s)"))
        for gib, misses in DS_MISS_BY_BUDGET.items():
            rows.append((f"internal drive, {gib} GiB budget ({misses} misses a token)",
                         deepseek_token(DS_INTERNAL_GBPS, misses), "live misses from 18.52; the cliff above 48 GiB is not modelled"))
        rows.append(("every expert resident (0 misses)", DS_FLOOR_MS, "DS-FLOOR-SRV 79-80 ms measured; not reachable at 142 GiB of experts on 96 GiB of memory"))
        if floor_x:
            lo = DS_FLOOR_MS * (1 - DS_FLOOR_GPU_SHARE * BW_BOUND_SHARE_RANGE[1] * (1 - 1 / floor_x))
            rows.append((f"GPU memory bandwidth x{floor_x:g} (range)", lo,
                         f"range {lo:.0f}-{DS_FLOOR_MS:.0f} ms: from 'all GPU kernels are bandwidth-bound' to 'none are' (the floor is 27 % GPU idle between 44 syncs, "
                         "kernels at 29-45 % of peak); a hypothesis to test with L4(f)"))
    else:
        base = minimax_token(23.0)
        rows.append(("baseline: 23 misses, 68 GiB, both drives", base, "measured 115.6 (0.38.0), 8-ish tok/s live today"))
        for k in (storage_x,) if storage_x else (2, 5):
            rows.append((f"storage x{k:g}", minimax_token(23.0, storage_x=k), "unvalidated: assumes the 3.6 ms a miss is all bandwidth"))
        rows.append(("storage infinite", MM_FLOOR_MS, "limit: MM-FLOOR 44-49 ms measured"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("glm", "deepseek", "minimax"), default="glm")
    ap.add_argument("--storage-x", type=float, default=None)
    ap.add_argument("--floor-x", type=float, default=None, help="GPU memory bandwidth multiplier (DeepSeek floor range)")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        print(f"{'point':34s} {'predicted':>9s} {'measured':>9s} {'error':>7s}  used in the fit")
        for c in validation():
            print(f"{c.name:34s} {c.predicted:9.1f} {c.measured:9.1f} {100 * c.err:+6.1f}%  {'yes' if c.fitted else 'NO (held out)'}")
        return
    half, why = interval(args.model)
    print(f"{args.model}: band +-{100 * half:.0f} % ({why})")
    if args.model == "glm":
        print("note: GLM's rate axis was measured at one point (0.963 GB/s). DeepSeek's curve shows reads overlapping the floor and a knee near\n"
              "2.6 GB/s, so GLM storage multipliers above ~2 extrapolate beyond the validated regime.")
    print()
    for name, ms, note in scenarios(args.model, args.storage_x, args.floor_x):
        print(f"{name:66s} {ms:7.0f} ms  [{ms * (1 - half):.0f}-{ms * (1 + half):.0f}]  ({1000 / ms:.2f} tok/s)\n    {note}")


if __name__ == "__main__":
    main()
