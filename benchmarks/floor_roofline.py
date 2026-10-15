"""L4(f): split the DeepSeek all-resident floor into a bandwidth-bound part and a remainder.

Offline, no machine. Reads the saved per-row GPU times of the floor profile (`profile_decode_gpu.py`, HANDOFF 18.75) and, for every row,
derives the bytes its kernels read from the checkpoint's tensor shapes (verified against the safetensors headers on 2026-10-15) and divides
them by the rate that kernel class reaches at that launch size (LEDGER MC-READ-SIZE-Q: bf16 GEMV, FP8 GEMV, 2-bit quantized matmul, as a
fraction of 819 GB/s). The bandwidth-bound part of a row is min(bytes / rate, row time); the rest is dequantization compute, launch cost and
latency. Doubling memory bandwidth saves half of the bandwidth-bound sum. Record: docs/L4F-FLOOR-ROOFLINE-RECORD.md.

    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/floor_roofline.py
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

PEAK_GBPS = 819.0
MIB = 1024 * 1024

# LEDGER MC-READ-SIZE-Q: fraction of 819 GB/s at 1 / 2 / 3 / 4 / 8 / 16 / 128 MiB a launch (16 stands for the 11-32 MiB band).
SIZES_MIB = (1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 128.0)
RATE = {
    "bf16": (0.32, 0.64, 0.77, 0.77, 0.87, 0.87, 0.90),
    "fp8": (0.17, 0.33, 0.37, 0.46, 0.56, 0.60, 0.65),
    "q2": (0.31, 0.43, 0.56, 0.61, 0.68, 0.70, 0.72),
}

# ---- bytes, from the checkpoint headers (DeepSeek-V4.1-Flash, layer 0/2/5/25; sizes in bytes) ---------------------------------------
WQ_A = 6_553_600 + 6_400        # fp8 weight + ue8m0 scales
WQ_B = 41_943_040 + 40_960
WKV = 2_621_440 + 2_560
WO_A_BF16 = 8192 * 4096 * 2     # stored fp8 (33.5 MB) but the runtime reads wo_a_bf16 (attention_compressed.py, `Grouped BF16 wo_a`)
WO_B = 41_943_040 + 40_960
SHARED_PROJ = 11_796_480 + 11_520   # one of w1 / w2 / w3
GATE_BF16 = 3_932_160
HC_FN = 1_966_080               # hc_attn_fn / hc_ffn_fn, fp32
COMPRESSOR = 5_242_880 * 2      # wgate + wkv, bf16 (kv-source layers 2, 8, 14, 20)
INDEXER = 5_242_880 + 5_120 + 327_680 + 131_072  # wq_b fp8 + scales, weights_proj, wk (index-source layers)
EXPERT_BYTES = 9.49 * MIB       # 2-bit g128 bank record (9.95 MB), three projections
HEAD_BF16 = 129_280 * 5_120 * 2


def rate_at(kind: str, nbytes: float) -> float:
    """Fraction of peak at a launch of nbytes, log-linear in size between the measured points."""
    mib = max(nbytes / MIB, 1e-9)
    xs, ys = SIZES_MIB, RATE[kind]
    if mib <= xs[0]:
        return ys[0]
    if mib >= xs[-1]:
        return ys[-1]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x0 <= mib <= x1:
            f = (math.log(mib) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + f * (y1 - y0)
    raise AssertionError


def ms_at_rate(kind: str, nbytes: float) -> float:
    return nbytes / (rate_at(kind, nbytes) * PEAK_GBPS * 1e9) * 1e3


@dataclass
class Row:
    name: str
    ms: float                  # measured per call, ms
    count: int                 # calls a token
    launches: list[tuple[str, float]] = field(default_factory=list)  # (kernel class, bytes) per call
    note: str = ""

    @property
    def bytes(self) -> float:
        return sum(b for _, b in self.launches)

    @property
    def bw_ms(self) -> float:
        """Time the row would take if every launch ran at its size-matched rate, capped by the row's measured time."""
        return min(sum(ms_at_rate(k, b) for k, b in self.launches), self.ms)


ATTN = [("fp8", WQ_A), ("fp8", WQ_B), ("fp8", WKV), ("bf16", WO_A_BF16), ("fp8", WO_B)]
SHARED = [("fp8", SHARED_PROJ)] * 3
EXPERTS = [("q2", 6 * EXPERT_BYTES / 3)] * 3   # gate, up, down projections of six experts; the size per launch matters, not the split

# Per-call times of the saved run (benchmarks/results/floor-profile-0.60.7/gpu.out, 17 positions); parsed when the file is there.
DEFAULT_TIMES = {
    "hc_mixes_1d (sinkhorn)": (0.023, 80), "hc_pre_norm_1d": (0.018, 80), "hc_post_1d": (0.021, 80),
    "compressed reuse attention, ratio 2": (0.530, 15), "compressed reuse attention, ratio 1": (0.524, 15),
    "router route_topk_fused (shipped)": (0.022, 40), "shared expert (fp8, 3 gemv)": (0.150, 40),
    "routed experts, 2-bit affine g128 (6), traced": (0.252, 40),
    "layer-0 attention (sliding window)": (0.603, 1), "layer-1 attention (sliding window)": (0.428, 1),
    "compressed attention, source ratio 2": (1.460, 3), "compressed attention, source ratio 1": (0.998, 1),
    "compressed attention, index-only source": (0.836, 4), "final norm + head logits": (1.635, 1),
    "engram forward, layer 1": (0.377, 1), "engram forward, layer 14": (0.362, 1),
}
ROW_RE = re.compile(r"^(?P<name>.+?)\s+(?P<ms>\d+\.\d+) ms \(build\s+[\d.]+\) x\s*(?P<n>\d+) =")


def parse_times(path: Path) -> dict[str, tuple[float, int]]:
    out = dict(DEFAULT_TIMES)
    if path.exists():
        for line in path.read_text().splitlines():
            m = ROW_RE.match(line.strip())
            if m and m["name"] in out:
                out[m["name"]] = (float(m["ms"]), int(m["n"]))
    return out


WINDOW = 128                # sliding-window entries (config sliding_window)
HEAD_DIM = 512              # one KV head
INDEX_DIM = 128             # indexer key width (index_head_dim)
INDEX_TOPK = 512            # entries a reuse layer gathers after selection (index_topk)


def context_bytes(kind: str, ctx: int, ratio: int) -> float:
    """Bytes of KV and compressed state a layer reads beyond its weights at `ctx` tokens (bf16). Derived from the config, not measured:
    reuse layers gather window + top-k compressed entries; source and index-only layers also scan every compressed key with the indexer."""
    if ctx <= 0:
        return 0.0
    n_c = ctx // ratio if ratio else 0
    window = min(ctx, WINDOW) * HEAD_DIM * 2
    gather = min(n_c, INDEX_TOPK) * HEAD_DIM * 2
    scan = n_c * INDEX_DIM * 2
    return {"window": window, "reuse": window + gather, "source": window + gather + scan, "index": window + gather + scan}[kind]


def build_rows(times: dict[str, tuple[float, int]], ctx: int = 0) -> list[Row]:
    def row(name, launches, note=""):
        ms, n = times[name]
        return Row(name, ms, n, launches, note)

    def kv(kind, ratio):
        b = context_bytes(kind, ctx, ratio)
        return [("bf16", b)] if b else []
    rows = [
        row("hc_mixes_1d (sinkhorn)", [("bf16", HC_FN)], "fp32 matrix read as a GEMV; the bf16 curve stands in"),
        row("hc_pre_norm_1d", [], "elementwise on 4 x 5120; bytes negligible"),
        row("hc_post_1d", [], "elementwise on 4 x 5120; bytes negligible"),
        row("compressed reuse attention, ratio 2", ATTN + kv("reuse", 2), "five projections; KV and window state added by context"),
        row("compressed reuse attention, ratio 1", ATTN + kv("reuse", 1), "five projections; twice the keys of ratio 2 (context term only)"),
        row("router route_topk_fused (shipped)", [("bf16", GATE_BF16)]),
        row("shared expert (fp8, 3 gemv)", SHARED),
        row("routed experts, 2-bit affine g128 (6), traced", EXPERTS),
        row("layer-0 attention (sliding window)", ATTN + kv("window", 0)),
        row("layer-1 attention (sliding window)", ATTN + kv("window", 0)),
        row("compressed attention, source ratio 2", ATTN + [("bf16", COMPRESSOR), ("fp8", INDEXER)] + kv("source", 2),
            "assumes the row includes the compressor and the indexer"),
        row("compressed attention, source ratio 1", ATTN + [("bf16", COMPRESSOR), ("fp8", INDEXER)] + kv("source", 1),
            "assumes the row includes the compressor and the indexer"),
        row("compressed attention, index-only source", ATTN + [("fp8", INDEXER)] + kv("index", 1), "assumes the row includes the indexer"),
        row("final norm + head logits", [("bf16", HEAD_BF16)]),
        row("engram forward, layer 1", [], "gathered embedding rows; bytes not derived (lower bound)"),
        row("engram forward, layer 14", [], "gathered embedding rows; bytes not derived (lower bound)"),
    ]
    return rows


def summarize(rows: list[Row]) -> dict[str, float]:
    token_bytes = sum(r.bytes * r.count for r in rows)
    gpu_ms = sum(r.ms * r.count for r in rows)
    bw_ms = sum(r.bw_ms * r.count for r in rows)
    return {"token_bytes_gb": token_bytes / 1e9, "gpu_ms": gpu_ms, "bw_ms": bw_ms, "share": bw_ms / gpu_ms,
            "saving_2x_ms": bw_ms / 2}


def report(rows: list[Row], floor_ms: float, label: str) -> None:
    print(f"== {label}")
    print(f"{'row':48s} {'ms/call':>8s} {'x':>3s} {'MB/call':>8s} {'GB/s':>6s} {'%peak':>6s} {'bw ms':>7s} {'of row':>7s} {'ms/token':>9s}")
    for r in rows:
        gbps = r.bytes / (r.ms * 1e-3) / 1e9 if r.bytes else 0.0
        share = r.bw_ms / r.ms if r.ms else 0.0
        print(f"{r.name:48s} {r.ms:8.3f} {r.count:3d} {r.bytes / 1e6:8.1f} {gbps:6.0f} {gbps / PEAK_GBPS * 100:5.0f}% "
              f"{r.bw_ms:7.3f} {share * 100:6.0f}% {r.ms * r.count:9.1f}")
    s = summarize(rows)
    print(f"bytes a token {s['token_bytes_gb']:.2f} GB; chained GPU rows {s['gpu_ms']:.1f} ms; bandwidth-bound {s['bw_ms']:.1f} ms "
          f"({s['share'] * 100:.0f} % of the rows); doubling bandwidth saves {s['saving_2x_ms']:.1f} ms "
          f"of a {floor_ms:.0f} ms floor ({s['saving_2x_ms'] / floor_ms * 100:.1f} %), floor -> {floor_ms - s['saving_2x_ms']:.1f} ms")


ATTN_ARMS = ("wq_a", "wkv", "wq_b", "wo_b", "wo_a (bf16)")
ATTN_ROW_PREFIXES = ("compressed reuse attention", "compressed attention, ", "layer-0 attention", "layer-1 attention")


def projection_variants(json_paths: list[str]) -> dict[str, float]:
    """Per-layer ms of the five attention projections from `micro_attn_projection_roofline.py` runs (mean of the best times):
    upper = the measured time of the kernels as issued; sum = each projection's bytes at the `mx.sum` rate over the same bytes, capped
    by its own time; peak77 = all bytes at the best rate any projection reached (the bf16 wo_a, ~77 % of 819 GB/s)."""
    import json

    runs = [json.loads(Path(p).read_text())["rows"] for p in json_paths]
    own = {a: sum(next(r for r in run if r["arm"] == a)["ms_best"] for run in runs) / len(runs) for a in ATTN_ARMS}
    sum_ms = {a: sum(next(r for r in run if r["arm"] == (a.split(" ")[0] + " sum"))["ms_best"] for run in runs) / len(runs)
              for a in ATTN_ARMS}
    nbytes = {a: next(r for r in runs[0] if r["arm"] == a)["bytes"] for a in ATTN_ARMS}
    best_rate = max(nbytes[a] / (own[a] * 1e-3) for a in ATTN_ARMS)
    return {
        "upper": sum(own.values()),
        "sum": sum(min(sum_ms[a], own[a]) for a in ATTN_ARMS),
        "peak77": sum(nbytes.values()) / best_rate * 1e3,
        "table": sum(ms_at_rate(k, b) for k, b in ATTN),
    }


def measured_attention_summary(rows: list[Row], variants: dict[str, float]) -> dict[str, float]:
    """Bandwidth-bound floor time when the five projections of every attention row use a measured variant instead of the rate tables."""
    out = {}
    for name, proj_ms in variants.items():
        total = 0.0
        for r in rows:
            bw = r.bw_ms
            if r.name.startswith(ATTN_ROW_PREFIXES):
                bw = min(r.ms, bw - variants["table"] + proj_ms)
            total += bw * r.count
        out[name] = total
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-out", default="benchmarks/results/floor-profile-0.60.7/gpu.out")
    ap.add_argument("--floor-ms", type=float, default=80.0, help="the server's all-resident token (DS-FLOOR-SRV)")
    ap.add_argument("--measured", nargs="*", default=[], help="micro_attn_projection_roofline.py JSON files (step 2)")
    ap.add_argument("--context", type=int, nargs="*", default=[22000], help="context lengths for the derived KV column")
    args = ap.parse_args()
    times = parse_times(Path(args.gpu_out))
    report(build_rows(times), args.floor_ms, "17 positions (the saved run)")
    if args.measured:
        v = projection_variants(args.measured)
        print()
        print("== step 2: the five attention projections, measured (ms a layer): "
              + ", ".join(f"{k} {x:.3f}" for k, x in v.items()))
        for name, total in measured_attention_summary(build_rows(times), v).items():
            print(f"   projections at '{name}': floor bandwidth-bound {total:.1f} ms, doubling bandwidth saves {total / 2:.1f} ms "
                  f"({total / 2 / args.floor_ms * 100:.1f} % of {args.floor_ms:.0f})")
    for ctx in args.context:
        print()
        report(build_rows(times, ctx), args.floor_ms,
               f"{ctx} tokens of context: weights as above plus derived KV bytes; row times are the 17-position ones, NOT measured at {ctx}")


if __name__ == "__main__":
    main()
