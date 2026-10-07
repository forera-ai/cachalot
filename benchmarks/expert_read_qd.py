"""
Read queue depth against bandwidth and latency on one expert bank (L3 sweep 6, 0.61.6).

K worker threads each read real expert records through the shipped ExpertReader (F_NOCACHE) from one shared,
shuffled, disjoint list, so a level never re-reads an expert (a repeat is a page-cache hit even with F_NOCACHE,
HANDOFF 18.1). Per read it records the time from submit to done; per level it reports throughput (bytes over
the level's wall clock) and p50 / p95 / p99 latency. `--raw-mib N` adds a second mode: K threads each pread N MiB
at random aligned offsets of one shard (the drive and link without the record size).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/expert_read_qd.py BANK --levels 1,2,4,8,16 --reads 400 --out OUT.jsonl

Order matters: pass the levels in the order to run, and repeat one at the end as a drift control.

Page cache: F_NOCACHE does not keep pages out of it (HANDOFF 18.1, 18.85). Experts are disjoint within a run, but
a second run, a raw run on a shard an earlier run read, or a file just written reads partly or wholly from RAM and
reports 1.1-26 GB/s on a 1 GB/s drive. Read a rate above the drive's wall as a cache hit; flush with a large read of
other files (never `sudo purge`) between runs that share files, and compare layouts in paired, interleaved (ABBA)
slices (`--skip` offsets each slice) so a drive-state drift hits both.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cachalot.storage.index import detect_expert_bank
from cachalot.storage.reader import ExpertReader


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99), "max": s[-1], "mean": sum(s) / len(s)}


def make_views(entry) -> dict[str, bytearray]:
    return {".".join(t.name.rsplit(".", 2)[-2:]): bytearray(t.size) for t in entry.tensors}


def run_level(k: int, jobs: list, work) -> dict:
    lat: list[float] = []
    nbytes = 0
    lock = threading.Lock()
    it = iter(jobs)

    def worker() -> None:
        nonlocal nbytes
        views = None
        while True:
            with lock:
                job = next(it, None)
            if job is None:
                return
            t0 = time.perf_counter()
            n, views = work(job, views)
            dt = time.perf_counter() - t0
            with lock:
                lat.append(dt * 1000)
                nbytes += n

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(k)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    return {"K": k, "reads": len(lat), "wall_s": round(wall, 3), "GBps": round(nbytes / wall / 1e9, 3),
            **{f"lat_{n}_ms": round(v, 2) for n, v in quantiles(lat).items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bank")
    ap.add_argument("--levels", default="1,2,4,8,16,6,3,12,1,4")
    ap.add_argument("--reads", type=int, default=400, help="reads a level")
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--raw-mib", type=float, default=0.0, help="also run a raw random pread mode of this many MiB")
    ap.add_argument("--glm", action="store_true", help="BANK is a GLM-5.3-Flash checkpoint (per-expert MLX tensors)")
    ap.add_argument("--glm-bank", default="", help="with --glm: read through this contiguous bank (cachalot.glm.bank)")
    ap.add_argument("--layers", default="", help="restrict to these layers, comma list (a bank may cover only some)")
    ap.add_argument("--skip", type=int, default=0, help="start this many experts into the shuffled list (paired runs over the same experts)")
    ap.add_argument("--no-expert", action="store_true", help="skip the expert-record mode (raw only)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    if args.glm:
        from cachalot.glm.experts import build_glm_expert_index

        fmt, index = build_glm_expert_index(args.bank)
        if args.glm_bank:
            from cachalot.glm.bank import apply_bank

            index = apply_bank(fmt, index, Path(args.glm_bank))
    else:
        fmt, index = detect_expert_bank(args.bank)
    keys = sorted(index)
    if args.layers:
        keep = {int(x) for x in args.layers.split(",")}
        keys = [k for k in keys if k[0] in keep]
    random.Random(args.seed).shuffle(keys)
    need = 0 if args.no_expert else args.skip + args.reads * len(levels)
    if need > len(keys):
        raise SystemExit(f"{need} reads need more experts than the bank has ({len(keys)})")
    expert_bytes = sum(t.size for t in index[keys[0]].tensors)
    print(f"bank {args.bank}: {len(keys):,} experts, {expert_bytes / 1e6:.2f} MB each, "
          f"{len(index[keys[0]].tensors)} tensors; {args.reads} reads a level, levels {levels}", flush=True)

    rd = ExpertReader(bypass_page_cache=True)

    def work(key, views):
        views = views or make_views(index[key])
        return rd.read_expert_into(index[key], views), views

    cursor = args.skip
    out = open(args.out, "a")
    print("| mode | K | reads | wall s | GB/s | p50 ms | p95 ms | p99 ms | max ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in [] if args.no_expert else levels:
        jobs = keys[cursor:cursor + args.reads]
        cursor += args.reads
        r = {"mode": "expert", "bank": args.bank, **run_level(k, jobs, work)}
        out.write(json.dumps(r) + "\n"); out.flush()
        print(f"| expert | {k} | {r['reads']} | {r['wall_s']} | {r['GBps']} | {r['lat_p50_ms']} | "
              f"{r['lat_p95_ms']} | {r['lat_p99_ms']} | {r['lat_max_ms']} |", flush=True)

    if args.raw_mib > 0:
        block = int(args.raw_mib * 2**20)
        shards = sorted(Path(args.bank).glob("*.safetensors"))
        shard = shards[len(shards) // 2]
        size = shard.stat().st_size
        fd = os.open(shard, os.O_RDONLY)
        import fcntl
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        rng = random.Random(args.seed + 1)
        slots = size // block
        offs = rng.sample(range(slots), min(slots, args.reads * len(levels)))

        def raw(off, buf):
            buf = buf or bytearray(block)
            return os.preadv(fd, [buf], off * block), buf

        c = 0
        for k in levels:
            jobs = offs[c:c + args.reads]
            c += args.reads
            if len(jobs) < args.reads:
                break
            r = {"mode": f"raw{args.raw_mib:g}MiB", "bank": args.bank, "shard": shard.name, **run_level(k, jobs, raw)}
            out.write(json.dumps(r) + "\n"); out.flush()
            print(f"| raw {args.raw_mib:g} MiB | {k} | {r['reads']} | {r['wall_s']} | {r['GBps']} | {r['lat_p50_ms']} | "
                  f"{r['lat_p95_ms']} | {r['lat_p99_ms']} | {r['lat_max_ms']} |", flush=True)
        os.close(fd)
    rd.close()


if __name__ == "__main__":
    main()
