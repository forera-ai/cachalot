"""
Write, check or scan a bias-free MiniMax-M3 expert bank (cachalot.minimax.coded_bank, HANDOFF 18.4).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    # every expert group's bias against (k * scale); prints the k histogram (~40 s, reads 24 GiB)
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_coded_bank.py --scan MODEL
    # write the bank for MoE layers 3-42 (or all with no --layers), then check 64 random experts byte for byte
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_coded_bank.py --write MODEL OUT [--layers 3-42]
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_coded_bank.py --verify MODEL OUT [--n 64]
    # compress every record's head into BANK/heads.zst + heads.json (HANDOFF 18.7; ~4.2 GiB, a few minutes),
    # checking each blob against the record's own head
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_coded_bank.py --zheads BANK [--level 19]
    # M24 (HANDOFF 18.16): rewrite the bank's records as slot images in place, layer by layer, each layer checked
    # byte for byte in a pair slot before the old file goes (~3 min for all 56 layers on the internal SSD)
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_coded_bank.py --to-slot BANK [--layers 3-42]
    # a tail-only mirror on a second drive: sparse layer files holding the last --tail of each weight piece
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/minimax_coded_bank.py --mirror-tail BANK MIRROR [--tail 0.25]

Records are written in expert order, one file per layer; a layer file is written to `.part` and renamed, and
`bank.json` is rewritten after every layer, so an interrupted run leaves a usable bank of the layers done.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cachalot.glm.experts import tensor_sizes
from cachalot.minimax.coded_bank import (
    BANK_VERSION,
    PROJS,
    CodedBankReader,
    encode_biases,
    format_to_json,
    write_zheads,
    layout_from_sizes,
)
from cachalot.minimax.experts import build_minimax_expert_index
from cachalot.storage.reader import ExpertReader


def read_pieces(reader: ExpertReader, entry, sizes) -> dict[str, np.ndarray]:
    views = {k: np.empty(v, np.uint8) for k, v in sizes.items()}
    reader.read_expert_into(entry, views)
    return views


def build_record(views, lay) -> tuple[str, bytes]:
    scales = [views[f"{p}.scales"].view(np.uint16) for p in PROJS]
    biases = [views[f"{p}.biases"].view(np.uint16) for p in PROJS]
    codes = [encode_biases(s, b) for s, b in zip(scales, biases, strict=True)]
    if all(c is not None for c in codes):
        head = b"".join(s.tobytes() for s in scales) + b"".join(c.tobytes() for c in codes)
        kind, head_size = "coded", lay.coded_head
    else:
        head = b"".join(s.tobytes() for s in scales) + b"".join(b.tobytes() for b in biases)
        kind, head_size = "raw", lay.raw_head
    head += b"\0" * (head_size - len(head))
    body = b"".join(views[f"{p}.weight"].tobytes() for p in PROJS)
    return kind, head + body


def write(model, out, layers):
    fmt, index = build_minimax_expert_index(model)
    sizes = tensor_sizes(fmt)
    lay = layout_from_sizes(sizes)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "bank.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "version": BANK_VERSION,
        "checkpoint": str(Path(model).resolve()),
        "layout": {"weight": lay.weight, "scales": lay.scales},
        "records": [],
    }
    meta["format"] = format_to_json(fmt)
    done = {r[0] for r in meta["records"]}
    all_layers = sorted({k[0] for k in index})
    todo = [L for L in all_layers if (layers is None or L in layers) and L not in done]
    reader = ExpertReader(bypass_page_cache=True)
    pool = ThreadPoolExecutor(8)
    n_exp = max(k[1] for k in index) + 1
    for L in todo:
        t0 = time.perf_counter()
        fname = f"layer-{L:03d}.bin"
        part = out / (fname + ".part")
        fd = os.open(part, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        offset, recs, raw = 0, [], 0

        def job(e, L=L):
            return build_record(read_pieces(reader, index[(L, e)], sizes), lay)

        for e, (kind, data) in enumerate(pool.map(job, range(n_exp))):
            os.pwrite(fd, data, offset)
            recs.append([L, e, fname, offset, kind])
            raw += kind == "raw"
            offset += len(data)
        os.fsync(fd)
        os.close(fd)
        os.replace(part, out / fname)
        meta["records"] += recs
        tmp = out / "bank.json.part"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, meta_path)
        dt = time.perf_counter() - t0
        print(f"layer {L}: {offset / 2**30:.2f} GiB, {raw} raw of {n_exp}, {dt:.1f} s", flush=True)
    reader.close()


def verify(model, out, n):
    fmt, index = build_minimax_expert_index(model)
    sizes = tensor_sizes(fmt)
    bank = CodedBankReader(out, bypass_page_cache=True)
    plain = ExpertReader(bypass_page_cache=True)
    keys = sorted(bank.records)
    random.seed(int(os.environ.get("SEED", "5")))
    sample = random.sample(keys, min(n, len(keys)))
    raw = [k for k in keys if bank.records[k][2] == "raw"]
    sample += random.sample(raw, min(4, len(raw)))
    bad = 0
    for key in sample:
        a = {k: np.full(v, 0xAB, np.uint8) for k, v in sizes.items()}
        b = {k: np.empty(v, np.uint8) for k, v in sizes.items()}
        bank.read_expert_into(index[key], a)
        plain.read_expert_into(index[key], b)
        same = all(np.array_equal(a[k], b[k]) for k in sizes)
        bad += not same
        if not same:
            print("MISMATCH", key, bank.records[key], [k for k in sizes if not np.array_equal(a[k], b[k])])
    print(f"VERIFY {len(sample)} experts ({sum(bank.records[k][2] == 'raw' for k in sample)} raw) mismatched {bad}; "
          f"bank covers {len(keys)} experts, {len(raw)} raw")
    return bad == 0


def to_slot(bank, layers, target="slot"):
    """Rewrite a bank's coded and raw records as slot-image records (M24, HANDOFF 18.16), in place, one layer at a
    time: the new layer file is written beside the old one, every expert of it is read back into a pair slot
    through both records and compared byte for byte, then bank.json points at the new file (atomic replace) and
    the old file is removed. An interrupted run leaves a usable bank at every step. `target="coded"` goes back:
    coded records where every bias is k x scale for k in -6..-3, raw ones otherwise, in `layer-NNN.bin` files, the
    bank --write would have written (for A/Bs against the old format)."""
    from cachalot.glm.experts import tensor_sizes
    from cachalot.minimax.coded_bank import (BankLayout, index_from_bank, raw_head_from_slot, slot_format,
                                             slot_head_from)

    bank = Path(bank)
    meta_path = bank / "bank.json"
    meta = json.loads(meta_path.read_text())
    lay = BankLayout(**meta["layout"])
    fmt, _ = index_from_bank(bank)
    sizes = tensor_sizes(slot_format(fmt, sidx=2))
    reader = CodedBankReader(bank, bypass_page_cache=True)
    pool = ThreadPoolExecutor(8)
    done = {"slot"} if target == "slot" else {"coded", "raw"}
    todo = sorted({r[0] for r in meta["records"] if r[4] not in done and (layers is None or r[0] in layers)})
    t_all = time.perf_counter()
    for L in todo:
        t0 = time.perf_counter()
        # APFS frees a deleted layer file lazily (and not while Spotlight holds it): wait for room for two layers
        need = 2 * lay.record("coded") * 128 + (2 << 30)
        while os.statvfs(bank).f_bavail * os.statvfs(bank).f_frsize < need and time.perf_counter() - t0 < 600:
            time.sleep(2)
        recs = sorted((r for r in meta["records"] if r[0] == L), key=lambda r: r[1])
        old_files = {r[2] for r in recs}
        fname = f"layer-{L:03d}-slot.bin" if target == "slot" else f"layer-{L:03d}.bin"
        part = bank / (fname + ".part")
        fds = {f: os.open(bank / f, os.O_RDONLY) for f in old_files}
        for fd in fds.values():
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)

        def job(r):
            fd, kind = fds[r[2]], r[4]
            head = os.pread(fd, lay.head_payload(kind), r[3])
            weights = os.pread(fd, 3 * lay.weight, r[3] + lay.head(kind))
            if len(head) != lay.head_payload(kind) or len(weights) != 3 * lay.weight:
                raise OSError(f"short read of {r}")
            if target == "slot":
                sh = slot_head_from(kind, head, lay)
                return "slot", sh + b"\0" * (lay.slot_head - len(sh)) + weights
            raw = raw_head_from_slot(head, lay)
            n = 3 * lay.groups
            views = {}
            for i, p in enumerate(PROJS):
                views[f"{p}.scales"] = np.frombuffer(raw, np.uint8, lay.scales, i * lay.scales)
                views[f"{p}.biases"] = np.frombuffer(raw, np.uint8, lay.scales, 2 * n + i * lay.scales)
                views[f"{p}.weight"] = np.frombuffer(weights, np.uint8, lay.weight, i * lay.weight)
            return build_record(views, lay)

        out = os.open(part, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        fcntl.fcntl(out, fcntl.F_NOCACHE, 1)
        offset, new = 0, []
        for r, (kind, data) in zip(recs, pool.map(job, recs), strict=True):
            os.pwrite(out, data, offset)
            new.append([L, r[1], fname, offset, kind])
            offset += len(data)
        os.fsync(out)
        os.close(out)
        for fd in fds.values():
            os.close(fd)
        os.replace(part, bank / fname)
        t_write = time.perf_counter() - t0

        def check(pair):
            old, rec = pair
            a = {k: np.full(v, 0xAB, np.uint8) for k, v in sizes.items()}
            b = {k: np.full(v, 0x5C, np.uint8) for k, v in sizes.items()}
            reader._read_record(tuple(old[2:5]), a)
            reader._read_record(tuple(rec[2:5]), b)
            return [k for k in sizes if not np.array_equal(a[k], b[k])]

        bad = [(o[1], d) for o, d in zip(recs, pool.map(check, zip(recs, new, strict=True)), strict=True) if d]
        if bad:
            (bank / fname).unlink()
            raise SystemExit(f"layer {L}: {len(bad)} experts differ, first {bad[:3]}; old layer kept")
        meta["records"] = [r for r in meta["records"] if r[0] != L] + new
        meta["version"] = 2 if any(r[4] == "slot" for r in meta["records"]) else 1
        tmp = bank / "bank.json.part"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, meta_path)
        for f in old_files - {fname}:
            # the check's reader holds the old file open, which would keep its blocks allocated until exit
            reader.close()
            (bank / f).unlink()
        print(f"layer {L}: {offset / 2**30:.2f} GiB {target} records, {len(recs)} experts byte-equal in a pair slot, "
              f"write {t_write:.1f} s, total {time.perf_counter() - t0:.1f} s", flush=True)
    reader.close()
    print(f"TO_SLOT {len(todo)} layers in {time.perf_counter() - t_all:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--to-slot", action="store_true", help="MODEL is the bank directory; rewrite it in place (M24)")
    ap.add_argument("--to-coded", action="store_true", help="the reverse of --to-slot (A/Bs)")
    ap.add_argument("--mirror-tail", action="store_true", help="MODEL is the bank, OUT the tail-only mirror to write")
    ap.add_argument("--tail", type=float, default=0.25)
    ap.add_argument("--zheads", action="store_true", help="MODEL is the bank directory")
    ap.add_argument("--level", type=int, default=19)
    ap.add_argument("model")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--layers", default=None, help="a-b, inclusive")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    layers = None
    if a.layers:
        lo, hi = (int(x) for x in a.layers.split("-"))
        layers = set(range(lo, hi + 1))
    if a.to_slot or a.to_coded:
        to_slot(a.model, layers, "slot" if a.to_slot else "coded")
        return
    if a.mirror_tail:
        from cachalot.minimax.coded_bank import write_mirror_tail

        n, written, dt = write_mirror_tail(a.model, a.out, a.tail)
        print(f"MIRROR_TAIL {n} records, {written / 2**30:.1f} GiB written (tail {a.tail}), {dt:.0f} s", flush=True)
        return
    if a.zheads:
        n, raw, packed, dt = write_zheads(a.model, a.level)
        print(f"ZHEADS {n} heads, {raw / 2**30:.2f} -> {packed / 2**30:.2f} GiB ({packed / raw:.3f}), "
              f"level {a.level}, {dt:.0f} s")
        return
    if a.scan:
        scan(a.model)
    if a.write:
        write(a.model, a.out, layers)
    if a.verify or a.write:
        sys.exit(0 if verify(a.model, a.out, a.n) else 1)


def scan(model):
    """Every routed-expert group's bias against bf16(k * scale), k = round(bias / scale); the k histogram."""
    fmt, index = build_minimax_expert_index(model)
    reader = ExpertReader(bypass_page_cache=True)
    hist, bad, total, t0 = {}, 0, 0, time.perf_counter()
    for L in sorted({k[0] for k in index}):
        for key in [k for k in sorted(index) if k[0] == L]:
            for p in PROJS:
                ranges = {t.name.rsplit(".", 2)[-2] + "." + t.name.rsplit(".", 1)[-1]: t for t in index[key].tensors}
                s_t, b_t = ranges[f"{p}.scales"], ranges[f"{p}.biases"]
                fd = reader._fd(s_t.shard)
                s = np.frombuffer(os.pread(fd, s_t.size, s_t.start), np.uint16)
                b = np.frombuffer(os.pread(reader._fd(b_t.shard), b_t.size, b_t.start), np.uint16)
                sf, bf = (s.astype(np.uint32) << 16).view(np.float32), (b.astype(np.uint32) << 16).view(np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    k = np.where(sf == 0, 0, np.round(bf / sf))
                from cachalot.minimax.coded_bank import bf16_bits_rne

                ok = (bf16_bits_rne((k * sf).astype(np.float32)) == b) | ((sf == 0) & (bf == 0))
                bad += int((~ok).sum())
                total += b.size
                for v, c in zip(*np.unique(k, return_counts=True), strict=True):
                    hist[int(v)] = hist.get(int(v), 0) + int(c)
    print(f"SCAN groups {total} mismatched {bad} k {hist} {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
