"""
Write or check a contiguous GLM-5.3-Flash expert bank (cachalot.glm.bank, HANDOFF 18.34).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    # one record per expert (the slot's own tensor order), layers a-b or all MoE layers; each layer checked byte for byte
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/glm_bank.py --write MODEL OUT [--layers 3-24]
    # byte-compare N random experts of every layer the bank holds against the checkpoint
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/glm_bank.py --verify MODEL OUT [--n 64]

A layer is written to `.part`, renamed when complete, and `bank.json` is rewritten after every layer, so an interrupted
run leaves a usable bank of the layers done.
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cachalot.glm.bank import ALIGN, BANK_VERSION, record_layout
from cachalot.glm.experts import build_glm_expert_index


def _read(fds, t):
    fd = fds.get(t.shard)
    if fd is None:
        fd = fds[t.shard] = os.open(t.shard, os.O_RDONLY)
    return os.pread(fd, t.size, t.start)


def _record(fds, entry, fmt, layout, record):
    by_name = {".".join(t.name.split(".")[-2:]): t for t in entry.tensors}
    buf = bytearray(record)
    for name, (off, size) in layout.items():
        data = _read(fds, by_name[name])
        assert len(data) == size, (entry.layer, entry.expert, name)
        buf[off:off + size] = data
    return bytes(buf)


def write(model, out, layers):
    fmt, index = build_glm_expert_index(model)
    layout, record = record_layout(fmt)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "bank.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "version": BANK_VERSION, "tensor_names": list(fmt.tensor_names), "record_bytes": record, "layers": {}}
    all_layers = sorted({k[0] for k in index})
    todo = [layer for layer in all_layers if layers is None or layer in layers]
    fds = {}
    print(f"record {record} bytes ({record / 1048576:.2f} MiB), layers {todo[0]}..{todo[-1]} ({len(todo)})", flush=True)
    for layer in todo:
        experts = sorted(e for (l, e) in index if l == layer)
        t0 = time.perf_counter()
        name = f"layer-{layer:03d}.bin"
        with ThreadPoolExecutor(8) as pool, open(out / (name + ".part"), "wb") as f:
            for chunk in range(0, len(experts), 8):
                for rec in pool.map(lambda e: _record(fds, index[(layer, e)], fmt, layout, record), experts[chunk:chunk + 8]):
                    f.write(rec)
            f.flush()
            os.fsync(f.fileno())
        os.replace(out / (name + ".part"), out / name)
        meta["layers"][str(layer)] = {"file": name, "experts": len(experts)}
        meta_path.write_text(json.dumps(meta))
        bad = check_layer(fds, index, fmt, layout, record, out / name, layer, experts, 16)
        print(f"layer {layer:2d} {len(experts)} experts {time.perf_counter() - t0:5.1f} s  check 16: {'ok' if not bad else bad}",
              flush=True)
        assert not bad


def check_layer(fds, index, fmt, layout, record, path, layer, experts, n):
    rng = random.Random(layer)
    bad = []
    with open(path, "rb") as f:
        for e in rng.sample(experts, min(n, len(experts))):
            f.seek(e * record)
            if f.read(record) != _record(fds, index[(layer, e)], fmt, layout, record):
                bad.append(e)
    return bad


def verify(model, out, n):
    fmt, index = build_glm_expert_index(model)
    layout, record = record_layout(fmt)
    meta = json.loads((Path(out) / "bank.json").read_text())
    fds, total = {}, 0
    for layer_s, info in sorted(meta["layers"].items(), key=lambda kv: int(kv[0])):
        layer = int(layer_s)
        bad = check_layer(fds, index, fmt, layout, record, Path(out) / info["file"], layer, list(range(info["experts"])), n)
        total += n
        print(f"layer {layer:2d} {'ok' if not bad else 'MISMATCH ' + str(bad)}")
    print(f"VERIFY layers {len(meta['layers'])} sampled {total} experts")


def _layers(spec):
    if not spec:
        return None
    a, _, b = spec.partition("-")
    return set(range(int(a), int(b or a) + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", nargs=2, metavar=("MODEL", "OUT"))
    ap.add_argument("--verify", nargs=2, metavar=("MODEL", "OUT"))
    ap.add_argument("--layers")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    if a.write:
        write(a.write[0], a.write[1], _layers(a.layers))
    elif a.verify:
        verify(a.verify[0], a.verify[1], a.n)
    else:
        ap.print_help()
