"""Compare the longest snapshot two directories share a token count with, tensor by tensor (docs/glm-speed/G2-RECORD.md). Usage: compare_blocks.py <dir a> <dir b>"""
import glob, hashlib, json, os, struct, sys


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def digests(path):
    head, base = header(path)
    out = {}
    with open(path, "rb") as f:
        for name, info in head.items():
            if name == "__metadata__":
                continue
            a, b = info["data_offsets"]
            f.seek(base + a)
            out[name] = (info["dtype"], tuple(info["shape"]), hashlib.sha256(f.read(b - a)).hexdigest())
    return out


def files(d):
    return {int(os.path.basename(p).split("-")[1]): p for p in glob.glob(os.path.join(d, "prefix-*.safetensors"))}


a, b = files(sys.argv[1]), files(sys.argv[2])
print("dir a token counts", sorted(a), "dir b", sorted(b))
n = max(set(a) & set(b))
da, db = digests(a[n]), digests(b[n])
diff = [k for k in da if da[k] != db.get(k)] + [k for k in db if k not in da]
print(f"compared {n} tokens: {len(da)} tensors, {len(diff)} differ; {'IDENTICAL' if not diff else diff[:5]}")
