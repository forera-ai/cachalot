"""
Check a downloaded Hugging Face checkpoint against the Hub's file list before anything is built from it.

A bank build reads every routed expert of the checkpoint, so a short or missing shard fails 40 minutes in, or worse,
builds a bank from a file that a later resume completed differently. This compares every file the Hub lists at one
revision with the local directory: presence and size always, and with --sha256 the content hash of every LFS file
(the safetensors shards; about 8 minutes for 475 GiB on the X10Pro at ~1 GB/s).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac && ~/venvs/deepseek-v41/bin/python -I \
      scripts/verify_hf_download.py deepseek-ai/DeepSeek-V4.1-Flash-1017 \
      /Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-1017 \
      --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --sha256

Exit status 0 when every listed file is present with the listed size (and hash), 1 otherwise. Local files the Hub
does not list (the `.cache` directory `hf download` writes) are ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def hub_files(repo: str, revision: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as resp:
        entries = json.load(resp)
    return [e for e in entries if e.get("type") == "file"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("local_dir")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--sha256", action="store_true", help="also hash every LFS file (slow)")
    args = ap.parse_args()

    root = Path(args.local_dir)
    files = hub_files(args.repo, args.revision)
    bad = 0
    total = 0
    for entry in sorted(files, key=lambda e: e["path"]):
        local = root / entry["path"]
        size = entry["size"]
        if not local.is_file():
            print(f"MISSING  {entry['path']}")
            bad += 1
            continue
        if local.stat().st_size != size:
            print(f"SIZE     {entry['path']}: {local.stat().st_size} local, {size} on the Hub")
            bad += 1
            continue
        total += size
        lfs = entry.get("lfs")
        if args.sha256 and lfs:
            digest = sha256_of(local)
            if digest != lfs["oid"]:
                print(f"HASH     {entry['path']}: {digest} local, {lfs['oid']} on the Hub")
                bad += 1
                continue
            print(f"ok       {entry['path']}", flush=True)
    print(f"{len(files)} files listed, {bad} bad, {total / 2**30:.1f} GiB checked"
          f"{' with sha256' if args.sha256 else ' by size'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
