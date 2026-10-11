"""
Per-layer rows from the raw tracer columns saved by `token_trace_raw.py` (docs/L4B-RAW-LAYER-RECORD.md, HANDOFF 18.136).
Offline: reads benchmarks/results/l4b-raw/turn*.npz, loads no model.

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    PYTHONPATH=src ~/venvs/deepseek-v41/bin/python benchmarks/layer_rows.py benchmarks/results/l4b-raw

A layer's window runs from its router-sync start to the next layer's router-sync start (the last layer's, to the token end).
Per window: sync time, blocked time (`get_many` wait), read busy time (union of read intervals clipped to the window), and the
part of it hidden (outside the blocked interval).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cachalot.metrics import token_trace as tt


def _clip(iv: list[tuple[int, int]], lo: int, hi: int) -> list[tuple[int, int]]:
    out = []
    for s, e in iv:
        s2, e2 = max(s, lo), min(e, hi)
        if e2 > s2:
            out.append((s2, e2))
    return out


def layer_rows(d: dict) -> list[dict]:
    """One dict per (token, layer) for the active tokens of one drained turn."""
    ts, code, a, b = d["ts"], d["code"], d["a"], d["b"]
    stamps, positions = d["token_stamps"], d["token_positions"]
    active = {int(t) for t in ts[code == tt.TOKEN].tolist()}
    rows: list[dict] = []
    for k in range(len(stamps) - 1):
        lo, hi = int(stamps[k]), int(stamps[k + 1])
        if lo not in active:
            continue
        sel = (ts >= lo) & (ts < hi)
        # sync and get intervals with their layer
        ev = []
        for c0, c1, tag in ((tt.SYNC_START, tt.SYNC_END, "sync"), (tt.GET_START, tt.GET_END, "get")):
            s_m, e_m = sel & (code == c0), sel & (code == c1)
            pend: dict[int, list[int]] = {}
            for t, kind, layer in sorted([(t, 0, l) for t, l in zip(ts[s_m].tolist(), a[s_m].tolist())]
                                         + [(t, 1, l) for t, l in zip(ts[e_m].tolist(), a[e_m].tolist())]):
                if kind == 0:
                    pend.setdefault(layer, []).append(t)
                elif pend.get(layer):
                    ev.append((tag, layer, pend[layer].pop(0), t))
        rs, re_ = np.sort(ts[sel & (code == tt.READ_START)]), np.sort(ts[sel & (code == tt.READ_END)])
        n = min(len(rs), len(re_))
        reads = tt._union(rs[:n], re_[:n]) if n else []
        sync = {l: (s, e) for tag, l, s, e in ev if tag == "sync"}
        get = {l: (s, e) for tag, l, s, e in ev if tag == "get"}
        layers = sorted(sync)
        for i, l in enumerate(layers):
            w_lo = sync[l][0]
            w_hi = sync[layers[i + 1]][0] if i + 1 < len(layers) else hi
            win = _clip(reads, w_lo, w_hi)
            busy = sum(e - s for s, e in win)
            g = get.get(l)
            blocked = (g[1] - g[0]) if g else 0
            ov = tt._overlap(win, [g]) if g and win else 0
            rows.append({"position": int(positions[k]), "layer": int(l), "sync_ms": (sync[l][1] - sync[l][0]) / 1e6,
                         "blocked_ms": blocked / 1e6, "busy_ms": busy / 1e6, "hidden_ms": (busy - ov) / 1e6,
                         "window_ms": (w_hi - w_lo) / 1e6})
    return rows


def load_all(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((path / "manifest.json").read_text())
    rows: list[dict] = []
    for f in sorted(path.glob("turn*.npz")):
        z = np.load(f)
        d = {k: z[k] for k in z.files}
        turn = int(f.stem[4:])
        for r in layer_rows(d):
            r["turn"] = turn
            rows.append(r)
    return manifest, rows


def report(manifest: dict, rows: list[dict]) -> dict:
    turns = sorted({r["turn"] for r in rows})
    by_tok: dict[tuple[int, int], list[dict]] = {}
    for r in rows:
        by_tok.setdefault((r["turn"], r["position"]), []).append(r)
    top10, tok_tot = [], {"sync": [], "blocked": [], "busy": []}
    for key, rs in by_tok.items():
        bl = np.array([r["blocked_ms"] for r in rs])
        if bl.sum() > 1.0:
            top10.append(np.sort(bl)[-10:].sum() / bl.sum())
        tok_tot["sync"].append(sum(r["sync_ms"] for r in rs))
        tok_tot["blocked"].append(bl.sum())
        tok_tot["busy"].append(sum(r["busy_ms"] for r in rs))
    layers = sorted({r["layer"] for r in rows})
    half = layers[len(layers) // 2]
    def share(sel):
        b = sum(r["busy_ms"] for r in rows if sel(r)); h = sum(r["hidden_ms"] for r in rows if sel(r))
        return h / b if b else float("nan")
    per_layer = {l: {"blocked": float(np.mean([r["blocked_ms"] for r in rows if r["layer"] == l])),
                     "sync": float(np.mean([r["sync_ms"] for r in rows if r["layer"] == l])),
                     "busy": float(np.mean([r["busy_ms"] for r in rows if r["layer"] == l])),
                     "hidden": float(np.mean([r["hidden_ms"] for r in rows if r["layer"] == l]))} for l in layers}
    return {
        "turns": len(turns), "tokens": len(by_tok), "layers": len(layers), "dropped": sum(r.get("dropped", 0) for r in manifest["rows"]),
        "median_top10_blocked_share": float(np.median(top10)) if top10 else float("nan"),
        "hidden_share_first_half": share(lambda r: r["layer"] < half), "hidden_share_second_half": share(lambda r: r["layer"] >= half),
        "split_layer": half,
        "per_token_means": {k: float(np.mean(v)) for k, v in tok_tot.items()},
        "per_layer": per_layer,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--layers", action="store_true")
    args = ap.parse_args()
    man, rows = load_all(args.path)
    rep = report(man, rows)
    pl = rep.pop("per_layer")
    print(json.dumps(rep, indent=2))
    if args.layers:
        for l, v in pl.items():
            print(l, {k: round(x, 2) for k, x in v.items()})
