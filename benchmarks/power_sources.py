"""Power sources on this Mac: a read-only probe and a uniform energy reader (charter L5b, docs/POWER-ACCOUNTING-PLAN.md, job E0).

Two sources are tried, neither needs root as far as this module knows (the probe records what it actually saw):

* IOReport (private `libIOReport`, loaded through the dyld cache): the "Energy Model" group holds cumulative energy counters per
  block (CPU clusters, GPU, ANE and, on Apple Silicon, DRAM and other fabric blocks under names the probe lists as found).
* SMC (`AppleSMC` through IOKit): keys whose names begin with `P` are power rails in watts (system total, DC input, ...).

    cd /Users/hamedprooshani/Projects/deepseek-v41-mac
    ~/venvs/deepseek-v41/bin/python benchmarks/power_sources.py                  # table of every channel and key found
    ~/venvs/deepseek-v41/bin/python benchmarks/power_sources.py --json OUT.json  # the same, machine readable
    ~/venvs/deepseek-v41/bin/python benchmarks/power_sources.py --watch 5        # 5 s of watts per channel and SMC power key

`read_energy()` returns cumulative joules by component with the source and a tag. A source that is unavailable is `None`, never zero
(charter section 7). Tags: IOReport energy counters are `measured` (the SoC's own estimate, as powermetrics', not a wall meter);
SMC rails are `measured` as power (watts) and are integrated here only by the caller.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import re
import struct
import sys
import time

from cachalot.metrics.power_sources import (  # noqa: E402,F401  (the readers live in the package since 0.63.0)
    SMC, IOReport, _UNIT_TO_J, _cf, _cfstr, _component, _decode, _fourcc, _fourcc_str, _pystr,
)


def smc_power_keys(smc: SMC, prefixes=("P",)) -> list[dict]:
    rows = []
    for i in range(smc.key_count()):
        k = smc.key_at(i)
        if not k.startswith(prefixes):
            continue
        try:
            typ, raw, val = smc.read(k)
        except Exception:
            continue
        rows.append({"key": k, "type": typ, "value": val, "hex": raw.hex()})
    return rows


# --------------------------------------------------------------------------- uniform reader
def read_energy() -> dict | None:
    """Cumulative joules by component {component: {"joules": float, "source": str, "tag": "measured", "channels": [names]}}.

    A component with no channel is absent; if IOReport cannot be read at all the result is None (never zeros)."""
    try:
        snap = IOReport("Energy Model").sample()
    except Exception:
        return None
    out: dict[str, dict] = {}
    for ch in snap:
        comp = _component(ch["group"], ch["name"])
        scale = _UNIT_TO_J.get(ch["unit"])
        if comp is None or scale is None or ch["value"] is None:
            continue
        d = out.setdefault(comp, {"joules": 0.0, "source": "IOReport Energy Model", "tag": "measured", "channels": []})
        d["joules"] += ch["value"] * scale
        d["channels"].append(ch["name"])
    return out or None


def read_system_watts() -> dict | None:
    """Whole-system input power from the SMC key PSTR (watts), or None. Read without root on this Mac (2026-10-10: 24-34 W while
    package power was ~4 W). What PSTR includes (PSU losses, fans, drives) is **to verify** against an inline meter; the tag is
    `measured` as a reading of the SMC's own sensor, not as wall power."""
    try:
        typ, raw, val = SMC().read("PSTR")
    except Exception:
        return None
    return None if val is None else {"watts": float(val), "source": "SMC PSTR", "tag": "measured"}


def advancing(seconds: float = 1.0) -> dict[str, bool] | None:
    """Which Energy Model components actually advance between two samples. On this Mac (macOS 27, 2026-10-10) only the GPU counter
    moved without root; CPU, ANE, DRAM, DCS and AMCC stayed frozen at the value they had at boot-time-ish, so `read_energy()` values
    for them are NOT usable until this says True. A caller must check this before trusting a joule figure."""
    try:
        rep = IOReport("Energy Model")
        a = rep.sample()
        time.sleep(seconds)
        b = rep.sample()
    except Exception:
        return None
    moved: dict[str, bool] = {}
    for x, y in zip(a, b):
        cm = _component(x["group"], x["name"])
        if cm and x["value"] is not None and y["value"] is not None:
            moved[cm] = moved.get(cm, False) or (y["value"] != x["value"])
    return moved or None


class FullSampler:
    """Every Energy Model channel by position (the order is stable between samples) plus, every `smc_every` rows, the SMC power (P), voltage (V),
    current (I), fan-speed (F?Ac) and temperature (T) keys that decode to a number. `header()` is written once; `row()` once a second."""

    def __init__(self, smc_every: int = 5):
        self.rep = IOReport("Energy Model")
        first = self.rep.sample()
        self._hdr = [[c["group"], c["name"], c["unit"], c["format"]] for c in first]
        self.smc = SMC()
        n = self.smc.key_count()
        self.keys = [k for k in (self.smc.key_at(i) for i in range(n)) if k[0] in "PVIT" or (k[0] == "F" and k.endswith("Ac"))]
        self.every, self.n = smc_every, 0

    def header(self) -> dict:
        return {"channels": self._hdr, "smc_keys": self.keys}

    def row(self) -> dict:
        snap = self.rep.sample()
        out = {"raw": [c["value"] for c in snap]}
        if self.n % self.every == 0:
            d = {}
            for k in self.keys:
                try:
                    v = self.smc.read(k)[2]
                except Exception:
                    continue
                if isinstance(v, (int, float)):
                    d[k] = v
            out["smc"] = d
        self.n += 1
        return out


# --------------------------------------------------------------------------- probe CLI
def probe() -> dict:
    res: dict = {"ioreport": None, "smc": None, "root": False, "errors": []}
    try:
        import os
        res["root"] = os.geteuid() == 0
    except Exception:
        pass
    try:
        res["ioreport"] = IOReport().sample()
    except Exception as e:
        res["errors"].append(f"ioreport: {e}")
    try:
        res["smc"] = smc_power_keys(SMC())
    except Exception as e:
        res["errors"].append(f"smc: {e}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", metavar="OUT", help="write the probe result as JSON")
    ap.add_argument("--watch", type=float, metavar="SECONDS", help="sample Energy Model channels and SMC P keys for SECONDS and print watts")
    ap.add_argument("--all", action="store_true", help="print every IOReport channel, not only the energy-like and memory-like ones")
    a = ap.parse_args()
    r = probe()
    print(f"euid root: {r['root']}  (the probe ran {'as' if r['root'] else 'without'} root)")
    for e in r["errors"]:
        print("ERROR", e)
    chans = r["ioreport"] or []
    groups: dict[str, int] = {}
    for c in chans:
        groups[c["group"]] = groups.get(c["group"], 0) + 1
    print(f"\nIOReport: {len(chans)} channels in {len(groups)} groups")
    for g, n in sorted(groups.items()):
        print(f"  {g}: {n}")
    interesting = ("DRAM", "DCS", "AMCC", "FABRIC", "DISK", "NAND", "ANS", "SSD", "PCIE", "FAB")
    print("\nEnergy-unit channels and anything memory/fabric/disk-like:")
    for c in chans:
        txt = f"{c['group']} {c['subgroup']} {c['name']}".upper()
        energy = c["group"] == "Energy Model" or c["unit"] in _UNIT_TO_J
        if a.all or energy or any(k in txt for k in interesting):
            print(f"  [{c['group']} / {c['subgroup']}] {c['name']}  unit={c['unit'] or '-'} format={c['format']} value={c['value']}")
    smc = r["smc"]
    print(f"\nSMC power keys (names beginning P): {'unavailable' if smc is None else len(smc)}")
    for k in smc or []:
        print(f"  {k['key']} type={k['type']!r} value={k['value']} raw={k['hex']}")
    dram = [c for c in chans if _component(c["group"], c["name"]) == "dram"]
    adv = advancing(1.0)
    print(f"\nEnergy Model components that advance in 1 s without root: {adv}")
    print(f"System power (SMC PSTR): {read_system_watts()}")
    print(f"\nDRAM energy channel found: {'yes: ' + ', '.join(c['name'] for c in dram) if dram else 'no'}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)
        print(f"wrote {a.json}")
    if a.watch:
        rep = IOReport("Energy Model")  # one subscription: counters are cumulative, so two samples give the interval
        n0, t0 = rep.sample(), time.time()
        sm = SMC()
        k0 = {k["key"]: k["value"] for k in smc_power_keys(sm)}
        time.sleep(a.watch)
        n1, t1 = rep.sample(), time.time()
        dt = t1 - t0
        comp: dict[str, float] = {}
        for c0, c1 in zip(n0, n1):
            s_ = _UNIT_TO_J.get(c1["unit"])
            cm = _component(c1["group"], c1["name"])
            if cm and s_ and c0["value"] is not None and c1["value"] is not None and c0["name"] == c1["name"]:
                comp[cm] = comp.get(cm, 0.0) + (c1["value"] - c0["value"]) * s_ / dt
        print(f"\nwatts by component over {dt:.1f} s (IOReport Energy Model delta / time):")
        for k, v in sorted(comp.items()):
            print(f"  {k:<14} {v:8.3f} W")
        k1 = {k["key"]: k["value"] for k in smc_power_keys(sm)}
        print("SMC power keys now (W where the key is a power rail):")
        for k, v in sorted(k1.items()):
            print(f"  {k}: {v}   (at start {k0.get(k)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
