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

_UNIT_TO_J = {"J": 1.0, "mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9, "pJ": 1e-12}


# --------------------------------------------------------------------------- CoreFoundation helpers
def _cf():
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    cf.CFDictionaryGetValue.restype = ctypes.c_void_p
    cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cf.CFArrayGetCount.restype = ctypes.c_long
    cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
    cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
    cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFRetain.restype = ctypes.c_void_p
    cf.CFRetain.argtypes = [ctypes.c_void_p]
    return cf


def _cfstr(cf, s: str):
    return cf.CFStringCreateWithCString(None, s.encode(), 0x08000100)  # kCFStringEncodingUTF8


def _pystr(cf, ref) -> str:
    if not ref:
        return ""
    buf = ctypes.create_string_buffer(256)
    return buf.value.decode(errors="replace") if cf.CFStringGetCString(ref, buf, 256, 0x08000100) else ""


# --------------------------------------------------------------------------- IOReport
class IOReport:
    """Subscribes to every IOReport channel (or one group) and reads cumulative raw values."""

    def __init__(self, group: str | None = None):
        self.cf = _cf()
        self.lib = ctypes.CDLL("/usr/lib/libIOReport.dylib")  # resolved from the dyld shared cache
        L = self.lib
        L.IOReportCopyAllChannels.restype = ctypes.c_void_p
        L.IOReportCopyAllChannels.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        L.IOReportCopyChannelsInGroup.restype = ctypes.c_void_p
        L.IOReportCopyChannelsInGroup.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        L.IOReportCreateSubscription.restype = ctypes.c_void_p
        L.IOReportCreateSubscription.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.c_void_p]
        L.IOReportCreateSamples.restype = ctypes.c_void_p
        L.IOReportCreateSamples.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        for name in ("IOReportChannelGetGroup", "IOReportChannelGetSubGroup", "IOReportChannelGetChannelName", "IOReportChannelGetUnitLabel"):
            fn = getattr(L, name)
            fn.restype = ctypes.c_void_p
            fn.argtypes = [ctypes.c_void_p]
        L.IOReportSimpleGetIntegerValue.restype = ctypes.c_int64
        L.IOReportSimpleGetIntegerValue.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        L.IOReportChannelGetFormat.restype = ctypes.c_uint8
        L.IOReportChannelGetFormat.argtypes = [ctypes.c_void_p]
        chans = L.IOReportCopyChannelsInGroup(_cfstr(self.cf, group), None, 0, 0, 0) if group else L.IOReportCopyAllChannels(0, 0)
        if not chans:
            raise RuntimeError("IOReport returned no channels")
        self.subbed = ctypes.c_void_p()
        self.sub = L.IOReportCreateSubscription(None, chans, ctypes.byref(self.subbed), 0, None)
        if not self.sub:
            raise RuntimeError("IOReportCreateSubscription failed")
        self._key_channels = _cfstr(self.cf, "IOReportChannels")

    def sample(self) -> list[dict]:
        """One snapshot: a list of {group, subgroup, name, unit, value, format}; value is the channel's cumulative integer."""
        cf, L = self.cf, self.lib
        samples = L.IOReportCreateSamples(self.sub, self.subbed, None)
        if not samples:
            raise RuntimeError("IOReportCreateSamples failed")
        try:
            arr = cf.CFDictionaryGetValue(samples, self._key_channels)
            out = []
            for i in range(cf.CFArrayGetCount(arr)):
                ch = cf.CFArrayGetValueAtIndex(arr, i)
                fmt = L.IOReportChannelGetFormat(ch)  # 1 = simple integer value
                out.append({
                    "group": _pystr(cf, L.IOReportChannelGetGroup(ch)),
                    "subgroup": _pystr(cf, L.IOReportChannelGetSubGroup(ch)),
                    "name": _pystr(cf, L.IOReportChannelGetChannelName(ch)),
                    "unit": _pystr(cf, L.IOReportChannelGetUnitLabel(ch)).strip(),
                    "format": int(fmt),
                    "value": int(L.IOReportSimpleGetIntegerValue(ch, 0)) if fmt == 1 else None,
                })
            return out
        finally:
            cf.CFRelease(samples)


# --------------------------------------------------------------------------- SMC
class _KeyInfo(ctypes.Structure):
    _fields_ = [("dataSize", ctypes.c_uint32), ("dataType", ctypes.c_uint32), ("dataAttributes", ctypes.c_uint8)]


class _Vers(ctypes.Structure):
    _fields_ = [("major", ctypes.c_char), ("minor", ctypes.c_char), ("build", ctypes.c_char), ("reserved", ctypes.c_char), ("release", ctypes.c_uint16)]


class _PLimit(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint16), ("length", ctypes.c_uint16), ("cpuPLimit", ctypes.c_uint32), ("gpuPLimit", ctypes.c_uint32), ("memPLimit", ctypes.c_uint32)]


class _KeyData(ctypes.Structure):
    _fields_ = [("key", ctypes.c_uint32), ("vers", _Vers), ("pLimitData", _PLimit), ("keyInfo", _KeyInfo),
                ("result", ctypes.c_uint8), ("status", ctypes.c_uint8), ("data8", ctypes.c_uint8), ("data32", ctypes.c_uint32),
                ("bytes", ctypes.c_uint8 * 32)]


def _fourcc(s: str) -> int:
    return struct.unpack(">I", s.encode()[:4].ljust(4))[0]


def _fourcc_str(v: int) -> str:
    return struct.pack(">I", v).decode(errors="replace")


class SMC:
    def __init__(self):
        io = ctypes.CDLL(ctypes.util.find_library("IOKit"))
        self.io = io
        io.IOServiceMatching.restype = ctypes.c_void_p
        io.IOServiceMatching.argtypes = [ctypes.c_char_p]
        io.IOServiceGetMatchingService.restype = ctypes.c_uint32
        io.IOServiceGetMatchingService.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        io.IOServiceOpen.restype = ctypes.c_int
        io.IOServiceOpen.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        io.IOConnectCallStructMethod.restype = ctypes.c_int
        io.IOConnectCallStructMethod.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        libsys = ctypes.CDLL(None)
        task = ctypes.c_uint32.in_dll(libsys, "mach_task_self_").value
        svc = io.IOServiceGetMatchingService(0, io.IOServiceMatching(b"AppleSMC"))
        if not svc:
            raise RuntimeError("AppleSMC service not found")
        self.conn = ctypes.c_uint32()
        rc = io.IOServiceOpen(svc, task, 0, ctypes.byref(self.conn))
        if rc != 0:
            raise RuntimeError(f"IOServiceOpen(AppleSMC) failed: {rc:#x}")

    def _call(self, inp: _KeyData) -> _KeyData:
        out = _KeyData()
        size = ctypes.c_size_t(ctypes.sizeof(_KeyData))
        rc = self.io.IOConnectCallStructMethod(self.conn, 2, ctypes.byref(inp), ctypes.sizeof(_KeyData), ctypes.byref(out), ctypes.byref(size))
        if rc != 0:
            raise RuntimeError(f"SMC call failed: {rc:#x}")
        return out

    def key_info(self, key: str) -> _KeyInfo:
        inp = _KeyData()
        inp.key, inp.data8 = _fourcc(key), 9
        out = self._call(inp)
        if out.result != 0:
            raise KeyError(key)
        return out.keyInfo

    def read(self, key: str):
        info = self.key_info(key)
        inp = _KeyData()
        inp.key, inp.data8, inp.keyInfo.dataSize = _fourcc(key), 5, info.dataSize
        out = self._call(inp)
        if out.result != 0:
            raise KeyError(key)
        raw = bytes(out.bytes[: info.dataSize])
        typ = _fourcc_str(info.dataType)
        return typ, raw, _decode(typ, raw)

    def key_count(self) -> int:
        return int.from_bytes(self.read("#KEY")[1], "big")

    def key_at(self, index: int) -> str:
        inp = _KeyData()
        inp.data8, inp.data32 = 8, index
        return _fourcc_str(self._call(inp).key)


def _decode(typ: str, raw: bytes):
    try:
        if typ == "flt " and len(raw) == 4:
            return struct.unpack("<f", raw)[0]  # Apple Silicon SMC floats are little endian
        if typ == "ui8 ":
            return raw[0]
        if typ in ("ui16", "ui32"):
            return int.from_bytes(raw, "big")
        if typ.startswith("sp") and len(raw) == 2:  # fixed point, e.g. sp78
            frac = int(typ[3], 16) if typ[3] in "0123456789abcdef" else 8
            return int.from_bytes(raw, "big", signed=True) / (1 << frac)
    except Exception:
        pass
    return None


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
def _component(group: str, name: str) -> str | None:
    """Map an Energy Model channel to ONE component, using only top-level channels so nothing is counted twice (the group also holds
    per-core, per-cluster and SRAM sub-channels that are parts of "CPU Energy"). Seen on this Mac (M3 Ultra, two dies, 2026-10-10):
    `DIE_n_CPU Energy` (mJ), `GPU Energy` (nJ), `ANE0_n`, `DRAM0_n`, `DCS0_n`, `AMCC0_n`, display and media blocks, `PCIe Port n Energy`.
    Whether DCS and AMCC are inside or beside DRAM is not known, so they are separate components, never summed into dram."""
    if group != "Energy Model":
        return None
    if re.fullmatch(r"DIE_\d+_CPU Energy", name):
        return "cpu"
    if name == "GPU Energy":
        return "gpu"
    if re.fullmatch(r"ANE\d+_\d+", name):
        return "ane"
    if re.fullmatch(r"DRAM\d+_\d+", name):
        return "dram"
    if re.fullmatch(r"DCS\d+_\d+", name):
        return "dcs"
    if re.fullmatch(r"AMCC\d+_\d+", name):
        return "amcc"
    if re.fullmatch(r"(DISP|DISPEXT|ISP|AVE|MSR)\d+_\d+", name):
        return "display_media"
    if re.fullmatch(r"(PCIe Port \d+|apciec\d+) Energy", name):
        return "pcie"
    return None


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
