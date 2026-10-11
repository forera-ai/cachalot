"""Apple Silicon power sources, read-only and without root: IOReport "Energy Model" cumulative energy counters and the SMC's whole-system power.

Moved here from `benchmarks/power_sources.py` (0.63.0) so the server's energy meter (`cachalot.metrics.energy`) and the benchmarks share one reader;
the benchmarks module imports these names and keeps the probe, the full-channel sampler and the command line (docs/POWER-ACCOUNTING-PLAN.md).
IOReport's DRAM, DCS, AMCC, CPU and ANE counters advance only while a root `powermetrics` is sampling beside the reader (HANDOFF 18.118); the SMC
key `PSTR` (whole-system watts) needs no root.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import re
import struct

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
