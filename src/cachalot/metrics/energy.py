"""Energy meter for the server (HANDOFF 18.129, docs/POWER-ACCOUNTING-PLAN.md section 5, route 1).

Off unless `CACHALOT_ENERGY=1`. A sampler thread reads, once a second (`CACHALOT_ENERGY_INTERVAL`), the SMC's whole-system power (`PSTR`, watts, no root
needed) and the IOReport "Energy Model" cumulative counters (CPU, GPU, ANE, DRAM, DCS, AMCC). It keeps cumulative joules at each sample in a ring, so a request
asks for the energy between two clock times with no syscall and no GPU work on the decode path: the engine takes two `time.time()` readings and one lookup.

Two honest limits are part of the contract:

* The IOReport CPU, ANE, DRAM, DCS and AMCC counters advance only while a root `powermetrics` is sampling beside the server (HANDOFF 18.118). Without it
  the components are reported `available: false` (never zeros) and only the system figure is given.
* System energy is the 1 Hz integral of `PSTR`; the components are the SoC's own estimates. The component fit that adds them up is about 15 % short of the system
  figure for quantized kernels (HANDOFF 18.128), so `system` is the total and `components` are parts, never summed to it. A request shorter than a few samples
  carries sampling error of up to one interval at each end.

Every figure carries its source and tag (`measured`). Nothing here changes what the model computes.
"""
from __future__ import annotations

import bisect
import os
import threading
import time
from typing import Any, Callable

COMPONENTS = ("cpu", "gpu", "ane", "dram", "dcs", "amcc")
SYSTEM_SOURCE = "SMC PSTR (whole-system power, integrated at the sampling interval)"
COMPONENT_SOURCE = "IOReport Energy Model"
COMPONENT_NOTE = "CPU, ANE, DRAM, DCS and AMCC counters advance only while a root powermetrics samples beside the server"


class _DefaultReader:
    """The two live sources. Either can fail to open; the other still works."""

    def __init__(self) -> None:
        from cachalot.metrics import power_sources as ps

        self._ps = ps
        self.rep = self.smc = None
        self.errors: dict[str, str] = {}
        try:
            self.rep = ps.IOReport("Energy Model")
        except Exception as e:  # noqa: BLE001
            self.errors["components"] = f"IOReport: {e}"
        try:
            self.smc = ps.SMC()
        except Exception as e:  # noqa: BLE001
            self.errors["system"] = f"SMC: {e}"

    def read(self) -> tuple[float | None, dict[str, float] | None]:
        watts = None
        if self.smc is not None:
            try:
                v = self.smc.read("PSTR")[2]
                watts = None if v is None else float(v)
            except Exception:  # noqa: BLE001
                watts = None
        joules = None
        if self.rep is not None:
            try:
                joules = {c: 0.0 for c in COMPONENTS}
                for ch in self.rep.sample():
                    comp = self._ps._component(ch["group"], ch["name"])
                    scale = self._ps._UNIT_TO_J.get(ch["unit"])
                    if comp in joules and scale is not None and ch["value"] is not None:
                        joules[comp] += ch["value"] * scale
            except Exception:  # noqa: BLE001
                joules = None
        return watts, joules


class EnergyMeter:
    def __init__(self, interval: float = 1.0, history: int = 7200, reader: Any = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.interval = max(0.1, float(interval))
        self.history = history
        self.clock = clock
        self.reader = reader if reader is not None else _DefaultReader()
        self.errors: dict[str, str] = dict(getattr(self.reader, "errors", {}))
        self._lock = threading.Lock()
        self._t: list[float] = []
        self._sys: list[float] = []  # cumulative system joules at each sample
        self._comp: list[dict[str, float]] = []  # cumulative component joules (counter minus the first reading)
        self._watts: list[float | None] = []
        self._raw0: dict[str, float] | None = None
        self._last_raw: dict[str, float] | None = None
        self._moved_at: dict[str, float] = {}  # component -> time its counter last advanced
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_env(cls, env: dict | None = None) -> "EnergyMeter | None":
        env = os.environ if env is None else env
        if env.get("CACHALOT_ENERGY", "0") != "1":
            return None
        return cls(interval=float(env.get("CACHALOT_ENERGY_INTERVAL", "1.0")))

    # ----------------------------------------------------------------- sampling
    def start(self) -> "EnergyMeter":
        self.sample_once()
        self._thread = threading.Thread(target=self._run, name="energy-meter", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample_once()
            except Exception:  # noqa: BLE001  the meter never hurts the server
                pass

    def sample_once(self) -> None:
        now = self.clock()
        watts, raw = self.reader.read()
        with self._lock:
            if self._t:
                dt = now - self._t[-1]
                prev_w = self._watts[-1]
                if watts is not None and prev_w is not None:
                    inc = 0.5 * (watts + prev_w) * dt  # trapezoid
                elif watts is not None:
                    inc = watts * dt
                else:
                    inc = 0.0
                sys_j = self._sys[-1] + inc
            else:
                sys_j = 0.0
            comp = dict(self._comp[-1]) if self._comp else {c: 0.0 for c in COMPONENTS}
            if raw is not None:
                if self._raw0 is None:
                    self._raw0 = dict(raw)
                for c in COMPONENTS:
                    comp[c] = raw[c] - self._raw0[c]
                    if self._last_raw is not None and raw[c] != self._last_raw[c]:
                        self._moved_at[c] = now
                self._last_raw = dict(raw)
            self._t.append(now)
            self._sys.append(sys_j)
            self._comp.append(comp)
            self._watts.append(watts)
            if len(self._t) > self.history:
                cut = len(self._t) - self.history
                for a in (self._t, self._sys, self._comp, self._watts):
                    del a[:cut]

    # ----------------------------------------------------------------- availability
    def _fresh(self, now: float) -> bool:
        return bool(self._t) and now - self._t[-1] <= 5 * self.interval + 1.0

    def system_available(self, now: float | None = None) -> bool:
        now = self.clock() if now is None else now
        with self._lock:
            return self._fresh(now) and self._watts[-1] is not None

    def components_available(self, now: float | None = None) -> bool:
        """True when the DRAM counter advanced within the last five intervals, i.e. a root powermetrics is sampling beside us."""
        now = self.clock() if now is None else now
        with self._lock:
            moved = self._moved_at.get("dram")
            return self._fresh(now) and self._raw0 is not None and moved is not None and now - moved <= 5 * self.interval + 1.0

    # ----------------------------------------------------------------- interpolation
    def _cum_at(self, t: float) -> tuple[float, dict[str, float]] | None:
        """Cumulative (system J, component J) at clock time t: linear between samples, held at the latest watts for up to 2 intervals past the last."""
        ts = self._t
        if not ts or t < ts[0]:
            return None
        i = bisect.bisect_right(ts, t) - 1
        if i >= len(ts) - 1:
            over = t - ts[-1]
            if over > 2 * self.interval:
                return None
            w = self._watts[-1] or 0.0
            return self._sys[-1] + w * over, dict(self._comp[-1])
        f = (t - ts[i]) / (ts[i + 1] - ts[i])
        sj = self._sys[i] + f * (self._sys[i + 1] - self._sys[i])
        cj = {c: self._comp[i][c] + f * (self._comp[i + 1][c] - self._comp[i][c]) for c in COMPONENTS}
        return sj, cj

    def between(self, t0: float, t1: float) -> dict[str, Any] | None:
        """Energy between two clock times, or None when the window is outside what was sampled."""
        with self._lock:
            a, b = self._cum_at(t0), self._cum_at(t1)
            if a is None or b is None or t1 <= t0:
                return None
            fresh = self._fresh(t1)
            sys_ok = fresh and self._watts[-1] is not None
            moved = self._moved_at.get("dram")
            comp_ok = fresh and moved is not None and moved >= t0 - 5 * self.interval - 1.0
            out: dict[str, Any] = {"seconds": t1 - t0, "system_joules": (b[0] - a[0]) if sys_ok else None}
            out["components_joules"] = {c: b[1][c] - a[1][c] for c in COMPONENTS} if comp_ok else None
            return out

    # ----------------------------------------------------------------- reporting
    def stats(self) -> dict[str, Any]:
        now = self.clock()
        sys_ok, comp_ok = self.system_available(now), self.components_available(now)
        with self._lock:
            last = self._t[-1] if self._t else None
            watts = self._watts[-1] if self._t else None
            sys_j = self._sys[-1] if self._t else None
            comp = dict(self._comp[-1]) if self._comp else None
            n = len(self._t)
            first = self._t[0] if self._t else None
        return {
            "enabled": True,
            "available": sys_ok,
            "sample_interval_seconds": self.interval,
            "samples": n,
            "last_sample_age_seconds": None if last is None else round(now - last, 3),
            "since": first,
            "system": {
                "available": sys_ok,
                "watts": watts if sys_ok else None,
                "joules": sys_j if sys_ok else None,
                "source": SYSTEM_SOURCE,
                "tag": "measured",
                "reason": None if sys_ok else self.errors.get("system", "SMC PSTR not readable"),
            },
            "components": {
                "available": comp_ok,
                "joules": comp if comp_ok else None,
                "source": COMPONENT_SOURCE,
                "tag": "measured",
                "reason": None if comp_ok else self.errors.get("components", COMPONENT_NOTE),
            },
        }

    def request_report(self, t_start: float, t_end: float, decode_seconds: float, completion_tokens: int) -> dict[str, Any]:
        """The `usage.cachalot.energy` object of one request: the whole request window and the decode window (the last `decode_seconds`)."""
        whole = self.between(t_start, t_end)
        dec = self.between(max(t_start, t_end - decode_seconds), t_end) if decode_seconds > 0 else None
        if whole is None or whole["system_joules"] is None:
            return {"available": False, "reason": "no usable energy samples for this request's window", "tag": "measured"}
        rep: dict[str, Any] = {
            "available": True,
            "tag": "measured",
            "source": SYSTEM_SOURCE,
            "sample_interval_seconds": self.interval,
            "request": {
                "seconds": round(whole["seconds"], 3),
                "system_joules": round(whole["system_joules"], 3),
                "mean_system_watts": round(whole["system_joules"] / whole["seconds"], 2),
                "components_joules": None if whole["components_joules"] is None else {c: round(v, 3) for c, v in whole["components_joules"].items()},
            },
            "decode": None,
            "components_available": whole["components_joules"] is not None,
            "components_source": COMPONENT_SOURCE,
        }
        if dec is not None and dec["system_joules"] is not None and completion_tokens > 0:
            rep["decode"] = {
                "seconds": round(dec["seconds"], 3),
                "system_joules": round(dec["system_joules"], 3),
                "mean_system_watts": round(dec["system_joules"] / dec["seconds"], 2),
                "system_joules_per_token": round(dec["system_joules"] / completion_tokens, 4),
                "components_joules": None if dec["components_joules"] is None else {c: round(v, 3) for c, v in dec["components_joules"].items()},
            }
        return rep


def log_fields(report: dict | None) -> str:
    """The ` J/tok=0.55 W=71.2` part of the `[request]` line ('' when the meter is off)."""
    if report is None:
        return ""
    d = report.get("decode") if report.get("available") else None
    if not d:
        return " J/tok=n/a"
    return f" J/tok={d['system_joules_per_token']:.2f} W={d['mean_system_watts']:.1f}"
