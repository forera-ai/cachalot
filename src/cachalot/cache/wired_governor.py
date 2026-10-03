"""A wired-memory governor for the DeepSeek expert cache (HANDOFF 18.52, 18.53).

Measured on the 96 GiB Mac Studio with the 2-bit bank: an all-resident token costs 70-80 ms while the whole
system's wired memory is up to about 74 GiB, 91 ms at 74.6 GiB and 158-182 ms at 75.1-75.3 GiB, with the same
residents and the same hits; the budget that crossed the edge (52 GiB) decoded at 4.8 tok/s against 8.3 at 48. The
edge does not move with the runtime's own settings (a wired limit of 84 GiB changed nothing) and is the same figure
the MiniMax notes found (~74.3 GiB, 18.17). Anything else that wires memory (other apps, a second model) moves a
fixed budget across it, so the budget alone cannot be right: this governor reads the system's wired memory
(`vm.page_wired_count`, a few microseconds) and gives expert slots back while it is above a ceiling, and takes them
back, slowly, once it has been comfortably below it for a while.

Pure logic here; the runtime calls `WiredGovernor.target` between tokens and applies the answer with
`ResidentExpertStore.set_capacity`. Output is unaffected: only which experts stay resident changes.
"""

from __future__ import annotations

import math
import os
import time

GIB = 1024**3

# The ceiling as a share of RAM: 0.76 x 96 GiB = 73.0 GiB. The floor is flat to about 72 GiB (70-77 ms) and creeps
# from there (85 ms with the system at 74.2 GiB, 91 at 74.6, 158+ at 75.1); a ceiling of 73.9 held the system at
# 74.1-74.2 GiB and gave a 92 ms floor on the mixed set, so this one sits lower. CACHALOT_WIRED_CEILING_GIB
# overrides it in GiB; zero or negative turns the governor off.
CEILING_FRACTION = 0.76
HYSTERESIS_GIB = float(os.environ.get("CACHALOT_WIRED_HYSTERESIS_GIB", "1.5"))
GROW_QUIET_S = float(os.environ.get("CACHALOT_WIRED_GROW_QUIET_S", "60"))
CHECK_EVERY_TOKENS = int(os.environ.get("CACHALOT_WIRED_CHECK_EVERY", "16"))
MIN_GROW_SLOTS = 16

_SYSCTL = None


def _sysctl(name: str) -> int | None:
    global _SYSCTL
    if _SYSCTL is None:
        import ctypes
        import ctypes.util

        try:
            _SYSCTL = ctypes, ctypes.CDLL(ctypes.util.find_library("c")).sysctlbyname
        except (OSError, AttributeError):
            _SYSCTL = False
    if not _SYSCTL:
        return None
    ctypes, fn = _SYSCTL
    value, size = ctypes.c_uint64(0), ctypes.c_size_t(8)
    if fn(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        return None
    return value.value if size.value == 8 else value.value & 0xFFFFFFFF


def system_wired_bytes() -> int:
    """Bytes of wired memory in the whole system (kernel, every process, the GPU's resident sets); -1 when unknown."""
    pages, page = _sysctl("vm.page_wired_count"), _sysctl("hw.pagesize")
    if pages is None or page is None:
        return -1
    return pages * page


def default_ceiling_bytes() -> int:
    """The wired ceiling in bytes (CACHALOT_WIRED_CEILING_GIB, else 77 % of RAM); -1 when off or unknown."""
    env = os.environ.get("CACHALOT_WIRED_CEILING_GIB")
    if env is not None:
        value = float(env)
        return int(value * GIB) if value > 0 else -1
    total = _sysctl("hw.memsize")
    return int(total * CEILING_FRACTION) if total else -1


class WiredGovernor:
    """Decides the expert capacity from the system's wired memory.

    Shrinks at once by the excess over the ceiling (whole slots, rounded up). Grows only after `grow_quiet`
    seconds without a shrink, and only into the room that stays `hysteresis` under the ceiling, so a fit never
    gives back and takes back the same slots in turn."""

    def __init__(self, ceiling: int, slot_bytes: int, hysteresis: int | None = None, grow_quiet: float | None = None,
                 clock=time.monotonic):
        self.ceiling = ceiling
        self.slot_bytes = max(1, slot_bytes)
        self.hysteresis = int((HYSTERESIS_GIB if hysteresis is None else hysteresis / GIB) * GIB)
        self.grow_quiet = GROW_QUIET_S if grow_quiet is None else grow_quiet
        self.clock = clock
        self.last_shrink = float("-inf")
        self.shrinks = 0
        self.grows = 0

    @property
    def enabled(self) -> bool:
        return self.ceiling > 0

    def target(self, capacity: int, full: int, wired: int) -> int:
        """The capacity to move to, given the current one, the built one and the system's wired bytes."""
        if not self.enabled or wired < 0:
            return capacity
        excess = wired - self.ceiling
        if excess > 0:
            self.last_shrink = self.clock()
            self.shrinks += 1
            return max(1, capacity - math.ceil(excess / self.slot_bytes))
        room = self.ceiling - self.hysteresis - wired
        if capacity < full and room >= MIN_GROW_SLOTS * self.slot_bytes and self.clock() - self.last_shrink >= self.grow_quiet:
            self.grows += 1
            return min(full, capacity + int(room // self.slot_bytes))
        return capacity
