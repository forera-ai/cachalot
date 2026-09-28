"""
Pre-allocated, wired expert slots.

Every routed expert of DeepSeek V4.1 Flash has the same six tensors with the
same byte sizes, so the resident cache can be a fixed pool of slots allocated
once at start-up. Experts are read from SSD straight into a slot's unified
memory through a writable NumPy view of the MLX buffer:

    SSD ── preadv ──▶ slot.views[name] (== slot.arrays[name] memory) ──▶ GPU

This removes per-expert mx.array allocation, the memcpy, allocator churn and,
with mx.set_wired_limit, the per-allocation Metal residency-set updates that
made concurrent promotion 25x slower.

Safety rule: a slot may only be released/overwritten after every MLX
operation that read it has been evaluated. The store enforces this by
evicting only after the per-layer synchronisation point (route eval) and by
holding bypass ("transient") slots until the caller reports evaluation.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

TENSOR_NAMES = (
    "w1.weight",
    "w1.scale",
    "w2.weight",
    "w2.scale",
    "w3.weight",
    "w3.scale",
)


@dataclass(eq=False)
class ExpertSlot:
    index: int
    arrays: dict[str, mx.array]
    views: dict[str, np.ndarray]
    typed: dict[str, tuple[mx.array, mx.array, mx.array]] = field(default_factory=dict)
    """
    Per-projection (weight, scales, biases) views of an affine bank's slot
    bytes, built once and reused for every expert that later occupies this
    slot. `.view(dtype).reshape(shape)` on a contiguous buffer is zero-copy in
    MLX, so these arrays alias the slot's memory exactly as `views` does and
    keep reflecting it after a refill -- there is nothing to invalidate. Only
    the shapes and dtypes matter, and one bank has one format for every expert.
    Without this, moe_layer_forward rebuilt nine views per expert per layer:
    4,320 MLX op constructions per decoded token on the critical path.
    """

    @property
    def size(self) -> int:
        return sum(a.nbytes for a in self.arrays.values())


def _writable_view(array: mx.array) -> np.ndarray:
    view = np.array(array, copy=False)
    view.setflags(write=True)
    return view


class ExpertSlotPool:
    def __init__(
        self,
        tensor_sizes: dict[str, int],
        n_slots: int,
        *,
        chunk: int = 64,
        verbose: bool = False,
    ) -> None:
        if n_slots <= 0:
            raise ValueError(f"n_slots must be positive, got {n_slots}")

        missing = {"w1.weight", "w2.weight", "w3.weight"} - set(tensor_sizes)
        if missing:
            raise ValueError(f"tensor_sizes missing {sorted(missing)}")

        # Slot layout follows the expert bank's tensor set (FP4: weight+scale,
        # affine: weight+scales+biases per projection), see storage.index.
        self.tensor_sizes = dict(tensor_sizes)
        self.tensor_names = tuple(self.tensor_sizes)
        self.slot_bytes = sum(self.tensor_sizes.values())
        self._slots: list[ExpertSlot] = []
        self._free: deque[int] = deque()
        self._cond = threading.Condition()
        self._parked: list[int] = []

        for start in range(0, n_slots, chunk):
            batch = []
            for index in range(start, min(start + chunk, n_slots)):
                arrays = {
                    name: mx.zeros((self.tensor_sizes[name],), dtype=mx.uint8)
                    for name in self.tensor_names
                }
                batch.append((index, arrays))
            mx.eval(*(a for _, arrays in batch for a in arrays.values()))
            for index, arrays in batch:
                views = {name: _writable_view(a) for name, a in arrays.items()}
                self._slots.append(ExpertSlot(index, arrays, views))
                self._free.append(index)
            if verbose:
                print(f"  slots {len(self._slots)}/{n_slots}", flush=True)

    @property
    def capacity(self) -> int:
        return len(self._slots)

    @property
    def parked(self) -> int:
        """Slots whose memory was given back (park); they are never handed out until unparked."""
        with self._cond:
            return len(self._parked)

    def park(self, n: int) -> int:
        """Free the memory of up to `n` free slots (HANDOFF 18.6: a long context's KV cache needs the room).

        Only free slots are parked, so nothing still reads them; their arrays, views and typed views are
        dropped and MLX can return the buffers. Returns how many were parked."""
        done = 0
        with self._cond:
            while done < n and self._free:
                slot = self._slots[self._free.pop()]
                slot.arrays, slot.views, slot.typed = {}, {}, {}
                self._parked.append(slot.index)
                done += 1
        return done

    def unpark(self, n: int) -> int:
        """Allocate memory for up to `n` parked slots again and make them free. Returns how many."""
        with self._cond:
            indices = [self._parked.pop() for _ in range(min(n, len(self._parked)))]
        if not indices:
            return 0
        fresh = []
        for index in indices:
            arrays = {name: mx.zeros((self.tensor_sizes[name],), dtype=mx.uint8) for name in self.tensor_names}
            fresh.append((index, arrays))
        mx.eval(*(a for _, arrays in fresh for a in arrays.values()))
        with self._cond:
            for index, arrays in fresh:
                slot = self._slots[index]
                slot.arrays = arrays
                slot.views = {name: _writable_view(a) for name, a in arrays.items()}
                slot.typed = {}
                self._free.append(index)
            self._cond.notify_all()
        return len(indices)

    @property
    def free_count(self) -> int:
        with self._cond:
            return len(self._free)

    def try_acquire(self) -> ExpertSlot | None:
        with self._cond:
            if not self._free:
                return None
            return self._slots[self._free.popleft()]

    def acquire(self, timeout: float | None = None) -> ExpertSlot:
        """Block until a slot is free."""
        with self._cond:
            if not self._cond.wait_for(lambda: bool(self._free), timeout=timeout):
                raise TimeoutError("no free expert slot")
            return self._slots[self._free.popleft()]

    def release(self, slot: ExpertSlot) -> None:
        with self._cond:
            self._free.append(slot.index)
            self._cond.notify()

    def release_many(self, slots) -> None:
        with self._cond:
            for slot in slots:
                self._free.append(slot.index)
            self._cond.notify_all()


class SlabSlotPool(ExpertSlotPool):
    """
    Slots as rows of a few large arrays ("slabs", HANDOFF 18.12): each slab holds `slab_slots` slots, each slot
    one contiguous record with its tensors at fixed, page-aligned offsets. A decode kernel can then reach any
    resident expert from a slot index computed on the GPU (the slab is `index // slab_slots`, the row
    `index % slab_slots`), so the host no longer has to read every layer's routing before its experts run.
    `slot.arrays` / `slot.views` are views into the slab, so every other path (reads, prefill, `_typed`) is
    unchanged.

    Memory comes back a whole slab at a time: `park_slab` / `unpark_slab` (the store picks the slab and empties
    it first, see ResidentExpertStore.set_capacity). One MLX dimension is an int32, hence (slots, record) slabs.
    """

    PAGE = 16384
    # a slab kernel binds every slab plus its slot index, input and outputs; Metal allows 31 buffers, and 28
    # slabs failed to build (HANDOFF 18.17 item 2), so a larger pool gets larger slabs instead of more of them
    MAX_SLABS = 27

    def __init__(self, tensor_sizes: dict[str, int], n_slots: int, *, slab_slots: int = 128,
                 verbose: bool = False) -> None:
        if n_slots <= 0:
            raise ValueError(f"n_slots must be positive, got {n_slots}")
        self.tensor_sizes = dict(tensor_sizes)
        self.tensor_names = tuple(self.tensor_sizes)
        self.slot_bytes = sum(self.tensor_sizes.values())
        self.slab_slots = max(int(slab_slots), -(-n_slots // self.MAX_SLABS))
        self.offsets: dict[str, int] = {}
        pos = 0
        for name in self.tensor_names:
            self.offsets[name] = pos
            pos += -(-self.tensor_sizes[name] // self.PAGE) * self.PAGE
        self.record_bytes = pos
        self._slots: list[ExpertSlot] = []
        self._free: deque[int] = deque()
        self._cond = threading.Condition()
        self._parked: list[int] = []
        self.slabs: list[mx.array | None] = []
        # stands in for a parked slab in a kernel's inputs; not one element, which MLX would bind as `constant`
        self._placeholder = mx.zeros((1, 4096), dtype=mx.uint8)
        for start in range(0, n_slots, self.slab_slots):
            n = min(self.slab_slots, n_slots - start)
            self._slots.extend(ExpertSlot(start + j, {}, {}) for j in range(n))
            self.slabs.append(None)
            self._fill_slab(len(self.slabs) - 1)
            self._free.extend(range(start, start + n))
            if verbose:
                print(f"  slots {len(self._slots)}/{n_slots}", flush=True)
        mx.eval(self._placeholder)

    def _slab_range(self, s: int) -> range:
        return range(s * self.slab_slots, min((s + 1) * self.slab_slots, len(self._slots)))

    def _fill_slab(self, s: int) -> None:
        """Allocate slab `s` and point its slots' arrays and views into it."""
        rows = self._slab_range(s)
        slab = mx.zeros((len(rows), self.record_bytes), dtype=mx.uint8)
        mx.eval(slab)
        whole = _writable_view(slab)
        views = []
        for j, index in enumerate(rows):
            slot = self._slots[index]
            slot.arrays = {n: slab[j, o:o + self.tensor_sizes[n]] for n, o in self.offsets.items()}
            slot.views = {n: whole[j, o:o + self.tensor_sizes[n]] for n, o in self.offsets.items()}
            slot.typed = {}
            views.extend(slot.arrays.values())
        mx.eval(*views)
        self.slabs[s] = slab

    def slab_of(self, index: int) -> int:
        return index // self.slab_slots

    def kernel_slabs(self) -> list[mx.array]:
        """Every slab in order, a placeholder for a parked one (no table entry points there)."""
        return [s if s is not None else self._placeholder for s in self.slabs]

    def active_slabs(self) -> list[int]:
        return [s for s, slab in enumerate(self.slabs) if slab is not None]

    def park(self, n: int) -> int:  # slot-granular parking cannot give a slab back
        return 0

    def unpark(self, n: int) -> int:
        return 0

    def park_slab(self, s: int) -> int:
        """Give slab `s`'s memory back; every one of its slots must be free. Returns how many slots it held."""
        rows = set(self._slab_range(s))
        with self._cond:
            if self.slabs[s] is None or not rows <= set(self._free):
                raise RuntimeError(f"slab {s} still has slots in use")
            self._free = deque(i for i in self._free if i not in rows)
            for i in rows:
                slot = self._slots[i]
                slot.arrays, slot.views, slot.typed = {}, {}, {}
            self._parked.extend(sorted(rows))
            self.slabs[s] = None
        return len(rows)

    def unpark_slab(self, s: int) -> int:
        """Allocate parked slab `s` again and make its slots free. Returns how many."""
        rows = list(self._slab_range(s))
        if self.slabs[s] is not None:
            return 0
        self._fill_slab(s)
        with self._cond:
            self._parked = [i for i in self._parked if i not in set(rows)]
            self._free.extend(rows)
            self._cond.notify_all()
        return len(rows)

    def try_acquire_outside(self, slabs: set[int]) -> ExpertSlot | None:
        """A free slot that is not in any of `slabs` (migration target), or None."""
        with self._cond:
            for i in self._free:
                if self.slab_of(i) not in slabs:
                    self._free.remove(i)
                    return self._slots[i]
        return None
