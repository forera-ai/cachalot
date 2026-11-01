from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock

from cachalot.cache.resident import ResidentExpert
from cachalot.cache.resident_store import ResidentExpertStore
from cachalot.storage.index import ExpertEntry


class ResidentExpertPrefetcher:
    def __init__(
        self,
        store: ResidentExpertStore,
        workers: int = 2,
        depth: int | None = None,
    ) -> None:
        if workers <= 0:
            raise ValueError(
                f"workers must be > 0, got {workers}"
            )

        self.store = store
        self.workers = int(workers)

        # Jobs kept submitted ahead of consumption. Measured on USB 3.2:
        # depth == workers gave 66% SSD occupancy on cold prefill, depth 48
        # dropped it to 38%, so keep the queue shallow until the lookahead
        # is re-measured on faster storage.
        env_depth = os.environ.get("CACHALOT_PREFETCH_DEPTH")
        if depth:
            self.depth = int(depth)
        elif env_depth:
            self.depth = int(env_depth)
        else:
            self.depth = self.workers

        self._pool = ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="resident-prefetch",
        )

        self._pending: dict[
            tuple[int, int],
            Future[ResidentExpert],
        ] = {}

        self._lock = RLock()

        # Finished loads whose record the store had already let go of when they were collected (see get).
        self.stale = 0

    def prefetch(
        self,
        entry: ExpertEntry,
    ) -> bool:
        """
        Submit a background load unless the expert is resident or already
        pending. Returns True when a load is outstanding for this expert
        (newly submitted or already pending), False when it is resident.
        """
        key = (entry.layer, entry.expert)

        # Avoid touching the store's hit/miss statistics merely
        # to check residency.
        with self.store._lock:
            if key in self.store._items:
                return False

        with self._lock:
            future = self._pending.get(key)

            if future is not None:
                return True

            self._pending[key] = self._pool.submit(
                self.store.get_prefill,
                entry,
            )

        return True

    def get(
        self,
        entry: ExpertEntry,
    ) -> ResidentExpert:
        key = (entry.layer, entry.expert)

        with self._lock:
            future = self._pending.pop(
                key,
                None,
            )

        if future is not None:
            resident = future.result()
            # A load can outlive its record: DI-1's last encoder layer speculates layer 20, which runs only in finish_decoder, after the next
            # chunk's layer 0 has released that transient slot. The slot then holds another expert, or the wired governor parked it and its
            # arrays are gone (KeyError 'w1.weight' in the 1017 live session, HANDOFF section 18.244). Such a record is read again.
            if self.store.holds(resident):
                return resident
            self.stale += 1

        return self.store.get_prefill(entry)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def discard_pending(self, layer: int, keep: set[tuple[int, int]]) -> int:
        """
        Forget pending futures of `layer` whose key is not in `keep`
        (speculative loads the routing did not need). Running loads finish
        into transient slots that the store releases at the next layer.
        """
        with self._lock:
            victims = [k for k in self._pending if k[0] == layer and k not in keep]
            futures = [self._pending.pop(k) for k in victims]
        for future in futures:
            # queued loads are dropped before they start; running ones finish
            future.cancel()
        return len(victims)

    def pending_keys(self) -> list[tuple[int, int]]:
        """Keys with a submitted load, in submission order."""
        with self._lock:
            return list(self._pending)

    def close(self) -> None:
        with self._lock:
            futures = list(self._pending.values())
            self._pending.clear()

        for future in futures:
            future.cancel()

        self._pool.shutdown(
            wait=True,
            cancel_futures=True,
        )

    def __enter__(
        self,
    ) -> ResidentExpertPrefetcher:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
