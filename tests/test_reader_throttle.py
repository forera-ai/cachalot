"""The emulated-bandwidth throttle (CACHALOT_READ_THROTTLE_GBPS) used by storage sweeps."""
import time

import cachalot.storage.reader as reader


def test_throttle_holds_a_drive_read_for_its_bytes(monkeypatch):
    monkeypatch.setattr(reader, "READ_THROTTLE_BPS", 1e9)
    monkeypatch.setattr(reader, "_throttle_free_at", 0.0)
    started = time.perf_counter() - 0.002  # a 2 ms read: served by the drive
    reader._throttle(10_000_000, started)  # 10 MB at 1 GB/s = 10 ms on the pipe
    assert time.perf_counter() - started >= 0.0095


def test_throttle_queues_concurrent_reads_on_one_pipe(monkeypatch):
    monkeypatch.setattr(reader, "READ_THROTTLE_BPS", 1e9)
    monkeypatch.setattr(reader, "_throttle_free_at", 0.0)
    started = time.perf_counter() - 0.002
    reader._throttle(5_000_000, started)
    reader._throttle(5_000_000, started)  # same start: waits behind the first
    assert time.perf_counter() - started >= 0.0095


def test_throttle_lets_page_cache_hits_pass(monkeypatch):
    monkeypatch.setattr(reader, "READ_THROTTLE_BPS", 1e6)
    monkeypatch.setattr(reader, "_throttle_free_at", 0.0)
    started = time.perf_counter()
    reader._throttle(10_000_000, started)  # would be 10 s if charged
    assert time.perf_counter() - started < 0.5
    assert reader._throttle_free_at == 0.0
