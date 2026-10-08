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


def test_coded_bank_record_read_is_throttled(monkeypatch):
    from cachalot.minimax import coded_bank

    bank = object.__new__(coded_bank.CodedBankReader)
    bank.records = {(3, 7): ("f", 0, "raw")}
    monkeypatch.setattr(coded_bank, "ENABLED", True)
    monkeypatch.setattr(bank, "_read_record", lambda rec, views, key=None: (time.sleep(0.002), 10_000_000)[1])
    entry = type("E", (), {"layer": 3, "expert": 7, "tensors": ()})()
    monkeypatch.setattr(reader, "READ_THROTTLE_BPS", 0.0)
    t0 = time.perf_counter()
    assert bank.read_expert_into(entry, {}) == 10_000_000
    assert time.perf_counter() - t0 < 0.008  # off: no hold
    monkeypatch.setattr(reader, "READ_THROTTLE_BPS", 1e9)
    monkeypatch.setattr(reader, "_throttle_free_at", 0.0)
    t0 = time.perf_counter()
    assert bank.read_expert_into(entry, {}) == 10_000_000
    assert time.perf_counter() - t0 >= 0.0095  # 10 MB at 1 GB/s
