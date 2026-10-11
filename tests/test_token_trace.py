"""The per-token trace (HANDOFF 18.132): columns, thread merge, and the exposed/hidden read split."""
import threading
import time

from cachalot.metrics import token_trace as tt


def _burst(trace, layer, seconds):
    trace.emit(tt.READ_START, layer, 1)
    time.sleep(seconds)
    trace.emit(tt.READ_END, layer, 100)


def test_summary_splits_exposed_and_hidden_reads():
    trace = tt.TokenTrace()
    trace.begin_token(10)
    trace.emit(tt.SYNC_START, 3)
    trace.emit(tt.SYNC_END, 3)
    worker = threading.Thread(target=_burst, args=(trace, 3, 0.03))
    worker.start()
    trace.emit(tt.GET_START, 3, 1)
    time.sleep(0.01)  # the main thread waits only for the first 10 ms of the 30 ms read
    trace.emit(tt.GET_END, 3, 1)
    worker.join()
    trace.begin_token(11)
    rows = tt.summarize(trace.drain())
    assert len(rows) == 1
    row = rows[0]
    assert row["position"] == 10 and row["reads"] == 1 and row["bytes"] == 100
    assert 25 < row["read_busy_ms"] < 60
    assert 8 < row["blocked_ms"] < 30
    assert row["hidden_read_ms"] > row["read_busy_ms"] - row["blocked_ms"] - 1.0
    assert row["hidden_read_ms"] < row["read_busy_ms"]


def test_inactive_tokens_stamp_but_do_not_record():
    trace = tt.TokenTrace()
    trace.active = False
    trace.begin_token(1)
    trace.emit(tt.SYNC_START, 0)
    trace.active = True
    trace.begin_token(2)
    trace.emit(tt.SYNC_START, 0)
    trace.emit(tt.SYNC_END, 0)
    trace.begin_token(3)
    drained = trace.drain()
    assert list(drained["token_positions"]) == [1, 2, 3]
    rows = tt.summarize(drained)
    assert [r["position"] for r in rows] == [2]
    assert rows[0]["events"] == 3


def test_capacity_drops_are_counted_not_raised():
    trace = tt.TokenTrace(capacity=4)
    for i in range(10):
        trace.emit(tt.SYNC_START, i)
    drained = trace.drain()
    assert len(drained["ts"]) == 4 and drained["dropped"] == 6


def test_alternate_records_only_the_chosen_parity_and_minus_one_records_nothing():
    trace = tt.TokenTrace()
    trace.alternate = 0
    for pos in range(6):
        trace.begin_token(pos)
        trace.emit(tt.SYNC_START, 0)
        trace.emit(tt.SYNC_END, 0)
    drained = trace.drain()
    assert list(drained["token_positions"]) == [0, 1, 2, 3, 4, 5]
    rows = tt.summarize(drained)
    assert [r["position"] for r in rows] == [0, 2, 4]
    trace.alternate = -1
    for pos in range(4):
        trace.begin_token(pos)
        trace.emit(tt.SYNC_START, 0)
    assert len(trace.drain()["ts"]) == 0
