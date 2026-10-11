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


def test_unsampled_tokens_stamp_but_do_not_record():
    trace = tt.TokenTrace(every=2)
    trace.begin_token(1)  # odd position: not sampled
    trace.emit(tt.SYNC_START, 0)
    trace.begin_token(2)  # sampled
    trace.emit(tt.SYNC_START, 0)
    trace.emit(tt.SYNC_END, 0)
    trace.begin_token(3)
    drained = trace.drain()
    assert list(drained["token_positions"]) == [1, 2, 3]
    rows = tt.summarize(drained)
    assert [r["position"] for r in rows] == [2]
    assert rows[0]["events"] == 3


def test_nothing_is_recorded_outside_a_sampled_token():
    trace = tt.TokenTrace(every=4)
    trace.emit(tt.READ_START, 1, 1)  # a prefill read before any decode token
    trace.begin_token(5)  # not sampled
    trace.emit(tt.READ_START, 1, 1)
    assert len(trace.drain()["ts"]) == 0


def test_capacity_drops_are_counted_not_raised():
    trace = tt.TokenTrace(capacity=4)
    trace.active = True
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


def test_meter_samples_every_fourth_token_and_reports_running_means():
    meter = tt.TokenTraceMeter(every=4)
    trace = meter.trace
    for request in range(2):
        meter.begin_request()
        for pos in range(12):
            trace.begin_token(100 * request + pos)
            if trace.active:
                trace.emit(tt.SYNC_START, 0)
                trace.emit(tt.SYNC_END, 0)
        meter.collect()
    stats = meter.stats()
    assert stats["every"] == 4 and stats["requests"] == 2
    # per request: positions 0, 4, 8 are sampled and the last stamp has no successor, so tokens 0, 4 and 8 give rows; 8 -> 11 ends at stamp 11
    assert stats["sampled_tokens"] == 6
    assert stats["mean_per_sampled_token"]["events"] == 3
    assert stats["exposed_read_share"] is None  # no reads recorded
    assert stats["dropped_events"] == 0


def test_begin_request_discards_what_a_cancelled_request_left():
    meter = tt.TokenTraceMeter(every=1)
    meter.trace.begin_token(1)
    meter.trace.emit(tt.SYNC_START, 0)
    meter.begin_request()
    assert len(meter.trace.drain()["ts"]) == 0
