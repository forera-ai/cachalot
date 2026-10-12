import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import layer_rows as lr  # noqa: E402
from cachalot.metrics import token_trace as tt  # noqa: E402


def _drained():
    # one token of two layers: layer 0 syncs 0-10, waits 10-30 on reads 5-25 (blocked, read 5-10 hidden before get);
    # layer 1 syncs 40-50 with a read 60-70 after its get window (hidden entirely)
    ev = [(0, tt.TOKEN, 28), (0, tt.SYNC_START, 0), (10, tt.SYNC_END, 0), (10, tt.GET_START, 0), (30, tt.GET_END, 0),
          (5, tt.READ_START, 0), (25, tt.READ_END, 0), (40, tt.SYNC_START, 1), (50, tt.SYNC_END, 1), (50, tt.GET_START, 1),
          (51, tt.GET_END, 1), (60, tt.READ_START, 1), (70, tt.READ_END, 1)]
    ev.sort()
    ts, code, a = (np.array([e[i] for e in ev], dtype=np.int64) for i in (0, 1, 2))
    return {"ts": ts, "code": code, "a": a, "b": np.zeros(len(ev), dtype=np.int64),
            "token_stamps": np.array([0, 100], dtype=np.int64), "token_positions": np.array([28, 29], dtype=np.int64), "dropped": 0}


def test_layer_windows_blocked_and_hidden():
    rows = lr.layer_rows(_drained())
    assert [r["layer"] for r in rows] == [0, 1]
    r0, r1 = rows
    assert r0["blocked_ms"] == 20 / 1e6 and r0["busy_ms"] == 20 / 1e6
    assert abs(r0["hidden_ms"] - 5 / 1e6) < 1e-12  # the read's first 5 before the get window
    assert r1["busy_ms"] == 10 / 1e6 and r1["hidden_ms"] == 10 / 1e6


def test_sync_read_overlap_is_the_read_time_inside_the_sync_interval():
    rows = lr.layer_rows(_drained())
    # layer 0 syncs 0-10 with a read 5-25: 5 ns inside the sync interval; layer 1 syncs 40-50 with a read 60-70: none
    assert abs(rows[0]["sync_read_ms"] - 5 / 1e6) < 1e-12 and rows[1]["sync_read_ms"] == 0.0
