"""DI-1 server A/B driver (docs/DI1-SERVER-AB-RECORD.md): the replayed bodies, the watchdog's stop rules, a full arm against a stub server,
and the scorer's quantities."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import di1_server_ab_driver as d  # noqa: E402


def _dump(tmp_path, n=5):
    lines = []
    for i in range(n):
        lines.append(json.dumps({"t": i, "body": {"model": "m", "messages": [{"role": "user", "content": f"q{i}"}], "stream": True,
                                                  "stream_options": {"include_usage": True}, "temperature": 0.7, "reasoning_effort": "none"}}))
        lines.append(json.dumps({"t": i + 0.5, "reply": {"prompt_tokens": [1, 2]}}))
    p = tmp_path / "dump.jsonl"
    p.write_text("\n".join(lines))
    return p


def test_bodies_takes_the_first_four_request_bodies_and_skips_reply_records(tmp_path):
    b = d.bodies(_dump(tmp_path))
    assert [x["messages"][0]["content"] for x in b] == ["q0", "q1", "q2", "q3"]


def test_prepare_keeps_messages_and_effort_and_asks_one_greedy_unstreamed_token():
    body = {"model": "m", "messages": [1], "tools": [2], "stream": True, "stream_options": {}, "temperature": 0.7, "max_tokens": 9,
            "reasoning_effort": "none", "n": 1}
    p = d.prepare(body)
    assert p == {"model": "m", "messages": [1], "tools": [2], "reasoning_effort": "none", "n": 1, "max_tokens": 1, "temperature": 0,
                 "stream": False}
    assert body["stream"] is True                               # the dump's body is not modified


def test_swap_parse():
    assert d.swap_used_mb("total = 2048.00M  used = 1204.50M  free = 843.50M  (encrypted)") == 1204.5
    assert d.swap_used_mb("garbage") == 0.0


def _sampler(seq):
    it = iter(seq)
    return lambda: next(it)


def test_watchdog_trips_on_two_critical_samples_not_one():
    w = d.Watchdog(None, sampler=_sampler([(1, 100.0), (4, 100.0), (1, 100.0), (4, 100.0), (4, 100.0)]))
    w.check()                                                   # 4: one critical sample
    w.check()                                                   # 1: the streak resets
    w.check()                                                   # 4
    assert w.tripped is None
    w.check()                                                   # 4: two in a row
    assert "critical" in w.tripped and w.summary()["pressure_max"] == 4


def test_watchdog_trips_on_swap_growth():
    w = d.Watchdog(None, sampler=_sampler([(1, 100.0), (2, 600.0), (2, 700.0)]))
    w.check()
    assert w.tripped is None
    w.check()
    assert w.tripped.startswith("swap grew 600")


class _Stub(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.seen.append(body)
        i = len(_Stub.seen)
        if i == 3 and getattr(_Stub, "fail", False):
            self.send_response(500)
            self.end_headers()
            return
        out = {"choices": [{"message": {"content": "x", "tool_calls": None}}],
               "usage": {"prompt_tokens": 100 * i, "completion_tokens": 1,
                         "cachalot": {"reused_prefix_tokens": 10 * (i - 1), "prefill_seconds": float(i)}}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def _serve():
    s = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}/v1"


def test_an_arm_against_a_stub_server_records_every_request(tmp_path):
    _Stub.seen, _Stub.fail = [], False
    s, url = _serve()
    out = tmp_path / "arm-1-off.json"
    try:
        rc = d.run(_dump(tmp_path), out, 1, 0, None, url=url, sampler=lambda: (1, 0.0))
    finally:
        s.shutdown()
    rec = json.loads(out.read_text())
    assert rc == 0 and len(rec["requests"]) == 4 and all(b["max_tokens"] == 1 and b["stream"] is False for b in _Stub.seen)
    assert [q["prefilled"] for q in rec["requests"]] == [100, 190, 280, 370]
    assert [q["prefill_s"] for q in rec["requests"]] == [1.0, 2.0, 3.0, 4.0]
    assert rec["watchdog"]["tripped"] is None and "finished" in rec


def test_a_request_error_stops_the_arm_with_its_rows_kept(tmp_path):
    _Stub.seen, _Stub.fail = [], True
    s, url = _serve()
    out = tmp_path / "arm-1-off.json"
    try:
        rc = d.run(_dump(tmp_path), out, 1, 0, None, url=url, sampler=lambda: (1, 0.0))
    finally:
        s.shutdown()
        _Stub.fail = False
    rec = json.loads(out.read_text())
    assert rc == 2 and len(rec["requests"]) == 2 and "500" in rec["error"]


def _arm(tmp_path, n, knob, pre, tok="x"):
    rec = {"arm": n, "knob": knob, "requests": [{"request": r, "prefill_s": s, "prefilled": 1000 * r, "token": tok} for r, s in
                                                 enumerate(pre, start=1)]}
    (tmp_path / f"arm-{n}-{'on' if knob else 'off'}.json").write_text(json.dumps(rec))


def test_score_means_ratios_spreads_and_equalities(tmp_path):
    _arm(tmp_path, 1, 0, [160, 2, 110, 60])
    _arm(tmp_path, 2, 1, [100, 2, 66, 45])
    _arm(tmp_path, 3, 1, [104, 2, 64, 45], tok="y")
    _arm(tmp_path, 4, 0, [168, 2, 114, 66])
    s = d.score(tmp_path)
    assert s["mean_prefill_s"]["off"] == [164.0, 2.0, 112.0, 63.0]
    assert s["ratio"]["request 1"] == round(102 / 164, 3)
    assert s["ratio"]["requests 1+3+4"] == round((102 + 65 + 45) / (164 + 112 + 63), 3)
    assert s["request 1 spread off"] == round(8 / 160, 3) and s["request 1 spread on"] == 0.04
    assert s["prefilled equal across arms"] == [True] * 4
    assert s["token equal across arms"] == [False] * 4
