"""The server's energy meter (0.63.0, HANDOFF 18.129): a fake clock and a scripted reader, no hardware."""
from fastapi.testclient import TestClient

from cachalot.metrics.energy import COMPONENTS, EnergyMeter, log_fields
from cachalot.server.app import ServerConfig, create_app
from cachalot.server.engine import Engine
from fake_model import FakeEncoding, FakeModel, ScriptedRuntime


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class Reader:
    """watts constant; the DRAM counter advances `dram_rate` J per second unless frozen."""

    def __init__(self, clock, watts=60.0, dram_rate=10.0, frozen=False):
        self.clock, self.watts, self.dram_rate, self.frozen = clock, watts, dram_rate, frozen
        self.errors = {}

    def read(self):
        t = self.clock.t - 1000.0
        raw = {c: 0.0 for c in COMPONENTS}
        if not self.frozen:
            raw["dram"] = self.dram_rate * t
            raw["gpu"] = 20.0 * t
        else:
            raw["gpu"] = 20.0 * t  # the GPU counter moves without root, the DRAM counter does not
        return self.watts, raw


def run(meter, clock, seconds, step=1.0):
    for _ in range(int(seconds / step)):
        clock.t += step
        meter.sample_once()


def test_off_unless_the_knob_is_set():
    assert EnergyMeter.from_env({}) is None
    assert EnergyMeter.from_env({"CACHALOT_ENERGY": "0"}) is None


def test_between_integrates_system_power_and_interpolates_components():
    clock = Clock()
    m = EnergyMeter(interval=1.0, reader=Reader(clock), clock=clock)
    m.sample_once()
    run(m, clock, 10)
    r = m.between(1002.5, 1007.5)
    assert abs(r["system_joules"] - 60.0 * 5.0) < 1e-6
    assert abs(r["components_joules"]["dram"] - 50.0) < 1e-6 and abs(r["components_joules"]["gpu"] - 100.0) < 1e-6
    assert abs(r["seconds"] - 5.0) < 1e-9


def test_a_window_outside_the_samples_is_none_not_zero():
    clock = Clock()
    m = EnergyMeter(interval=1.0, reader=Reader(clock), clock=clock)
    m.sample_once()
    run(m, clock, 5)
    assert m.between(990.0, 1003.0) is None  # before the first sample
    assert m.between(1002.0, 1020.0) is None  # far past the last sample
    assert m.between(1003.0, 1003.0) is None


def test_frozen_dram_counter_means_components_unavailable_but_system_still_reported():
    clock = Clock()
    m = EnergyMeter(interval=1.0, reader=Reader(clock, frozen=True), clock=clock)
    m.sample_once()
    run(m, clock, 12)
    assert m.system_available() and not m.components_available()
    r = m.between(1003.0, 1010.0)
    assert r["components_joules"] is None and abs(r["system_joules"] - 420.0) < 1e-6
    st = m.stats()
    assert st["system"]["available"] and st["system"]["tag"] == "measured" and st["system"]["source"].startswith("SMC")
    assert st["components"]["available"] is False and st["components"]["joules"] is None and "root powermetrics" in st["components"]["reason"]


def test_components_become_available_when_the_counter_moves():
    clock = Clock()
    m = EnergyMeter(interval=1.0, reader=Reader(clock), clock=clock)
    m.sample_once()
    run(m, clock, 6)
    assert m.components_available()
    assert m.stats()["components"]["joules"]["dram"] > 0


def test_stale_samples_make_everything_unavailable():
    clock = Clock()
    m = EnergyMeter(interval=1.0, reader=Reader(clock), clock=clock)
    m.sample_once()
    run(m, clock, 3)
    clock.t += 60.0  # the sampler thread died
    assert not m.system_available() and not m.components_available()
    assert m.stats()["available"] is False


def test_request_report_decode_window_and_log_fields():
    clock = Clock()
    m = EnergyMeter(interval=1.0, reader=Reader(clock), clock=clock)
    m.sample_once()
    run(m, clock, 20)
    rep = m.request_report(1004.0, 1016.0, decode_seconds=8.0, completion_tokens=40)
    assert rep["available"] and rep["tag"] == "measured"
    assert abs(rep["request"]["system_joules"] - 720.0) < 1e-6
    d = rep["decode"]
    assert abs(d["system_joules"] - 480.0) < 1e-6 and abs(d["system_joules_per_token"] - 12.0) < 1e-6 and abs(d["mean_system_watts"] - 60.0) < 1e-6
    assert log_fields(rep) == " J/tok=12.00 W=60.0"
    assert log_fields(None) == ""
    gone = m.request_report(900.0, 910.0, 5.0, 10)
    assert gone["available"] is False and log_fields(gone) == " J/tok=n/a"


def test_history_is_bounded():
    clock = Clock()
    m = EnergyMeter(interval=1.0, history=50, reader=Reader(clock), clock=clock)
    m.sample_once()
    run(m, clock, 200)
    assert m.stats()["samples"] == 50


def test_usage_and_stats_carry_the_energy_only_when_the_meter_is_on():
    rt = ScriptedRuntime(reply="Hello, whale!")
    engine = Engine(FakeModel(rt), model_id="test-model", encoding=FakeEncoding())
    client = TestClient(create_app(engine, ServerConfig()))
    body = {"messages": [{"role": "user", "content": "hi"}]}
    off = client.post("/v1/chat/completions", json=body).json()
    assert "energy" not in off["usage"]["cachalot"] and "energy" not in client.get("/v1/stats").json()

    import time

    class Live:
        errors = {}

        def read(self):
            return 55.0, {c: 0.0 for c in COMPONENTS}

    meter = EnergyMeter(interval=0.05, reader=Live(), clock=time.time).start()
    try:
        engine.energy = meter
        time.sleep(0.2)
        on = client.post("/v1/chat/completions", json=body).json()
        e = on["usage"]["cachalot"]["energy"]
        assert e["tag"] == "measured" and "available" in e
        stats = client.get("/v1/stats").json()["energy"]
        assert stats["enabled"] is True and stats["components"]["available"] is False and stats["system"]["available"] is True
    finally:
        meter.stop()


def test_a_window_ending_after_the_last_sample_extends_components_at_their_last_rate():
    clock = Clock()
    m = EnergyMeter(interval=1.0, reader=Reader(clock), clock=clock)
    m.sample_once()
    run(m, clock, 10)  # last sample at 1010.0; dram +10 J/s, gpu +20 J/s, system 60 W
    r = m.between(1006.0, 1010.5)
    assert abs(r["system_joules"] - 60.0 * 4.5) < 1e-6
    assert abs(r["components_joules"]["dram"] - 45.0) < 1e-6 and abs(r["components_joules"]["gpu"] - 90.0) < 1e-6
