"""DI-1 split analysis (Mac only; record docs/DI1-SPLIT-RECORD.md): the pure functions on planted marks. The runtime is never loaded."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import prefill_half_split as hs  # noqa: E402


def marks(per_layer, moe_frac=0.8, t0=100.0):
    out, t = [], t0
    for k, d in enumerate(per_layer):
        start = t + d * (1 - moe_frac)
        out += [(start, f"moe_start L{k}"), (t + d, f"moe_end L{k}")]
        t += d
    return out, t0


def test_layer_times_are_the_intervals_between_moe_ends():
    m, t0 = marks([1.0] * 20 + [0.5] * 20)
    lt = hs.layer_times(m, t0)
    assert len(lt["layers"]) == 40 and lt["layers"][0]["total_s"] == pytest.approx(1.0) and lt["layers"][25]["total_s"] == pytest.approx(0.5)
    assert lt["layers"][3]["moe_s"] == pytest.approx(0.8) and lt["last_moe_end"] == pytest.approx(100.0 + 30.0)


def test_the_halves_split_at_layer_20_and_the_decoder_share_is_the_second_half():
    m, t0 = marks([1.0] * 20 + [0.5] * 20)
    h = hs.halves(hs.layer_times(m, t0)["layers"])
    assert h["complete"] and h["encoder_s"] == pytest.approx(20.0) and h["decoder_s"] == pytest.approx(10.0)
    assert h["decoder_share"] == pytest.approx(10.0 / 30.0) and h["encoder_moe_s"] == pytest.approx(16.0) and h["decoder_other_s"] == pytest.approx(2.0)
    m2, t2 = marks([1.0] * 40)
    assert hs.halves(hs.layer_times(m2, t2)["layers"])["decoder_share"] == pytest.approx(0.5)


def test_a_missing_layer_makes_the_split_incomplete_not_a_number():
    m, t0 = marks([1.0] * 40)
    m = [x for x in m if x[1] not in ("moe_end L7", "moe_start L7")]
    assert hs.halves(hs.layer_times(m, t0)["layers"]) == {"complete": False}


def test_reads_are_counted_before_and_after_the_end_of_layer_19():
    m, t0 = marks([1.0] * 40)
    cut = 100.0 + 20.0
    reads = [(110.0, 110.5), (119.9, 120.1), (120.2, 120.5), (130.0, 131.0)]
    r = hs.reads_per_half(reads, m)
    assert r == {"complete": True, "encoder_reads": 2, "decoder_reads": 2} and cut > 119.9
    assert hs.reads_per_half(reads, [])["complete"] is False


def test_the_sync_profile_flag_stops_the_run(monkeypatch, capsys):
    monkeypatch.setenv("CACHALOT_PROFILE_SYNC", "1")
    assert hs.main(["128", "1"]) == 2 and "STOP" in capsys.readouterr().err
