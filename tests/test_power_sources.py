import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import power_sources as ps  # noqa: E402


def test_component_uses_only_top_level_channels():
    c = ps._component
    assert c("Energy Model", "DIE_0_CPU Energy") == "cpu"
    assert c("Energy Model", "DIE_1_PACC0_CPU3_SRAM") is None  # a part of the cluster, would double count
    assert c("Energy Model", "PCPUDTL00") is None
    assert c("Energy Model", "GPU Energy") == "gpu"
    assert c("Energy Model", "GPU0_0") is None
    assert c("Energy Model", "DRAM0_1") == "dram"
    assert c("Energy Model", "DCS0_0") == "dcs" and c("Energy Model", "AMCC0_1") == "amcc"
    assert c("Other Group", "DRAM0_0") is None


def test_decode_float_and_fixed_point():
    assert ps._decode("flt ", struct.pack("<f", 23.5)) == 23.5
    assert ps._decode("ui8 ", b"\x07") == 7
    assert ps._decode("sp78", b"\x01\x80") == 1.5
    assert ps._decode("zzzz", b"\x00") is None


def test_fourcc_round_trip():
    assert ps._fourcc_str(ps._fourcc("PSTR")) == "PSTR"


def test_unit_scale():
    assert ps._UNIT_TO_J["mJ"] == 1e-3 and ps._UNIT_TO_J["nJ"] == 1e-9
