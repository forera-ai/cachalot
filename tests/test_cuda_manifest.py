"""CUDA lane (docs/CUDA-RESEARCH-PLAN.md section 7): manifest, redaction and doctor parsers. No GPU, no torch needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "cuda"))

import doctor  # noqa: E402
import manifest as mf  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "cuda"


def fx(name):
    return (FIX / name).read_text()


def test_parse_v100_csv_and_missing_fields():
    gpu = doctor.parse_nvidia_smi_csv(fx("smi_v100.csv"))[0]
    assert gpu["name"] == "Tesla V100-PCIE-32GB"
    assert gpu["compute_cap"] == "7.0"
    assert gpu["memory_total_mib"] == 32768 and gpu["memory_free_mib"] == 32500
    assert gpu["pcie_link_gen_current"] == 3 and gpu["pcie_link_width_current"] == 16
    assert gpu["power_limit_w"] == 250


def test_parse_5090_csv_na_becomes_none_not_zero():
    gpu = doctor.parse_nvidia_smi_csv(fx("smi_rtx5090.csv"))[0]
    assert gpu["compute_cap"] == "12.0" and gpu["pcie_link_gen_current"] == 5
    assert gpu["ecc_mode_current"] is None


def test_parse_banner_nvcc_cpu_mem():
    assert doctor.parse_nvidia_smi_header(fx("smi_banner.txt")) == {
        "driver_version": "575.57.08", "driver_cuda_version": "12.9"}
    assert doctor.parse_nvcc(fx("nvcc.txt")) == {"release": "12.8", "build": "12.8.93"}
    cpu = doctor.parse_lscpu(fx("lscpu.txt"))
    assert cpu["model"] == "AMD EPYC 7B13" and cpu["logical_cpus"] == 32 and cpu["numa_nodes"] == 1
    assert doctor.parse_meminfo(fx("meminfo.txt"))["ram_total_gib"] == pytest.approx(251.65, abs=0.01)


def test_garbage_input_yields_empty_not_exception():
    assert doctor.parse_nvidia_smi_csv("not csv at all") == []
    assert doctor.parse_nvcc("") == {} and doctor.parse_lscpu("") == {} and doctor.parse_meminfo("") == {}


def test_env_allowlist_drops_everything_else_and_secret_names():
    env = {"CUDA_VISIBLE_DEVICES": "0", "CACHALOT_X": "1", "OMP_NUM_THREADS": "8", "HOME": "/home/u",
           "AWS_SECRET_ACCESS_KEY": "x", "CACHALOT_API_KEY": "k", "CUDA_PROMO_CODE": "p", "PATH": "/bin"}
    assert mf.filter_env(env) == {"CUDA_VISIBLE_DEVICES": "0", "CACHALOT_X": "1", "OMP_NUM_THREADS": "8"}


@pytest.mark.parametrize("text,kind", [
    ("10.1.2.3", "ipv4"), ("fe80:0:0:0:1:2:3:4", "ipv6"), ("aa:bb:cc:dd:ee:ff", "mac"),
    ("123e4567-e89b-12d3-a456-426614174000", "uuid"), ("GPU-12345678-aaaa", "gpu_uuid"),
    ("me@example.com", "email"), ("/Users/someone/x", "home_path"), ("rift_ABCDEFGH12345678", "api_token"),
    ("ssh-ed25519 AAAAC3Nza", "private_key"),
])
def test_forbidden_shapes_are_found_and_never_echoed(text, kind):
    hits = mf.find_forbidden({"a": {"b": [text]}})
    assert any(k == kind for _, k in hits)
    with pytest.raises(mf.RedactionError) as err:
        mf.assert_publishable({"a": text})
    assert text not in str(err.value)


def test_sha256_digest_and_plain_numbers_are_allowed():
    assert mf.find_forbidden({"id": "a" * 64, "v": "575.57.08", "cc": "12.0", "gen": "driver 12.8.93"}) == []


def test_seal_is_content_addressed_and_tamper_evident():
    base = {"schema": "cuda-manifest", "schema_version": 1, "platform": "cuda", "kind": "machine"}
    a = mf.seal(base)
    assert a["manifest_id"] == mf.seal(dict(base))["manifest_id"]
    a["kind"] = "run"
    assert mf.content_id(a) != a["manifest_id"]


def test_doctor_collect_on_a_machine_without_cuda_is_valid_and_clean():
    doc = doctor.collect(note="unit test")
    assert mf.validate_machine(doc) == []
    assert doc["gpu"]["gpu_identity"].startswith("not collected")
    ok, why = doctor.cuda_ready(doc)
    if not ok:                                   # the Mac case: informative reason, no exception
        assert "CUDA" in why or "torch" in why
    mf.assert_publishable(doc)


def test_row_rules():
    r = mf.row("r1", "arm", "time_ms", 1.5, "ms", "measured", source="cuda_events", n=30, p50=1.5)
    assert mf.validate_row(r) == []
    u = mf.row("r1", "arm", "time_ms", None, "ms", "unavailable", source="x", note="no counters")
    assert "value" not in u and mf.validate_row(u) == []
    assert mf.validate_row(mf.row("r1", "a", "f", 0, "ms", "unavailable", source="x"))      # needs a reason
    assert mf.validate_row({**r, "tag": "estimated"})                                       # not in the vocabulary
