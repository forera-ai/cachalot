"""B1 instrument: the pure helpers, the smaps parser and every cell on the CPU rehearsal (never a CUDA number)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))

import backing_probe as bp  # noqa: E402
import manifest as mf  # noqa: E402

torch = pytest.importorskip("torch")

SMAPS = """\
7f0000000000-7f0000200000 rw-p 00000000 00:00 0
Size:               2048 kB
Rss:                2048 kB
AnonHugePages:         0 kB
THPeligible:           0
7f0000200000-7f0000c00000 rw-p 00000000 00:00 0
Size:              10240 kB
Rss:               10240 kB
AnonHugePages:      8192 kB
THPeligible:           1
7f0000c00000-7f0000e00000 r--p 00000000 00:00 0 [vvar]
Size:               2048 kB
AnonHugePages:         0 kB
"""


def test_parse_smaps_finds_the_mapping_that_contains_the_address():
    e = bp.parse_smaps(SMAPS, 0x7F0000300000)
    assert e == {"start": 0x7F0000200000, "end": 0x7F0000C00000, "size_kb": 10240, "rss_kb": 10240,
                 "anon_huge_kb": 8192, "thp_eligible": 1}
    assert bp.parse_smaps(SMAPS, 0x7F0000000010)["anon_huge_kb"] == 0
    assert bp.parse_smaps(SMAPS, 0x1000) is None
    assert bp.parse_smaps("", 1) is None


def test_huge_fraction_is_capped_and_none_when_unreadable():
    assert bp.huge_fraction({"anon_huge_kb": 8192}, 10 * 1024 * 1024) == pytest.approx(0.8)
    assert bp.huge_fraction({"anon_huge_kb": 99999}, 1024 * 1024) == 1.0
    assert bp.huge_fraction(None, 1) is None and bp.huge_fraction({"size_kb": 1}, 1) is None


def test_slow_flags_by_threshold_and_by_ratio():
    ms = [0.182, 0.183, 0.34, 0.182, 0.31]
    assert bp.slow_flags(ms, 0.28) == [False, False, True, False, True]
    assert bp.slow_flags(ms) == [False, False, True, False, True]            # below 0.8 x the fastest buffer's rate
    assert bp.slow_flags([]) == [] and bp.share([True, False, False, False]) == 0.25
    assert bp.third_mode_share([0.18, 0.22, 0.25, 0.34]) == 0.5


def test_order_gap_lag_agreement_and_runs():
    flags = [False] * 6 + [True] * 2
    assert bp.order_gap(flags) == pytest.approx(6.5 - 2.5)
    assert bp.order_gap([False, False]) is None
    assert bp.lag1_agreement([True, True, False, False]) == pytest.approx(2 / 3)
    assert bp.lag1_agreement([True]) is None
    assert bp.expected_agreement(0.2) == pytest.approx(0.68)
    assert bp.run_lengths([True, True, False, True, False, False, True, True, True]) == [2, 1, 3]
    assert bp.run_lengths([False, True, False], value=False) == [1, 1]


def test_find_cudart_lib_prefers_the_pip_runtime_wheel(tmp_path):
    lib = tmp_path / "nvidia" / "cuda_runtime" / "lib"
    lib.mkdir(parents=True)
    (lib / "libcudart.so.12").write_text("")
    assert bp.find_cudart_lib([str(tmp_path / "missing"), str(tmp_path)]).endswith("libcudart.so.12")
    assert bp.find_cudart_lib([str(tmp_path / "missing")]) is None


@pytest.fixture(scope="module")
def bench():
    return bp.BackingBench(rehearsal_cpu=True)


def validate(rows):
    assert rows and all(mf.validate_row(r) == [] for r in rows), [mf.validate_row(r) for r in rows if mf.validate_row(r)]


def test_alloc_order_cell_and_rows(bench):
    res = bench.alloc_order(k=6, nbytes=1 << 16)
    assert len(res["ms"]) == 6 and res["roundtrip_ok"]
    rows = bp.rows_alloc_order("r", "a", res)
    validate(rows)
    assert {"slow_share", "copy_ms_min", "copy_ms_max", "third_mode_share"} <= {r["field"] for r in rows}


@pytest.mark.parametrize("advice", ["thp", "nothp", "default"])
def test_mmap_cell_runs_and_reports_what_it_could_not_do(bench, advice):
    res = bench.mmap_cell(advice, k=3, nbytes=1 << 16)
    assert len(res["ms"]) == 3 and res["roundtrip_ok"]
    rows = bp.rows_mmap_cell("r", "m", res)
    validate(rows)
    fields = {r["field"] for r in rows}
    assert "huge_fraction_median" in fields
    if advice != "default" and not hasattr(__import__("mmap"), "MADV_HUGEPAGE"):
        assert "advice_or_register_error" in fields                        # the Mac cannot advise: said, not hidden


def test_reroll_cell(bench):
    res = bench.reroll("default", cycles=4, nbytes=1 << 16)
    assert len(res["ms"]) == 4
    validate(bp.rows_mmap_cell("r", "x", res))


def test_slab_chunks_cell_and_rows(bench):
    res = bench.slab_chunks(slab_bytes=1 << 20, chunk_bytes=1 << 16)
    assert len(res["chunk_ms"]) == 16 and res["roundtrip_ok"]
    rows = bp.rows_slab("r", "s", res)
    validate(rows)
    assert {"slow_chunk_share", "whole_slab_gbps", "slow_run_mean_length"} <= {r["field"] for r in rows}


def test_curve_cell_and_rows(bench):
    res = bench.curve(sizes=(1 << 16, 1 << 17), ks=(3, 2))
    assert set(res) == {str(1 << 16), str(1 << 17)}
    # the byte check must hold with K > 1 distinct buffers (it failed on the first GPU run when made after the timing loop)
    assert all(v["roundtrip_ok"] for v in res.values())
    rows = bp.rows_curve("r", "c", res)
    validate(rows)
    assert {"pinned_h2d_gbps_fastest", "pageable_h2d_gbps_median", "pinned_d2h_gbps_fastest_buffer"} <= {r["field"] for r in rows}


def test_a_deadline_stops_the_cells(bench):
    import suite
    assert bench.alloc_order(k=5, nbytes=1 << 16, deadline=suite.Deadline(-1))["ms"] == []
    assert bench.curve(sizes=(1 << 16,), ks=(2,), deadline=suite.Deadline(-1)) == {}


def test_plan_and_dry_run(capsys):
    assert bp.main(["plan"]) == 0
    assert bp.main(["run", "--cells", "b1"]) == 0
    assert "b1" in capsys.readouterr().out


def test_curve_records_an_allocation_failure_and_keeps_going(bench, monkeypatch):
    real = bench.pinned
    calls = {"n": 0}

    def flaky(nbytes, seed):
        calls["n"] += 1
        if nbytes == 1 << 17:
            raise RuntimeError("cudaHostAlloc failed: out of memory")
        return real(nbytes, seed)

    monkeypatch.setattr(bench, "pinned", flaky)
    res = bench.curve(sizes=(1 << 16, 1 << 17, 1 << 18), ks=(2, 2, 2))
    assert "error" in res[str(1 << 17)] and "pinned_ms" in res[str(1 << 16)] and "pinned_ms" in res[str(1 << 18)]
    rows = bp.rows_curve("r", "c", res)
    validate(rows)
    assert any(r["tag"] == "unavailable" and r["arm"].endswith(str(1 << 17)) for r in rows)


# ---------------------------------------------------------------------------------------------- B2 (private mapping, populate, prefix)

SMAPS_SHM = SMAPS.replace("THPeligible:           1", "ShmemPmdMapped:     4096 kB\nFilePmdMapped:         0 kB\nTHPeligible:           1")


def test_parse_smaps_reads_the_shmem_and_file_pmd_fields():
    e = bp.parse_smaps(SMAPS_SHM, 0x7F0000300000)
    assert e["shmem_pmd_kb"] == 4096 and e["file_pmd_kb"] == 0
    assert "shmem_pmd_kb" not in bp.parse_smaps(SMAPS, 0x7F0000300000)            # absent stays absent, B1's equality test holds
    assert bp.shmem_fraction(e, 10 * 1024 * 1024) == pytest.approx(0.4)
    assert bp.shmem_fraction(None, 1) is None and bp.shmem_fraction({"size_kb": 1}, 1) is None


def test_host_thp_reads_policy_and_counters_or_none():
    h = bp.host_thp()
    assert set(bp.THP_FILES) <= set(h) and set(bp.MEMINFO_KEYS) <= set(h)


def _fake_smaps(monkeypatch, make=2048, register=2048, copy=2048):             # kB; a 64 KiB buffer registers 2 MiB
    seq = {"n": 0}

    def entry(self):
        seq["n"] += 1
        kb = (make, register, copy)[(seq["n"] - 1) % 3]
        return {"start": 1, "end": 2, "size_kb": 4096, "rss_kb": 4096, "anon_huge_kb": kb, "thp_eligible": 1}

    monkeypatch.setattr(bp.MmapBuffer, "smaps_entry", entry)


def test_private_cell_checks_every_buffer_and_aligns_the_smaps_points(bench, monkeypatch):
    _fake_smaps(monkeypatch, make=2048, register=1024, copy=1024)
    res = bench.private_cell("thp", "private", False, k=3, nbytes=1 << 16)
    assert len(res["ms"]) == 3 and len(res["smaps"]) == 3 and res["roundtrip_ok"]
    assert all(set(p) == {"make", "register", "copy"} for p in res["smaps"])
    rows = bp.rows_private_cell("r", "p", res)
    validate(rows)
    by = {r["field"]: r["value"] for r in rows if r["tag"] != "unavailable"}
    assert by["huge_verified_count"] == 3 and by["huge_verified_count_post_register"] == 0
    assert by["huge_fraction_max_drop_after_register"] == pytest.approx(0.5)
    assert {"huge_verified_slow_count", "huge_verified_ms_max", "huge_verified_ms_min"} <= set(by)


def test_private_cell_small_page_share_and_unreadable_smaps(bench, monkeypatch):
    _fake_smaps(monkeypatch, make=0, register=0, copy=0)
    rows = bp.rows_private_cell("r", "p", bench.private_cell("nothp", "private", False, k=3, nbytes=1 << 16))
    validate(rows)
    by = {r["field"]: r["value"] for r in rows if r["tag"] != "unavailable"}
    assert by["huge_verified_count"] == 0 and "small_page_slow_share" in by and "huge_verified_ms_max" not in by
    monkeypatch.undo()
    monkeypatch.setattr(bp.MmapBuffer, "smaps_entry", lambda self: None)
    rows = bp.rows_private_cell("r", "p", bench.private_cell("default", "private", False, k=2, nbytes=1 << 16))
    validate(rows)
    assert any(r["field"] == "huge_fraction_median" and r["tag"] == "unavailable" for r in rows)


def test_the_private_mapping_asks_for_map_private_and_the_shared_one_does_not(bench, monkeypatch):
    seen = []
    real = bp.mmap.mmap

    def spy(fileno, length, *a, **kw):
        seen.append(kw.get("flags"))
        return real(fileno, length, *a, **kw)

    monkeypatch.setattr(bp.mmap, "mmap", spy)
    bench.private_cell("default", "private", False, k=1, nbytes=1 << 16)
    bench.private_cell("default", "shared", False, k=1, nbytes=1 << 16)
    assert seen[0] is not None and seen[0] & bp.mmap.MAP_PRIVATE and seen[1] is None


def test_populate_unavailable_is_said_and_is_not_a_failure(bench, monkeypatch):
    monkeypatch.setattr(bp, "MADV_POPULATE_WRITE", 9999)                           # an option no kernel has
    res = bench.private_cell("thp", "private", True, k=3, nbytes=1 << 16)
    assert res["unavailable"] and res["ms"] == []
    rows = bp.rows_private_cell("r", "p", res)
    validate(rows)
    assert any(r["field"] == "populate_unavailable" for r in rows)


def test_populate_available_path_times_every_buffer(bench, monkeypatch):
    monkeypatch.setattr(bp, "MADV_POPULATE_WRITE", bp.mmap.MADV_NORMAL)            # an option every platform accepts
    res = bench.private_cell("thp", "private", True, k=3, nbytes=1 << 16)
    assert res["unavailable"] is None and len(res["ms"]) == 3 and res["roundtrip_ok"]


def test_prefix_cell_times_every_buffer_and_checks_each_copy(bench):
    res = bench.prefix_cell(small_sizes=(1 << 14, 1 << 15), k_small=2, k=4, nbytes=1 << 16)
    assert len(res["ms"]) == 4 and res["roundtrip_ok"] and set(res["prefix_ms"]) == {str(1 << 14), str(1 << 15)}
    rows = bp.rows_prefix("r", "pf", res)
    validate(rows)
    assert any(r["field"] == "prefix_h2d_gbps_median" for r in rows) and any(r["field"] == "slow_share" for r in rows)


@pytest.mark.parametrize("group", list(bp.GROUPS))
def test_every_group_runs_through_the_real_run_group_with_every_gate_true(bench, group):
    """B1's lesson: the rehearsal must run the code path the GPU session runs, gates included, with K above 1."""
    import suite
    gates, failures, raw = {"roundtrip": True, "sanity": True}, [], {}
    rows = bp._run_group(bench, group, "r", suite.Deadline(120), gates, failures, raw, small=True)
    assert rows and gates == {"roundtrip": True, "sanity": True} and failures == []
    validate(rows)


@pytest.mark.parametrize("group", ["h1", "h6", "h7", "b1", "curve"])
def test_a_failing_byte_check_trips_the_roundtrip_gate(bench, monkeypatch, group):
    import suite
    monkeypatch.setattr(bench, "roundtrip_ok", lambda *a, **k: False)
    gates, failures, raw = {"roundtrip": True, "sanity": True}, [], {}
    bp._run_group(bench, group, "r", suite.Deadline(120), gates, failures, raw, small=True)
    assert gates["roundtrip"] is False


def test_the_b2_groups_are_in_the_plan_and_the_dry_run(capsys):
    assert set(bp.B2_GROUPS) <= set(bp.GROUPS)
    assert bp.main(["plan"]) == 0 and bp.main(["run", "--cells", "h5"]) == 0
    assert "h5" in capsys.readouterr().out
