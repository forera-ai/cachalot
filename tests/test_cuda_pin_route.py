"""Session K (Mac only; record docs/CUDA-PIN-ROUTE-RECORD.md): the committed classification rule, the DC-3 route cells and the DC-5 guard,
every one through the code path the GPU session runs on the CPU rehearsal (never a CUDA number)."""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))
sys.path.insert(0, str(ROOT / "tests"))

import backing_probe as bp  # noqa: E402
import manifest as mf  # noqa: E402
import overlap_arena as oa  # noqa: E402
import pin_route as pr  # noqa: E402
import suite  # noqa: E402

from test_cuda_overlap_arena import _gates, synth  # noqa: E402

torch = pytest.importorskip("torch")


def validate(rows):
    assert rows and all(mf.validate_row(r) == [] for r in rows), [mf.validate_row(r) for r in rows if mf.validate_row(r)]


# ------------------------------------------------------------------------------------------------ DC-6: the committed rule

def test_the_rule_labels_buffers_against_the_64_mib_plateau():
    plateau_ms = [64 * 2**20 / 52e9 * 1e3, 64 * 2**20 / 40e9 * 1e3, 64 * 2**20 / 51e9 * 1e3]      # fastest 52 GB/s
    fast_ms, slow_ms = bp.RECORD_BYTES / 54.6e9 * 1e3, bp.RECORD_BYTES / 27.7e9 * 1e3
    c = pr.classify_buffers([fast_ms] * 11 + [slow_ms], plateau_ms)
    assert c["plateau_gbps"] == pytest.approx(52.0, rel=1e-6)
    assert c["slow"] == [False] * 11 + [True] and c["slow_count"] == 1 and c["instance_class"] == "fast"
    assert pr.classify_buffers([slow_ms] * 12, plateau_ms)["instance_class"] == "slow"
    assert pr.classify_buffers([slow_ms] * 11 + [fast_ms], plateau_ms)["instance_class"] == "slow"
    assert pr.classify_buffers([fast_ms] * 6 + [slow_ms] * 6, plateau_ms)["instance_class"] == "mixed"
    # exactly at the threshold is not slow (rate below 0.75 x plateau is)
    edge = bp.RECORD_BYTES / (0.75 * 52e9) * 1e3
    assert pr.classify_buffers([edge], plateau_ms)["slow"] == [False]
    assert pr.classify_buffers([edge * 1.001], plateau_ms)["slow"] == [True]


def test_no_plateau_or_no_buffers_is_unavailable_not_a_class():
    assert pr.classify_buffers([0.2] * 12, [])["instance_class"] == "unavailable"
    assert pr.classify_buffers([], [1.0])["instance_class"] == "unavailable"
    assert pr.instance_class([]) == "unavailable"


def test_the_class_scales_with_the_buffer_count_for_the_mac_rehearsal():
    assert pr.instance_class([False] * 4) == "fast" and pr.instance_class([True] * 4) == "slow"
    assert pr.instance_class([True, False, True, False]) == "mixed"


def test_label_agreement_between_the_two_rules():
    assert pr.label_agreement([True, False, False], [True, False, True]) == pytest.approx(2 / 3)
    assert pr.label_agreement([], []) is None and pr.label_agreement([True], [True, False]) is None


def test_the_stream_rule_needs_more_than_15_percent_and_more_than_3_microseconds():
    # 256 MiB read plus write: 1,517 GB/s is 354 microseconds a copy
    assert pr.stream_changed(1517.0, 1517.0) is False
    assert pr.stream_changed(1517.0, 1517.0 * 0.90) is False          # 10 %: below the relative threshold
    assert pr.stream_changed(1517.0, 1517.0 * 0.80) is True           # 25 % slower, 88 microseconds
    assert pr.stream_changed(100000.0, 100000.0 * 0.80, nbytes=1 << 10) is False   # a 20 % change of a 20 ns copy is under 3 microseconds


def test_the_link_sampler_lines_are_summarised_and_garbage_is_skipped():
    s = pr.parse_link_lines("pcie.link.gen.current, pcie.link.width.current\n5, 16\n5, 16\n4, 16\nnot a line\n\n")
    assert s["samples"] == 3 and s["gens"] == {"4": 1, "5": 2} and s["widths"] == {"16": 3} and s["min_gen"] == 4 and s["max_gen"] == 5
    assert pr.parse_link_lines("")["samples"] == 0 and pr.parse_link_lines("")["max_gen"] is None
    rows = pr.rows_link("r", "a", s)
    validate(rows)
    assert next(r for r in rows if r["field"] == "link_samples_below_max_gen")["value"] == 1


def test_the_link_sampler_reports_unavailable_when_nvidia_smi_is_missing(monkeypatch):
    monkeypatch.setattr(pr.shutil, "which", lambda name: None)
    sm = pr.LinkSampler()
    sm.start()
    out = sm.stop()
    assert out["samples"] == 0 and "nvidia-smi" in out["unavailable"]
    rows = pr.rows_link("r", "a", out)
    validate(rows)
    assert rows[0]["tag"] == "unavailable" and rows[0].get("value") is None


class _FakeProc:
    def __init__(self, text):
        self.text = text

    def terminate(self):
        pass

    def communicate(self, timeout=None):
        return self.text, ""


def test_the_link_sampler_collects_what_the_process_printed(monkeypatch):
    monkeypatch.setattr(pr.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(pr.subprocess, "Popen", lambda *a, **k: _FakeProc("5, 16\n5, 16\n"))
    sm = pr.LinkSampler()
    sm.start()
    out = sm.stop()
    assert out["samples"] == 2 and out["max_gen"] == 5 and "unavailable" not in out


# ------------------------------------------------------------------------------------------------ DC-3: routes

@pytest.fixture(scope="module")
def bench():
    return pr.PinRouteBench(rehearsal_cpu=True)


class _NoSampler:
    def start(self):
        pass

    def stop(self):
        return {"samples": 2, "gens": {"5": 2}, "widths": {"16": 2}, "min_gen": 5, "max_gen": 5, "interval_ms": 100}


@pytest.mark.parametrize("group", list(pr.GROUPS))
def test_every_k_group_runs_through_the_real_run_group_with_every_gate_true(bench, group):
    gates, failures, raw = {"roundtrip": True, "sanity": True}, [], {}
    rows = pr.run_group(bench, group, "r", suite.Deadline(120), gates, failures, raw, small=True, sampler_factory=_NoSampler)
    assert rows and gates == {"roundtrip": True, "sanity": True} and failures == []
    validate(rows)


def test_the_classification_cell_reports_both_rules_the_stream_and_the_link(bench):
    gates, failures, raw = {"roundtrip": True, "sanity": True}, [], {}
    rows = pr.run_group(bench, "k0", "r", suite.Deadline(120), gates, failures, raw, small=True, sampler_factory=_NoSampler)
    fields = {r["field"] for r in rows}
    assert {"plateau_gbps", "slow_count_relative", "slow_count_absolute", "label_agreement", "stream_d2d_gbps", "stream_d2d_ms",
            "link_gen_min", "link_gen_max", "slow_share"} <= fields
    assert "instance_class_code" in fields
    assert "classify_start" in raw and "classify_end" not in raw
    gates, failures, raw = {"roundtrip": True, "sanity": True}, [], {}
    pr.run_group(bench, "k5", "r", suite.Deadline(120), gates, failures, raw, small=True, sampler_factory=_NoSampler)
    assert "classify_end" in raw


def test_the_ordinary_route_registers_what_torch_allocated_and_carries_smaps(bench):
    res = bench.ordinary_cell(0, k=3, nbytes=1 << 16)
    assert len(res["ms"]) == 3 and res["roundtrip_ok"] and res["mapping"] == "ordinary" and res["reg_flags"] == 0
    rows = pr.rows_route("r", "p", res)
    validate(rows)
    assert "h2d_gbps_median" in {r["field"] for r in rows}


def _stub_bench(calls):
    return types.SimpleNamespace(torch=torch, rehearsal=False, register=lambda a, n, f: calls.append(("register", f)),
                                 unregister=lambda a: calls.append(("unregister", None)))


def test_the_registration_flag_reaches_cudahostregister_on_both_registered_routes():
    seed = torch.randint(0, 256, (1 << 21,), dtype=torch.uint8)
    for flags in (0, 1):
        calls: list = []
        mb = bp.MmapBuffer(_stub_bench(calls), 1 << 16, "default", seed, mapping="private", reg_flags=flags)
        assert ("register", flags) in calls and mb.is_registered
        mb.release()
        assert ("unregister", None) in calls
        calls = []
        ob = pr.OrdinaryBuffer(_stub_bench(calls), 1 << 16, seed, flags)
        assert ("register", flags) in calls and ob.is_registered
        ob.release()
    calls = []
    bp.MmapBuffer(_stub_bench(calls), 1 << 16, "default", seed, mapping="private")
    assert ("register", 1) in calls                                            # B1 and B2's default stays the portable flag


def test_a_registration_failure_is_unavailable_never_a_number():
    seed = torch.randint(0, 256, (1 << 21,), dtype=torch.uint8)

    def boom(a, n, f):
        raise RuntimeError("cudaHostRegister returned 1")

    stub = types.SimpleNamespace(torch=torch, rehearsal=False, register=boom, unregister=lambda a: None)
    ob = pr.OrdinaryBuffer(stub, 1 << 16, seed, 0)
    assert ob.register_error and not ob.is_registered
    mb = bp.MmapBuffer(stub, 1 << 16, "default", seed, mapping="private", reg_flags=0)
    assert mb.register_error and not mb.is_registered


@pytest.mark.parametrize("group", ["k0", "k1", "k3", "k4"])
def test_a_failing_byte_check_trips_the_roundtrip_gate(bench, monkeypatch, group):
    monkeypatch.setattr(bench, "roundtrip_ok", lambda *a, **k: False)
    gates, failures, raw = {"roundtrip": True, "sanity": True}, [], {}
    pr.run_group(bench, group, "r", suite.Deadline(120), gates, failures, raw, small=True, sampler_factory=_NoSampler)
    assert gates["roundtrip"] is False


def test_a_cell_that_raises_is_a_failed_cell_not_a_crash(bench, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no CUDA")

    monkeypatch.setattr(bench, "alloc_order", boom)
    gates, failures, raw = {"roundtrip": True, "sanity": True}, [], {}
    rows = pr.run_group(bench, "k4", "r", suite.Deadline(120), gates, failures, raw, small=True, sampler_factory=_NoSampler)
    assert rows == [] and failures and failures[0]["cell"] == "alloc_control"


def test_plan_and_dry_run_list_every_group(capsys):
    assert pr.main(["plan"]) == 0 and pr.main(["run", "--cells", "k3"]) == 0
    out = capsys.readouterr().out
    assert all(g in out for g in pr.ORDER) and "DRY RUN" not in out.split("\n")[0]
    assert list(pr.ORDER) == ["k0", "k1", "k2", "k3", "k4", "k5"] and set(pr.ORDER) | set(pr.CONTEXT) | {"m0", "m1", "m4", "m6", "m7", "m8", "m9", "n1", "n2", "n3", "n4", "o1", "o2", "o3", "o4"} == set(pr.GROUPS)
    assert list(pr.FOLLOWUP_ORDER) == ["k4", "k3", "k2", "k1", "l1", "l2", "l3", "l4", "k4"] and "l4" in out



# ------------------------------------------------------------------------------------------------ follow-up N

def test_follow_up_n_order_and_the_wait_list_are_as_registered():
    assert list(pr.ORDER_N) == ["k4", "l1", "n1", "n2", "n3", "n4", "k4"]
    assert len(pr.WAITS_MS) == 28 and all(pr.WAITS_MS.count(t) == 4 for t in (0, 5, 20, 50, 100, 200, 500))
    assert pr.EVICT_BYTES == (0, 64 * pr.MIB, 256 * pr.MIB) and pr.RETOUCH == ("none", "read", "write")


def test_follow_up_n_cells_report_their_keys(bench):
    gates, failures, raw = {"roundtrip": True, "sanity": True}, [], {}
    fields = set()
    for g in ("n1", "n2", "n3", "n4"):
        fields |= {r["field"] for r in pr.run_group(bench, g, "r", suite.Deadline(120), gates, failures, raw, small=True, sampler_factory=_NoSampler)}
    assert gates == {"roundtrip": True, "sanity": True} and failures == []
    assert {"slow_share_wait_0", "slow_share_wait_2", "slow_share_evict_0", "slow_share_evict_256", "fill_ms_evict_64",
            "slow_share_touch_none", "slow_share_touch_read", "slow_share_touch_write", "train_copies_b0", "train_first_ms_b1"} <= fields
    assert raw["retouch"]["mode"] == ["none", "read", "write"] and raw["wait_sweep"]["wait_ms"] == [0, 2, 0, 2]
    assert all(len(t["ms"]) == len(t["t_ms"]) > 0 for t in raw["settle_train"]["trains"])


def test_the_rewritten_buffer_is_checked_against_its_new_bytes(bench):
    res = bench.retouch_cell(k=3, age_ms=0.0, nbytes=1 << 12)
    assert res["roundtrip_ok"] and res["mode"] == ["none", "read", "write"]


def test_train_summary_finds_the_settle_point_and_the_tail():
    t = {"ms": [0.36] * 20 + [0.18] * 120, "t_ms": [float(i) for i in range(140)]}
    s = pr.train_summary(t)
    assert s["settle_ms"] == 20.0 and s["tail_slow_share"] == 0.0 and s["first_ms"] == 0.36 and s["copies"] == 140
    s = pr.train_summary({"ms": [0.36] * 150, "t_ms": [float(i) for i in range(150)]})
    assert s["settle_ms"] is None and s["tail_slow_share"] == 1.0


def test_a_train_that_never_settles_still_writes_valid_rows():
    res = {"ms": [0.36], "roundtrip_ok": True, "nbytes": 1 << 12, "kept": 1, "train_ms": 1.0,
           "trains": [{"ms": [0.36] * 30, "t_ms": [float(i) for i in range(30)]}]}
    rows = pr.rows_train("r", "pinroute.settle_train", res)
    validate(rows)
    assert any(r["field"] == "train_settle_ms_b0" and r["tag"] == "unavailable" and r["note"] for r in rows)


# ------------------------------------------------------------------------------------------------ follow-up O

def test_follow_up_o_order_and_lists_are_as_registered():
    assert list(pr.ORDER_O) == ["k4", "l1", "o1", "o2", "o3", "o4", "k4"]
    assert len(pr.FINE_WAITS_MS) == 36 and all(pr.FINE_WAITS_MS.count(t) == 4 for t in (0, 5, 7.5, 10, 12.5, 15, 17.5, 20, 50))
    assert len(pr.GAPS_MS) == 10 and all(pr.GAPS_MS.count(g) == 2 for g in (1, 2, 5, 10, 20))
    assert pr.FLUSH_MODES == ("none", "memcpy", "nt", "clflushopt", "clwb") and pr.READ_MODES == ("none", "direct", "buffered", "direct_wait")
    assert pr.PAD_BYTES == 9_953_280 and pr.PAD_BYTES % 4096 == 0 and pr.PAD_BYTES >= pr.RECORD_BYTES
    assert pr.wait_key(7.5) == "7p5" and pr.wait_key(20) == "20" and pr.wait_key(0.5) == "0p5"


def test_follow_up_o_cells_report_their_keys_and_their_unavailable_modes(bench, tmp_path):
    gates, failures, raw = {"roundtrip": True, "sanity": True}, [], {}
    rows = []
    for g in ("o1", "o2", "o3", "o4"):
        rows += pr.run_group(bench, g, "r", suite.Deadline(120), gates, failures, raw, small=True, sampler_factory=_NoSampler, work_dir=str(tmp_path))
    validate(rows)
    assert gates == {"roundtrip": True, "sanity": True} and failures == []
    fields = {r["field"] for r in rows}
    assert {"slow_share_wait_0", "slow_share_wait_0p5", "gap_ms_b0", "gap_second_half_slow_share_b1", "gap_after_first_fast_share_b0",
            "slow_share_mode_none", "op_ms_none", "read_ms_buffered", "content_ok"} <= fields
    assert all(len(t["ms"]) == len(t["t_ms"]) > 0 for t in raw["gap_train"]["trains"]) and [t["gap_ms"] for t in raw["gap_train"]["trains"]] == [0.5, 1.0]
    assert not list(tmp_path.glob("o4-records-*"))                        # o4's record file is removed
    assert list(tmp_path.glob("cache_touch_*/cache_touch.so"))           # o3's helper is built inside the run directory
    # every mode the Mac cannot run is an unavailable row with its reason, never a number and never a failed cell
    un = {r["field"]: r for r in rows if r["tag"] == "unavailable" and r["field"].startswith("slow_share_mode_")}
    for mode in list(raw["flush"]["unavailable"]) + list(raw["direct_read"]["unavailable"]):
        assert un[f"slow_share_mode_{mode}"]["note"] and "value" not in un[f"slow_share_mode_{mode}"]
    assert "none" in raw["direct_read"]["mode"] and "buffered" in raw["direct_read"]["mode"]


def test_the_flush_cell_without_a_compiler_times_only_the_control(bench, monkeypatch):
    import cache_touch as ct
    monkeypatch.setattr(ct, "compiler", lambda: None)
    res = bench.flush_cell(reps=1, nbytes=1 << 12)
    assert res["mode"] == ["none"] and res["helper"] is None and res["roundtrip_ok"]
    assert set(res["unavailable"]) == {"memcpy", "nt", "clflushopt", "clwb"} and all("compiler" in v for v in res["unavailable"].values())
    rows = pr.rows_flush("r", "pinroute.flush", res)
    validate(rows)
    assert sum(1 for r in rows if r["tag"] == "unavailable" and r["field"].startswith("slow_share_mode_")) == 4


def test_the_helper_compiles_and_its_memcpy_writes_the_new_bytes():
    import cache_touch as ct
    helper, why = ct.CacheTouch.load()
    if helper is None:
        pytest.skip(why)
    src = torch.randint(0, 256, (1 << 12,), dtype=torch.uint8)
    dst = torch.zeros(1 << 12, dtype=torch.uint8)
    helper.run("memcpy", dst.data_ptr(), src.data_ptr(), 1 << 12)
    assert torch.equal(dst, src)
    import platform
    if platform.machine() not in ("x86_64", "AMD64"):
        assert helper.features == 0 and helper.available("nt") and helper.available("clflushopt")
        with pytest.raises(RuntimeError):
            helper.run("clflush", dst.data_ptr(), 0, 1 << 12)


def test_a_refused_direct_read_is_unavailable_not_a_failed_cell(bench, monkeypatch, tmp_path):
    def refuse(fd, view, offset):
        if fd == state["direct"]:
            raise OSError(22, "Invalid argument")
        return real(fd, view, offset)

    real, state = pr._pread, {"direct": None}
    real_open = pr.os.open

    def fake_open(path, flags, *a):
        fd = real_open(path, flags & ~getattr(pr.os, "O_DIRECT", 0), *a)
        if flags & 0x4000_0000:
            state["direct"] = fd
        return fd

    monkeypatch.setattr(pr.os, "O_DIRECT", 0x4000_0000, raising=False)
    monkeypatch.setattr(pr.os, "open", fake_open)
    monkeypatch.setattr(pr, "_pread", refuse)
    res = bench.direct_read_cell(str(tmp_path), reps=1, nbytes=1 << 12, pad_bytes=1 << 12, age_ms=0.0)
    assert res["mode"] == ["none", "buffered"] and res["roundtrip_ok"] and res["content_ok"]
    assert set(res["unavailable"]) == {"direct", "direct_wait"} and all("refused" in v for v in res["unavailable"].values())


def test_gap_summary_reads_both_halves_and_the_tail_after_the_first_copy():
    s = pr.gap_summary({"ms": [0.36] + [0.18] * 9, "t_ms": [float(i) for i in range(10)], "gap_ms": 20})
    assert s["after_first_fast_share"] == 1.0 and s["second_half_slow_share"] == 0.0 and s["gap_ms"] == 20 and s["settle_ms"] is None
    s = pr.gap_summary({"ms": [0.36] * 30, "t_ms": [float(i) for i in range(30)], "gap_ms": 1})
    assert s["after_first_fast_share"] == 0.0 and s["second_half_slow_share"] == 1.0
    rows = pr.rows_gap_train("r", "pinroute.gap_train", {"ms": [0.36], "roundtrip_ok": True, "nbytes": 1, "kept": 1, "train_ms": 1.0,
                                                          "trains": [{"ms": [0.36] * 30, "t_ms": [float(i) for i in range(30)], "gap_ms": 1}]})
    validate(rows)
    assert any(r["field"] == "gap_settle_ms_b0" and r["tag"] == "unavailable" and r["note"] for r in rows)


# ------------------------------------------------------------------------------------------------ DC-5: the guard

def test_the_guard_brackets_the_window_and_counts_only_synchronisation_warnings():
    import warnings

    modes: list[str] = []
    g = oa.SyncGuard(rehearsal=True, set_mode=modes.append)
    with g:
        g.fake_sync()                                              # window closed: nothing
        g.open()
        g.fake_sync()
        warnings.warn("something unrelated", UserWarning, stacklevel=2)
        g.close()
        g.token_end()
        g.open()
        g.close()
        g.token_end()
    assert modes == ["warn", "default", "warn", "default", "default"]
    s = g.summary()
    assert s["calls"] == 1 and s["guarded_rows"] == 2 and s["tokens"] == 2 and s["tokens_with_hit"] == 1 and s["token_hits"] == [1, 0]
    assert any(k.startswith("test_cuda_pin_route.py:") for k in s["sources"]) and "synchronizing CUDA operation" in s["first_message"]


def test_the_guard_restores_the_default_mode_even_when_the_cell_raises():
    modes: list[str] = []
    g = oa.SyncGuard(rehearsal=True, set_mode=modes.append)
    with pytest.raises(RuntimeError), g:
        g.open()
        raise RuntimeError("boom")
    assert modes[-1] == "default"


def test_the_k_profiles_register_the_guard_cells():
    full = oa.cells_of(oa.cell_defs("k-full")[0])
    assert len(full) == 27 and len({c["id"] for c in full}) == 27
    assert sum(1 for c in full if c["guard"]) == 15 and sum(1 for c in full if c["planted"]) == 3
    assert {(c["level"], c["c"], c["guard"], c["planted"]) for c in full} == {
        (3, 1.0, False, False), (3, 1.0, True, False), (4, 1.0, False, False), (4, 1.0, True, False),
        (4, 0.3, False, False), (4, 0.3, True, False), (4, 0.3, True, True)}
    assert all(c["arm"] in ("N", "D", "P0L") for c in full) and all(c["tokens"] == 400 for c in full)
    assert {c["level"] for c in full if c["rows"]} == {3} and not any(c["rows"] and c["guard"] for c in full)
    smoke = oa.cells_of(oa.cell_defs("k-smoke")[0])
    assert len(smoke) == 7 and all(c["tokens"] == 40 and c["guard"] for c in smoke)


def test_the_expected_file_has_the_cells_the_k_profiles_need():
    exp = json.loads((ROOT / "benchmarks" / "cuda" / "n3d-expected.json").read_text())
    for prof in ("k-smoke", "k-full"):
        for g in oa.cell_defs(prof):
            for arm in {a for e in g["plan"] for a in e["arms"] if a in oa.ARM_MODE}:
                assert f"{g['percent']}.{g['tokens']}.{oa.ARM_MODE[arm]}" in exp["cells"]


def test_plan_lists_the_k_profile_and_the_tags_know_it(capsys):
    oa.cmd_plan(None)
    out = capsys.readouterr().out
    assert "k-full" in out and "p17k" in out and "cells=27" in out and "cells=7" in out
    text = (ROOT / "benchmarks" / "cuda" / "overlap_arena.py").read_text()
    assert '"k": "n3k"' in text and '"n3k": "N3K-1"' in text and text.count('"k-smoke", "k-full"') >= 2


def _rehearse_k(arrays, **kw):
    gates, failures, raw = _gates(), [], {}
    rows, status = oa.run_group(arrays, oa.cell_defs("k-rehearsal")[0], None, "r", suite.Deadline(300), gates, failures, raw, rehearsal=True,
                                weight_start=1 << 16, calib_tolerance=50.0, nbytes=8192, readback_every=20, drift_limit=10.0, **kw)
    return rows, status, gates, failures, raw


def test_the_k_rehearsal_runs_the_real_path_with_every_gate_and_a_clean_issue_loop():
    rows, status, gates, failures, raw = _rehearse_k(synth(tokens=8))
    assert status == "complete" and failures == [] and all(gates.values()), (gates, failures)
    assert gates["guard_sees_planted"] is True
    validate(rows)
    guarded = [k for k in raw if ".G." in k or ".GP." in k]
    assert guarded and all("sync_guard" in raw[k] for k in guarded)
    plain = [k for k in guarded if ".GP." not in k and not k.endswith((".N0", ".N1"))]
    assert plain and all(raw[k]["sync_guard"]["calls"] == 0 for k in plain)             # nothing hidden in the rehearsal's loop
    planted = [k for k in guarded if ".GP." in k and k.endswith(".P0L")]
    assert planted and raw[planted[0]]["sync_guard"]["calls"] == raw[planted[0]]["sync_guard"]["tokens_with_hit"] > 0
    calls = [r for r in rows if r["field"] == "sync_calls_in_issue"]
    assert calls and any(r["value"] > 0 for r in calls) and all(r["tag"] == "measured" for r in calls)


def test_a_hidden_synchronisation_in_the_issue_loop_is_counted_with_its_source(monkeypatch):
    live: list = []

    class Tracked(oa.SyncGuard):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            live.append(self)

    monkeypatch.setattr(oa, "SyncGuard", Tracked)
    orig = oa.OverlapArena.enqueue

    def sneaky(self, loads, marks=None, parity=0):
        orig(self, loads, marks, parity)
        if live and loads:
            live[-1].fake_sync()                                    # a hidden blocking call inside the copy issue

    monkeypatch.setattr(oa.OverlapArena, "enqueue", sneaky)
    rows, status, gates, failures, raw = _rehearse_k(synth(tokens=8))
    dirty = [k for k in raw if ".G." in k and raw[k]["sync_guard"]["calls"] > 0]
    assert dirty, "a guarded cell with copies must count the hidden call"
    s = raw[dirty[0]]["sync_guard"]
    assert s["sources"] and next(iter(s["sources"])).startswith("test_cuda_pin_route.py:")
    unguarded = [k for k in raw if "sync_guard" not in raw[k] and k.startswith("p17k.L4.C0.05.") and ".G" not in k]
    assert unguarded                                                # unguarded cells keep no guard and cannot fail on it
    assert gates["guard_sees_planted"] is True


def test_a_blind_guard_trips_its_own_gate(monkeypatch):
    monkeypatch.setattr(oa.SyncGuard, "fake_sync", lambda self: None)       # the planted synchronisation is no longer seen
    rows, status, gates, failures, raw = _rehearse_k(synth(tokens=8))
    assert gates["guard_sees_planted"] is False
    assert any("guard saw the planted synchronisation" in f["error"] for f in failures)


def test_session_sh_has_the_k_commands_in_the_registered_order():
    text = (ROOT / "benchmarks" / "cuda" / "session.sh").read_text()
    assert "pinroute)" in text and "n3ksmoke)" in text and "n3kfull)" in text and "k-smoke" in text and "k-full" in text and "N3K" in text
    loop = next(line for line in text.splitlines() if line.strip().startswith("for g in k0"))
    assert loop.split("in", 1)[1].replace("; do", "").split() == list(pr.ORDER)
    assert "pin_route.py" in text and "--execute" in text.split("pinroute)", 1)[1].split("n3ksmoke)", 1)[0]
    assert (ROOT / "benchmarks" / "cuda" / "SESSION-RUNBOOK-PIN-ROUTE.md").exists()


# ------------------------------------------------------------------------------------------------ follow-up L

def test_the_context_cells_switch_the_three_factors_one_at_a_time_and_together():
    assert pr.CONTEXT == {"l1": (True, False, False), "l2": (False, True, False), "l3": (False, False, True), "l4": (True, True, True)}


def test_the_context_cell_allocates_in_the_requested_order_and_reports_its_context(bench, monkeypatch):
    events: list[str] = []
    orig_pinned, orig_time = bench.pinned, bench.time_copy_on

    def pinned(n, seed):
        events.append("alloc")
        return orig_pinned(n, seed)

    def timed(dst, src, stream, **k):
        events.append("time")
        return orig_time(dst, src, stream, **k)

    monkeypatch.setattr(bench, "pinned", pinned)
    monkeypatch.setattr(bench, "time_copy_on", timed)
    r = bench.context_cell(True, False, False, k=3, nbytes=1 << 16)
    assert events == ["alloc"] * 3 + ["time"] * 3 and r["context"]["batch_first"] and r["roundtrip_ok"] and r["kept"] == 3
    events.clear()
    r = bench.context_cell(False, True, True, k=3, nbytes=1 << 16, arena_bytes=1 << 20)
    assert events == ["alloc", "time"] * 3 and r["context"] == {"batch_first": False, "arena": True, "arena_bytes": 1 << 20, "side_stream": True}


def test_the_arena_destination_is_a_slice_of_the_big_buffer(bench, monkeypatch):
    seen = {}
    orig = bench.time_copy_on

    def timed(dst, src, stream, **k):
        seen["storage"] = dst.untyped_storage().nbytes()
        return orig(dst, src, stream, **k)

    monkeypatch.setattr(bench, "time_copy_on", timed)
    bench.context_cell(False, True, False, k=1, nbytes=1 << 16, arena_bytes=1 << 20)
    assert seen["storage"] == 1 << 20
    bench.context_cell(False, False, False, k=1, nbytes=1 << 16)
    assert seen["storage"] == 1 << 16


def test_session_sh_runs_the_followup_in_the_registered_order_into_its_own_directory():
    text = (ROOT / "benchmarks" / "cuda" / "session.sh").read_text()
    block = text.split("pinroute2)", 1)[1].split("n3ksmoke)", 1)[0]
    loop = next(line for line in block.splitlines() if line.strip().startswith("for g in k4"))
    assert loop.split("in", 1)[1].replace("; do", "").split() == list(pr.FOLLOWUP_ORDER)
    assert "cuda-followup" in block and "--execute" in block


# ------------------------------------------------------------------------------------------------ follow-up M

def test_the_m_session_order_and_session_sh_agree():
    assert list(pr.ORDER_M) == ["m0", "k4", "l1", "m1", "m4", "m6", "m7", "m8", "m9"] and set(pr.ORDER_M) <= set(pr.GROUPS)
    text = (ROOT / "benchmarks" / "cuda" / "session.sh").read_text()
    block = text.split("pinroute3)", 1)[1].split("n3ksmoke)", 1)[0]
    loop = next(line for line in block.splitlines() if line.strip().startswith("for g in m0"))
    assert loop.split("in", 1)[1].replace("; do", "").split() == list(pr.ORDER_M)
    assert "cuda-order" in block and "--execute" in block


def test_half_copy_half_runs_one_copy_between_two_allocation_batches(bench, monkeypatch):
    events: list[str] = []
    orig_pinned = bench.pinned
    monkeypatch.setattr(bench, "pinned", lambda n, seed: (events.append("alloc"), orig_pinned(n, seed))[1])
    monkeypatch.setattr(bench, "sync", lambda: events.append("sync"))
    orig_time = bench.time_copy
    monkeypatch.setattr(bench, "time_copy", lambda *a, **k: (events.append("time"), orig_time(*a, **k))[1])
    r = bench.half_copy_half(3, 1 << 16)
    assert events[:5] == ["alloc"] * 3 + ["sync"] + ["alloc"] and r["split"] == 3 and len(r["ms"]) == 6 and r["roundtrip_ok"]
    assert events.index("time") > max(i for i, e in enumerate(events) if e == "alloc")        # every allocation before the first timed copy
    rows = pr.rows_half("r", "a", r)
    validate(rows)
    assert {"slow_share_first_half", "slow_share_second_half"} <= {x["field"] for x in rows}


def test_copy_first_copies_once_then_allocates_in_batch(bench, monkeypatch):
    events: list[str] = []
    orig_pinned = bench.pinned
    monkeypatch.setattr(bench, "pinned", lambda n, seed: (events.append(f"alloc{n}"), orig_pinned(n, seed))[1])
    r = bench.copy_first(3, 1 << 16, small_bytes=1 << 12)
    assert events == ["alloc4096"] + ["alloc65536"] * 3 and len(r["ms"]) == 3 and r["context"]["copy_first_bytes"] == 1 << 12 and r["context"]["batch_first"]


def test_the_batch_first_registered_cells_make_every_buffer_before_timing_any(bench, monkeypatch):
    events: list[str] = []
    orig_time = bench.time_copy
    monkeypatch.setattr(bench, "time_copy", lambda *a, **k: (events.append("time"), orig_time(*a, **k))[1])
    orig_init = bp.MmapBuffer.__init__

    def init(self, *a, **k):
        events.append("make")
        orig_init(self, *a, **k)

    monkeypatch.setattr(bp.MmapBuffer, "__init__", init)
    r = bench.registered_batch_cell("private", "thp", k=3, nbytes=1 << 16)
    assert events == ["make"] * 3 + ["time"] * 3 and r["pattern"] == "batch_first" and r["advice"] == "thp" and len(r["ms"]) == 3
    r = bench.registered_batch_cell("ordinary", "none", k=3, nbytes=1 << 16, reg_flags=0)
    assert r["mapping"] == "ordinary" and r["reg_flags"] == 0 and len(r["ms"]) == 3
    validate(pr.rows_route("r", "p", r))


def test_the_batch_classification_cell_allocates_everything_before_timing(bench, monkeypatch):
    events: list[str] = []
    orig_pinned, orig_time = bench.pinned, bench.time_copy
    monkeypatch.setattr(bench, "pinned", lambda n, seed: (events.append("alloc"), orig_pinned(n, seed))[1])
    monkeypatch.setattr(bench, "time_copy", lambda *a, **k: (events.append("time"), orig_time(*a, **k))[1])
    r = bench.classify_batch_cell(4, 2, 1 << 16, 1 << 18, 1 << 18)
    assert events == ["alloc"] * 6 + ["time"] * 7 and r["pattern"] == "batch_first" and len(r["ms"]) == 4 and len(r["plateau_ms"]) == 2
    validate(pr.rows_classify("r", "a", r))
