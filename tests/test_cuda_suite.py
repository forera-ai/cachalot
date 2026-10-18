"""CUDA lane microbenchmark suite: planning, caps, statistics, runner and gates, on a scripted bench (no GPU)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "cuda"))

import cuda_bench  # noqa: E402
import doctor  # noqa: E402
import manifest as mf  # noqa: E402
import suite  # noqa: E402


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class ScriptedBench:
    """Stands in for TorchBench. Every cell returns fixed samples; knobs inject failures."""

    def __init__(self, ms=1.0, n=30, overhead=0.001, bad_validation=(), raise_on=(), burn=0.0, clock=None):
        self.ms, self.n, self.overhead = ms, n, overhead
        self.bad, self.raise_on, self.burn, self.clock = set(bad_validation), set(raise_on), burn, clock
        self.seen = []

    def timer_overhead_ms(self):
        return self.overhead

    def execute(self, cell, dl, limits):
        self.seen.append(cell.id)
        if self.clock is not None:
            self.clock.t += self.burn
        if cell.id in self.raise_on:
            raise RuntimeError("boom")
        samples = [self.ms + 0.01 * (i % 3) for i in range(self.n)]
        ok = {"ok": cell.id not in self.bad, "detail": "scripted"}
        wb = None if cell.kind in ("launch", "overlap") else 10_000_000
        if cell.kind == "overlap":
            arms = {"compute_alone": samples, "copy_alone": samples, "compute_with_copy": samples}
        else:
            arms = {"events": samples, "host": samples}
        return suite.CellResult(arms, "both", wb, ok)


def run(bench, profile="smoke", clock=None):
    return suite.run(profile, suite.PROFILE_DEFAULTS[profile], bench, run_id="r1", machine_id="m" * 16,
                     clock=clock or Clock())


def test_percentiles_and_summary_rules():
    assert suite.percentile([1, 2, 3, 4], 50) == 2.5
    assert suite.percentile([5.0], 95) == 5.0
    s = suite.summarize([float(i) for i in range(30)], seed=1)
    assert s["n"] == 30 and "p99" not in s and s["ci_low"] <= s["p50"] <= s["ci_high"]
    assert "p99" in suite.summarize([float(i) for i in range(200)])
    assert suite.summarize([float(i) for i in range(30)], seed=1) == s          # bootstrap is seeded


def test_limits_hard_ceilings_cannot_be_lifted():
    for bad in (dict(max_seconds=0), dict(max_seconds=901), dict(max_iters=0), dict(max_iters=1001),
                dict(max_alloc_mib=8), dict(max_alloc_mib=5000)):
        with pytest.raises(ValueError):
            suite.Limits(**bad).validated()
    suite.Limits().validated()


def test_plan_is_deterministic_and_respects_the_allocation_cap():
    lim = suite.PROFILE_DEFAULTS["baseline"]
    a, _ = suite.plan("baseline", lim)
    b, _ = suite.plan("baseline", lim)
    assert [c.id for c in a] == [c.id for c in b]
    cells, skipped = suite.plan("baseline", suite.Limits(max_alloc_mib=64, max_iters=5))
    assert skipped and all(c.alloc_mib <= 64 for c in cells)
    assert all("cap" in s["reason"] for s in skipped)
    assert a[0].control and a[-1].control and a[0].id == a[-1].id            # drift bracket


def test_plan_covers_the_required_families_and_record_sizes():
    cells, _ = suite.plan("baseline", suite.PROFILE_DEFAULTS["baseline"])
    assert {c.kind for c in cells} == {"copy", "launch", "device_bw", "matmul", "overlap"}
    sizes = {c.params["nbytes"] for c in cells if c.kind == "copy"}
    assert set(suite.RECORD_SIZES.values()) <= sizes and suite.READ_3MIB in sizes
    assert {c.params["mode"] for c in cells if c.kind == "copy"} == {"pageable", "pinned"}


def test_dtype_gate_by_capability_not_by_name():
    assert suite.dtype_support((7, 0)) == {"fp16": True, "bf16": False, "fp8": False}
    assert suite.dtype_support((12, 0)) == {"fp16": True, "bf16": True, "fp8": True}


def test_run_with_scripted_bench_produces_valid_sealed_artifacts():
    manifest, rows, raw = run(ScriptedBench())
    assert mf.validate_run(manifest) == []
    assert manifest["status"] == "complete" and manifest["gates"]["copy_checksum"]
    assert all(mf.validate_row(r) == [] for r in rows)
    assert any(r["field"] == "measured_device_peak_gb_s" for r in rows)
    assert any(r["field"] == "share_of_measured_peak" for r in rows)
    drift = [r for r in rows if r["field"] == "control_drift"]
    assert len(drift) == 1 and drift[0]["value"] == pytest.approx(0.0)
    assert set(raw) >= {"copy.h2d.pinned.9950000.control", "device_bw.d2d"}
    json.dumps(manifest)
    mf.assert_publishable(manifest)
    mf.assert_publishable(rows)


def test_failed_validation_gate_voids_the_timing_row():
    cell = "copy.h2d.pinned.1048576"
    manifest, rows, _ = run(ScriptedBench(bad_validation={cell}))
    assert manifest["gates"]["copy_checksum"] is False
    assert not any(r["arm"].startswith(cell) and r["field"] == "time_ms" for r in rows)
    assert any(r["field"] == "validation_failed" and r["arm"] == cell for r in rows)


def test_exception_in_a_cell_is_recorded_not_hidden_and_the_run_continues():
    manifest, rows, _ = run(ScriptedBench(raise_on={"device_bw.d2d"}))
    assert manifest["gates"]["failed_cells"] == [{"cell": "device_bw.d2d", "error": "RuntimeError: boom"}]
    assert any(r["tag"] == "unavailable" and "cell failed" in r["note"] for r in rows)
    assert not any(r["field"] == "share_of_measured_peak" for r in rows)       # no peak, so no share


def test_wall_clock_cap_truncates_and_reports_what_was_skipped():
    clock = Clock()
    lim = suite.Limits(max_seconds=10, max_iters=5, max_alloc_mib=1024, warmup=1)
    bench = ScriptedBench(clock=clock, burn=3.0)
    manifest, _, _ = suite.run("smoke", lim, bench, run_id="r", machine_id="m" * 16, clock=clock)
    assert manifest["status"] == "truncated"
    assert len(bench.seen) <= 4
    assert any(s["reason"] == "wall-clock cap reached" for s in manifest["skipped"])


def test_sanity_bound_flags_an_impossible_pcie_rate_and_fails_the_gate():
    manifest, rows, _ = run(ScriptedBench(ms=0.01))      # 10 MB in 10 us = 1000 GB/s: not a PCIe result
    assert manifest["gates"]["sanity"] is False and manifest["gates"]["sanity_violations"]
    assert any("SANITY FAIL" in (r.get("note") or "") for r in rows)


def test_timer_overhead_above_ten_percent_means_below_resolution_no_number():
    _, rows, _ = run(ScriptedBench(ms=1.0, overhead=0.5))
    t = [r for r in rows if r["field"] == "time_ms"]
    assert t and all(r["tag"] == "unavailable" and "resolution" in r["note"] for r in t)


def test_small_n_is_flagged_as_below_the_baseline_minimum():
    _, rows, _ = run(ScriptedBench(n=12))
    assert any("below the 30-iteration" in (r.get("note") or "") for r in rows if r["field"] == "time_ms")


def test_cli_plan_and_dry_run_need_no_gpu_and_write_nothing(tmp_path, capsys):
    assert cuda_bench.main(["plan", "--profile", "smoke"]) == 0
    assert json.loads(capsys.readouterr().out)["cells"] > 10
    assert cuda_bench.main(["run", "--profile", "smoke", "--out-dir", str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == []
    assert cuda_bench.main(["plan", "--profile", "smoke", "--max-seconds", "99999"]) == 2


def test_cli_execute_without_cuda_is_an_informative_skip(tmp_path, capsys):
    if doctor.cuda_ready(doctor.collect())[0]:
        pytest.skip("this host has CUDA; the skip path is a no-GPU behaviour")
    assert cuda_bench.main(["run", "--profile", "smoke", "--execute", "--out-dir", str(tmp_path)]) == 3
    assert "SKIP" in capsys.readouterr().err and list(tmp_path.iterdir()) == []


def test_cli_validate_accepts_good_and_rejects_leaky_files(tmp_path):
    manifest, rows, _ = run(ScriptedBench())
    good = tmp_path / "run-manifest.json"
    good.write_text(json.dumps(manifest))
    rj = tmp_path / "rows.jsonl"
    rj.write_text("\n".join(json.dumps(r) for r in rows))
    assert cuda_bench.main(["validate", str(good), str(rj)]) == 0
    leaky = mf.seal({k: v for k, v in manifest.items() if k != "manifest_id"} | {"operator_note": "host 10.0.0.7"})
    bad = tmp_path / "leaky.json"
    bad.write_text(json.dumps(leaky))
    assert cuda_bench.main(["validate", str(bad)]) == 4


def test_torch_is_imported_lazily_only():
    src = Path(suite.__file__).read_text()
    assert "\nimport torch" not in src and "\nfrom torch" not in src
    assert "\nimport torch" not in Path(doctor.__file__).read_text()


def test_cli_validate_accepts_raw_samples_and_rejects_unknown_documents(tmp_path):
    _, _, raw = run(ScriptedBench())
    ok = tmp_path / "raw-samples.json"
    ok.write_text(json.dumps({"run": "r1", "samples": raw}))
    assert cuda_bench.main(["validate", str(ok)]) == 0
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hello": 1}))
    assert cuda_bench.main(["validate", str(other)]) == 4
