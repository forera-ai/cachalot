"""CPU rehearsal of the TorchBench cells: the real cell code runs on the CPU with an event shim (never a CUDA result)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "cuda"))

import manifest as mf  # noqa: E402
import suite  # noqa: E402

LIM = suite.Limits(max_seconds=60, max_iters=4, max_alloc_mib=256, warmup=1)


@pytest.fixture(scope="module")
def bench():
    return suite.TorchBench(rehearsal_cpu=True)


def cell(kind, **params):
    return suite.Cell(f"{kind}.test", kind, params, 1.0, 0.1)


def execute(bench, c):
    return bench.execute(c, suite.Deadline(60), LIM)


@pytest.mark.parametrize("mode", ["pageable", "pinned"])
@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_copy_cell(bench, mode, direction):
    res = execute(bench, cell("copy", direction=direction, mode=mode, nbytes=1 << 20))
    assert res.validation["ok"] and res.work_bytes == 1 << 20
    assert len(res.arms["events"]) == len(res.arms["host"]) == 4
    assert all(x >= 0 for x in res.arms["events"])


@pytest.mark.parametrize("name", ["tiny_launch", "sync_idle", "event_pair", "router_chain_40"])
def test_launch_cells(bench, name):
    res = execute(bench, cell("launch", name=name))
    assert len(res.arms["host"]) == 4
    if name == "tiny_launch":
        assert res.derived["per_launch_us"][1] == "us"
    if name == "router_chain_40":
        assert res.derived["per_layer_sync_us"][1] == "us"


def test_device_bw_cell(bench):
    res = execute(bench, cell("device_bw", nbytes=4 << 20))
    assert res.validation["ok"] and res.work_bytes == 8 << 20


@pytest.mark.parametrize("mode,g", [("loop", 1), ("loop", 3), ("bmm", 3)])
def test_matmul_cell_matches_the_float32_reference(bench, mode, g):
    res = execute(bench, cell("matmul", m=4, d=64, f=48, g=g, mode=mode))
    assert res.validation["ok"], res.validation["detail"]
    assert res.work_bytes == g * 64 * 48 * 2 and res.derived["tflops"][1] == "TFLOP/s"


def test_overlap_cell(bench):
    res = execute(bench, cell("overlap", nbytes=1 << 20, m=4, d=64, f=48))
    assert res.validation["ok"] and set(res.arms) == {"compute_alone", "copy_alone", "compute_with_copy"}
    assert "compute_slowdown_with_copy" in res.derived and res.derived["copies_per_compute_window"][0] >= 1


def test_deadline_truncates_a_cell(bench):
    res = bench.execute(cell("copy", direction="h2d", mode="pageable", nbytes=1 << 20), suite.Deadline(-1), LIM)
    assert res.truncated and res.arms["events"] == []


def test_results_flow_into_valid_rows(bench):
    c = cell("copy", direction="h2d", mode="pinned", nbytes=1 << 20)
    res = execute(bench, c)
    rows, _ = suite._rows_for("r", c, res, bench.timer_overhead_ms(), None, 1, [])
    assert rows and all(mf.validate_row(r) == [] for r in rows)


@pytest.mark.parametrize("nweights,interferer", [(1, "h2d"), (3, "h2d"), (1, "d2d"), (3, "d2d")])
def test_overlap_variants(bench, nweights, interferer):
    res = execute(bench, cell("overlap", nbytes=1 << 20, m=4, d=64, f=48, nweights=nweights, interferer=interferer))
    assert res.validation["ok"] and "compute_slowdown_with_copy" in res.derived
    assert interferer in res.derived["compute_slowdown_with_copy"][2]


@pytest.mark.parametrize("k,mode", [(1, "single"), (4, "single"), (4, "loop")])
def test_batchcopy_cell(bench, k, mode):
    res = execute(bench, cell("batchcopy", k=k, mode=mode, nbytes=1 << 16))
    assert res.validation["ok"] and res.work_bytes == k * (1 << 16)
    assert res.derived["per_expert_ms"][1] == "ms" and len(res.arms["events"]) == 4
    c = cell("batchcopy", k=k, mode=mode, nbytes=1 << 16)
    rows, _ = suite._rows_for("r", c, res, bench.timer_overhead_ms(), None, 1, [])
    assert rows and all(mf.validate_row(r) == [] for r in rows)
