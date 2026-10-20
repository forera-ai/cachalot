"""Compute-slowdown fit (Mac only; record docs/CUDA-SLOWDOWN-RECORD.md): the fit recovers a planted slope from fabricated runs."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))

import compute_slowdown_fit as cs  # noqa: E402
import overlap_arena as oa  # noqa: E402
import row_timing as rt  # noqa: E402

S_TRUE = 0.05


def _cell(role: str, c: float, rng, tokens: int = 60, rows: int = 12, s: float = S_TRUE) -> dict:
    out = []
    for _ in range(tokens + 1):
        t, trows = 0.0, []
        for _r in range(rows):
            n_dem = int(rng.integers(0, 3)) if role != "N" else 0
            n_pf = int(rng.integers(0, 8)) if role == "P0L" else 0
            pre = t
            dem_a, dem_b = pre + 0.01, pre + 0.01 + 0.19 * n_dem
            pf_a = dem_b
            pf_b = pf_a + 0.2 * n_pf
            start = dem_b
            end_nominal = start + c
            ov = max(0.0, min(end_nominal, pf_b) - max(start, pf_a))
            end = end_nominal + (s * ov if role == "P0L" else 0.0)
            trows.append([n_dem, n_pf, 0, pre, start, (start + end) / 2, end, dem_a, dem_b, pf_a, pf_b])
            t = end
        out.append(trows)
    return {"rows": out, "wall_ms": [r[-1][6] for r in out]}


def _run(c: float, seed: int, level: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    cells = {}
    for role in ("N0", "N1", "D", "P0L"):
        cells[rt.cell_id(level, c, role)] = _cell("N" if role[0] == "N" else role, c, rng)
    return cells


def test_row_field_order_is_the_one_the_script_uses():
    assert oa.ROW_FIELDS[:2] == ("n_dem", "n_pf") and cs.RF["pf_b"] == oa.ROW_FIELDS.index("pf_b")


def test_overlap_of_a_single_row():
    tok = np.full((2, len(oa.ROW_FIELDS)), np.nan)
    tok[0, [cs.RF["start"], cs.RF["end"], cs.RF["n_pf"], cs.RF["pf_a"], cs.RF["pf_b"]]] = [1.0, 2.0, 3, 0.0, 1.5]
    tok[1, [cs.RF["start"], cs.RF["end"], cs.RF["n_pf"], cs.RF["pf_a"], cs.RF["pf_b"]]] = [2.0, 3.0, 2, 1.8, 2.4]
    ov = cs.row_terms(tok)["overlap"]
    assert abs(ov[0] - 0.7) < 1e-12 and abs(ov[1] - 0.4) < 1e-12        # row 0 meets batch 0 for 0.5 and batch 1 for 0.2; row 1 meets batch 1 for 0.4
    assert list(cs.row_terms(tok)["count"]) == [3, 5]


def test_the_fit_recovers_a_planted_slope_and_beats_the_count_term():
    a = _run(1.0, 1)
    b = _run(1.0, 2)
    res = cs.score_level(a, b, 1.0, 3)
    m = res["models"]
    assert res["available"] and abs(m["M1_overlap"]["slope_run_a"] - S_TRUE) < 0.006
    assert m["M1_overlap"]["held_out_r2"] > 0.9 and m["M1_overlap"]["held_out_sse"] < m["M2_count"]["held_out_sse"]
    assert m["M0_none"]["held_out_r2"] < 0.05
    assert abs(res["token_busy"]["error"]) < 0.05


def test_a_run_with_no_slowdown_gives_a_zero_slope():
    rng = np.random.default_rng(3)
    mk = lambda seed: {rt.cell_id(3, 0.3, r): _cell("N" if r[0] == "N" else r, 0.3, np.random.default_rng(seed), s=0.0)  # noqa: E731
                       for r in ("N0", "N1", "D", "P0L")}
    res = cs.score_level(mk(4), mk(5), 0.3, 3)
    assert abs(res["models"]["M1_overlap"]["slope_run_a"]) < 1e-9 and rng is not None


def test_verdicts_use_only_the_registered_numbers():
    a, b = _run(1.0, 1), _run(1.0, 2)
    res = {"level3": {1.0: cs.score_level(a, b, 1.0, 3)}, "hidden": {}, "level6": {}}
    v = cs.verdicts(res)
    assert v["CS-1@1.0"] == "in interval" and v["CS-2@1.0"] == "in interval" and v["CS-3@1.0"] == "in interval"
    assert v["CS-4@1.0"] == "in interval" and v["CS-5@1.0"] == "in interval"
