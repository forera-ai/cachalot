"""Predicted-overlap fit (Mac only; record docs/CUDA-PREDICTED-OVERLAP-RECORD.md): fabricated runs with a planted first-in first-out copy stream."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "cuda"))

import compute_slowdown_fit as cs  # noqa: E402
import overlap_arena as oa  # noqa: E402
import predicted_overlap_fit as po  # noqa: E402
import row_timing as rt  # noqa: E402

S_TRUE, C_DEM, C_PF, GAP = 0.06, 0.19, 0.2, 0.03


def _token(role: str, c: float, rng, rows: int = 14) -> list:
    """A token on a planted FIFO copy stream: demand batch, then prefetch batch queued behind earlier prefetch work; compute stretched by S_TRUE x overlap."""
    free, pre, out = 0.0, 0.05, []
    pf_a_all, pf_b_all = [], []
    for _ in range(rows):
        n_dem = int(rng.integers(0, 3)) if role != "N" else 0
        n_pf = int(rng.integers(0, 6)) if role == "P0L" else 0
        dem_a = max(pre, free)
        dem_b = dem_a + n_dem * C_DEM
        free = dem_b
        start = dem_b
        pf_a = pf_b = None
        if n_pf:
            pf_a = max(dem_b, free)
            pf_b = pf_a + n_pf * C_PF
            free = pf_b
            pf_a_all.append(pf_a)
            pf_b_all.append(pf_b)
        end = start + c
        if role == "P0L":
            for _ in range(30):
                ov = sum(max(0.0, min(end, b) - max(start, a)) for a, b in zip(pf_a_all, pf_b_all))
                new_end = start + c + S_TRUE * ov
                if abs(new_end - end) < 1e-9:
                    break
                end = new_end
        out.append([n_dem, n_pf, 0, round(pre, 4), round(start, 4), round((start + end) / 2, 4), round(end, 4),
                    round(dem_a, 4) if n_dem or True else None, round(dem_b, 4), None if pf_a is None else round(pf_a, 4),
                    None if pf_b is None else round(pf_b, 4)])
        pre = end + GAP
    return out


def _cell(role: str, c: float, rng, tokens: int = 50) -> dict:
    toks = [_token(role, c, rng) for _ in range(tokens + 1)]
    return {"rows": toks, "wall_ms": [t[-1][6] for t in toks]}


def _run(c: float, seed: int, level: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    return {rt.cell_id(level, c, role): _cell("N" if role[0] == "N" else role, c, rng) for role in ("N0", "N1", "D", "P0L")}


def test_row_field_order_matches_the_arena():
    assert po.RF["pf_b"] == oa.ROW_FIELDS.index("pf_b") and po.RF["n_dem"] == 0


def test_parameters_are_recovered_from_a_planted_run():
    a = _run(1.0, 1)
    par = po.fit_params(a[rt.cell_id(3, 1.0, "P0L")], a[rt.cell_id(3, 1.0, "D")])
    assert abs(par["c_pf"] - C_PF) < 0.005 and abs(par["gap"] - GAP) < 0.005 and abs(par["b0"] - 1.0) < 0.01


def test_simulation_matches_a_hand_computed_token():
    par = {"c_pf": 0.2, "c_dem": 0.1, "b0": 1.0, "gap": 0.0, "pre0": 0.0}
    sim = po.simulate(np.array([2.0, 0.0]), np.array([3.0, 0.0]), par)
    # row 0: demand 0.0-0.2, compute 0.2-1.2, prefetch 0.2-0.8 (overlap 0.6); row 1: pre 1.2, no copies, no overlap
    assert abs(sim["start"][0] - 0.2) < 1e-12 and abs(sim["end"][0] - 1.2) < 1e-12
    assert abs(sim["overlap"][0] - 0.6) < 1e-12 and sim["overlap"][1] == 0.0


def test_a_prefetch_batch_queues_behind_an_earlier_one():
    par = {"c_pf": 1.0, "c_dem": 0.0, "b0": 1.0, "gap": 0.0, "pre0": 0.0}
    sim = po.simulate(np.zeros(2), np.array([3.0, 1.0]), par)
    # batch 0 runs 0-3; row 1 computes 1-2 and its batch waits until 3, so row 1 overlaps only batch 0 for its whole window
    assert abs(sim["overlap"][1] - 1.0) < 1e-12


def test_closed_loop_stretches_the_window():
    par = {"c_pf": 1.0, "c_dem": 0.0, "b0": 1.0, "gap": 0.0, "pre0": 0.0}
    open_loop = po.simulate(np.zeros(1), np.array([5.0]), par)
    closed = po.simulate(np.zeros(1), np.array([5.0]), par, s=0.1)
    assert abs(open_loop["overlap"][0] - 1.0) < 1e-12 and abs(closed["overlap"][0] - 1.0 / 0.9) < 1e-6   # ov = 1 + 0.1 ov


def test_prediction_reads_only_the_counts():
    a, b = _run(1.0, 1), _run(1.0, 2)
    par = po.fit_params(a[rt.cell_id(3, 1.0, "P0L")], a[rt.cell_id(3, 1.0, "D")])
    cell = b[rt.cell_id(3, 1.0, "P0L")]
    base = po.predict_cell(cell, par)
    scrambled = {"rows": [[[r[0], r[1], r[2], *np.random.default_rng(9).random(8).tolist()] for r in tok] for tok in cell["rows"]]}
    again = po.predict_cell(scrambled, par)
    assert all(np.allclose(x, y) for x, y in zip(base, again))


def test_the_fit_recovers_the_planted_stream_and_scores_well():
    a, b = _run(1.0, 1), _run(1.0, 2)
    res = po.score_level(a, b, 1.0, 3)
    assert res["available"] and res["overlap_row_r_run_b"] > 0.9
    assert 0.85 <= res["overlap_ratio_run_b"] <= 1.1 and abs(res["s_pred_run_a"] - S_TRUE) < 0.02
    assert res["held_out_r2_v1"] > 0.7 and abs(res["token_busy"]["v1_error"]) < 0.15
    assert 0.9 < res["v2_over_v1"] < 1.3


def test_a_run_with_no_prefetch_gives_a_zero_prediction():
    rng = np.random.default_rng(3)
    cell = {"rows": [[[0, 0, 0, 0.0, 0.1, 0.6, 1.1, 0.0, 0.1, None, None]] * 5 for _ in range(4)], "wall_ms": [1.1] * 4}
    par = {"c_pf": 0.2, "c_dem": 0.1, "b0": 1.0, "gap": 0.0, "pre0": 0.0}
    assert all(np.all(p == 0) for p in po.predict_cell(cell, par)) and rng is not None


def test_verdicts_use_only_the_registered_numbers():
    a, b = _run(1.0, 1), _run(1.0, 2)
    e = po.score_level(a, b, 1.0, 3)
    res = {"level3": {1.0: e}, "hidden": {1.0: {"available": True, "error_predicted": 0.04}},
           "level6": {1.0: {"available": True, "error": 0.5}}}
    v = po.verdicts(res)
    assert v["PO-5@1.0"] == "in interval" and v["PO-6@1.0"] == "in interval"
    assert v["PO-7@1.0"] == "past falsifier"
    res["hidden"][1.0]["error_predicted"] = 0.07
    assert po.verdicts(res)["PO-5@1.0"] == "outside interval"
    res["hidden"][1.0]["error_predicted"] = 0.09
    assert po.verdicts(res)["PO-5@1.0"] == "past falsifier"


def test_hidden_share_runs_on_a_planted_pair_of_runs():
    a, b = _run(1.0, 1), _run(1.0, 2)
    for run in (a, b):
        for role in ("N0", "N1", "D", "P0L"):
            run[rt.cell_id(4, 1.0, role)] = _cell("N" if role[0] == "N" else role, 1.0, np.random.default_rng(7))
    e = po.score_level(a, b, 1.0, 3)
    h = po.hidden_share(a, b, 1.0, e["s_pred_run_a"], e["params"])
    assert h["available"] and h["error_predicted"] is not None and np.isfinite(h["H_predicted_overlap"])
    assert cs.RF is po.RF
