import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import whatif  # noqa: E402


def _checks():
    return {c.name: c for c in whatif.validation()}


def test_glm_held_out_arms_within_the_quoted_band():
    held = [c for c in whatif.validation() if c.name.startswith("GLM") and not c.fitted]
    assert len(held) == 4
    assert all(abs(c.err) < whatif.interval("glm")[0] for c in held)


def test_deepseek_bandwidth_curve_reproduces_the_fitted_points():
    pts = [c for c in whatif.validation() if "emulated" in c.name]
    assert len(pts) == 4 and all(abs(c.err) < 0.03 for c in pts)


def test_deepseek_real_x10pro_is_over_predicted_by_about_a_fifth_and_inside_the_band():
    c = _checks()["DeepSeek real X10Pro"]
    assert 0.15 < c.err < whatif.interval("deepseek")[0] + 0.01


def test_storage_never_beats_the_floor_and_more_storage_never_hurts():
    for fn, floor in ((lambda k: whatif.glm_token(100.0, storage_x=k), whatif.GLM_FLOOR_MS),
                      (lambda k: whatif.minimax_token(23.0, storage_x=k), whatif.MM_FLOOR_MS)):
        vals = [fn(k) for k in (1, 2, 5, 20, 1e9)]
        assert vals == sorted(vals, reverse=True) and vals[-1] >= floor - 1e-6
    vals = [whatif.deepseek_token(b) for b in (0.5, 1, 2, 4, 6.8, 20)]
    assert vals == sorted(vals, reverse=True)


def test_deepseek_below_the_knee_is_bytes_over_rate():
    assert abs(whatif.deepseek_token(1.0) - whatif.DS_BYTES_TOKEN_MB / 1.0) < 1e-6


def test_every_scenario_row_has_a_note_and_a_positive_time():
    for m in ("glm", "deepseek", "minimax"):
        rows = whatif.scenarios(m, None, 2.0 if m == "deepseek" else None)
        assert rows and all(ms > 0 and note for _, ms, note in rows)
