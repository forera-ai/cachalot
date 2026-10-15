import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import floor_roofline as fr  # noqa: E402


def test_rate_matches_table_points_and_interpolates():
    assert fr.rate_at("fp8", 3 * fr.MIB) == 0.37
    assert fr.rate_at("q2", 0.5 * fr.MIB) == 0.31       # below the first point: clamped
    assert fr.rate_at("bf16", 1000 * fr.MIB) == 0.90    # above the last point: clamped
    assert 0.37 < fr.rate_at("fp8", 3.5 * fr.MIB) < 0.46


def test_row_bandwidth_time_is_capped_by_the_row():
    r = fr.Row("x", ms=0.01, count=1, launches=[("bf16", 1e9)])
    assert r.bw_ms == 0.01


def test_expert_launch_bytes_are_a_9_95_mb_record_times_six():
    rows = fr.build_rows(fr.DEFAULT_TIMES)
    experts = next(r for r in rows if r.name.startswith("routed experts"))
    assert abs(experts.bytes - 6 * 9.49 * fr.MIB) < 1


def test_no_row_reads_faster_than_the_chip():
    for r in fr.build_rows(fr.DEFAULT_TIMES):
        if r.bytes:
            assert r.bytes / (r.ms * 1e-3) / 1e9 <= fr.PEAK_GBPS * 1.02, r.name


def test_token_bytes_cover_the_headers_total():
    s = fr.summarize(fr.build_rows(fr.DEFAULT_TIMES))
    assert 11.0 < s["token_bytes_gb"] < 13.0


def test_context_adds_little_and_never_removes_bytes():
    base = fr.summarize(fr.build_rows(fr.DEFAULT_TIMES))
    long = fr.summarize(fr.build_rows(fr.DEFAULT_TIMES, 22000))
    assert 0 < long["token_bytes_gb"] - base["token_bytes_gb"] < 0.2


def test_saving_is_half_the_bandwidth_bound_sum():
    s = fr.summarize(fr.build_rows(fr.DEFAULT_TIMES))
    assert abs(s["saving_2x_ms"] - s["bw_ms"] / 2) < 1e-9


def test_measured_variants_order_and_cap(tmp_path):
    import json

    def run(scale):
        rows = []
        for arm, nbytes, ms, sum_ms in (("wq_a", 6.6e6, 0.04, 0.03), ("wkv", 2.6e6, 0.03, 0.02), ("wq_b", 42e6, 0.10, 0.087),
                                        ("wo_b", 42e6, 0.115, 0.087), ("wo_a (bf16)", 67e6, 0.106, 0.122)):
            rows.append({"arm": arm, "bytes": nbytes, "ms_best": ms * scale})
            rows.append({"arm": arm.split(" ")[0] + " sum", "bytes": nbytes, "ms_best": sum_ms * scale})
        return {"rows": rows}

    paths = []
    for i, sc in enumerate((1.0, 1.0)):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(run(sc)))
        paths.append(str(p))
    v = fr.projection_variants(paths)
    assert v["peak77"] <= v["sum"] <= v["upper"]
    rows = fr.build_rows(fr.DEFAULT_TIMES)
    out = fr.measured_attention_summary(rows, v)
    assert out["upper"] >= out["sum"] >= out["peak77"]
    assert abs(out["table"] - fr.summarize(rows)["bw_ms"]) < 1e-9
