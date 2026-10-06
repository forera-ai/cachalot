import numpy as np

from cachalot.metrics.routing_trace import PHASE_DECODE, PHASE_PREFILL, RoutingTracer, load_trace


def test_tracer_records_prefill_and_decode(tmp_path):
    t = RoutingTracer()
    t.mark("turn1")
    t.record("prefill", 3, 0, np.array([[1, 2, 3], [4, 5, 6]]))
    t.record("decode", 3, 2, np.array([7, 8, 9]))
    assert t.records == 3

    path = t.save(tmp_path / "x.trace.npz")
    arrays, segments = load_trace(path)

    assert arrays["phase"].tolist() == [PHASE_PREFILL, PHASE_PREFILL, PHASE_DECODE]
    assert arrays["layer"].tolist() == [3, 3, 3]
    assert arrays["position"].tolist() == [0, 1, 2]
    assert arrays["experts"].tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert segments == [{"label": "turn1", "at": 0}]


def test_empty_tracer_saves(tmp_path):
    t = RoutingTracer()
    arrays, segments = load_trace(t.save(tmp_path / "e.trace.npz"))
    assert arrays["experts"].shape[0] == 0
    assert segments == []


def test_tracer_saves_weights_only_when_every_record_has_them(tmp_path):
    t = RoutingTracer()
    t.record("decode", 0, 0, np.array([1, 2]), np.array([0.7, 0.3]))
    t.record("decode", 1, 0, np.array([3, 4]), np.array([0.6, 0.4]))
    arrays, _ = load_trace(t.save(tmp_path / "w.trace.npz"))
    assert np.allclose(arrays["weights"], [[0.7, 0.3], [0.6, 0.4]])

    t = RoutingTracer()
    t.record("decode", 0, 0, np.array([1, 2]), np.array([0.7, 0.3]))
    t.record("decode", 1, 0, np.array([3, 4]))
    arrays, _ = load_trace(t.save(tmp_path / "m.trace.npz"))
    assert "weights" not in arrays


def test_tracer_rejects_mismatched_weights():
    import pytest

    t = RoutingTracer()
    with pytest.raises(ValueError):
        t.record("decode", 0, 0, np.array([1, 2, 3]), np.array([0.5, 0.5]))


def test_tracer_saves_predicted_sets_and_old_traces_load(tmp_path):
    t = RoutingTracer()
    t.decode_position = 41
    t.record("decode", 5, 41, np.array([1, 2, 3]), np.array([0.5, 0.3, 0.2]))
    t.record_predicted(4, 5, [1, 2, 9], [0.4, 0.35, 0.25])
    t.record_predicted(5, 6, [7, 8, 9], [0.5, 0.3, 0.2], position=42)
    assert t.predicted_records == 2

    arrays, _ = load_trace(t.save(tmp_path / "p.trace.npz"))
    assert arrays["pred_source"].tolist() == [4, 5]
    assert arrays["pred_target"].tolist() == [5, 6]
    assert arrays["pred_position"].tolist() == [41, 42]
    assert arrays["pred_experts"].tolist() == [[1, 2, 9], [7, 8, 9]]
    assert np.allclose(arrays["pred_weights"], [[0.4, 0.35, 0.25], [0.5, 0.3, 0.2]])

    plain = RoutingTracer()
    plain.record("decode", 0, 0, np.array([1, 2]))
    arrays, _ = load_trace(plain.save(tmp_path / "n.trace.npz"))
    assert not any(k.startswith("pred_") for k in arrays)


def test_tracer_rejects_mismatched_predicted_weights():
    import pytest

    with pytest.raises(ValueError):
        RoutingTracer().record_predicted(0, 1, [1, 2, 3], [0.5, 0.5])


def test_predicted_used_mask_joins_by_token_across_requests(tmp_path):
    from cachalot.metrics.routing_trace import predicted_used_mask

    t = RoutingTracer()
    # Two requests whose decode positions both start at 10: position alone is ambiguous.
    for request_experts in ([1, 2], [5, 6]):
        t.decode_position = 10
        t.record("decode", 0, 10, np.array(request_experts))
        t.record_predicted(0, 1, [request_experts[0], 99], [0.6, 0.4])
        t.record("decode", 1, 10, np.array(request_experts))

    arrays, _ = load_trace(t.save(tmp_path / "j.trace.npz"))
    assert predicted_used_mask(arrays).tolist() == [[True, False], [True, False]]
