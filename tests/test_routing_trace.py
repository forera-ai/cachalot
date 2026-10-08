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


def test_record_next_positions_group_a_decode_token_across_layers(tmp_path):
    import numpy as np

    from cachalot.metrics.routing_trace import RoutingTracer, load_trace

    t = RoutingTracer()
    t.record_next("prefill", 3, np.arange(12).reshape(3, 4))  # 3 tokens, top-4
    for _ in range(2):  # two decode tokens through two layers
        for layer in (3, 4):
            t.record_next("decode", layer, [1, 2, 3, 4])
    arrays, _ = load_trace(t.save(tmp_path / "t.npz"))
    dec = arrays["phase"] == 1
    assert arrays["position"][~dec].tolist() == [0, 1, 2]
    assert arrays["position"][dec].tolist() == [0, 0, 1, 1]  # rows of one token share a position
    assert arrays["experts"].shape == (7, 4) and "weights" not in arrays


def test_streaming_switch_traces_routes_and_honours_forced_phase():
    import numpy as np

    from cachalot.glm.experts import StreamingSwitchGLU
    from cachalot.metrics.routing_trace import RoutingTracer

    from types import SimpleNamespace

    mod = SimpleNamespace(_layer=5, tracer=RoutingTracer())  # _trace only reads these two
    StreamingSwitchGLU._trace(mod, np.arange(8, dtype=np.int32), 2, 4)  # two tokens: prefill
    StreamingSwitchGLU._trace(mod, np.arange(4, dtype=np.int32), 1, 4)  # one token: decode
    mod.tracer.forced_phase = "prefill"
    StreamingSwitchGLU._trace(mod, np.arange(4, dtype=np.int32), 1, 4)  # a one-token prefill chunk stays prefill
    assert mod.tracer.arrays()["phase"].tolist() == [0, 0, 1, 0]


def test_tap_gate_leaves_weights_in_the_sink_and_changes_nothing_else():
    import mlx.core as mx
    import mlx.nn as nn

    from cachalot.glm.experts import ScoreSink, tap_gate, untap_gate

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.ones((4, 3))

        def __call__(self, x):
            return mx.array([[1, 2]]), mx.array([[0.25, 0.75]])

    gate = Gate()
    before = sorted(k for k, _ in nn.utils.tree_flatten(gate.parameters()))
    sink = ScoreSink()
    tap_gate(gate, sink)
    inds, w = gate(mx.zeros((1, 3)))
    assert inds.tolist() == [[1, 2]] and sink.scores is w
    assert type(gate).__name__ == "Gate" and sorted(k for k, _ in nn.utils.tree_flatten(gate.parameters())) == before
    assert not any(isinstance(m, ScoreSink) for m in gate.modules())
    untap_gate(gate)
    assert type(gate) is Gate and not hasattr(gate, "_sink")


def test_weights_and_predictions_are_recorded_and_joined_for_layers_starting_at_three(tmp_path):
    import numpy as np
    from types import SimpleNamespace

    from cachalot.glm.experts import StreamingSwitchGLU
    from cachalot.metrics.routing_trace import RoutingTracer, load_trace, predicted_used_mask

    t = RoutingTracer()
    layers = (3, 4, 5)
    for token in range(2):
        for layer in layers:
            mod = SimpleNamespace(_layer=layer, tracer=t)
            routes = np.array([layer, 10 + token, 20, 30], dtype=np.int32)
            w = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
            # predict the next layer: its routes will be [layer + 1, 10 + token, 99, 98] in this test's convention
            pred = None if layer == 5 else (np.array([layer + 1, 10 + token, 99], np.int32), np.array([0.3, 0.5, 0.2], np.float32))
            StreamingSwitchGLU._trace(mod, routes, 1, 4, w, pred)
    arrays, _ = load_trace(t.save(tmp_path / "t.npz"))
    assert arrays["weights"].shape == arrays["experts"].shape == (6, 4)
    assert arrays["pred_source"].tolist() == [3, 4, 3, 4] and arrays["pred_target"].tolist() == [4, 5, 4, 5]
    assert arrays["pred_position"].tolist() == [0, 0, 1, 1]  # the position of the token being routed
    # predictions are written best first: layer 3 predicted [10, 4, 99]; layer 4 actually routed [4, 10, 20, 30]
    mask = predicted_used_mask(arrays)
    assert mask[0].tolist() == [True, True, False]
    assert mask.shape == (4, 3) and mask[2].tolist() == [True, True, False]  # token 1 joins its own token, not token 0


def test_switch_call_traces_decode_and_prefill_through_the_real_call_path():
    """StreamingSwitchGLU.__call__ with a stub store: routes, tapped weights and the predictor's set reach the tracer
    in the one sync, for a decode token and for a prefill chunk of one token (forced phase) and of three."""
    import mlx.core as mx
    import numpy as np
    from types import SimpleNamespace

    from cachalot.glm import experts
    from cachalot.glm.experts import ScoreSink, StreamingSwitchGLU
    from cachalot.metrics.routing_trace import RoutingTracer

    dim, k = 8, 2
    index = {(layer, e): SimpleNamespace(layer=layer, expert=e) for layer in (3, 4) for e in range(16)}

    class Store:
        def get_many(self, entries, **kw):
            return [SimpleNamespace(slot=None) for _ in entries]

    mod = StreamingSwitchGLU(3, Store(), index, None, None)
    mod._expert_out = lambda x, slot: mx.zeros((1, dim))
    tracer, sink = RoutingTracer(), ScoreSink()
    mod.tracer, mod.score_sink = tracer, sink
    mod.predict = lambda x, topk: (mx.array([[5, 6]]), mx.array([[0.2, 0.9]]))

    sink.scores = mx.array([[0.7, 0.3]])
    mod(mx.zeros((1, dim)), mx.array([[1, 2]]))  # one decode token
    assert experts.PREDICT_TOPK > 0
    a = tracer.arrays()
    assert a["phase"].tolist() == [1] and a["experts"].tolist() == [[1, 2]]
    assert np.allclose(a["weights"], [[0.7, 0.3]])
    assert a["pred_experts"].tolist() == [[6, 5]] and np.allclose(a["pred_weights"], [[0.9, 0.2]])  # best first
    assert a["pred_source"].tolist() == [3] and a["pred_target"].tolist() == [4] and a["pred_position"].tolist() == [0]

    tracer.forced_phase = "prefill"  # a one-token prefill chunk: routed and weighted, no decode prediction
    sink.scores = mx.array([[0.6, 0.4]])
    mod(mx.zeros((1, dim)), mx.array([[3, 4]]))
    assert tracer.arrays()["phase"].tolist() == [1, 0] and tracer.predicted_records == 1


def test_decode_miss_budget_drops_the_lightest_misses_and_rescales_the_kept_outputs():
    """HANDOFF 18.106: with a budget, the store is asked for at most that many misses ranked by router weight, a dropped
    expert contributes zero and the kept ones are scaled by total / kept router weight. No budget: untouched."""
    import mlx.core as mx
    import numpy as np
    from types import SimpleNamespace

    from cachalot.glm.experts import ScoreSink, StreamingSwitchGLU

    dim = 4
    index = {(3, e): SimpleNamespace(layer=3, expert=e) for e in range(16)}
    seen = {}

    class Store:
        def get_many(self, entries, *, max_misses=None, priorities=None, **kw):
            seen.update(max_misses=max_misses, priorities=priorities)
            out = [SimpleNamespace(slot=e.expert) for e in entries]
            resident = {1}  # expert 1 is a hit; the rest are misses
            misses = [i for i, e in enumerate(entries) if e.expert not in resident]
            if max_misses is not None and len(misses) > max_misses:
                keep = set(sorted(misses, key=lambda i: -priorities[i])[:max_misses])
                for i in misses:
                    if i not in keep:
                        out[i] = None
            return out

    mod = StreamingSwitchGLU(3, Store(), index, None, None)
    mod._expert_out = lambda x, slot: mx.full((1, dim), float(slot))
    sink = ScoreSink()
    mod.score_sink = sink
    sink.scores = mx.array([[0.4, 0.3, 0.2, 0.1]])
    inds = mx.array([[1, 2, 3, 4]])

    exact = np.array(mod(mx.zeros((1, dim)), inds))
    assert seen["max_misses"] is None and exact[0, :, 0].tolist() == [1, 2, 3, 4]

    mod.miss_budget = 1  # of three misses (2, 3, 4) keep the heaviest: expert 2 (0.3)
    y = np.array(mod(mx.zeros((1, dim)), inds))
    assert seen["max_misses"] == 1 and np.allclose(seen["priorities"], [0.4, 0.3, 0.2, 0.1])
    factor = 1.0 / (0.4 + 0.3)  # kept router mass is experts 1 and 2
    assert np.allclose(y[0, :, 0], [1 * factor, 2 * factor, 0, 0])
