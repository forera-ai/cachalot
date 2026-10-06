import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import batch_union as bu  # noqa: E402
from simulate_policies import N_LAYERS  # noqa: E402


def _trace(streams):
    """streams: list of lists of tokens; a token is a list of experts, the same for every layer."""
    phase, layer, position, experts = [], [], [], []
    pos = 0
    for s in streams:
        phase.append(0); layer.append(0); position.append(pos); experts.append([0, 1])  # a tiny prefill
        pos += 1
        for tok in s:
            for la in range(N_LAYERS):
                phase.append(1); layer.append(la); position.append(pos); experts.append(tok)
            pos += 1
    return {"phase": np.array(phase, np.uint8), "layer": np.array(layer), "position": np.array(position),
            "experts": np.array(experts)}


def test_split_and_lanes():
    arrays = _trace([[[1, 2], [3, 4]], [[1, 2]]])
    first, streams = bu.split_streams(arrays)
    assert first == (0, 1) and [len(s) for s in streams] == [2, 1]
    assert [len(lane) for lane in bu.lanes_of(streams, 2)] == [2, 1]
    assert [len(lane) for lane in bu.lanes_of(streams, 1)] == [3]


def test_identical_streams_halve_requests_at_batch_two():
    arrays = _trace([[[5, 6], [7, 8]], [[5, 6], [7, 8]]])
    one = bu.replay(arrays, slots=10_000, batch=1)
    two = bu.replay(arrays, slots=10_000, batch=2)
    assert one["requests_per_token"] == 2 * N_LAYERS / 1
    assert two["union_share"] == 0.5 and two["requests_per_token"] == N_LAYERS
    # with room for everything, the second stream hits what the first read: a batch saves requests, not misses
    assert two["misses_per_token"] == one["misses_per_token"]
