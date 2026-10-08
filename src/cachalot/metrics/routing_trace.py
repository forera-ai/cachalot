from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PHASE_PREFILL = 0
PHASE_DECODE = 1

_PHASE_IDS = {"prefill": PHASE_PREFILL, "decode": PHASE_DECODE}


@dataclass
class RoutingTracer:
    """
    Records routed-expert selections for offline cache-policy analysis.

    One record per (token position, layer). Storage is columnar so a
    multi-thousand-token session costs a few megabytes.

    Enabled explicitly by the caller; the hot path pays one attribute check
    when no tracer is installed.
    """

    _phase: list[np.ndarray] = field(default_factory=list)
    _layer: list[np.ndarray] = field(default_factory=list)
    _position: list[np.ndarray] = field(default_factory=list)
    _experts: list[np.ndarray] = field(default_factory=list)
    _weights: list[np.ndarray] = field(default_factory=list)
    _segments: list[dict] = field(default_factory=list)
    _count: int = 0
    # Decode-time next-layer predictions (what the prefetch would submit), one
    # record per (source layer, target layer, token). Empty unless the model
    # path calls record_predicted, so a trace without them is unchanged.
    _pred_source: list[np.ndarray] = field(default_factory=list)
    _pred_target: list[np.ndarray] = field(default_factory=list)
    _pred_position: list[np.ndarray] = field(default_factory=list)
    _pred_experts: list[np.ndarray] = field(default_factory=list)
    _pred_weights: list[np.ndarray] = field(default_factory=list)
    # Position of the decode token being traced; set by the runtime at the
    # start of each decode token so predictions can be joined to routes.
    decode_position: int = -1
    _next_pos: dict = field(default_factory=dict)
    # set by a model's prefill so that a one-token prefill chunk is not recorded as a decode token
    forced_phase: str | None = None

    def record(
        self,
        phase: str,
        layer: int,
        start_pos: int,
        indices,
        weights=None,
    ) -> None:
        """
        indices: [n_tokens, topk] or [topk] array-like of expert ids for
        consecutive positions starting at start_pos.
        weights: optional router weights, same shape as indices. Stored only
        when every record carries them, so a trace is either fully weighted
        or has no weights at all.
        """
        arr = np.asarray(indices, dtype=np.int16)

        if arr.ndim == 1:
            arr = arr[None, :]

        if weights is not None:
            w = np.asarray(weights, dtype=np.float32)

            if w.ndim == 1:
                w = w[None, :]

            if w.shape != arr.shape:
                raise ValueError(f"weights shape {w.shape} != indices shape {arr.shape}")

            self._weights.append(w)

        n_tokens = arr.shape[0]

        self._phase.append(
            np.full(n_tokens, _PHASE_IDS[phase], dtype=np.int8)
        )
        self._layer.append(np.full(n_tokens, layer, dtype=np.int16))
        self._position.append(
            np.arange(start_pos, start_pos + n_tokens, dtype=np.int32)
        )
        self._experts.append(arr)
        self._count += n_tokens

    def last_position(self, phase: str, layer: int) -> int:
        """Position of the token `record_next` last recorded for (phase, layer); -1 before the first."""
        return self._next_pos.get((phase, layer), 0) - 1

    def record_next(self, phase: str, layer: int, indices, weights=None) -> None:
        """
        `record` for a model path that does not know the sequence position (GLM and MiniMax, which route
        inside the expert module). Positions are a per-(phase, layer) counter: every layer that routes a token
        advances its own, so rows of one decode token share a position, which is what `cache_sim.py` groups by.
        Positions are unique within a run but do not restart with a request; the trace has no router weights.
        """
        arr = np.asarray(indices, dtype=np.int16)
        arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim != 1 else arr[None, :]
        key = (phase, layer)
        start = self._next_pos.get(key, 0)
        self._next_pos[key] = start + arr.shape[0]
        self.record(phase, layer, start, arr, None if weights is None else np.asarray(weights).reshape(arr.shape))

    def record_predicted(
        self,
        source_layer: int,
        target_layer: int,
        indices,
        weights,
        position: int | None = None,
    ) -> None:
        """
        One decode-time prediction: the experts that `source_layer`'s input
        scored highest under `target_layer`'s router, with the predictor's own
        router weights (the confidence a precision-gated prefetch would use).
        Shapes are [topk]. Join to the target layer's decode route by
        (position, target_layer) to learn which predicted experts were used.
        """
        idx = np.asarray(indices, dtype=np.int16).reshape(-1)
        w = np.asarray(weights, dtype=np.float32).reshape(-1)

        if w.shape != idx.shape:
            raise ValueError(f"weights shape {w.shape} != indices shape {idx.shape}")

        pos = self.decode_position if position is None else position

        self._pred_source.append(np.array([source_layer], dtype=np.int16))
        self._pred_target.append(np.array([target_layer], dtype=np.int16))
        self._pred_position.append(np.array([pos], dtype=np.int32))
        self._pred_experts.append(idx[None, :])
        self._pred_weights.append(w[None, :])

    @property
    def predicted_records(self) -> int:
        return len(self._pred_experts)

    def mark(self, label: str, **meta) -> None:
        """Annotate a boundary (new turn, new prompt) at the current record count."""
        self._segments.append({"label": label, "at": self._count, **meta})

    @property
    def records(self) -> int:
        return self._count

    def arrays(self) -> dict[str, np.ndarray]:
        if not self._experts:
            return {
                "phase": np.zeros(0, np.int8),
                "layer": np.zeros(0, np.int16),
                "position": np.zeros(0, np.int32),
                "experts": np.zeros((0, 0), np.int16),
            }

        out = {
            "phase": np.concatenate(self._phase),
            "layer": np.concatenate(self._layer),
            "position": np.concatenate(self._position),
            "experts": np.concatenate(self._experts, axis=0),
        }

        if self._weights and len(self._weights) == len(self._experts):
            out["weights"] = np.concatenate(self._weights, axis=0)

        out.update(self._predicted_arrays())

        return out

    def _predicted_arrays(self) -> dict[str, np.ndarray]:
        if not self._pred_experts:
            return {}

        return {
            "pred_source": np.concatenate(self._pred_source),
            "pred_target": np.concatenate(self._pred_target),
            "pred_position": np.concatenate(self._pred_position),
            "pred_experts": np.concatenate(self._pred_experts, axis=0),
            "pred_weights": np.concatenate(self._pred_weights, axis=0),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            segments=np.array(json.dumps(self._segments)),
            **self.arrays(),
        )

        return path


def load_trace(path: str | Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    with np.load(path) as data:
        arrays = {
            key: data[key]
            for key in ("phase", "layer", "position", "experts")
        }

        if "weights" in data.files:
            arrays["weights"] = data["weights"]

        for key in ("pred_source", "pred_target", "pred_position", "pred_experts", "pred_weights"):
            if key in data.files:
                arrays[key] = data[key]
        segments = json.loads(str(data["segments"]))

    return arrays, segments


def _token_ordinals(layers: np.ndarray) -> np.ndarray:
    """0, 1, 2 ... per decode token for records in generation order: a token ends when the layer stops ascending."""
    if layers.size == 0:
        return np.zeros(0, np.int64)
    new = np.concatenate([[True], layers[1:] <= layers[:-1]])
    return np.cumsum(new) - 1


def predicted_used_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """
    For each predicted record, a bool [n, topk] mask: was that predicted expert
    among the experts the target layer then routed to, in the same decode token.

    Decode positions restart with every request, so (position, layer) is not a
    unique key across a multi-request trace. Records are written in generation
    order and the layers of one token ascend, so a new decode token starts wherever
    a record's layer is not above the previous one's (DeepSeek's start at layer 0,
    GLM's and MiniMax's at their first MoE layer, 3); the token ordinal is the
    join key, for routes and for predictions alike.
    """
    decode = np.where(arrays["phase"] == PHASE_DECODE)[0]
    layers = arrays["layer"][decode]
    route_token = _token_ordinals(layers)
    routed = {
        (int(t), int(l)): set(e.tolist())
        for t, l, e in zip(route_token, layers, arrays["experts"][decode])
    }

    pred_token = _token_ordinals(arrays["pred_source"])
    mask = np.zeros(arrays["pred_experts"].shape, dtype=bool)

    for i, (t, target, experts) in enumerate(
        zip(pred_token, arrays["pred_target"], arrays["pred_experts"])
    ):
        used = routed.get((int(t), int(target)), set())
        mask[i] = [int(e) in used for e in experts]

    return mask
