"""Which rows the decoder half of a deferred (encoder-only) prefill has to run, derived from the sliding-window cascade.

The checkpoint is a causal encoder-decoder (layers 0 to 19 encoder, 20 to 39 decoder, `kv_source_layer_ids [2, 8, 14, 20]`): every layer after 20 reads the
global KV that layer 20 and the encoder sources wrote from the encoder's output, so a prompt token's decoder-layer hidden state is read again only through
the sliding window of the next 128 positions at each later layer (docs/DEEPINFER-LEVER-LANE.md, DI-1). A prefill may therefore run the encoder over every row and
the decoder over only the last `TAIL_ROWS` rows, starting each decoder layer with an empty window ring, and still end in the state a full prefill ends in.

The bound: output rows of decoder layer j are exact from row `start + WINDOW_CONTEXT * j` on (a row needs the 127 rows before it at every layer, and the rows before
`start` are missing), so layer 38's output is exact from `start + 127 * 19`, and the layer-39 window needs rows `n - 128` on: `start <= n - (127 * 19 + 128)`.
"""

from __future__ import annotations

WINDOW = 128
WINDOW_CONTEXT = WINDOW - 1                 # rows of history a sliding-window row needs
ENCODER_LAYERS = 20                         # layers 0 to 19 run on every row; 20 to 39 are the decoder half
N_LAYERS = 40
DECODER_LAYERS = N_LAYERS - ENCODER_LAYERS
# Decoder layers whose window outputs a later layer reads: 20..38 (19 hops) and the last layer's own window.
TAIL_ROWS = WINDOW_CONTEXT * (DECODER_LAYERS - 1) + WINDOW      # 127 * 19 + 128 = 2,541


def tail_start(deferred_start: int, end: int, tail_rows: int = TAIL_ROWS) -> tuple[int, bool]:
    """First row the decoder half runs and whether it starts from empty window rings.

    `deferred_start` is the position at which the decoder state was last exact (a snapshot, a reset, the previous finish), `end` the prompt position now. When the
    rows since `deferred_start` fit in the tail the decoder runs all of them with the real windows (nothing is truncated); otherwise only the last `tail_rows`
    rows run, with zeroed rings."""
    if end <= deferred_start:
        raise ValueError(f"no rows to finish: start {deferred_start}, end {end}")
    if end - deferred_start <= tail_rows:
        return deferred_start, False
    return end - tail_rows, True


def retained_rows(deferred_start: int, end: int, tail_rows: int = TAIL_ROWS) -> int:
    """Encoder-output rows the deferral has to keep at `end`: the rows `tail_start` will run."""
    return end - tail_start(deferred_start, end, tail_rows)[0]


def exact_from(start: int, layers: int = DECODER_LAYERS) -> list[int]:
    """The oracle behind TAIL_ROWS, written as a propagation, not as the closed form: the first row whose output is exact after each decoder layer when the
    layer-20 input is exact on every row from `start` and every decoder layer begins with an empty window ring."""
    first = start                              # input of the first decoder layer: exact from `start`
    out = []
    for _ in range(layers):
        first = first + WINDOW_CONTEXT         # a row needs its 127 predecessors, which exist (and are exact) only from `first`
        out.append(first)
    return out
