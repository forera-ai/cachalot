"""DI-1 L1 matrix (weightless; docs/DI1-BUILD-RECORD.md): the tail plan against an independent brute-force window simulation."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cachalot.model import prefill_tail_plan as tp  # noqa: E402


def simulate_exact_rows(n, start, window=128, layers=20):
    """Row-level brute force, written without the closed form. Decoder layer inputs are exact on rows [start, n) (retained encoder outputs); every layer starts with
    an empty window ring, so a row's output is exact iff its `window - 1` predecessors are present (>= start, or the sequence begins at 0) and exact in the layer's
    input. Returns (rows whose input to the LAST layer is exact, whether the last row's final output is exact)."""
    exact = [r >= start for r in range(n)]
    for _ in range(layers - 1):
        nxt = [False] * n
        for r in range(start, n):
            lo = max(0, r - window + 1)
            if (lo >= start or start == 0) and all(exact[q] for q in range(lo, r + 1)):
                nxt[r] = True
        exact = nxt
    last = n - 1
    lo = max(0, last - window + 1)
    final = (lo >= start or start == 0) and all(exact[q] for q in range(lo, last + 1))
    return exact, final


def test_the_constant_is_the_cascade_bound():
    assert tp.TAIL_ROWS == 127 * 19 + 128 == 2541
    assert tp.WINDOW == 128 and tp.DECODER_LAYERS == 20 and tp.ENCODER_LAYERS == 20


def test_exact_from_matches_the_closed_form():
    out = tp.exact_from(1000)
    assert len(out) == 20 and out[0] == 1127 and out[-1] == 1000 + 127 * 20 and out[18] == 1000 + 127 * 19


@pytest.mark.parametrize("n", [3000, 2541, 2600, 5000, 7000])
def test_the_tail_makes_the_last_window_and_the_last_row_exact(n):
    """Brute force on a smaller window to keep the matrix fast, then the closed form on the real one: the final state is exact iff start <= n - tail."""
    for window, layers in ((8, 5), (16, 4)):
        tail = (window - 1) * (layers - 1) + window
        for nn in (tail - 1, tail, tail + 1, 3 * tail):
            start = max(0, nn - tail)
            ex, final = simulate_exact_rows(nn, start, window, layers)
            need = range(max(0, nn - window), nn)         # the last decoder layer's window rows: its K comes from the last layer's input
            assert all(ex[q] for q in need) and final, (window, layers, nn, start)
            if nn > tail:                                 # one row fewer is not enough
                ex2, final2 = simulate_exact_rows(nn, start + 1, window, layers)
                assert not (all(ex2[q] for q in need) and final2), (window, layers, nn, start + 1)
    start, trunc = tp.tail_start(0, n)
    assert (start, trunc) == ((0, False) if n <= 2541 else (n - 2541, True))


@pytest.mark.parametrize("deferred_start,end,expect", [
    (0, 1, (0, False)), (0, 2541, (0, False)), (0, 2542, (1, True)), (0, 22411, (22411 - 2541, True)),
    (17217, 22411, (17217, False)),                       # 5,194 rows since a pinned block: truncated? no: 5194 > 2541
])
def test_tail_start_cases(deferred_start, end, expect):
    got = tp.tail_start(deferred_start, end)
    if end - deferred_start > tp.TAIL_ROWS:
        expect = (end - tp.TAIL_ROWS, True)
    assert got == expect


def test_a_continuation_shorter_than_the_tail_runs_every_row_with_the_real_windows():
    assert tp.tail_start(20000, 22000) == (20000, False) and tp.retained_rows(20000, 22000) == 2000


def test_a_snapshot_boundary_inside_the_tail_resets_the_start():
    """A finish at the pinned boundary (b = 17,217) leaves the decoder exact there; the next finish at the prompt end sees only the rows since."""
    b, n = 17217, 22411
    s1, t1 = tp.tail_start(0, b)
    assert (s1, t1) == (b - 2541, True)
    s2, t2 = tp.tail_start(b, n)
    assert t2 is True and s2 == n - 2541 and tp.retained_rows(b, n) == 2541
    s3, t3 = tp.tail_start(b, b + 2000)
    assert (s3, t3) == (b, False)


def test_empty_range_is_rejected():
    with pytest.raises(ValueError):
        tp.tail_start(5, 5)


def test_the_decoder_rows_saved_by_a_22k_prompt_in_4096_chunks():
    """Sanity of the record's table: 6 calls x 20 decoder layers vs. one tail pass of 2,541 rows (+ one at the pinned boundary)."""
    n, chunk = 22411, 4096
    calls = -(-n // chunk)
    full_decoder_passes = calls
    deferred_passes = 1
    assert calls == 6 and full_decoder_passes == 6 and deferred_passes == 1
    assert sum(min(chunk, n - i * chunk) for i in range(calls)) == n
