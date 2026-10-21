"""DI-4 grain replay (Mac only; docs/DEEPINFER-LEVER-LANE.md): the pure helpers on planted rows. The tokenizer and the cache are not loaded."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import prefix_grain_replay as g  # noqa: E402


def test_common_prefix_and_grain_hit():
    assert g.common_prefix([1, 2, 3], [1, 2, 9, 4]) == 2 and g.common_prefix([], [1]) == 0
    assert g.grain_hit(0) == 0 and g.grain_hit(255) == 0 and g.grain_hit(256) == 128 and g.grain_hit(22420) == 22272


def test_summary_counts_only_divergence_inside_a_chunk():
    rows = [{"prompt": 22411, "reused": 0, "shared": 0}, {"prompt": 22439, "reused": 22420, "shared": 22420},
            {"prompt": 30000, "reused": 20480, "shared": 24500}]            # the last diverges inside a chunk: the cache stopped at 20,480
    s = g.summarize(rows)
    assert s["diverged_in_chunk"] == 1 and s["extra_tokens"] == [24320 - 20480] and s["requests"] == 3
    assert s["grain_prefilled"] == s["real_prefilled"] - (24320 - 20480)
    assert g.summarize([])["diverged_share"] is None
