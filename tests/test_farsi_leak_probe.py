"""benchmarks/farsi_leak_probe.py helpers: the Farsi section start and the KL estimate (no model, no dumps)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import farsi_leak_probe as probe  # noqa: E402


class CharTokenizer:
    """One token per character: id = code point."""

    def decode(self, ids, **_):
        return "".join(chr(i) for i in ids)

    def __len__(self):
        return 0x0700


def ids(text):
    return [ord(c) for c in text]


def test_farsi_start_is_the_line_where_persian_begins():
    tok = CharTokenizer()
    reply = ids("English story.\n\n**فارسی**\n\nسلام")
    s = probe.farsi_start(tok, reply)
    assert tok.decode(reply[s:]).startswith("**فارسی**")
    assert probe.farsi_start(tok, ids("only English\n")) is None


def test_latin_mask_marks_two_ascii_letters():
    mask = probe.latin_mask(CharTokenizer())
    assert not mask[ord("a")] and not mask[ord("س")]  # single characters never hold two letters


def test_kl_on_union_is_zero_for_equal_and_positive_otherwise():
    i = np.array([1, 2, 3])
    lp = np.log(np.array([0.7, 0.2, 0.1]))
    assert probe.kl_on_union(i, lp, i, lp) == pytest.approx(0.0, abs=1e-12)
    lq = np.log(np.array([0.5, 0.3, 0.2]))
    assert probe.kl_on_union(i, lp, i, lq) > 0
    # Q missing an id: floored at Q's smallest kept value (a truncated estimate, finite but not bounded below by 0)
    assert np.isfinite(probe.kl_on_union(i, lp, np.array([1, 2]), np.log(np.array([0.7, 0.2]))))
