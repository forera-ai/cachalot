import pytest

from cachalot.cli import _decode_miss_budget_from_env as parse


def test_unset_empty_and_negative_are_off():
    assert parse({}) is None
    assert parse({"CACHALOT_DECODE_MISS_BUDGET": ""}) is None
    assert parse({"CACHALOT_DECODE_MISS_BUDGET": "  "}) is None
    assert parse({"CACHALOT_DECODE_MISS_BUDGET": "-1"}) is None


def test_non_negative_integers_are_the_cap():
    assert parse({"CACHALOT_DECODE_MISS_BUDGET": "0"}) == 0
    assert parse({"CACHALOT_DECODE_MISS_BUDGET": " 2 "}) == 2


def test_garbage_is_an_error_not_silently_off():
    with pytest.raises(SystemExit):
        parse({"CACHALOT_DECODE_MISS_BUDGET": "two"})
