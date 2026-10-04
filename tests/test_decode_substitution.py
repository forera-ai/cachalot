import numpy as np

from cachalot.model.decode_substitution import plan_substitution, ranked_candidates


def test_ranked_candidates_skip_chosen_and_keep_order():
    sel = np.array([0.1, 0.9, 0.5, 0.8, 0.7, 0.2])
    assert ranked_candidates(sel, [1, 3], 2) == [4, 2]
    assert ranked_candidates(sel, [1, 3], 10) == [4, 2, 5, 0]


def test_weak_miss_is_replaced_by_best_resident_candidate():
    ids, n = plan_substitution([10, 11, 12], [0.6, 0.3, 0.1], [20, 21, 22], lambda e: e in (10, 21, 22), tau=0.2)
    assert ids == [10, 11, 21] and n == 1


def test_strong_misses_and_resident_experts_are_untouched():
    ids, n = plan_substitution([1, 2, 3], [0.5, 0.3, 0.2], [7, 8], lambda e: e in (1, 7, 8), tau=0.1)
    assert ids == [1, 2, 3] and n == 0


def test_no_resident_candidate_means_the_miss_stays():
    ids, n = plan_substitution([1, 2], [0.9, 0.1], [5, 6], lambda e: e == 1, tau=0.5)
    assert ids == [1, 2] and n == 0


def test_each_candidate_is_used_once_weakest_miss_first():
    ids, n = plan_substitution([1, 2, 3], [0.5, 0.3, 0.2], [8], lambda e: e == 8, tau=1.0)
    assert ids == [1, 2, 8] and n == 1  # only one spare: the weakest miss (0.2) takes it


def test_candidate_already_routed_is_not_a_substitute():
    ids, n = plan_substitution([1, 2], [0.9, 0.1], [1, 5], lambda e: e in (1, 5), tau=0.5)
    assert ids == [1, 5] and n == 1


def test_parse_decode_arm():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
    from pareto import parse_decode_arm as parse

    assert parse("exact") == {"miss_budget": None, "substitute_tau": None}
    assert parse("0") == {"miss_budget": 0, "substitute_tau": None}
    assert parse("s0.25") == {"miss_budget": None, "substitute_tau": 0.25}
    assert parse("s1.0b0") == {"miss_budget": 0, "substitute_tau": 1.0}
    try:
        parse("x")
    except ValueError:
        pass
    else:
        raise AssertionError
