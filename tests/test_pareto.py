import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import pareto  # noqa: E402


def test_wilson_bounds():
    lo, hi = pareto.wilson(10, 10)
    assert hi == 1.0 and 0.6 < lo < 0.8
    assert pareto.wilson(0, 0) == (0.0, 1.0)
    lo, hi = pareto.wilson(5, 10)
    assert lo < 0.5 < hi


def test_block_bootstrap_centres_on_mean_and_widens_with_noise():
    rng = np.random.default_rng(1)
    groups = np.repeat([0, 1, 2], 100)
    quiet = rng.normal(0.01, 0.01, 300)
    loud = rng.normal(0.01, 0.5, 300)
    m1, lo1, hi1 = pareto.block_bootstrap_ci(quiet, groups)
    m2, lo2, hi2 = pareto.block_bootstrap_ci(loud, groups)
    assert lo1 < m1 < hi1 and lo2 < m2 < hi2
    assert (hi2 - lo2) > 5 * (hi1 - lo1)
    assert lo1 > 0  # a consistent +0.01 shift is detected


def test_kl_is_zero_for_equal_and_positive_otherwise():
    lp = np.log(np.array([[0.5, 0.3, 0.1]]))
    assert pareto.kl_top_tail(lp, lp)[0] == pytest.approx(0.0, abs=1e-9)
    other = np.log(np.array([[0.3, 0.3, 0.3]]))
    assert pareto.kl_top_tail(lp, other)[0] > 0.01


def test_kl_matches_full_distribution_when_tail_is_empty():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.5, 0.3, 0.2])
    expected = float((p * np.log(p / q)).sum())
    got = pareto.kl_top_tail(np.log(p)[None], np.log(q)[None])[0]
    assert got == pytest.approx(expected, rel=1e-6)


def test_pareto_front_drops_dominated():
    pts = [("a", (1.0, 0.0)), ("b", (2.0, 0.0)), ("c", (0.5, 1.0)), ("d", (2.0, 2.0))]
    assert pareto.pareto_front(pts) == {"a", "c"}


def test_classify_arm():
    assert pareto.classify_arm(0.0, 0.0, True, None) == "identical"
    assert pareto.classify_arm(0.01, 0.001, False, None) == "no band"
    assert pareto.classify_arm(0.01, 0.001, False, (0.02, 0.002)) == "inside noise"
    assert pareto.classify_arm(0.05, 0.001, False, (0.02, 0.002)) == "outside noise"
    assert pareto.classify_arm(0.01, 0.01, False, (0.02, 0.002)) == "outside noise"


def test_exact_and_json_checks():
    assert pareto.check_exact("The answer is 391.", {"contains": ["391"]})
    assert not pareto.check_exact("392", {"contains": ["391"]})
    assert pareto.check_exact("9.9", {"regex": r"^\W*9\.9\b"})
    assert not pareto.check_exact("9.11 is larger", {"regex": r"^\W*9\.9\b"})
    assert pareto.check_json('```json\n{"x": 1, "y": 2}\n```', {"keys": {"x": 1, "y": None}})
    assert pareto.check_json('Here: {"paid": false}', {"keys": {"paid": False}})
    assert not pareto.check_json("no json", {"keys": {"x": 1}})
    assert not pareto.check_json('{"x": 2}', {"keys": {"x": 1}})


def test_python_check_runs_the_asserts():
    good = "```python\ndef f(x):\n    return x + 1\n```"
    bad = "```python\ndef f(x):\n    return x\n```"
    task = {"asserts": "assert f(1) == 2"}
    assert pareto.check_python(good, task)
    assert not pareto.check_python(bad, task)
    assert not pareto.check_python("no code", task)
    assert not pareto.check_python("```python\nwhile True:\n    pass\n```", task, timeout=1.0)


def test_extract_code_prefers_the_language_and_the_last_block():
    text = "```python\nA\n```\n```csharp\nB\n```\n```csharp\nC\n```"
    assert pareto.extract_code(text, "csharp").strip() == "C"
    assert pareto.extract_code("```\nplain\n```", "python").strip() == "plain"


def test_task_file_is_well_formed():
    tasks = json.loads(pareto.TASKS_FILE.read_text())["tasks"]
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids))
    for t in tasks:
        assert t["kind"] in pareto.CHECKS
        assert t["prompt"] and t["max_tokens"] > 0
    assert any(t["quick"] for t in tasks)


def test_reference_solutions_pass_the_python_checks():
    """Each python task's asserts must be satisfiable: a known-good solution passes them."""
    solutions = {
        "palindrome": "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]",
        "merge-intervals": (
            "def merge_intervals(iv):\n    out = []\n    for a, b in sorted(iv):\n"
            "        if out and a <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n"
            "        else:\n            out.append([a, b])\n    return out"),
        "flatten": (
            "def flatten(x):\n    r = []\n    for i in x:\n        r.extend(flatten(i) if isinstance(i, list) else [i])\n"
            "    return r"),
        "rle": (
            "def rle(s):\n    import itertools\n    return ''.join(c + str(len(list(g))) for c, g in itertools.groupby(s))"),
        "parse-duration": (
            "def parse_duration(s):\n    import re\n    u = {'h': 3600, 'm': 60, 's': 1}\n"
            "    return sum(int(n) * u[k] for n, k in re.findall(r'(\\d+)([hms])', s))"),
        "top-k-words": (
            "def top_k(words, k):\n    from collections import Counter\n    c = Counter(words)\n"
            "    return sorted(c, key=lambda w: (-c[w], w))[:k]"),
        "roman": (
            "def roman_to_int(s):\n    v = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)\n    t = 0\n"
            "    for a, b in zip(s, s[1:] + ' '):\n        t += -v[a] if b in v and v[b] > v[a] else v[a]\n    return t"),
        "two-sum": (
            "def two_sum(n, t):\n    for i in range(len(n)):\n        for j in range(i + 1, len(n)):\n"
            "            if n[i] + n[j] == t:\n                return [i, j]"),
    }
    tasks = {t["id"]: t for t in json.loads(pareto.TASKS_FILE.read_text())["tasks"] if t["kind"] == "python"}
    assert set(solutions) == set(tasks)
    for name, code in solutions.items():
        assert pareto.check_python(f"```python\n{code}\n```", tasks[name]), name


def test_arms_file_loads_and_names_a_reference():
    cfg = pareto.load_arms(Path(__file__).resolve().parents[1] / "benchmarks" / "pareto_arms.example.json")
    assert cfg["reference"] in cfg["arms"]
    assert any(a.get("noise") for a in cfg["arms"].values())


def _write_arm(out, name, target, logp, group, top1, ms, passed=True):
    np.savez_compressed(out / f"{name}.npz", target_logp=target, group=group, ids=np.zeros(logp.shape, dtype=int),
                        logp=logp, top1=top1, ms=ms, misses=np.ones(len(ms)))
    tasks = [{"id": str(i), "kind": "exact", "passed": passed} for i in range(3)]
    (out / f"{name}.json").write_text(json.dumps({"texts_hash": "h", "tasks": tasks, "complete": True}))


def test_summarise_end_to_end(tmp_path):
    n = 64
    rng = np.random.default_rng(0)
    group = np.repeat([0, 1], n // 2)
    target = -np.abs(rng.normal(1.0, 0.3, n))
    logp = np.log(np.tile([0.5, 0.3, 0.1], (n, 1)))
    top1 = np.zeros(n, dtype=int)
    _write_arm(tmp_path, "ref", target, logp, group, top1, np.full(n, 100.0))
    _write_arm(tmp_path, "same", target, logp, group, top1, np.full(n, 80.0))
    _write_arm(tmp_path, "worse", target - 0.5, np.log(np.tile([0.3, 0.3, 0.3], (n, 1))), group, top1,
               np.full(n, 70.0), passed=False)
    cfg = {"reference": "ref", "order": ["ref", "same", "worse"], "arms": {"ref": {}, "same": {}, "worse": {}}}
    rows = {r["name"]: r for r in pareto.summarise(tmp_path, cfg)}
    assert rows["same"]["verdict"] == "identical" and rows["same"]["frontier"]
    assert rows["worse"]["d_nll"] == pytest.approx(0.5)
    assert rows["worse"]["verdict"] == "no band"
    assert rows["worse"]["kl_mean"] > 0.01
    assert not rows["ref"]["frontier"]  # "same" is faster at equal quality
