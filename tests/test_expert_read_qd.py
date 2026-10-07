import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from expert_read_qd import quantiles, run_level  # noqa: E402


def test_quantiles_pick_sorted_ranks():
    q = quantiles([float(i) for i in range(1, 101)])
    assert q["p50"] == 51.0 and q["p95"] == 96.0 and q["p99"] == 100.0 and q["max"] == 100.0


def test_run_level_reads_every_job_once_and_counts_bytes():
    seen = []

    def work(job, views):
        seen.append(job)
        time.sleep(0.001)
        return 1000, views or object()

    r = run_level(4, list(range(40)), work)
    assert sorted(seen) == list(range(40))
    assert r["K"] == 4 and r["reads"] == 40 and r["GBps"] > 0 and r["lat_p50_ms"] >= 1.0
