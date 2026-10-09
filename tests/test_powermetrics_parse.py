import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import powermetrics_parse as pp  # noqa: E402

SAMPLE = """
*** Sampled system activity (Fri Oct  9 22:19:59 2026 +0200) (1000.00ms elapsed) ***
CPU Power: 9000 mW
GPU Power: 1000 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 10000 mW

*** Sampled system activity (Fri Oct  9 22:20:00 2026 +0200) (1000.00ms elapsed) ***
CPU Power: 1000 mW
GPU Power: 200 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 1200 mW

*** Sampled system activity (Fri Oct  9 22:20:01 2026 +0200) (2000.00ms elapsed) ***
CPU Power: 2000 mW
GPU Power: 400 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 2400 mW
"""


def test_parse_reads_every_sample_and_component():
    s = pp.parse(SAMPLE)
    assert [x.clock for x in s] == ["22:19:59", "22:20:00", "22:20:01"]
    assert (s[1].cpu_mw, s[1].gpu_mw, s[1].ane_mw, s[1].combined_mw) == (1000, 200, 0, 1200)
    assert s[2].elapsed_ms == 2000


def test_window_drops_the_first_sample_and_filters_by_clock():
    s = pp.parse(SAMPLE)
    assert [x.clock for x in pp.window(s, None, None)] == ["22:20:00", "22:20:01"]
    assert [x.clock for x in pp.window(s, "22:20:01", None)] == ["22:20:01"]


def test_summary_weights_power_by_each_samples_own_elapsed_time():
    r = pp.summarize(pp.window(pp.parse(SAMPLE), None, None), tokens=10)
    # 1.2 W for 1 s + 2.4 W for 2 s = 6.0 J over 3 s = 2.0 W mean
    assert abs(r["combined"]["joules"] - 6.0) < 1e-9 and abs(r["combined"]["mean_w"] - 2.0) < 1e-9
    assert abs(r["joules_per_token"] - 0.6) < 1e-9 and r["seconds"] == 3.0
