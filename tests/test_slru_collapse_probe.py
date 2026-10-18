"""SLRU collapse probe (docs/CUDA-SLRU-COLLAPSE-RECORD.md): the default is preserved and the instrumentation changes nothing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
sys.path.insert(0, str(Path(__file__).parent))

import cuda_offline_policy_table as t  # noqa: E402
import slru_collapse_probe as p  # noqa: E402
from simulate_policies import Store  # noqa: E402
from test_cuda_offline_policy_table import synthetic  # noqa: E402


def test_default_protected_fraction_is_the_old_hard_coded_cap():
    arrays = synthetic()
    assert Store(10, "slru").protected_fraction == 0.8
    a = t.replay(arrays, 20, "slru")
    b = p.run_cell(arrays, 20, "slru", 0.8)[0]
    assert a["per_token"] == b["per_token"]


def test_probe_store_counts_without_changing_the_result():
    arrays = synthetic(decode=120)
    plain = p.run_cell(arrays, 16, "slru", 0.8)[0]
    probed, store = p.run_cell(arrays, 16, "slru", 0.8, probe=True)
    assert probed["per_token"] == plain["per_token"]
    assert store.hit_from["protected"] + store.hit_from["probation"] > 0
    assert store.evicted_inserted == len(store.lifetimes)
    assert all(life >= 0 for life in store.lifetimes)
    info = p.internals(store, probed["tokens"], probed["requests"] / probed["tokens"], 0.8)
    assert 0 <= info["share_evicted_never_rehit"] <= 1 and 0 <= info["share_hits_from_protected"] <= 1
