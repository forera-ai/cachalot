import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import run_manifest as rm  # noqa: E402


def test_swap_parse():
    assert rm.parse_swap_used_gib("total = 2048.00M  used = 1536.00M  free = 512.00M  (encrypted)") == 1.5
    assert rm.parse_swap_used_gib("total = 0.00M  used = 0.00M  free = 0.00M") == 0.0
    assert rm.parse_swap_used_gib("used = 2.50G") == 2.5
    assert rm.parse_swap_used_gib(None) is None


def test_runtime_processes_skips_self_and_others():
    ps = "\n".join([
        "100 /Users/x/venvs/deepseek-v41/bin/python -m cachalot.cli serve",
        "200 /Users/x/venvs/deepseek-v41/bin/python benchmarks/run_manifest.py",
        "300 /bin/zsh -c run ~/venvs/deepseek-v41/bin/python -m cachalot.cli serve",
        "400 python -m cachalot.cli chat",
    ])
    assert rm.runtime_processes(ps, own_pid=400) == ["/Users/x/venvs/deepseek-v41/bin/python -m cachalot.cli serve"]


def test_environment_filter():
    env = {"CACHALOT_X": "1", "MLX_METAL_FAST_SYNCH": "1", "HOME": "/h", "AI_GATEWAY_API_KEY": "secret"}
    assert rm.environment(env) == {"CACHALOT_X": "1", "MLX_METAL_FAST_SYNCH": "1"}


def test_row_schema_rules():
    r = rm.row("abc", "exact", {"ms_token": (120.0, "measured"), "hit_rate": (None, "measured")})
    assert r["fields"] == {"ms_token": {"value": 120.0, "unit": "ms", "tag": "measured"}}
    assert r["schema"] == rm.SCHEMA_VERSION
    with pytest.raises(KeyError):
        rm.row("abc", "exact", {"speed": (1, "measured")})
    with pytest.raises(ValueError):
        rm.row("abc", "exact", {"ms_token": (1, "guessed")})


def test_percentiles_need_enough_tokens():
    assert rm.percentiles([1.0] * 199) == {}
    p = rm.percentiles([float(i) for i in range(1, 201)])
    assert p["ms_token_p50"] == pytest.approx(100.0, abs=1)
    assert p["ms_token_p99"] == pytest.approx(198.0, abs=1)


def test_collect_and_write(tmp_path):
    with pytest.raises(ValueError):
        rm.collect(workload="decode", env={})
    man = rm.collect(workload="W4:csharp", workload_text="body", budget_gib=48, env={"CACHALOT_MODEL_PATH": "/m"})
    assert man["workload"] == {"id": "W4:csharp", "class": "W4", "hash": rm.content_hash("body")}
    assert man["model_path"] == "/m" and man["budget_gib"] == 48
    mid = rm.write_manifest(tmp_path, man)
    saved = json.loads((tmp_path / f"manifest-{mid}.json").read_text())
    assert saved["id"] == mid and rm.manifest_id(man) == mid
    rm.append_rows(tmp_path / "s.jsonl", [rm.row(mid, "a", {"tokens": (5, "measured")})])
    assert json.loads((tmp_path / "s.jsonl").read_text())["manifest"] == mid
