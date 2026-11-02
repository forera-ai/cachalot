"""DeepSeek-V4.1-Flash-1017 as the default checkpoint (0.73.0, runbook step 8): serve.sh, chat.sh and pareto.serve_defaults pick the
1017 directory, its own bank and its own hotlist, and an explicit CACHALOT_MODEL_PATH for the first release still pairs that trunk with
the first release's bank and hotlist."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
V1017 = "/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-1017"
OLD = "/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash"


def _exports(script: str, env: dict, home: Path) -> dict:
    """The model, bank and hotlist a launch script exports, from its environment block only (no one-runtime guard, no exec)."""
    text = (REPO / script).read_text()
    body = text[text.index("# DeepSeek-V4.1-Flash-1017 is the default"):text.index("exec ")]
    probe = '\nprintf "%s\\n%s\\n%s" "$CACHALOT_MODEL_PATH" "$CACHALOT_EXPERT_BANK" "$CACHALOT_HOTLIST"'
    out = subprocess.run(["bash", "-c", body + probe], capture_output=True, text=True,
                         env={"HOME": str(home), "PATH": "/usr/bin:/bin", **env}, cwd=REPO)
    assert out.returncode == 0, out.stderr
    model, bank, hotlist = out.stdout.split("\n")
    return {"model": model, "bank": bank, "hotlist": hotlist}


def _fake_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (home / "DeepSeek-V4.1-Flash-1017-q2g128").mkdir(parents=True)
    (home / "DeepSeek-V4.1-Flash-1017-q2g128" / ".cachalot-verified").touch()
    (home / "cachalot-hotlist-DeepSeek-V4.1-Flash-1017.json").write_text("{}")
    (home / "cachalot-hotlist.json").write_text("{}")
    return home


@pytest.mark.parametrize("script", ["serve.sh", "chat.sh"])
def test_launch_scripts_default_to_1017_with_its_own_bank_and_hotlist(script, tmp_path):
    home = _fake_home(tmp_path)
    got = _exports(script, {}, home)
    assert got == {"model": V1017, "bank": f"{home}/DeepSeek-V4.1-Flash-1017-q2g128",
                   "hotlist": f"{home}/cachalot-hotlist-DeepSeek-V4.1-Flash-1017.json"}


@pytest.mark.parametrize("script", ["serve.sh", "chat.sh"])
def test_launch_scripts_still_serve_the_first_release_when_asked(script, tmp_path):
    home = _fake_home(tmp_path)
    got = _exports(script, {"CACHALOT_MODEL_PATH": OLD}, home)
    # No verified internal copy of the first release's bank: it is read beside the checkpoint on the X10Pro.
    assert got == {"model": OLD, "bank": "/Volumes/X10Pro/Flash4-1/DeepSeek-V4.1-Flash-q2g128",
                   "hotlist": f"{home}/cachalot-hotlist.json"}


def _serve_defaults(env: dict, home: Path) -> dict:
    code = ("import os, sys; sys.path[:0] = ['benchmarks', 'src']; import pareto; pareto.serve_defaults(); "
            "print(os.environ['CACHALOT_MODEL_PATH']); print(os.environ['CACHALOT_EXPERT_BANK']); "
            "print(os.environ['CACHALOT_HOTLIST'])")
    clean = {k: v for k, v in os.environ.items() if not k.startswith("CACHALOT_")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**clean, "HOME": str(home), **env}, cwd=REPO)
    assert out.returncode == 0, out.stderr
    model, bank, hotlist = out.stdout.split("\n")[:3]
    return {"model": model, "bank": bank, "hotlist": hotlist}


def test_pareto_serve_defaults_match_serve_sh(tmp_path):
    home = _fake_home(tmp_path)
    assert _serve_defaults({}, home) == _exports("serve.sh", {}, home)
    assert _serve_defaults({"CACHALOT_MODEL_PATH": OLD}, home) == _exports("serve.sh", {"CACHALOT_MODEL_PATH": OLD}, home)


def test_a_checkpoint_without_a_hotlist_starts_without_one(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    assert _exports("serve.sh", {}, home)["hotlist"] == ""
    assert _serve_defaults({}, home)["hotlist"] == ""
