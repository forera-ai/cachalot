"""serve-glm.sh prefers a verified internal bank and falls back to the X10Pro copy; an environment value still wins."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bank_for(home: Path, env_bank: str | None = None) -> str:
    text = (ROOT / "serve-glm.sh").read_text()
    start, end = text.index("GLM_BANK_DEFAULT=/Volumes"), text.index("# Decode prefetch off")
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    if env_bank is not None:
        env["CACHALOT_GLM_BANK"] = env_bank
    probe = "set -u\n" + text[start:end].replace('/Volumes/X10Pro/models/GLM-5.3-Flash-bank', str(home / "x10"), 1) + 'printf "%s" "${CACHALOT_GLM_BANK-}"\n'
    return subprocess.run(["bash", "-c", probe], env=env, capture_output=True, text=True, check=True).stdout


def test_verified_internal_bank_wins(tmp_path):
    (tmp_path / "x10").mkdir()
    (tmp_path / "x10" / "bank.json").write_text("{}")
    internal = tmp_path / "GLM-5.3-Flash-bank"
    internal.mkdir()
    (internal / "bank.json").write_text("{}")
    assert bank_for(tmp_path) == str(tmp_path / "x10")  # an unverified copy never wins
    (internal / ".cachalot-verified").write_text("")
    assert bank_for(tmp_path) == str(internal)


def test_environment_still_wins(tmp_path):
    internal = tmp_path / "GLM-5.3-Flash-bank"
    internal.mkdir()
    (internal / "bank.json").write_text("{}")
    (internal / ".cachalot-verified").write_text("")
    assert bank_for(tmp_path, "") == ""
    assert bank_for(tmp_path, "/elsewhere") == "/elsewhere"
