"""serve-glm.sh reads a short prefill's routed experts only, below 512 tokens (HANDOFF 18.274-18.275); an environment value still wins,
and the code default (chat-glm.sh, direct runs) stays 128."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def threshold_for(env_value: str | None = None) -> str:
    text = (ROOT / "serve-glm.sh").read_text()
    line = next(l for l in text.splitlines() if l.startswith("export CACHALOT_GLM_SPECULATE_MIN_TOKENS="))
    env = {"PATH": "/usr/bin:/bin"}
    if env_value is not None:
        env["CACHALOT_GLM_SPECULATE_MIN_TOKENS"] = env_value
    probe = "set -u\n" + line + '\nprintf "%s" "$CACHALOT_GLM_SPECULATE_MIN_TOKENS"\n'
    return subprocess.run(["bash", "-c", probe], env=env, capture_output=True, text=True, check=True).stdout


def test_serve_glm_default_is_512():
    assert threshold_for() == "512"


def test_environment_still_wins():
    assert threshold_for("128") == "128"
    assert threshold_for("1200") == "1200"


def test_code_default_unchanged():
    src = (ROOT / "src" / "cachalot" / "glm" / "experts.py").read_text()
    assert 'os.environ.get("CACHALOT_GLM_SPECULATE_MIN_TOKENS", "128")' in src
