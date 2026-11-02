"""Tool-exact default-on (0.77.0, HANDOFF 18.256): serve.sh lifts the decode miss budget inside tool-call blocks unless the environment says
otherwise (0 or an explicit empty value keeps budget 0 there), chat.sh does not set it, and the code default stays off."""

import subprocess
from pathlib import Path

from cachalot.server.engine import Engine
from fake_model import FakeEncoding, FakeModel, ScriptedRuntime

REPO = Path(__file__).resolve().parent.parent


def _resolved(script: str, env: dict) -> str:
    """The value a launch script exports, from its environment block only (no one-runtime guard, no exec)."""
    text = (REPO / script).read_text()
    body = text[text.index("export CACHALOT_MODEL_PATH"):text.index("exec ")]
    out = subprocess.run(["bash", "-c", body + '\nprintf %s "${CACHALOT_DECODE_TOOL_EXACT-unset}"'],
                         capture_output=True, text=True, env={"HOME": str(Path.home()), "PATH": "/usr/bin:/bin", **env}, cwd=REPO)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_serve_sh_turns_it_on_and_keeps_an_explicit_value():
    assert _resolved("serve.sh", {}) == "1"
    assert _resolved("serve.sh", {"CACHALOT_DECODE_TOOL_EXACT": "0"}) == "0"
    assert _resolved("serve.sh", {"CACHALOT_DECODE_TOOL_EXACT": ""}) == ""


def test_chat_sh_does_not_set_it():
    assert _resolved("chat.sh", {}) == "unset"


def test_only_1_turns_the_engine_knob_on(monkeypatch):
    for value, on in (("1", True), ("0", False), ("", False)):
        monkeypatch.setenv("CACHALOT_DECODE_TOOL_EXACT", value)
        assert Engine(FakeModel(ScriptedRuntime()), model_id="t", encoding=FakeEncoding()).tool_exact_decode is on
