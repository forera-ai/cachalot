"""Vision upscale default-on (0.80.0, HANDOFF 18.266): serve.sh upscales images under 810,000 pixels unless the environment says otherwise
(0 or an explicit empty value is the checkpoint's preprocessing), chat.sh does not set it, and the code default stays off."""

import subprocess
from pathlib import Path

from cachalot.model import image_processor_mlx as ip

REPO = Path(__file__).resolve().parent.parent


def _resolved(script: str, env: dict) -> str:
    """The value a launch script exports, from its environment block only (no one-runtime guard, no exec)."""
    text = (REPO / script).read_text()
    body = text[text.index("export CACHALOT_MODEL_PATH"):text.index("exec ")]
    out = subprocess.run(["bash", "-c", body + '\nprintf %s "${CACHALOT_VISION_MIN_PIXELS-unset}"'],
                         capture_output=True, text=True, env={"HOME": str(Path.home()), "PATH": "/usr/bin:/bin", **env}, cwd=REPO)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_serve_sh_turns_it_on_and_keeps_an_explicit_value():
    assert _resolved("serve.sh", {}) == "810000"
    assert _resolved("serve.sh", {"CACHALOT_VISION_MIN_PIXELS": "0"}) == "0"
    assert _resolved("serve.sh", {"CACHALOT_VISION_MIN_PIXELS": ""}) == ""


def test_chat_sh_does_not_set_it():
    assert _resolved("chat.sh", {}) == "unset"


def test_the_off_values_give_the_checkpoint_preprocessing(monkeypatch):
    for value, expect in (("810000", 810000), ("0", None), ("", None)):
        monkeypatch.setenv("CACHALOT_VISION_MIN_PIXELS", value)
        assert ip.min_pixels_from_env() == expect
