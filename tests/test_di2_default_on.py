"""DI-2 default-on (0.75.0, HANDOFF 18.253): serve.sh replays 140 decoder rows unless the environment says otherwise (an explicit empty value is DI-1's
exact tail), chat.sh does not set it, the code default stays off, and a server's snapshot key follows it."""

import subprocess
from pathlib import Path

from cachalot.model import snapshot_store
from cachalot.model.prefill_tail_plan import decoder_keep

REPO = Path(__file__).resolve().parent.parent


def _resolved(script: str, env: dict) -> str:
    """The value a launch script exports, from its environment block only (no one-runtime guard, no exec)."""
    text = (REPO / script).read_text()
    body = text[text.index("export CACHALOT_MODEL_PATH"):text.index("exec ")]
    out = subprocess.run(["bash", "-c", body + '\nprintf %s "${CACHALOT_PREFILL_DECODER_KEEP-unset}"'],
                         capture_output=True, text=True, env={"HOME": str(Path.home()), "PATH": "/usr/bin:/bin", **env}, cwd=REPO)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_serve_sh_replays_140_rows_by_default_and_keeps_an_explicit_value():
    assert _resolved("serve.sh", {}) == "140"
    assert _resolved("serve.sh", {"CACHALOT_PREFILL_DECODER_KEEP": "256"}) == "256"
    assert _resolved("serve.sh", {"CACHALOT_PREFILL_DECODER_KEEP": ""}) == ""        # empty: the exact tail


def test_chat_sh_does_not_set_it():
    assert _resolved("chat.sh", {}) == "unset"


def test_the_code_default_stays_off_and_empty_means_the_exact_tail(monkeypatch):
    monkeypatch.delenv("CACHALOT_PREFILL_DECODER_KEEP", raising=False)
    assert decoder_keep() is None
    monkeypatch.setenv("CACHALOT_PREFILL_DECODER_KEEP", "")
    assert decoder_keep() is None


def test_the_served_default_keys_its_snapshots_apart(monkeypatch):
    monkeypatch.setenv("CACHALOT_PREFILL_DECODER_KEEP", "140")
    assert snapshot_store.numerics_version(True) == snapshot_store.NUMERICS_VERSION + "-di1.1-keep140"
