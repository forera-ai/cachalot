"""DI-1 default-on (0.72.0, docs/DI1-DEFAULT-ON-PLAN.md): serve.sh turns CACHALOT_PREFILL_ENCODER_ONLY on unless the environment says
otherwise, chat.sh does not, the code default stays off, and a DeepSeek server keys its disk snapshots on the mode (M1)."""

import subprocess
import types
from pathlib import Path

from cachalot import cli
from cachalot.model import generation, snapshot_store

REPO = Path(__file__).resolve().parent.parent


def _resolved(script: str, env: dict) -> str:
    """The value a launch script exports, from its environment block only (no one-runtime guard, no exec)."""
    text = (REPO / script).read_text()
    body = text[text.index("export CACHALOT_MODEL_PATH"):text.index("exec ")]
    out = subprocess.run(["bash", "-c", body + '\nprintf %s "${CACHALOT_PREFILL_ENCODER_ONLY-unset}"'],
                         capture_output=True, text=True, env={"HOME": str(Path.home()), "PATH": "/usr/bin:/bin", **env}, cwd=REPO)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_serve_sh_turns_it_on_by_default_and_keeps_an_explicit_value():
    assert _resolved("serve.sh", {}) == "1"
    assert _resolved("serve.sh", {"CACHALOT_PREFILL_ENCODER_ONLY": "0"}) == "0"


def test_chat_sh_keeps_the_full_prefill():
    assert _resolved("chat.sh", {}) == "unset"


def test_the_code_default_stays_off(monkeypatch):
    monkeypatch.delenv("CACHALOT_PREFILL_ENCODER_ONLY", raising=False)
    assert generation.encoder_only_enabled() is False


def test_the_numerics_tag_separates_the_modes_and_leaves_the_full_prefill_tag_alone():
    assert snapshot_store.numerics_version(False) == snapshot_store.NUMERICS_VERSION
    assert snapshot_store.numerics_version(True) == snapshot_store.NUMERICS_VERSION + "-di1"


def _runtime(tmp_path):
    model, bank = tmp_path / "model", tmp_path / "bank"
    model.mkdir()
    bank.mkdir()
    (model / "config.json").write_text("{}")
    (bank / "config.json").write_text("{}")
    cache = types.SimpleNamespace(add=lambda *a, **k: None)
    return types.SimpleNamespace(model_path=model, expert_bank_path=bank, max_seq_len=65536, prefix_cache=cache)


def test_the_server_keys_its_snapshot_store_on_the_mode(tmp_path, monkeypatch):
    rt = _runtime(tmp_path)
    ids = {}
    for knob in ("0", "1"):
        monkeypatch.setenv("CACHALOT_PREFILL_ENCODER_ONLY", knob)
        cli._attach_snapshot_store(rt, str(tmp_path / "snaps"))
        ids[knob] = rt.prefix_cache.persist.__self__.identity
    full = snapshot_store.runtime_identity(rt.model_path, rt.expert_bank_path, 65536, snapshot_store.NUMERICS_VERSION)
    assert ids["0"] == full                                        # the full prefill finds the snapshots it wrote before 0.72.0
    assert ids["1"] != full
