"""HANDOFF 18.12 (M21, M22): the checkpoint's sampling defaults, and the GLM/MiniMax chat's prompt head."""

import json

from fastapi.testclient import TestClient

from cachalot.cli import _glm_prompt_head
from cachalot.glm.model import GlmModel
from cachalot.server.app import ServerConfig, create_app
from cachalot.server.engine import Engine
from fake_model import FakeEncoding, FakeModel, ScriptedRuntime


def _top_p_seen(config: ServerConfig, body: dict) -> float:
    engine = Engine(FakeModel(ScriptedRuntime(reply="ok")), model_id="m", encoding=FakeEncoding())
    seen = []
    chat = engine.chat

    def recording(req):
        seen.append(req.params.top_p)
        return chat(req)

    engine.chat = recording
    r = TestClient(create_app(engine, config)).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], **body})
    assert r.status_code == 200, r.text
    return seen[0]


def test_a_request_without_top_p_takes_the_servers_default():
    assert _top_p_seen(ServerConfig(), {}) == 1.0
    assert _top_p_seen(ServerConfig(default_top_p=0.95), {}) == 0.95


def test_a_requests_own_top_p_wins():
    assert _top_p_seen(ServerConfig(default_top_p=0.95), {"top_p": 1.0}) == 1.0
    assert _top_p_seen(ServerConfig(default_top_p=0.95), {"top_p": 0.5}) == 0.5


def test_generation_defaults_read_the_checkpoints_config(tmp_path):
    model = GlmModel.__new__(GlmModel)
    model.model_path = tmp_path
    assert model.generation_defaults() == {}
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"temperature": 1.0, "top_p": 0.95, "do_sample": True, "eos_token_id": 3}))
    assert model.generation_defaults() == {"temperature": 1.0, "top_p": 0.95}


def test_notices_go_to_the_hook_when_one_is_set(capsys):
    model = GlmModel.__new__(GlmModel)
    model._notify("idle warm: 3 experts in 0.1s")
    assert "idle warm" in capsys.readouterr().out
    got = []
    model.notice = got.append
    model._notify("idle warm: 4 experts in 0.1s")
    assert got == ["idle warm: 4 experts in 0.1s"] and capsys.readouterr().out == ""


class _Template:
    """A chat template with a fixed header: one token per character."""

    class tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            return [ord(c) for c in text]

    @staticmethod
    def render_chat(messages, thinking=False, effort=None):
        out = "<head>"
        for m in messages:
            out += f"<{m['role']}>{m['content']}"
        return out + "<assistant>"


def test_the_prompt_head_is_what_every_first_turn_starts_with():
    head = _glm_prompt_head(_Template, [], False, None)
    assert "".join(map(chr, head)) == "<head><user>"
    head = _glm_prompt_head(_Template, [{"role": "system", "content": "be brief"}], False, None)
    assert "".join(map(chr, head)) == "<head><system>be brief<user>"
