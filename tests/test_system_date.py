"""HANDOFF section 18.66: an agent's date line shows a recent saved date so a new day reuses the saved block."""

import json

from cachalot.cli import _decode_miss_budget_from_env
from cachalot.server.system_date import SystemDateReuse


def _msgs(date_text, tail="Model: x"):
    system = f"You are an agent.\n\nConversation started: {date_text} (Europe/Vienna, CEST, UTC+02:00)\n{tail}"
    return [{"role": "system", "content": system}, {"role": "user", "content": "Conversation started: Friday, January 01, 2027"}]


def test_a_new_day_shows_the_saved_date_and_only_in_the_system_message(tmp_path):
    reuse = SystemDateReuse(tmp_path / "d.json", max_days=7)
    m, note = reuse.apply(_msgs("Monday, October 05, 2026"))
    assert note is None and "October 05" in m[0]["content"]  # first sighting: true date, recorded
    m, note = reuse.apply(_msgs("Tuesday, October 06, 2026"))
    assert "Monday, October 05, 2026" in m[0]["content"] and "October 06" not in m[0]["content"]
    assert note and reuse.reused == 1
    assert m[1]["content"] == "Conversation started: Friday, January 01, 2027"  # the user message is untouched
    assert m[0]["content"].endswith("Model: x")


def test_the_window_ends_and_the_true_date_starts_a_new_one(tmp_path):
    reuse = SystemDateReuse(tmp_path / "d.json", max_days=7)
    reuse.apply(_msgs("Monday, October 05, 2026"))
    m, note = reuse.apply(_msgs("Tuesday, October 13, 2026"))  # 8 days
    assert note is None and "October 13" in m[0]["content"]
    m, note = reuse.apply(_msgs("Wednesday, October 14, 2026"))
    assert "Tuesday, October 13, 2026" in m[0]["content"]  # the old window is out, the new date serves


def test_the_record_survives_a_restart(tmp_path):
    SystemDateReuse(tmp_path / "d.json").apply(_msgs("Monday, October 05, 2026"))
    assert json.loads((tmp_path / "d.json").read_text()) == ["Monday, October 05, 2026"]
    m, note = SystemDateReuse(tmp_path / "d.json").apply(_msgs("Tuesday, October 06, 2026"))
    assert note and "October 05" in m[0]["content"]


def test_off_switch_no_date_line_and_no_system_message(tmp_path):
    off = SystemDateReuse.from_env(tmp_path / "d.json", {"CACHALOT_SYSTEM_DATE_REUSE": "0"})
    assert not off.enabled
    assert off.apply(_msgs("Monday, October 05, 2026"))[1] is None
    reuse = SystemDateReuse(tmp_path / "e.json")
    plain = [{"role": "system", "content": "no date here"}, {"role": "user", "content": "hi"}]
    assert reuse.apply(plain) == (plain, None)
    user_only = [{"role": "user", "content": "Conversation started: Monday, October 05, 2026"}]
    assert reuse.apply(user_only) == (user_only, None)
    assert reuse.seen == []


def test_the_miss_budget_env_accepts_off():
    assert _decode_miss_budget_from_env({"CACHALOT_DECODE_MISS_BUDGET": "0"}) == 0
    assert _decode_miss_budget_from_env({"CACHALOT_DECODE_MISS_BUDGET": "off"}) is None
    assert _decode_miss_budget_from_env({}) is None


def test_a_new_day_through_the_server_reuses_the_whole_system_block(tmp_path):
    from fastapi.testclient import TestClient

    from cachalot.server.app import create_app
    from cachalot.server.engine import Engine
    from fake_model import FakeModel, ScriptedRuntime
    from test_system_boundary_snapshots import SystemPrefixEncoding

    engine = Engine(FakeModel(ScriptedRuntime(reply="ok")), model_id="m", encoding=SystemPrefixEncoding())
    engine.system_date = SystemDateReuse(tmp_path / "d.json")
    client = TestClient(create_app(engine))

    def body(day, user):
        system = _msgs(day)[0]["content"]
        return {"model": "m", "max_tokens": 2, "messages": [{"role": "system", "content": system},
                                                           {"role": "user", "content": user}]}

    client.post("/v1/chat/completions", json=body("Monday, October 05, 2026", "first question"))
    r = client.post("/v1/chat/completions", json=body("Tuesday, October 06, 2026", "an unrelated second one"))
    system_tokens = len("<system>" + _msgs("Monday, October 05, 2026")[0]["content"])
    assert r.json()["usage"]["cachalot"]["reused_prefix_tokens"] == system_tokens
    engine.system_date.enabled = False
    r = client.post("/v1/chat/completions", json=body("Wednesday, October 07, 2026", "third"))
    assert r.json()["usage"]["cachalot"]["reused_prefix_tokens"] < system_tokens  # the true date breaks the match
