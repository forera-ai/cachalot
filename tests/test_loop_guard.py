"""The runaway-repetition guard of the GLM/MiniMax server engine (HANDOFF 18.21 item 9)."""

from types import SimpleNamespace

from cachalot.glm import engine as glm_engine
from cachalot.glm.engine import GlmEngine, repeating_tail
from cachalot.minimax.model import MiniMaxSplitter
from cachalot.server.engine import ChatRequest
from cachalot.model.generation import SamplingParams


def test_repeating_tail_finds_a_block_repeated_back_to_back():
    block = list(range(100, 112))  # 12 tokens
    tokens = list(range(40)) + block * 6
    assert repeating_tail(tokens, 6) == 12
    assert repeating_tail(tokens[:-1], 6) == 0  # the sixth copy is not complete yet
    assert repeating_tail(list(range(40)) + block * 5, 6) == 0


def test_repeating_tail_ignores_short_blocks_and_varied_lists():
    assert repeating_tail([7] * 59, 6) == 0  # blocks are at least 10 tokens
    assert repeating_tail([7] * 60, 6) == 10  # 60 copies of one token is a loop too
    listing = []
    for i in range(60):  # a long list whose rows share a template but differ in a field
        listing += [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000 + i, 11, 12]
    assert repeating_tail(listing, 6) == 0


class _Tok:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        return "".join("<tool_call>" if i == 7 else f"w{i} " for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


class _Model:
    splitter_cls = MiniMaxSplitter
    eos_ids = {0}
    max_seq_len = 1 << 16

    def __init__(self, script):
        self.tokenizer, self.script, self.fed = _Tok(), script, 0
        self.store = SimpleNamespace(stats=lambda: SimpleNamespace(cache_hits=0, cache_misses=0, reads=0,
                                                                    fast_reads=0, read_wall_seconds=0.0))

    def render_chat(self, *a, **k):
        return "x"

    def parse_tool_calls(self, text, tools=None):
        return []  # the fake tokenizer never closes a call

    def stream(self, prompt, *, max_new_tokens, temperature, top_p, cancel=None, boundary=0):
        yield ("prefill", 0, 0.0)
        finish = "length"
        for t in self.script[:max_new_tokens]:
            if cancel is not None and cancel.is_set():
                finish = "cancel"
                break
            self.fed += 1
            yield ("token", t)
        yield ("done", finish, 0.1)


def _run(script, repeats):
    glm_engine.LOOP_GUARD_REPEATS = repeats
    model = _Model(script)
    eng = GlmEngine(model, model_id="m")
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}], params=SamplingParams(max_new_tokens=5000))
    deltas = list(eng.stream_chat(req))
    return model, deltas[-1]


def test_a_looping_reply_stops_and_finishes_normally():
    script = list(range(10, 60)) + list(range(200, 230)) * 100  # 3,050 tokens, a 30-token block looping
    model, last = _run(script, 6)
    assert last.finish_reason == "stop"
    assert model.fed == 50 + 30 * 6  # stopped at the sixth complete copy
    model, last = _run(script, 0)
    assert model.fed == len(script) and last.finish_reason == "length"
    glm_engine.LOOP_GUARD_REPEATS = 6


def test_a_normal_reply_is_untouched():
    script = list(range(10, 900))
    model, last = _run(script, 6)
    assert model.fed == len(script)


def test_a_reply_looping_inside_a_tool_call_drops_the_unclosed_block():
    # HANDOFF 18.23 item 8: the loop guard cut a reply inside its (never closed) tool call, which reached Hermes as text
    script = list(range(10, 20)) + [7] + list(range(300, 330)) * 100
    glm_engine.LOOP_GUARD_REPEATS = 6
    model = _Model(script)
    eng = GlmEngine(model, model_id="m")
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}], params=SamplingParams(max_new_tokens=5000))
    deltas = list(eng.stream_chat(req))
    text = "".join(d.content for d in deltas)
    assert "<tool_call>" not in text and "w300" not in text and text.startswith("w10 ")
    assert deltas[-1].finish_reason == "stop" and not deltas[-1].tool_calls


def test_incrementing_tail_finds_an_invented_counting_list():
    # HANDOFF 18.29 item 5: 18.28's runaway listed `noto 1` ... `noto 495`, which never repeats a block exactly
    from cachalot.glm.engine import incrementing_tail
    noto = "Here is the Desktop:\n" + "".join(f"- `noto {i}` \n" for i in range(1, 70))
    assert incrementing_tail(noto, 64) == 64
    assert incrementing_tail("".join(f"- `noto {i}` \n" for i in range(1, 64)), 64) == 0  # 63 items
    websites = ", ".join(f"`shaahin.website {i}.0/`" for i in range(2, 70)) + ", "
    assert incrementing_tail(websites, 64) == 64  # comma-separated, one number of two counting
    assert incrementing_tail(noto, 64, prompt_text="noto 12\nnoto 13") == 0  # copied from the prompt, not invented


def test_incrementing_tail_ignores_lists_that_do_not_count_by_one():
    from cachalot.glm.engine import incrementing_tail
    assert incrementing_tail(", ".join(str(i) for i in range(200)) + ", ", 64) == 0  # no letters
    assert incrementing_tail("".join(f"{i}. Step {i}: part {i}\n" for i in range(1, 200)), 64) == 0  # three numbers move
    assert incrementing_tail("".join(f"Chapter {i}\n" for i in range(200, 0, -1)), 64) == 0  # counts down
    assert incrementing_tail("".join(f"Screenshot 2026-09-{i % 28 + 1:02d} at 10.{i % 60:02d}.00.png\n"
                                     for i in range(200)), 64) == 0


class _LineTok(_Tok):
    """Token i >= 1000 is the line "- `noto <i - 1000>`"; the prompt is `prompt_lines`."""

    prompt_lines = ""

    def decode(self, ids, skip_special_tokens=False):
        if list(ids) == [1, 2, 3]:
            return self.prompt_lines
        return "".join(f"- `noto {i - 1000}`\n" if i >= 1000 else f"w{i} " for i in ids)


def _run_lines(script, prompt_lines=""):
    model = _Model(script)
    model.tokenizer = _LineTok()
    model.tokenizer.prompt_lines = prompt_lines
    eng = GlmEngine(model, model_id="m")
    req = ChatRequest(messages=[{"role": "user", "content": "hi"}], params=SamplingParams(max_new_tokens=5000))
    deltas = list(eng.stream_chat(req))
    return model, deltas[-1]


def test_an_invented_counting_list_stops_the_reply():
    script = list(range(10, 20)) + list(range(1001, 1500))
    glm_engine.LOOP_GUARD_INCREMENTING = 64
    model, last = _run_lines(script)
    assert last.finish_reason == "stop"
    assert model.fed == 10 + 65  # the 65th line completes the 64th item (the last line is still growing)
    model, last = _run_lines(script, prompt_lines="".join(f"noto {i}\n" for i in range(1, 500)))
    assert model.fed == len(script)  # the listing is in the prompt
    glm_engine.LOOP_GUARD_INCREMENTING = 0
    model, last = _run_lines(script)
    assert model.fed == len(script) and last.finish_reason == "length"
    glm_engine.LOOP_GUARD_INCREMENTING = 64
