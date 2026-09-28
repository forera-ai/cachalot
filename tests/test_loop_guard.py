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
        return "".join(f"w{i} " for i in ids)

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
