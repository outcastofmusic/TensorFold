"""The token loop detector: a reasoning block that repeats itself is closed with a hidden phrase, once."""

import math
import random

import pytest

from tensorfold.engine.think_loop import CLOSE_TEXT, LoopDetector, LoopGate, hide

END = 9999


def first_fire(tokens):
    d = LoopDetector()
    for i, t in enumerate(tokens):
        why = d.observe(t)
        if why:
            return i, why
    return None, None


def varied(n, seed=0, vocab=5000):
    rng = random.Random(seed)
    return [rng.randrange(vocab) for _ in range(n)]


@pytest.mark.parametrize("size", [1, 2, 3, 7, 16])
def test_a_pattern_fires_once_it_repeats_enough_times_to_span_128_tokens(size):
    count = max(4, math.ceil(128 / size))
    unit = list(range(100, 100 + size))
    head = varied(50, 1)
    at, why = first_fire(head + unit * (count + 3))
    # the count-th copy's last token: each token then equals the one size back for (count - 1) * size tokens
    assert at == len(head) + count * size - 1 and why.startswith(f"loop(size={size}")


def test_a_short_pattern_that_spans_less_than_128_tokens_does_not_fire():
    assert first_fire(varied(50, 2) + [5, 6] * 60 + varied(400, 3))[0] is None      # 120 tokens of "5 6"


def test_varied_text_never_fires():
    assert first_fire(varied(6000, 4))[0] is None


def test_the_history_arm_fires_on_two_checks_in_a_row_past_2048():
    rng = random.Random(5)
    block = [rng.randrange(5000) for _ in range(700)]           # repeated whole: no pattern of 16 tokens or fewer
    seq = varied(1700, 6) + block + block + block
    at, why = first_fire(seq)
    assert why.startswith("repeat16") and at % 64 == 63 and at >= 2047
    # the first check that sees 70% of its window repeated does not fire alone
    d, shares = LoopDetector(), []
    for i, t in enumerate(seq[:at]):
        d.observe(t)
        if (i + 1) % 64 == 0 and (i + 1) >= 2048:
            shares.append(d.share)
    assert shares[-1] >= 0.7 and (len(shares) < 2 or shares[-2] < 0.7)


def test_the_novelty_arm_fires_when_the_last_128_tokens_bring_almost_no_new_8_grams():
    rng = random.Random(7)
    cycle = [rng.randrange(5000) for _ in range(100)]           # a 100-token cycle: longer than any pattern
    at, why = first_fire(varied(200, 8) + cycle * 6)
    assert why.startswith("novelty") and at < 200 + 600


def test_the_gate_cuts_at_the_firing_token_without_committing_its_lookahead():
    unit = [11, 12, 13, 14]
    seq = varied(30, 9) + unit * 40
    at, _ = first_fire(seq)
    close = [1, 2, END, 3]
    gate = LoopGate(close, END)
    for t in seq[:at - 5]:
        assert gate.cut([t]) is None
        gate.observe(t)
    batch = seq[at - 5:at + 3]
    assert gate.cut(batch) == (5, close)
    assert gate.cut(batch) == (5, close)                         # a look ahead changes nothing
    for t in batch[:5] + close:
        gate.observe(t)
    assert not gate.open and gate.cut([11] * 50) is None         # one close a reply
    assert gate.record["reason"].startswith("loop(size=4") and gate.record["at"] == at


def test_the_gate_stops_watching_once_the_model_closes_the_block():
    gate = LoopGate([1, END], END)
    for t in varied(20, 10) + [END]:
        gate.observe(t)
    assert gate.cut([5] * 400) is None


def test_hiding_takes_the_phrase_out_and_holds_back_a_partial_one():
    assert hide("abc" + CLOSE_TEXT, CLOSE_TEXT, finished=True) == "abc"
    assert hide("abc" + CLOSE_TEXT[:20], CLOSE_TEXT, finished=False) == "abc"
    assert hide("abc\n", CLOSE_TEXT, finished=False) == "abc"


# ---- the CUDA server: a looping reply is closed, the phrase reaches the model and not the client --------------

from tests.test_cuda_thinking_controls import ChainEngine, after, app_for, ask  # noqa: E402
from tests.test_cuda_tool_choice import END as THINK_END  # noqa: E402
from tests.test_cuda_tool_choice import THINK, Tokens  # noqa: E402


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("width", [1, 4])
def test_the_server_closes_a_looping_think_block_with_a_hidden_phrase(tmp_path, monkeypatch, stream, width):
    monkeypatch.setenv("TENSORFOLD_THINK_LOOP", "1")
    status, body = ask(app_for(tmp_path, ChainEngine(width)), stream, max_tokens=400)
    assert status == 200
    ids = body["tensorfold"]["token_ids"]
    close = Tokens().encode(CLOSE_TEXT).ids + [THINK_END]
    at = next(i for i in range(len(ids)) if ids[i:i + len(close)] == close)
    reasoning = body["reasoning"] if stream else body["choices"][0]["message"]["reasoning_content"]
    assert reasoning == Tokens().decode(ids[:at])                    # the phrase never reaches the client
    assert "Considering" not in reasoning and at < 200
    chain, last = [], THINK                                          # the reply up to the cut is the engine's own
    for _ in range(at):
        last = after(last)
        chain.append(last)
    assert ids[:at] == chain
    assert body["tensorfold"]["think_loop"]["at"] == at


def test_the_server_leaves_a_looping_reply_alone_when_the_flag_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv("TENSORFOLD_THINK_LOOP", raising=False)
    status, body = ask(app_for(tmp_path, ChainEngine(4)), False, max_tokens=400)
    assert status == 200 and THINK_END not in body["tensorfold"]["token_ids"]
    assert "think_loop" not in body["tensorfold"]


def test_an_inline_cut_writes_the_close_into_the_live_stream_once():
    from tensorfold.engine.call_gate import generate_gated
    from tests.test_inline_cut import _engine

    seq = varied(30, 11) + [21, 22, 23] * 60
    at, _ = first_fire(seq)
    close = [1, 2, END, 3]
    gate, sent, runs = LoopGate(close, END), [], []
    rounds = [seq[i:i + 5] for i in range(0, len(seq), 5)]
    generate_gated(None, [7], 400, [gate], lambda new: sent.extend(new) or False, _engine(rounds, runs))
    assert sent == seq[:at] + close and runs == [[7]]
    assert gate.record["at"] == at and not gate.open


def test_hiding_takes_out_a_phrase_that_max_tokens_cut_short():
    # streamed, the tail waited as a possible phrase start; finished, it must not appear either
    assert hide("abc" + CLOSE_TEXT[:20], CLOSE_TEXT, finished=True) == "abc"


def test_hiding_keeps_the_phrase_where_the_model_wrote_it_itself():
    mine = "a" + CLOSE_TEXT + "b"                              # the model's own words, before the loop fired
    assert hide(mine + CLOSE_TEXT, CLOSE_TEXT, finished=True) == mine
    assert hide(mine + CLOSE_TEXT[:9], CLOSE_TEXT, finished=False) == mine
