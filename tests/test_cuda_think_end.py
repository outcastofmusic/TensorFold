"""The CUDA server's thinking prefill and its think-end gate: a request's ``thinking_prefill`` opens the think block
with its text, and an end token the model samples inside the think block closes the block instead, so the answer
follows (tf-runs/gpu-1105: 5 of 80 prefilled replies ended with their answer in the reasoning)."""

import pytest

from tests.test_cuda_thinking_controls import app_for, ask
from tests.test_cuda_tool_choice import END, EOS, Tokens

PREFILL = "## 1. Understand the problem\n"


def text(s):
    return [ord(c) for c in s]


class EndingEngine:
    """Thinks, then samples the end token without closing the block (``closes``: closes it first, as it should);
    after a close it answers. ``width`` tokens a round; records every prompt."""

    eos = (EOS,)

    def __init__(self, width=3, closes=False):
        self.width, self.closes, self.prompts = width, closes, []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.prompts.append(list(prompt))
        written = Tokens().decode(prompt)
        if written.endswith("</think>\n\n"):
            reply = text("The answer is 42.") + [EOS]
        elif self.closes:
            reply = text("thought") + [END] + text("\n\nThe answer is 42.") + [EOS]
        else:
            reply = text("thought, and the answer is 42.") + [EOS]
        reply = reply[:max_tokens]
        width = self.width if draft else 1
        for at in range(0, len(reply), width):
            if on_tokens(reply[at:at + width]) or EOS in reply[at:at + width]:
                break
        return {"rounds": 1}


def message(body, stream):
    if stream:
        return body["reasoning"], body["content"]
    got = body["choices"][0]["message"]
    return got.get("reasoning_content") or "", got.get("content") or ""


@pytest.mark.parametrize("width", [1, 3, 4, 7])
@pytest.mark.parametrize("stream", [False, True])
def test_an_end_token_inside_the_think_block_closes_it_and_the_answer_follows(tmp_path, width, stream):
    engine = EndingEngine(width)
    status, body = ask(app_for(tmp_path, engine), stream, max_tokens=200)
    assert status == 200
    assert message(body, stream) == ("thought, and the answer is 42.\n", "The answer is 42.")
    assert body["tensorfold"]["token_ids"] == [*text("thought, and the answer is 42.\n"), END, *text("\n\n"),
                                               *text("The answer is 42."), EOS]
    assert len(engine.prompts) == 2


def test_a_reply_that_closes_its_own_block_is_left_alone(tmp_path):
    engine = EndingEngine(closes=True)
    status, body = ask(app_for(tmp_path, engine), max_tokens=200)
    assert status == 200 and message(body, False) == ("thought", "The answer is 42.")
    assert len(engine.prompts) == 1


def test_with_thinking_off_the_end_token_ends_the_reply(tmp_path):
    engine = EndingEngine()
    status, body = ask(app_for(tmp_path, engine, thinking=False), max_tokens=200)
    assert status == 200 and message(body, False) == ("", "thought, and the answer is 42.")
    assert len(engine.prompts) == 1


def test_ignore_eos_leaves_the_end_token_to_the_request(tmp_path):
    engine = EndingEngine()
    status, body = ask(app_for(tmp_path, engine), max_tokens=200, ignore_eos=True)
    assert status == 200 and len(engine.prompts) == 1 and END not in body["tensorfold"]["token_ids"]


@pytest.mark.parametrize("stream", [False, True])
def test_the_prefill_opens_the_think_block_and_the_reasoning_is_the_models_text_after_it(tmp_path, stream):
    engine = EndingEngine(closes=True)
    status, body = ask(app_for(tmp_path, engine), stream, max_tokens=200, thinking_prefill=PREFILL)
    assert status == 200
    assert Tokens().decode(engine.prompts[0]).endswith("assistant:<think>" + PREFILL)
    assert engine.prompts[0][-len(PREFILL):] == text(PREFILL)
    assert message(body, stream) == ("thought", "The answer is 42.")


def test_an_empty_prefill_is_none(tmp_path):
    engine = EndingEngine(closes=True)
    status, _ = ask(app_for(tmp_path, engine), max_tokens=200, thinking_prefill="")
    assert status == 200 and Tokens().decode(engine.prompts[0]).endswith("assistant:<think>")


@pytest.mark.parametrize("fields, why", [
    ({"thinking_prefill": 7}, "must be a string"),
    ({"thinking_prefill": "plan</think>answer"}, "special token"),
    ({"thinking_prefill": PREFILL, "chat_template_kwargs": {"enable_thinking": False}}, "needs thinking on"),
    ({"thinking_prefill": PREFILL, "continue_final_message": True,
      "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "So"}]}, "needs thinking on"),
])
def test_a_prefill_the_server_cannot_place_is_refused(tmp_path, fields, why):
    engine = EndingEngine()
    status, body = ask(app_for(tmp_path, engine), max_tokens=200, **fields)
    assert status == 400 and why in str(body) and not engine.prompts


SENTENCE = "I will multiply the numbers.\n"


class EarlyCloseEngine:
    """Samples ``</think>`` first when its prompt ends in the prefill, as Qwen3.8 Flash Next does after a sentence and
    one newline; after a blank line it thinks and closes the block itself. Records every prompt."""

    eos = (EOS,)

    def __init__(self, width=3):
        self.width, self.prompts = width, []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.prompts.append(list(prompt))
        if Tokens().decode(prompt).endswith(SENTENCE):
            reply = [END] + text("\n\nThe answer is 391.") + [EOS]
        else:
            reply = text("17 times 23 is 391.") + [END] + text("\n\nThe answer is 391.") + [EOS]
        reply = reply[:max_tokens]
        width = self.width if draft else 1
        for at in range(0, len(reply), width):
            if on_tokens(reply[at:at + width]) or EOS in reply[at:at + width]:
                break
        return {"rounds": 1}


@pytest.mark.parametrize("width", [1, 3, 5])
@pytest.mark.parametrize("stream", [False, True])
def test_a_close_straight_after_the_prefill_becomes_a_blank_line_and_the_model_thinks(tmp_path, width, stream):
    engine = EarlyCloseEngine(width)
    status, body = ask(app_for(tmp_path, engine), stream, max_tokens=200, thinking_prefill=SENTENCE)
    assert status == 200
    reasoning, content = message(body, stream)
    assert reasoning.strip() == "17 times 23 is 391." and content == "The answer is 391."
    assert body["tensorfold"]["token_ids"][:2] == text("\n\n")
    assert len(engine.prompts) == 2 and engine.prompts[1][-2:] == text("\n\n")


def test_without_a_prefill_a_first_close_is_left_alone(tmp_path):
    class Closes(EarlyCloseEngine):
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            self.prompts.append(list(prompt))
            on_tokens([END, *text("\n\nThe answer is 391."), EOS])
            return {"rounds": 1}

    engine = Closes()
    status, body = ask(app_for(tmp_path, engine), max_tokens=200)
    assert status == 200 and message(body, False) == ("", "The answer is 391.")
    assert len(engine.prompts) == 1


def test_a_close_after_the_prefilled_reply_has_begun_is_left_alone(tmp_path):
    engine = EndingEngine(closes=True)
    status, body = ask(app_for(tmp_path, engine), max_tokens=200, thinking_prefill=PREFILL)
    assert status == 200 and message(body, False) == ("thought", "The answer is 42.")
    assert len(engine.prompts) == 1
