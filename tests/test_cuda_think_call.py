"""A tool call opened inside the think block moves to the answer, where the parser reads it, and a call the parser
leaves as text (an undeclared name, or a request without tools) is named in the reply's ``tensorfold`` block. After a
thinking prefill 13 of 63 agent replies wrote a call inside the thinking, and 2 of those calls never ran."""

import pytest

from tensorfold.cuda.server import unparsed_calls
from tests.test_cuda_think_end import message, text
from tests.test_cuda_thinking_controls import app_for, ask
from tests.test_cuda_tool_choice import CALL, CLOSE, END, EOS, OPEN, TOOLS, Tokens


class CallingEngine:
    """Thinks, then opens a call inside the think block (``closes``: closes the block first, as it should). After a
    close it writes the call. ``width`` tokens a round; records every prompt."""

    eos = (EOS,)

    def __init__(self, width=3, closes=False, tool="get_weather"):
        self.width, self.closes, self.tool, self.prompts = width, closes, tool, []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.prompts.append(list(prompt))
        call = Tokens().encode(CALL.replace("get_weather", self.tool)).ids + [CLOSE, EOS]
        if prompt[-1] == OPEN:                        # the gate's fix ends in the opener: the call's body follows
            reply = call
        elif self.closes:
            reply = text("I need the weather.") + [END] + text("\n\n") + [OPEN] + call
        else:
            reply = text("I need the weather.\n\n") + [OPEN] + call
        reply = reply[:max_tokens]
        width = self.width if draft else 1
        for at in range(0, len(reply), width):
            if on_tokens(reply[at:at + width]) or EOS in reply[at:at + width]:
                break
        return {"rounds": 1}


def calls(body, stream):
    if stream:
        return body.get("calls")
    return body["choices"][0]["message"].get("tool_calls")


@pytest.mark.parametrize("width", [1, 3, 7])
def test_a_call_opened_inside_the_think_block_moves_to_the_answer_and_is_parsed(tmp_path, width):
    engine = CallingEngine(width)
    status, body = ask(app_for(tmp_path, engine), max_tokens=200, tools=TOOLS)
    assert status == 200
    got = body["choices"][0]
    assert got["finish_reason"] == "tool_calls"
    assert [c["function"]["name"] for c in got["message"]["tool_calls"]] == ["get_weather"]
    assert (got["message"].get("reasoning_content") or "").strip() == "I need the weather."
    assert "<tool_call>" not in (got["message"].get("content") or "")
    assert body["tensorfold"]["think_call_moved"] is True
    assert body["tensorfold"]["token_ids"][:len(text("I need the weather.\n\n\n"))] == text("I need the weather.\n\n\n")
    assert len(engine.prompts) == 2


def test_a_call_after_the_block_closes_is_left_alone(tmp_path):
    engine = CallingEngine(closes=True)
    status, body = ask(app_for(tmp_path, engine), max_tokens=200, tools=TOOLS)
    assert status == 200 and body["choices"][0]["finish_reason"] == "tool_calls"
    assert len(engine.prompts) == 1 and "think_call_moved" not in body["tensorfold"]


def test_without_tools_the_gate_stays_off_and_the_text_call_is_named(tmp_path):
    engine = CallingEngine(closes=True, tool="bash")
    status, body = ask(app_for(tmp_path, engine), max_tokens=200)
    assert status == 200 and len(engine.prompts) == 1
    assert "<tool_call>" in message(body, False)[1]
    assert body["tensorfold"]["unparsed_tool_calls"] == ["bash"]


def test_an_undeclared_tool_stays_text_and_is_named(tmp_path):
    engine = CallingEngine(closes=True, tool="memory")
    status, body = ask(app_for(tmp_path, engine), max_tokens=200, tools=TOOLS)
    assert status == 200 and body["choices"][0]["finish_reason"] == "stop"
    assert body["tensorfold"]["unparsed_tool_calls"] == ["memory"]


def test_a_parsed_call_leaves_no_marker(tmp_path):
    engine = CallingEngine(closes=True)
    status, body = ask(app_for(tmp_path, engine), max_tokens=200, tools=TOOLS)
    assert status == 200 and "unparsed_tool_calls" not in body["tensorfold"]


@pytest.mark.parametrize("content, names", [
    ("plain answer", []),
    ("<tool_call>\n<function=memory>\n<parameter=fact>x", ["memory"]),
    ('<tool_call>{"name": "lookup", "arguments": {}}</tool_call> and <tool_call>broken', ["lookup", ""]),
])
def test_unparsed_calls_names_each_call_left_in_the_text(content, names):
    assert unparsed_calls(content) == names


@pytest.mark.parametrize("width", [1, 3, 7])
def test_a_streamed_reply_keeps_the_moved_call_out_of_reasoning_and_content(tmp_path, width):
    status, body = ask(app_for(tmp_path, CallingEngine(width)), True, max_tokens=200, tools=TOOLS)
    assert status == 200 and body["tensorfold"]["think_call_moved"] is True
    assert body["reasoning"].strip() == "I need the weather." and "<tool_call>" not in body["content"]
