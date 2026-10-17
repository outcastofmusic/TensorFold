"""LFM2.5's chat template through the fork's ChatTemplate (the checkpoint's own file, tests/fixtures/lfm2_moe, sha256
ea663864) and its Pythonic tool calls through the CUDA server's reply parser: a call the template renders parses back
to the same name and arguments, and the reasoning splits off at ``</think>``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("jinja2")

from tensorfold.cuda import reply_text
from tensorfold.server import pythonic_calls as P
from tensorfold.server.text import split_thinking

FIXTURES = Path(__file__).parent / "fixtures" / "lfm2_moe"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather in a city",
                                           "parameters": {"type": "object", "properties": {
                                               "city": {"type": "string"}, "days": {"type": "integer"}}}}},
         {"type": "function", "function": {"name": "web-search", "parameters": {"type": "object", "properties": {
             "query": {"type": "string"}}}}}]
ARGS = {"city": "Pa'ris \"Nord\"\nline\\2", "days": 3, "ratio": -0.5, "exact": True, "none": None,
        "tags": ["a", 1, False], "opts": {"deep": {"x": None, "y": [True]}}}


@pytest.fixture(scope="module")
def template():
    from tensorfold.cuda.chat_template import ChatTemplate

    return ChatTemplate(FIXTURES)


def _history(arguments):
    return [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": "", "reasoning_content": "I need the tool.",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather", "arguments": arguments}}]},
            {"role": "tool", "content": "18C"}]


def test_the_fixture_is_the_24b_a2b_template():
    digest = hashlib.sha256((FIXTURES / "chat_template.jinja").read_bytes()).hexdigest()
    assert digest == "ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef"


def test_a_tool_turn_renders_as_liquid_documents_it(template):
    text = template.render(_history({"city": "Paris"}), tools=TOOLS, enable_thinking=True)
    assert text.startswith("<|startoftext|><|im_start|>system\nBe brief.\nList of tools: [{\"type\": \"function\"")
    assert ("<|im_start|>assistant\n<think>I need the tool.</think><|tool_call_start|>[get_weather(city='Paris')]"
            "<|tool_call_end|><|im_end|>\n<|im_start|>tool\n18C<|im_end|>\n") in text
    assert text.endswith("<|im_start|>assistant\n<think>")                  # every reply opens its thinking


def test_past_thinking_is_dropped_unless_preserved(template):
    turns = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "r1"},
             {"role": "user", "content": "c"}]
    plain = template.render(turns, tools=None, enable_thinking=True)
    assert "r1" not in plain and "<|im_start|>assistant\nb<|im_end|>" in plain
    kept = template.render(turns, tools=None, enable_thinking=True, extra={"preserve_thinking": True})
    assert "<think>r1</think>b" in kept


def test_the_fork_renders_what_transformers_renders(template):
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    messages = _history(ARGS)
    hf, _ = utils.render_jinja_template(conversations=[messages], tools=TOOLS, add_generation_prompt=True,
                                        chat_template=(FIXTURES / "chat_template.jinja").read_text(),
                                        bos_token="<|startoftext|>", eos_token="<|im_end|>")
    assert template.render(messages, tools=TOOLS, enable_thinking=True) == hf[0]


def test_a_rendered_call_parses_back_to_its_arguments(template):
    text = template.render(_history(ARGS), tools=TOOLS, enable_thinking=True)
    start = text.index(P.CALL_OPEN)
    block = text[start:text.index(P.CALL_CLOSE) + len(P.CALL_CLOSE)]
    content, calls = reply_text.parse_tool_calls("Checking." + block, TOOLS)
    assert content == "Checking." and len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == ARGS


@pytest.mark.parametrize(("written", "calls"), [
    ('[get_weather(city="Paris", days=2)]', [("get_weather", {"city": "Paris", "days": 2})]),
    ("[get_weather('Oslo', 1), web-search(query='a, b (c)')]",
     [("get_weather", {"city": "Oslo", "days": 1}), ("web-search", {"query": "a, b (c)"})]),
    ("get_weather(city='Rome')", [("get_weather", {"city": "Rome"})]),
    ('[GET_WEATHER(city="x", extra={"k": [true, null]})]',
     [("get_weather", {"city": "x", "extra": {"k": [True, None]}})]),
])
def test_the_model_s_call_forms_parse(written, calls):
    content, got = reply_text.parse_tool_calls(f"{P.CALL_OPEN}{written}{P.CALL_CLOSE}", TOOLS)
    assert content == ""
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in got] == calls


@pytest.mark.parametrize("written", ["[unknown_tool(a=1)]", "[get_weather(city=open('x'))]", "[get_weather(**k)]",
                                     "[get_weather(city='x'", "not a call", "[1, 2]"])
def test_a_call_that_does_not_parse_or_names_no_offered_tool_stays_text(written):
    text = f"Answer {P.CALL_OPEN}{written}{P.CALL_CLOSE}"
    content, calls = reply_text.parse_tool_calls(text, TOOLS)
    assert calls is None and content == text


def test_one_call_when_parallel_calls_are_off():
    text = f"{P.CALL_OPEN}[get_weather(city='a'), get_weather(city='b')]{P.CALL_CLOSE}"
    _, calls = reply_text.parse_tool_calls(text, TOOLS, max_calls=1)
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [{"city": "a"}]


QWEN_CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"


def test_a_reply_that_names_the_other_format_s_opener_keeps_its_own_calls():
    mention = f"See {P.CALL_OPEN} in LFM docs. "
    content, calls = reply_text.parse_tool_calls(mention + QWEN_CALL, TOOLS)
    assert content == mention.strip() and json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    content, calls = reply_text.parse_tool_calls(mention + QWEN_CALL, TOOLS, max_calls=1)
    assert len(calls) == 1
    cut = mention + QWEN_CALL.removesuffix("</function>\n</tool_call>")
    assert reply_text.close_call(cut, TOOLS) == "</function></tool_call>"
    lfm = f"Qwen writes <tool_call> blocks. {P.CALL_OPEN}[get_weather(city='a')]{P.CALL_CLOSE}"
    content, calls = reply_text.parse_tool_calls(lfm, TOOLS)
    assert content == "Qwen writes <tool_call> blocks."
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "a"}
    assert reply_text.close_call(f"Qwen writes <tool_call> blocks. {P.CALL_OPEN}[get_weather(city='a')", TOOLS) == \
        "]" + P.CALL_CLOSE


@pytest.mark.parametrize("parameters", ["a string", ["city"], {"properties": ["city"]}, None])
def test_a_tool_whose_parameters_are_not_an_object_takes_keyword_calls(parameters):
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": parameters}}]
    text = f"{P.CALL_OPEN}[get_weather(city='a')]{P.CALL_CLOSE}"
    _, calls = reply_text.parse_tool_calls(text, tools)
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "a"}
    content, calls = reply_text.parse_tool_calls(f"{P.CALL_OPEN}[get_weather('a')]{P.CALL_CLOSE}", tools)
    assert calls is None and P.CALL_OPEN in content


def test_a_call_cut_by_the_end_token_is_closed_when_it_parses():
    assert reply_text.close_call(f"x {P.CALL_OPEN}[get_weather(city='a')]", TOOLS) == P.CALL_CLOSE
    assert reply_text.close_call(f"x {P.CALL_OPEN}[get_weather(city='a')", TOOLS) == "]" + P.CALL_CLOSE
    assert reply_text.close_call(f"x {P.CALL_OPEN}[get_weather(city='a", TOOLS) == ""
    assert reply_text.close_call(f"{P.CALL_OPEN}[get_weather()]{P.CALL_CLOSE}", TOOLS) == ""


def test_streamed_text_hides_calls_and_holds_a_partial_opener():
    hide = reply_text.hide_tool_calls
    assert hide(f"Sure. {P.CALL_OPEN}[get_weather(city='a')]{P.CALL_CLOSE} Done.", finished=True) == "Sure.  Done."
    assert hide(f"Sure. {P.CALL_OPEN}[get_wea", finished=False) == "Sure. "
    assert hide("Sure. <|tool_ca", finished=False) == "Sure. "
    assert hide("Sure. <tool_call>{}</tool_call>", finished=True) == "Sure. "


def test_the_reply_s_reasoning_splits_off_at_the_think_close():
    reply = f"The user wants weather.</think>{P.CALL_OPEN}[get_weather(city='a')]{P.CALL_CLOSE}"
    reasoning, answer = split_thinking(reply, finished=True)
    assert reasoning == "The user wants weather." and answer.startswith(P.CALL_OPEN)
    # a call written before the think block closes is the answer once the reply ends
    early = f"Thinking {P.CALL_OPEN}[get_weather(city='a')]{P.CALL_CLOSE}"
    assert split_thinking(early, finished=True) == ("Thinking ", f"{P.CALL_OPEN}[get_weather(city='a')]{P.CALL_CLOSE}")


def test_requests_cannot_turn_thinking_off():
    pytest.importorskip("torch")
    from tensorfold.families.lfm2_moe.cuda.app import always_thinking

    body = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True}}
    assert always_thinking(body) == {"chat_template_kwargs": {"preserve_thinking": True}}
    assert always_thinking({"reasoning_effort": "high"}) == {"reasoning_effort": "high"}
