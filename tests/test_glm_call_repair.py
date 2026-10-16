"""GLM calls with a lost ``<arg_key>`` and calls the end token cut off come back as calls (upstream TensorFold 1.0.9,
#761, whose cases these are)."""

from __future__ import annotations

import json

import pytest

from tensorfold.cuda.reply_text import close_call, parse_tool_calls
from tensorfold.server.tools import repair_glm_keys

LOOKUP = [{"type": "function", "function": {"name": "lookup",
                                            "parameters": {"properties": {"n": {"type": "integer"}}}}}]
READ = [{"type": "function", "function": {"name": "read_file", "parameters": {"properties": {
    "filePath": {"type": "string"}, "limit": {"type": "integer"}}}}}]


@pytest.mark.parametrize("written, repaired", [
    ("read_file path</arg_key><arg_value>/x</arg_value>", "read_file <arg_key>path</arg_key><arg_value>/x</arg_value>"),
    ("read_file\npath</arg_key><arg_value>/x</arg_value>", "read_file\n<arg_key>path</arg_key><arg_value>/x</arg_value>"),
    ("read_file<arg_key>path</arg_key><arg_value>/x</arg_value>\nlimit</arg_key><arg_value>3</arg_value>",
     "read_file<arg_key>path</arg_key><arg_value>/x</arg_value>\n<arg_key>limit</arg_key><arg_value>3</arg_value>"),
    ("read_file path</arg_key><arg_value>/x</arg_value> limit</arg_key><arg_value>3</arg_value>",
     "read_file <arg_key>path</arg_key><arg_value>/x</arg_value> <arg_key>limit</arg_key><arg_value>3</arg_value>"),
])
def test_a_key_written_as_whitespace_gets_its_opener_back(written, repaired):
    assert repair_glm_keys(written) == repaired
    assert repair_glm_keys(repaired) == repaired


@pytest.mark.parametrize("block", [
    "read_file<arg_key>path</arg_key><arg_value>/x</arg_value>",       # nothing missing
    "read_file",
    "read_filepath</arg_key><arg_value>/x</arg_value>",                # no space: no name left before the key
    " path</arg_key><arg_value>/x</arg_value>",                        # no name at all
    '{"name": "run", "arguments": {"command": "ls"}}',
])
def test_other_blocks_come_back_as_written(block):
    assert repair_glm_keys(block) == block


@pytest.mark.parametrize("reply, arguments", [
    ("<tool_call>read_file filePath</arg_key><arg_value>/home/a/main.go</arg_value></tool_call>",
     {"filePath": "/home/a/main.go"}),
    ("<tool_call>read_file<arg_key>filePath</arg_key><arg_value>/a</arg_value>\nlimit</arg_key><arg_value>3</arg_value>"
     "</tool_call>", {"filePath": "/a", "limit": 3}),
])
@pytest.mark.parametrize("max_calls", [None, 1])
def test_a_repaired_call_parses_in_both_modes(reply, arguments, max_calls):
    content, calls = parse_tool_calls(reply, READ, max_calls=max_calls)
    assert content == "" and json.loads(calls[0]["function"]["arguments"]) == arguments


@pytest.mark.parametrize("reply, close, arguments, content", [
    ('Looking.<tool_call>\n{"name":"lookup","arguments":{"q":"x"}}\n', "</tool_call>", {"q": "x"}, "Looking."),
    ('<tool_call>{"name":"lookup","arguments":{"q":["x"', "]}}</tool_call>", {"q": ["x"]}, ""),
    ("<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n", "</function></tool_call>", {"n": 5}, ""),
    ("<tool_call>\n<function=lookup>\n", "</function></tool_call>", {}, ""),
    ("<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n", "</tool_call>", {"n": 5}, ""),
    ("<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value>", "</tool_call>", {"n": 7}, ""),
    ("Trying.<tool_call>lookup<arg_key>n</arg_key><arg_value>7", "</arg_value></tool_call>", {"n": 7}, "Trying."),
    ('<tool_call>{"name":"lookup"}</tool_call>\n<tool_call>{"name":"lookup","arguments":{"n":2}', "}</tool_call>",
     {"n": 2}, ""),
    ('<tool_call>{"name":"lookup","arguments":{"q":"<tool_call>"}', "}</tool_call>", {"q": "<tool_call>"}, ""),
])
def test_a_call_the_end_token_left_open_closes_when_it_parses_whole(reply, close, arguments, content):
    assert close_call(reply, LOOKUP) == close
    left, calls = parse_tool_calls(reply + close, LOOKUP)
    assert json.loads(calls[-1]["function"]["arguments"]) == arguments and left.strip() == content


@pytest.mark.parametrize("reply", [
    'Trying.<tool_call>{"name":"lookup","arguments":{"q":"fast ca',            # inside a string
    'Trying.<tool_call>{"name":"lookup","arguments":{"q":',                    # a key without its value
    "Trying.<tool_call>\n<function=lookup>\n<parameter=q>\nfast ca",           # inside a parameter
    "Trying.<tool_call>\n<function=lookup>\n<parameter=n>\n5\n",               # before its </parameter>
    'Trying.<tool_call>{"name":"launch rocket!","arguments":{}}',              # no function name
    "Trying.<tool_call>",                                                      # nothing written
    'Trying <tool_call> tags.<tool_call>{"name":"lookup"}',                    # an earlier opener takes the closer
])
def test_a_call_that_would_not_parse_whole_stays_text(reply):
    assert close_call(reply, LOOKUP) == ""


@pytest.mark.parametrize("reply", [
    "Done.", '<tool_call>{"name":"lookup"}</tool_call>', "<tool_call>\n<function=lookup>\n</function>\n</tool_call> Done.",
])
def test_replies_that_end_outside_a_call_close_nothing(reply):
    assert close_call(reply, LOOKUP) == ""


def test_no_tools_close_nothing():
    assert close_call("<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value>", []) == ""


class CutOffEngine:
    """Writes a JSON call and stops before its ``</tool_call>``: on the end token, or at the reply's length limit."""

    def __init__(self, end):
        from tests.test_cuda_tool_choice import EOS

        self.eos, self.end = (EOS,), end

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        from tests.test_cuda_tool_choice import OPEN, Tokens

        reply = [OPEN, *Tokens().encode('{"name":"plan_trip","arguments":{"days":3}').ids, *([self.eos[0]] if self.end else [])]
        on_tokens(reply[:max_tokens])
        return {"rounds": 1, "decode_s": 0.1}


@pytest.mark.parametrize("end", [True, False])
def test_the_server_closes_a_call_only_when_the_end_token_cut_it(tmp_path, end):
    from tests.test_cuda_admission import http_server, post
    from tests.test_cuda_tool_choice import app_for

    tools = [{"type": "function", "function": {"name": "plan_trip", "parameters": {
        "type": "object", "properties": {"days": {"type": "integer"}}}}}]
    body = {"messages": [{"role": "user", "content": "Plan it."}], "tools": tools, "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False}}
    with http_server(app_for(tmp_path, CutOffEngine(end))) as port:
        status, reply = post(port, body, True)
    assert status == 200, reply
    choice = json.loads(reply)["choices"][0]
    if end:
        assert choice["finish_reason"] == "tool_calls"
        assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"days": 3}
    else:
        assert choice["finish_reason"] == "length" and "tool_calls" not in choice["message"]
