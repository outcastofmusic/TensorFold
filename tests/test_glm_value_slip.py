"""A GLM value that runs into the next pair, or never closes, keeps the call as text instead of making a call with
corrupt arguments, in whole replies, streams and cut replies (upstream #2801)."""

import json

import pytest

from tensorfold.cuda.server import parse_tool_calls
from tensorfold.server.tools import parse_tool_calls_from_content

TOOLS = [{"type": "function", "function": {"name": "session_search", "parameters": {"type": "object", "properties": {
    "query": {"type": "string"}, "arguments": {"type": "string"}}}}}]
RAN_ON = ("<tool_call>session_search<arg_key>query</arg_key><arg_value>decided<arg_key>arguments</arg_key>"
          '<arg_value>{"query": "decided", "limit": 10}</arg_value></tool_call>')
UNCLOSED = ("<tool_call>session_search<arg_key>query</arg_key><arg_value>decided</arg_value>"
            '<arg_key>arguments</arg_key><arg_value>{"limit": 10}</tool_call>')
PARSERS = pytest.mark.parametrize("parser", [parse_tool_calls, parse_tool_calls_from_content],
                                  ids=["cuda-server", "http-server"])


@PARSERS
@pytest.mark.parametrize("text", [RAN_ON, UNCLOSED], ids=["ran-on", "unclosed"])
def test_a_slipped_value_keeps_the_envelope_as_text(parser, text):
    assert parser(text, TOOLS) == (text, None)


@PARSERS
def test_a_value_with_a_lone_angle_bracket_is_still_a_call(parser):
    text = "<tool_call>session_search<arg_key>query</arg_key><arg_value>2 < 3</arg_value></tool_call>"
    content, calls = parser(text, TOOLS)
    assert content == "" and json.loads(calls[0]["function"]["arguments"]) == {"query": "2 < 3"}


def test_whole_and_streamed_replies_agree_on_a_slipped_value(tmp_path):
    pytest.importorskip("jinja2")
    from tests.test_cuda_tool_cut import reply

    for text in (RAN_ON, UNCLOSED):
        text = text.replace("session_search", "write_file").replace("query", "path").replace("arguments", "content")
        end = len(text) + 3                                # the engine's x tokens after the text: never a call
        whole = reply(tmp_path, text, stream=False, max_tokens=end)
        streamed = reply(tmp_path, text, stream=True, max_tokens=end)
        assert whole[2] == [] and "<arg_key>" in whole[1]
        assert streamed == whole


def test_a_cut_value_that_ran_into_the_next_pair_stays_text(tmp_path):
    pytest.importorskip("jinja2")
    from tests.test_cuda_tool_cut import reply

    text = "Sure.\n<tool_call>write_file<arg_key>path</arg_key><arg_value>a.txt<arg_key>content</arg_key><arg_value>The"
    for stream in (False, True):
        finish, content, calls = reply(tmp_path, text, stream=stream)
        assert finish == "length" and calls == [] and content == text
