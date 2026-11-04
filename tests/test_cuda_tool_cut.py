"""A tool call that max_tokens cuts is the reply's tool call, streamed and whole, with finish_reason "length"
(upstream #1238 for XML calls, #1298 for GLM's)."""

import pytest

pytest.importorskip("jinja2")

from tests.test_cuda_server_disconnect import MESSAGES, PacedEngine, app_for

WRITE = [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}, "lines": {"type": "integer"}}}}}]
XML = ("Sure.\n<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n"
       "<parameter=content>\nThe history of")
GLM = "Sure.\n<tool_call>write_file<arg_key>path</arg_key><arg_value>notes.txt</arg_value><arg_key>content</arg_key><arg_value>The hist"
CUT_XML = '{"path":"notes.txt","content":"The history of'
CUT_GLM = '{"path":"notes.txt","content":"The hist'


def reply(tmp_path, text, *, stream, max_tokens=None, **extra):
    """The reply's finish, content and calls as a client puts them together, from the deltas and the end's calls."""

    app = app_for(tmp_path, PacedEngine(text))
    body = {"messages": MESSAGES, "tools": WRITE, "max_tokens": max_tokens or len(text), "stream": stream, **extra}
    deltas = []
    result = app.run(body, True, lambda d: deltas.append(d) or True, prepared=app.prepare(body, True))
    calls: dict[int, dict] = {}
    content = ""
    for delta in (deltas if stream else []):
        content += delta.get("content", "")
        for part in delta.get("tool_calls", []):
            call = calls.setdefault(part["index"], {"name": "", "arguments": ""})
            call["name"] += part.get("function", {}).get("name", "")
            call["arguments"] += part["function"].get("arguments", "")
    if stream:
        content += result["final"].get("content", "")
    else:
        content = result["content"]
    for i, call in enumerate(result["calls"] or []):
        if not stream or i >= result["calls_streamed"]:
            calls[i] = {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
    return result["finish"], content.strip(), [calls[i] for i in sorted(calls)]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text, arguments", [(XML, CUT_XML), (GLM, CUT_GLM)], ids=["xml", "glm"])
def test_a_cut_call_comes_back_as_the_call_with_its_arguments_up_to_the_cut(tmp_path, stream, text, arguments):
    assert reply(tmp_path, text, stream=stream) == ("length", "Sure.", [{"name": "write_file", "arguments": arguments}])


@pytest.mark.parametrize("stream", [False, True])
def test_whole_calls_before_the_cut_one_come_back_too(tmp_path, stream):
    first = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>\n"
    finish, content, calls = reply(tmp_path, "Two files.\n" + first + XML[len("Sure.\n"):], stream=stream)
    assert finish == "length" and content == "Two files."
    assert calls == [{"name": "write_file", "arguments": '{"path":"a.txt"}'},
                     {"name": "write_file", "arguments": CUT_XML}]


@pytest.mark.parametrize("stream", [False, True])
def test_a_cut_typed_value_is_sent_as_written(tmp_path, stream):
    text = "<tool_call>\n<function=write_file>\n<parameter=lines>\n12"
    assert reply(tmp_path, text, stream=stream)[2] == [{"name": "write_file", "arguments": '{"lines":12'}]


@pytest.mark.parametrize("text", [
    "Sure.\n<tool_call>write_fi",                                          # GLM: the name may be cut
    "Sure.\n<tool_call>write_file<arg_key>pa",                             # GLM: no key has closed
    "Sure.\n<tool_call>delete_all<arg_key>path</arg_key><arg_value>/",      # GLM: a tool not offered
    "Sure.\n<tool_call>\n<function=delete_all>\n<parameter=path>\n/",      # XML: a tool not offered
    "Sure.\n<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"pa",  # JSON: stays text, as malformed ones do
])
def test_other_cut_envelopes_stay_text(tmp_path, text):
    finish, content, calls = reply(tmp_path, text, stream=False)
    assert (finish, calls) == ("length", []) and content == text.strip()


def test_one_call_requests_keep_hiding_a_cut_call(tmp_path):
    finish, content, calls = reply(tmp_path, XML, stream=False, parallel_tool_calls=False)
    assert finish == "length" and calls == [] and "write_file" not in content


def test_a_call_the_end_token_closes_is_still_a_whole_call(tmp_path):
    whole = XML + "\n</parameter>\n</function>\n</tool_call>"
    finish, _, calls = reply(tmp_path, whole, stream=False, max_tokens=len(whole) + 5)
    assert finish == "tool_calls" and calls[0]["arguments"] == '{"path":"notes.txt","content":"The history of"}'
