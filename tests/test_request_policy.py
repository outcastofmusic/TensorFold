import copy
import json
from itertools import pairwise

import pytest

from tensorfold.engine.tool_draft import ToolCallStreamer
from tests.test_server_openai_compat import FakeApp, post_json, serve_fake


TOOLS = [{"type": "function", "function": {"name": "measure", "parameters": {
    "type": "object", "properties": {"value": {"type": "integer"}}}}}]
CALL = '<tool_call><function=measure><parameter=value>{}</parameter></function></tool_call>'


class PolicyApp(FakeApp):
    streams_prose_with_tools = True

    def __init__(self, *, content="Hello", incremental=False):
        super().__init__(content=content)
        self.incremental = incremental
        self.chats = 0

    def chat(self, messages, *, on_delta=None, tools=None, **kwargs):
        self.chats += 1
        self.messages = messages
        reply = super().chat(messages, tools=tools, **kwargs)
        if on_delta is not None and self.incremental:
            streamer = ToolCallStreamer(tools)
            for stop in range(1, len(self.content) + 1):
                for delta in streamer.feed(self.content[:stop]):
                    on_delta(delta)
            reply["tool_calls_streamed"] = streamer.streamed
        return reply


def request(app, body):
    server = serve_fake(app)
    try:
        return post_json(server, "/v1/chat/completions", body)
    finally:
        server.shutdown()
        server.server_close()


def stream_calls(body):
    calls = {}
    for line in body.splitlines():
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[5:])
        assert "tensorfold_error" not in chunk and "error" not in chunk
        for choice in chunk.get("choices", []):
            for call in choice["delta"].get("tool_calls", []):
                target = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                target["name"] += call.get("function", {}).get("name", "")
                target["arguments"] += call.get("function", {}).get("arguments", "")
    return calls


@pytest.mark.parametrize("stream", [False, True])
def test_developer_and_late_system_messages_are_preserved(stream):
    messages = [{"role": "developer", "content": "First."},
                {"role": "system", "content": "Second."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "developer", "content": "Third."},
                {"role": "user", "content": "Continue"}]
    original = copy.deepcopy(messages)
    app = PolicyApp()
    status, body = request(app, {"messages": messages, "stream": stream})
    assert status == 200
    if stream:
        stream_calls(body)
    # a later instruction keeps its place; the app's template decides its role (tests/test_message_roles.py)
    assert app.messages == [{"role": "system", "content": "First.\n\nSecond."},
                            messages[2], messages[3], {"role": "system", "content": "Third."}, messages[5]]
    assert messages == original


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("part", [
    {"type": "image_url", "image_url": {"url": "https://example.com/picture.png"}},
    {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
    {"type": "video_url", "video_url": {"url": "https://example.com/movie.mp4"}},
])
def test_unsupported_modalities_refuse_before_chat_or_stream(part, stream):
    app = PolicyApp()
    status, body = request(app, {"stream": stream, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe it."}, part]}]})
    assert status == 400
    assert app.chats == 0
    assert "text" in json.loads(body)["error"]["message"].lower()
    assert not body.startswith("data:")


def test_text_parts_join_without_losing_text_or_mutating_input():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "first "}, {"type": "text", "text": "second"}]}]
    original = copy.deepcopy(messages)
    app = PolicyApp()
    assert request(app, {"messages": messages})[0] == 200
    assert app.messages == [{"role": "user", "content": "first second"}]
    assert messages == original


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
def test_parallel_false_returns_only_first_complete_call(stream, incremental):
    app = PolicyApp(content=CALL.format(1) + CALL.format(2), incremental=incremental)
    status, body = request(app, {"messages": [{"role": "user", "content": "Measure twice"}],
                               "tools": TOOLS, "stream": stream, "parallel_tool_calls": False})
    assert status == 200
    if stream:
        calls = stream_calls(body)
        assert set(calls) == {0}
        assert calls[0]["name"] == "measure"
        assert json.loads(calls[0]["arguments"]) == {"value": 1}
        assert '"finish_reason": "tool_calls"' in body and "data: [DONE]" in body
    else:
        calls = json.loads(body)["choices"][0]["message"]["tool_calls"]
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"value": 1}


def test_parallel_false_does_not_emit_an_incomplete_call():
    app = PolicyApp(content='<tool_call><function=measure><parameter=value>1', incremental=True)
    status, body = request(app, {"messages": [{"role": "user", "content": "Measure"}],
                               "tools": TOOLS, "stream": True, "parallel_tool_calls": False})
    assert status == 200
    assert stream_calls(body) == {}


@pytest.mark.parametrize("parallel", [None, True])
def test_default_parallel_calls_remain_available(parallel):
    body = {"messages": [{"role": "user", "content": "Measure"}], "tools": TOOLS, "stream": True}
    if parallel is not None:
        body["parallel_tool_calls"] = parallel
    app = PolicyApp(content=CALL.format(1) + CALL.format(2), incremental=True)
    status, raw = request(app, body)
    assert status == 200 and set(stream_calls(raw)) == {0, 1}


def test_parallel_flag_requires_a_boolean_before_streaming():
    app = PolicyApp()
    status, body = request(app, {"messages": [{"role": "user", "content": "Hi"}],
                               "parallel_tool_calls": "false", "stream": True})
    assert status == 400 and app.chats == 0
    assert "parallel_tool_calls" in json.loads(body)["error"]["message"]


@pytest.mark.parametrize("stream", [False, True])
def test_single_call_skips_malformed_parameters_instead_of_inventing_empty_args(stream):
    malformed = '<tool_call><function=measure><parameter=value>broken</function></tool_call>'
    app = PolicyApp(content=malformed + CALL.format(2), incremental=True)
    status, body = request(app, {"messages": [{"role": "user", "content": "Measure"}],
        "tools": TOOLS, "stream": stream, "parallel_tool_calls": False})
    assert status == 200
    calls = stream_calls(body) if stream else json.loads(body)["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1
    arguments = calls[0]["arguments"] if stream else calls[0]["function"]["arguments"]
    assert json.loads(arguments) == {"value": 2}


@pytest.mark.parametrize("stream", [False, True])
def test_single_call_ignores_a_malformed_extra_call(stream):
    app = PolicyApp(content=CALL.format(1) + '<tool_call>broken</tool_call>', incremental=True)
    status, body = request(app, {"messages": [{"role": "user", "content": "Measure"}],
        "tools": TOOLS, "stream": stream, "parallel_tool_calls": False})
    assert status == 200
    calls = stream_calls(body) if stream else json.loads(body)["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1


@pytest.mark.parametrize("options", [
    {"images": ["https://example.com/picture.png"]}, {"modalities": ["audio"]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi",
        "image_url": {"url": "https://example.com/picture.png"}}]}]},
])
def test_hidden_or_requested_media_is_not_silently_ignored(options):
    app = PolicyApp()
    status, body = request(app, {"messages": [{"role": "user", "content": "Hi"}],
                               "stream": True, **options})
    assert status == 400 and app.chats == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("arguments", ['{"value":NaN}', '"{\\"value\\":1e999}"'])
def test_single_call_skips_nonfinite_json_arguments(stream, arguments):
    bad = '<tool_call>{"name":"measure","arguments":' + arguments + '}</tool_call>'
    app = PolicyApp(content=bad + CALL.format(2), incremental=True)
    status, body = request(app, {"messages": [{"role": "user", "content": "Measure"}],
        "tools": TOOLS, "stream": stream, "parallel_tool_calls": False})
    assert status == 200
    calls = stream_calls(body) if stream else json.loads(body)["choices"][0]["message"]["tool_calls"]
    arguments = calls[0]["arguments"] if stream else calls[0]["function"]["arguments"]
    assert len(calls) == 1 and json.loads(arguments) == {"value": 2}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("content, visible", [
    ('Before<m:tool_call><function=measure><parameter=value>1', 'Before'),
    ('Before<m', 'Before<m'), ('Before<math>x</math>', 'Before<math>x</math>'),
])
def test_single_call_hides_incomplete_namespaced_envelopes(stream, content, visible):
    class ProseApp(PolicyApp):
        def chat(self, messages, *, on_delta=None, **kwargs):
            reply = super().chat(messages, **kwargs)
            if on_delta is not None:
                for character in self.content:
                    on_delta(character)
            return reply

    app = ProseApp(content=content)
    status, body = request(app, {"messages": [{"role": "user", "content": "Measure"}],
        "tools": TOOLS, "stream": stream, "parallel_tool_calls": False})
    assert status == 200
    if stream:
        assert stream_calls(body) == {}
        content = "".join(json.loads(line[5:])["choices"][0]["delta"].get("content", "")
                          for line in body.splitlines() if line.startswith("data:") and line != "data: [DONE]")
    else:
        message = json.loads(body)["choices"][0]["message"]
        assert not message.get("tool_calls")
        content = message["content"]
    assert content == visible


LFM2_CALL = "<|tool_call_start|>[get_weather(city='a')]<|tool_call_end|>"


def test_single_call_hides_lfm2_s_pythonic_envelope():
    from tensorfold.server.tool_policy import ToolCallPolicy

    policy = ToolCallPolicy({"parallel_tool_calls": False})
    assert policy.content(f"Sure. {LFM2_CALL}", finished=False) == "Sure. "
    assert policy.content(f"Sure. {LFM2_CALL} Done.") == "Sure.  Done."
    assert policy.content("Sure. <|tool_ca", finished=False) == "Sure. "
    assert policy.content("a <|im_end|> b <|x") == "a <|im_end|> b <|x"
    streamed = ToolCallPolicy({"parallel_tool_calls": False})
    shown = "".join(streamed.delta(ch) for ch in f"Sure. {LFM2_CALL} Done.") + streamed.flush()
    assert shown == "Sure.  Done."


# Gemma 4's and DeepSeek-V4's envelopes, as upstream's server goldens script them (tools/gemma_tool, tools/dsml_tool)
LOOKUP = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {
    "query": {"type": "string"}, "limit": {"type": "integer"}}}}}]
GEMMA_CALL = '<|tool_call>call:lookup{query:<|"|>hi, there<|"|>,limit:2}<tool_call|>'
GEMMA_BARE_CALL = '<|tool_call>:lookup{query:<|"|>bare<|"|>,limit:5}<tool_call|>'      # the 26B's form (4447ac30)
DSML_CALL = ('<｜DSML｜tool_calls>\n<｜DSML｜invoke name="lookup">\n'
             '<｜DSML｜parameter name="query" string="true">fast</｜DSML｜parameter>\n'
             '<｜DSML｜parameter name="limit" string="false">4</｜DSML｜parameter>\n'
             '</｜DSML｜invoke>\n</｜DSML｜tool_calls>')
DSML_SECOND = DSML_CALL.replace("fast", "slow").replace(">4<", ">7<")
GEMMA_SECOND = GEMMA_CALL.replace("hi, there", "again").replace("limit:2", "limit:3")
QWEN_LOOKUP = '<tool_call>\n<function=lookup>\n<parameter=query>\nParis\n</parameter>\n</function>\n</tool_call>'


class RawProseApp(PolicyApp):
    """An app that streams its reply's raw text, markup included, cut at ``splits``, as a token decoder would."""

    def __init__(self, *, content, splits=None):
        super().__init__(content=content)
        self.splits = list(range(1, len(content))) if splits is None else splits

    def chat(self, messages, *, on_delta=None, **kwargs):
        reply = super().chat(messages, **kwargs)
        if on_delta is not None:
            cuts = [0, *self.splits, len(self.content)]
            for start, end in pairwise(cuts):
                on_delta(self.content[start:end])
        return reply


def reply_of(app, *, stream, parallel):
    status, body = request(app, {"messages": [{"role": "user", "content": "Look it up"}], "tools": LOOKUP,
                                 "stream": stream, "parallel_tool_calls": parallel})
    assert status == 200
    if not stream:
        message = json.loads(body)["choices"][0]["message"]
        return message["content"] or "", [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                                          for c in message.get("tool_calls") or []]
    content = "".join(json.loads(line[5:])["choices"][0]["delta"].get("content", "")
                      for line in body.splitlines() if line.startswith("data:") and line != "data: [DONE]")
    calls = [(c["name"], json.loads(c["arguments"])) for _, c in sorted(stream_calls(body).items())]
    return content, calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("first, second, arguments", [
    (GEMMA_CALL, GEMMA_SECOND, {"query": "hi, there", "limit": 2}),
    (GEMMA_BARE_CALL, GEMMA_SECOND, {"query": "bare", "limit": 5}),
    (DSML_CALL, DSML_SECOND, {"query": "fast", "limit": 4}),
])
def test_single_call_returns_gemma_s_and_dsml_s_first_call_and_hides_the_rest(stream, first, second, arguments):
    app = RawProseApp(content=f"Sure. {first}{second} Done.")
    content, calls = reply_of(app, stream=stream, parallel=False)
    assert calls == [("lookup", arguments)]
    assert content == ("Sure.  Done." if stream else "")         # a message with calls has null content


@pytest.mark.parametrize("first, second", [(GEMMA_CALL, GEMMA_SECOND), (DSML_CALL, DSML_SECOND)])
def test_parallel_calls_still_return_every_gemma_and_dsml_call(first, second):
    content, calls = reply_of(RawProseApp(content=f"{first}{second}"), stream=False, parallel=True)
    assert content == "" and len(calls) == 2


@pytest.mark.parametrize("envelope", [GEMMA_CALL, GEMMA_BARE_CALL, DSML_CALL])
def test_single_call_hides_an_envelope_split_at_every_boundary(envelope):
    from tensorfold.server.tool_policy import ToolCallPolicy

    text = f"Sure. {envelope}{envelope} Done."
    for cut in range(len(text) + 1):
        policy = ToolCallPolicy({"parallel_tool_calls": False})
        shown = policy.delta(text[:cut]) + policy.delta(text[cut:]) + policy.flush()
        assert shown == "Sure.  Done.", cut
    policy = ToolCallPolicy({"parallel_tool_calls": False})
    assert "".join(policy.delta(ch) for ch in text) + policy.flush() == "Sure.  Done."
    for cut in range(text.index(">") + 1, len("Sure. ") + len(envelope)):   # a reply that ends inside the envelope
        assert ToolCallPolicy({"parallel_tool_calls": False}).content(text[:cut]) == "Sure. ", cut


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mention", [
    "Gemma opens a call with <|tool_call> in its replies. ",
    "DeepSeek writes <｜DSML｜tool_calls> first. ",
    "Gemma's <|tool_call>\n and DeepSeek's <｜DSML｜tool_calls> differ. ",
])
def test_another_family_s_opener_in_prose_neither_hides_text_nor_takes_the_call(stream, mention):
    from tensorfold.server.tool_policy import ToolCallPolicy
    from tensorfold.server.tools import parse_tool_calls_from_content

    content, calls = reply_of(RawProseApp(content=mention + QWEN_LOOKUP), stream=stream, parallel=False)
    assert calls == [("lookup", {"query": "Paris"})]
    assert content == (mention if stream else "")
    reply = ToolCallPolicy({"parallel_tool_calls": False}).finish(
        {"content": mention + QWEN_LOOKUP}, LOOKUP, parse_tool_calls_from_content)
    assert reply["content"] == mention.strip()
    content, calls = reply_of(RawProseApp(content=mention + "That is all."), stream=stream, parallel=False)
    assert calls == [] and content == mention + "That is all."


def test_a_gemma_reply_that_names_the_dsml_opener_keeps_its_call():
    mention = "Not <｜DSML｜tool_calls> here. "
    content, calls = reply_of(RawProseApp(content=mention + GEMMA_CALL + GEMMA_SECOND), stream=True, parallel=False)
    assert calls == [("lookup", {"query": "hi, there", "limit": 2})] and content == mention


def test_an_opener_followed_by_whitespace_holds_a_bounded_tail():
    from tensorfold.server.tool_policy import ToolCallPolicy

    policy = ToolCallPolicy({"parallel_tool_calls": False})
    assert policy.delta("a <|tool_call>") == "a "
    for _ in range(20_000):
        assert policy.delta(" ") == ""
        assert len(policy._text.pending) == 0
    assert policy.delta("then prose") == "<|tool_call>" + " " * 20_000 + "then prose"
    assert policy.flush() == ""
