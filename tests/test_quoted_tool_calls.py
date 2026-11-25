"""Tool-call markup quoted in Markdown inline code or a fence stays prose, whole and streamed (upstream #3588)."""

from __future__ import annotations

import json
import random

from tensorfold.cuda.reply_text import close_call, open_call, parse_tool_calls
from tensorfold.engine.call_gate import ThinkCall
from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.server import pythonic_calls
from tensorfold.server.code_spans import PENDING, Quoted, scan
from tensorfold.server.text import hide_tool_calls, split_thinking
from tensorfold.server.tool_policy import ToolCallPolicy
from tensorfold.server.tools import parse_tool_calls_from_content

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}}}}]
CALL = "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
REAL = CALL.replace("a.txt", "b.txt")
INLINE = "Use `" + CALL + "` to read."
FENCED = "Example:\n```xml\n" + CALL + "\n```\nDone."
LFM2 = '<|tool_call_start|>[read_file(path="a.txt")]<|tool_call_end|>'


def names(calls):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls or []]


def test_a_quoted_call_stays_text_whole():
    for text in (INLINE, FENCED, "~~~\n" + CALL + "\n~~~", "``a ` " + CALL + "``"):
        assert parse_tool_calls(text, TOOLS) == (text, None)
        assert parse_tool_calls_from_content(text, TOOLS) == (text, None)
        assert parse_tool_calls(text, TOOLS, max_calls=1)[1] is None


def test_a_real_call_after_a_quoted_opener_still_runs():
    for prose in ("Write `<tool_call>` to call a tool.\n\n", "```\n<tool_call>\n```\n"):
        content, calls = parse_tool_calls(prose + REAL, TOOLS)
        assert names(calls) == [("read_file", {"path": "b.txt"})] and content == prose.strip()


def test_markdown_inside_a_calls_arguments_is_not_read_as_prose():
    first = CALL.replace("a.txt", "```python")              # a fence opened inside the first call's value
    _, calls = parse_tool_calls(first + "\n" + REAL, TOOLS)
    assert names(calls) == [("read_file", {"path": "```python"}), ("read_file", {"path": "b.txt"})]


def test_a_backtick_that_never_closes_is_literal():
    text = "A lone ` backtick.\n\n" + REAL
    assert names(parse_tool_calls(text, TOOLS)[1]) == [("read_file", {"path": "b.txt"})]
    text = "A lone ` backtick, then " + REAL                 # same paragraph, no closer: literal once finished
    assert names(parse_tool_calls(text, TOOLS)[1]) == [("read_file", {"path": "b.txt"})]


def test_lfm2_quoted_list_stays_text():
    for text in ("Use `" + LFM2 + "` here.", "```\n" + LFM2 + "\n```"):
        assert pythonic_calls.parse_tool_calls(text, TOOLS) == (text, None)
        assert parse_tool_calls(text, TOOLS) == (text, None)
    assert names(pythonic_calls.parse_tool_calls("`x` " + LFM2, TOOLS)[1]) == [("read_file", {"path": "a.txt"})]
    assert pythonic_calls.close_call("```\n" + LFM2[:-len("<|tool_call_end|>")], TOOLS) == ""


def test_a_quoted_call_streams_as_text():
    for text in (INLINE, FENCED):
        shown = ""
        for n in range(1, len(text) + 1):
            now = hide_tool_calls(text[:n], finished=False, opener="<tool_call>")
            assert now.startswith(shown)
            shown = now
        assert hide_tool_calls(text, finished=True, opener="<tool_call>") == text
        assert text.startswith(shown) and CALL in shown


def test_a_quoted_opener_waits_while_its_span_may_close():
    assert hide_tool_calls("Use `<tool_call>", finished=False, opener="<tool_call>") == "Use `"
    # a closing run at the end may still grow (upstream holds it too), so the span is code once text follows
    assert hide_tool_calls("Use `<tool_call>` ", finished=False, opener="<tool_call>") == "Use `<tool_call>` "
    # finished without a closer: a literal backtick, so the call is live and hides
    assert hide_tool_calls("Use `" + REAL, finished=True, opener="<tool_call>") == "Use `"


def test_end_token_inside_a_quoted_call_does_not_close_it():
    text = "Like this:\n```\n" + CALL[:-len("\n</tool_call>")]
    assert open_call(text) is None
    assert close_call(text, TOOLS) == ""
    assert close_call(CALL[:-len("</tool_call>")], TOOLS) == "</tool_call>"      # a live one still closes


def test_the_streamer_skips_quoted_calls():
    for text in (INLINE + "\n\n" + REAL, FENCED + "\n" + REAL):
        streamer, deltas = ToolCallStreamer(TOOLS), []
        for n in range(1, len(text) + 1):
            deltas += streamer.feed(text[:n])
        args = "".join(d["tool_calls"][0]["function"].get("arguments", "") for d in deltas)
        assert streamer.index == 0 and json.loads(args) == {"path": "b.txt"}


def test_one_call_requests_show_quoted_calls():
    policy = ToolCallPolicy({"parallel_tool_calls": False})
    accept = lambda envelope: parse_tool_calls(envelope, TOOLS, max_calls=1)[1] is not None
    assert policy.content(FENCED, accept=accept) == FENCED
    assert policy.content(INLINE + "\n\n" + REAL, accept=accept) == INLINE + "\n\n"


def test_a_quoted_call_in_the_thinking_stays_reasoning():
    reasoning = "The format is `" + CALL + "`, so I will answer in prose.\n"
    shown = ""
    for n in range(1, len(reasoning) + 1):
        now, answer = split_thinking(reasoning[:n], finished=False)
        assert now.startswith(shown) and answer == ""
        shown = now
    assert shown == reasoning
    assert split_thinking(reasoning, finished=True) == (reasoning, "")


PIECES = {1: "Think: `", 2: "<tool_call>", 3: "`", 4: "</think>", 5: "\n", 6: "```\n", 7: "x"}


def decode(ids):
    return "".join(PIECES[int(t)] for t in ids)


def test_the_think_call_gate_leaves_a_quoted_opener():
    for prior in ([1], [6, 7, 5]):
        gate = ThinkCall(2, [5, 4], 4, decode=decode)
        for t in prior:
            gate.observe(t)
        assert gate.cut([2, 3]) is None
    gate = ThinkCall(2, [5, 4], 4, decode=decode)
    gate.observe(7)
    assert gate.cut([5, 2]) == (1, [5, 4, 2])                # a live opener still moves


def test_streaming_and_whole_scans_agree():
    rng = random.Random(0x5EEDC0DE)
    parts = ["`", "``", "```", "~~~", "<", "x", " ", "\n", "\n\n", "a<b"]
    for _ in range(3000):
        text = "".join(rng.choice(parts) for _ in range(rng.randrange(18)))
        whole = Quoted(text)
        for i, ch in enumerate(text):
            if ch in "`~":
                continue
            for n in range(i + 1, len(text) + 1):
                got = Quoted(text[:n], finished=False).check(i)
                if got != PENDING:
                    final = whole.check(i)
                    assert (got is None) == (final is None), (text, i, n)
                    break
        spans, hold = scan(text)
        assert hold == len(text) and all(a < b for a, b in spans)
