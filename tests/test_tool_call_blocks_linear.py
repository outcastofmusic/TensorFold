"""Tool-call envelopes found in linear time and equal to the lazy regexes they replace (upstream #913)."""

import random
import re
import time

import pytest

from tensorfold.cuda import reply_text
from tensorfold.server import tools

# the regexes tools._envelopes and reply_text._parse_tagged ran, which the server no longer carries
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_NAMESPACED_TOOL_CALL_BLOCK_RE = re.compile(r"<([A-Za-z_][\w.-]*):tool_call>\s*(.*?)\s*</\1:tool_call>",
                                            re.IGNORECASE | re.DOTALL)
_GEMMA_TOOL_CALL_BLOCK_RE = re.compile(r"<\|tool_call>\s*(.*?)\s*<tool_call\|>", re.DOTALL)
_DSML_BLOCK_RE = re.compile(r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>", re.DOTALL)


def _regex_envelopes(text):
    found = sorted([(m.start(), m.end(), m.group(1).strip()) for m in _TOOL_CALL_BLOCK_RE.finditer(text)]
                   + [(m.start(), m.end(), m.group(2).strip()) for m in _NAMESPACED_TOOL_CALL_BLOCK_RE.finditer(text)]
                   + [(m.start(), m.end(), m.group(1).strip()) for m in _GEMMA_TOOL_CALL_BLOCK_RE.finditer(text)]
                   + [(m.start(), m.end(), m.group(0)) for m in _DSML_BLOCK_RE.finditer(text)])
    kept = []
    for envelope in found:
        if not kept or envelope[0] >= kept[-1][1]:
            kept.append(envelope)
    return kept


def _regex_blocks(text):
    return [(m.start(), m.end(), m.group(1)) for m in _TOOL_CALL_BLOCK_RE.finditer(text)]


TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"q": {"type": "integer"}}}}}]

PIECES = [
    "<tool_call>", "</tool_call>", "<TOOL_CALL>", "</Tool_Call>", "<tool_call", "</tool_call",
    "<ns:tool_call>", "</ns:tool_call>", "</NS:TOOL_CALL>", "<a.b-c:tool_call>", "</a.B-c:tool_call>",
    "<nſ:tool_call>", "</ns:tool_call>", "<İ:tool_call>", "</i:tool_call>", "</ı:tool_call>",
    "<K:tool_call>", "</k:tool_call>", "<Σx:tool_call>", "</σx:tool_call>", "</ςx:tool_call>",
    "<|tool_call>", "<tool_call|>", "<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>",
    '{"name": "lookup", "arguments": {"q": 1}}', '{"name": "other", "arguments": {}}',
    "<function=lookup><parameter=q>\n1\n</parameter></function>", "lookup<arg_key>q</arg_key><arg_value>1</arg_value>",
    "call:lookup{q:1}", ('<｜DSML｜invoke name="lookup"><｜DSML｜parameter name="q" string="false">1'
                         "</｜DSML｜parameter></｜DSML｜invoke>"),
    " ", "\n", " ", "\x1c", "x", "<", "</", ":tool_call>", "ns", "|>",
]


def _parsed(parse, text, max_calls):
    """The parse without its random call ids, or the error it raised: a ``<tool_call>`` whose payload begins with a
    DSML opener raises AttributeError in ``parse_tool_calls_from_content``, before and after the scan."""

    try:
        content, calls = parse(text, TOOLS, max_calls=max_calls)
    except AttributeError as error:
        return type(error)
    return content, None if calls is None else [{**call, "id": None} for call in calls]


# whole calls, so that random replies also hold calls each parser reads
CALLS = ['<tool_call>\n{"name": "lookup", "arguments": {"q": 1}}\n</tool_call>',
         "<tool_call><function=lookup><parameter=q>\n2\n</parameter></function></tool_call>",
         "<tool_call>lookup<arg_key>q</arg_key><arg_value>3</arg_value></tool_call>",
         '<ns:tool_call>{"name": "lookup", "arguments": {"q": 4}}</NS:tool_call>',
         "<|tool_call>call:lookup{q:5}<tool_call|>",
         ('<｜DSML｜tool_calls><｜DSML｜invoke name="lookup"><｜DSML｜parameter name="q" string="false">6'
          "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>")]


def _random_reply(rng):
    return "".join(rng.choice(CALLS if rng.random() < 0.15 else PIECES) for _ in range(rng.randint(0, 20)))


def test_envelopes_equal_the_regexes_on_random_replies():
    rng = random.Random(913)
    for _ in range(20_000):
        text = _random_reply(rng)
        assert tools._envelopes(text) == _regex_envelopes(text), text
        assert [(s, e, b.strip()) for s, e, b in tools.tool_call_blocks(text)] == \
               [(s, e, b.strip()) for s, e, b in _regex_blocks(text)], text


@pytest.mark.parametrize("max_calls", [None, 1])
def test_parsed_calls_and_visible_text_equal_the_regex_parse(monkeypatch, max_calls):
    rng = random.Random(9130 + (max_calls or 0))
    replies = [_random_reply(rng) for _ in range(5_000)]
    def both(text):
        return (_parsed(tools.parse_tool_calls_from_content, text, max_calls),
                _parsed(reply_text.parse_tool_calls, text, max_calls))

    new = [both(t) for t in replies]
    monkeypatch.setattr(tools, "_envelopes", _regex_envelopes)
    monkeypatch.setattr(reply_text, "tool_call_blocks", _regex_blocks)
    old = [both(t) for t in replies]
    assert sum(1 for server, _ in new if isinstance(server, tuple) and server[1]) > 1_000       # replies with calls
    assert sum(1 for _, cuda in new if cuda[1]) > 1_000
    for text, got, want in zip(replies, new, old):
        assert got == want, text


PAD = "x" * 500


@pytest.mark.parametrize("opener", ["<tool_call>", "<ns:tool_call>", "<ns{i}:tool_call>", "<|tool_call>",
                                    "<｜DSML｜tool_calls>"])
def test_two_thousand_unclosed_openers_in_a_megabyte_parse_quickly(opener):
    # the regexes took 7.5-10 s each on these replies (18 Oct 2026, 6 CPUs)
    text = "".join(opener.format(i=i) + PAD for i in range(2_000)) + "<tool_call>" + '{"name": "lookup"}'
    assert len(text) > 1_000_000
    start = time.perf_counter()
    assert tools.parse_tool_calls_from_content(text, TOOLS) == (text, None)
    assert reply_text.parse_tool_calls(text, TOOLS) == (text, None)
    assert time.perf_counter() - start < 0.5
