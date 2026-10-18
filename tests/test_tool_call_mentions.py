"""A reply that names another model family's tool-call opener in prose keeps its text, and its own call is still
returned and hidden, for every family as the server, streamed at every split and finished.

The streamed text hides the serving template's own call opener at once. Another family's opener hides only once its
payload starts as that family writes it (Gemma 4's ``call:``, DSML's invoke); Qwen's and GLM's ``<tool_call>`` and
LFM2's list cannot be told from prose that way, so they hide only while the serving family is unknown."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorfold.cuda import reply_text
from tensorfold.server.stopping import StopPolicy
from tensorfold.server.text import CALL_PROBE, call_opener, hide_tool_calls
from tensorfold.server.tools import parse_tool_calls_from_content

LOOKUP = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {
    "query": {"type": "string"}}}}}]
QWEN, GEMMA, DSML, LFM2 = "<tool_call>", "<|tool_call>", "<｜DSML｜tool_calls>", "<|tool_call_start|>"
OPENERS = (QWEN, GEMMA, DSML, LFM2)

# (opener, a call to lookup(query="Paris") as the family writes it, the parsers that read the family's calls)
MAC, CUDA = parse_tool_calls_from_content, reply_text.parse_tool_calls
FAMILIES = {
    "qwen": (QWEN, "<tool_call>\n<function=lookup>\n<parameter=query>\nParis\n</parameter>\n</function>\n</tool_call>",
             (MAC, CUDA)),
    # Nemotron 3 writes Qwen 3.5's XML calls: the same text, kept as its own case
    "nemotron": (QWEN, ("<tool_call>\n<function=lookup>\n<parameter=query>\nParis\n</parameter>\n</function>\n"
                        "</tool_call>"), (MAC, CUDA)),
    "glm": (QWEN, "<tool_call>lookup\n<arg_key>query</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>",
            (MAC, CUDA)),
    "gemma": (GEMMA, '<|tool_call>call:lookup{query:<|"|>Paris<|"|>}<tool_call|>', (MAC,)),
    "gemma-26b": (GEMMA, '<|tool_call>:lookup{query:<|"|>Paris<|"|>}<tool_call|>', (MAC,)),
    "dsml": (DSML, ('<｜DSML｜tool_calls>\n<｜DSML｜invoke name="lookup">\n'
                    '<｜DSML｜parameter name="query" string="true">Paris</｜DSML｜parameter>\n'
                    '</｜DSML｜invoke>\n</｜DSML｜tool_calls>'), (MAC,)),
    "lfm2": (LFM2, '<|tool_call_start|>[lookup(query="Paris")]<|tool_call_end|>', (CUDA,)),
}
# the ways a reply names an opener: mid-sentence, before a newline, in backticks, at the end of a line
MENTIONS = ("Other models open a call with {o} and then the payload. ", "The opener is {o}\nwhich I will not use. ",
            "Their marker `{o}` differs from mine. ", "Some write {o}")
CASES = [(family, other, mention) for family, (own, _, _) in FAMILIES.items() for other in OPENERS if other != own
         for mention in MENTIONS]


def streamed(reply: str, opener: str | None) -> str:
    """The text a client receives: each prefix's visible text only grows, then the final flush adds the rest."""

    shown = ""
    for n in range(1, len(reply) + 1):
        visible = hide_tool_calls(reply[:n], finished=False, opener=opener)
        assert visible.startswith(shown), (n, shown, visible)
        shown = visible
    sent: list[str] = []
    StopPolicy({}, None, None, frozenset()).flush(sent.append, reply, None, shown, "", True, opener=opener)
    return shown + "".join(sent)


@pytest.mark.parametrize("family, other, mention", CASES)
def test_a_mention_of_another_family_s_opener_keeps_its_text_and_the_call(family, other, mention):
    own, call, parsers = FAMILIES[family]
    prose = mention.format(o=other) + " After the mention.\n"
    reply = prose + call + "\nDone."
    want = prose + "\nDone."
    assert hide_tool_calls(reply, finished=True, opener=own) == want
    for n in range(len(reply) + 1):         # no prefix shows the call's markup or anything the final text lacks
        assert want.startswith(hide_tool_calls(reply[:n], finished=False, opener=own)), n
    assert streamed(reply, own) == want
    for parse in parsers:
        content, calls = parse(reply, LOOKUP)
        assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == [
            ("lookup", {"query": "Paris"})], parse
        assert "After the mention." in content and "Paris" not in content


@pytest.mark.parametrize("family", FAMILIES)
def test_the_serving_family_s_own_calls_still_hide_at_every_split(family):
    own, call, _ = FAMILIES[family]
    reply = f"Sure. {call}{call} Done."
    for opener in (own, None):
        for n in range(len(reply) + 1):
            assert "Sure.  Done.".startswith(hide_tool_calls(reply[:n], finished=False, opener=opener)), (opener, n)
        assert hide_tool_calls(reply, finished=True, opener=opener) == "Sure.  Done."
        assert streamed(reply, opener) == "Sure.  Done."
        for cut in range(len("Sure. ") + len(own), len("Sure. ") + len(call)):   # a reply cut inside the call
            assert hide_tool_calls(reply[:cut], finished=True, opener=opener) == "Sure. ", (opener, cut)


@pytest.mark.parametrize("other", [GEMMA, DSML])
def test_an_unknown_family_keeps_a_gemma_or_dsml_mention_and_hides_its_calls(other):
    _, call, _ = FAMILIES["qwen"]
    reply = f"They write {other} first. {call} Done."
    assert streamed(reply, None) == f"They write {other} first.  Done."


@pytest.mark.parametrize("other", [QWEN, LFM2])
def test_an_unknown_family_still_hides_from_an_opener_without_a_lead(other):
    reply = f"They write {other} first. Then more."
    assert hide_tool_calls(reply, finished=True) == "They write "


def test_an_opener_ending_a_finished_reply_stays_when_not_the_family_s_own():
    assert hide_tool_calls("Gemma writes <|tool_call>", finished=False, opener=QWEN) == "Gemma writes "
    assert hide_tool_calls("Gemma writes <|tool_call>  ", finished=True, opener=QWEN) == "Gemma writes <|tool_call>  "
    assert hide_tool_calls("Gemma writes <|tool_call> ca", finished=True, opener=QWEN) == "Gemma writes <|tool_call> ca"
    assert hide_tool_calls("Mine is <tool_call>", finished=True, opener=QWEN) == "Mine is "
    assert hide_tool_calls("Cut <|tool_call> ca", finished=True) == "Cut "     # unknown family: a call cut short


def test_many_mentions_scan_in_linear_time():
    reply = f"{GEMMA} x {DSML} y {LFM2} z " * 20_000 + FAMILIES["qwen"][1]
    assert hide_tool_calls(reply, finished=True, opener=QWEN) == f"{GEMMA} x {DSML} y {LFM2} z " * 20_000


def test_the_opener_is_read_from_the_family_s_rendered_call():
    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.families.deepseek_v4.prompts import render

    lfm2 = ChatTemplate(Path(__file__).parent / "fixtures" / "lfm2_moe")
    assert call_opener(lfm2.render(CALL_PROBE, tools=None, enable_thinking=False)) == LFM2
    assert call_opener(render(CALL_PROBE)) == DSML
    assert call_opener('<|turn>model\n<|tool_call>call:tfprobe_fn{}<tool_call|>') == GEMMA
    assert call_opener("<|im_start|>assistant\n<tool_call>\n<function=tfprobe_fn>\n</function>\n</tool_call>") == QWEN
    assert call_opener("<|assistant|><tool_call>tfprobe_fn</tool_call>") == QWEN
    assert call_opener("<|im_start|>assistant\n<|im_end|>") is None


class MentionEngine:
    """Writes prose that names LFM2's and Gemma 4's openers, then a Qwen call, three tokens a round."""

    PROSE = f"LFM2 writes {LFM2} and Gemma {GEMMA} before a call. Mine follows.\n"

    def __init__(self):
        from tests.test_cuda_tool_choice import EOS

        self.eos = (EOS,)

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        from tests.test_cuda_tool_choice import CALL, CLOSE, EOS, OPEN, Tokens

        reply = Tokens().encode(self.PROSE).ids + [OPEN, *Tokens().encode(CALL).ids, CLOSE, EOS]
        for at in range(0, len(reply), 3):
            if on_tokens(reply[at:at + 3]):
                break
        return {"rounds": 1 + len(reply) // 3, "decode_s": 0.5}


@pytest.mark.parametrize("stream", [False, True])
def test_the_cuda_server_streams_a_qwen_reply_that_names_other_openers(tmp_path, stream):
    from tests.test_cuda_admission import http_server, post
    from tests.test_cuda_tool_choice import TOOLS, app_for, events

    body = {"messages": [{"role": "user", "content": "Weather?"}], "tools": TOOLS, "stream": stream}
    with http_server(app_for(tmp_path, MentionEngine())) as port:
        status, reply = post(port, body, True)
    assert status == 200, reply
    if not stream:
        message = json.loads(reply)["choices"][0]["message"]
        assert message["content"] == MentionEngine.PROSE.strip()
        assert [c["function"]["name"] for c in message["tool_calls"]] == ["get_weather"]
        return
    chunks = [c["choices"][0] for c in events(reply) if c.get("choices")]
    content = "".join(c["delta"].get("content") or "" for c in chunks)
    names = "".join(d["function"].get("name", "") for c in chunks for d in c["delta"].get("tool_calls", []))
    assert content == MentionEngine.PROSE and names == "get_weather"
    assert chunks[-1]["finish_reason"] == "tool_calls"
