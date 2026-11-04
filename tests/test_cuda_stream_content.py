"""A tool stream carries the content the whole reply carries: an unoffered call's envelope stays, and prose after a
hidden envelope waits for the final parse (upstream #1365)."""

import pytest

pytest.importorskip("jinja2")

from tests.test_cuda_tool_cut import reply

UNOFFERED = ("Let me.\n<tool_call>\n<function=delete_all>\n<parameter=path>\n/\n</parameter>\n</function>\n"
             "</tool_call>\nThen I will stop.")
OFFERED = ("Let me.\n<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n"
           "</tool_call>\nDone.")


@pytest.mark.parametrize("text", [UNOFFERED, OFFERED, "\n" + UNOFFERED, "Plain prose, no calls."],
                         ids=["unoffered", "offered", "leading-newline", "prose"])
def test_the_stream_and_the_whole_reply_carry_the_same_content_and_calls(tmp_path, text):
    end = len(text) + 3                                    # the engine's x tokens after the text: never a call
    whole = reply(tmp_path, text, stream=False, max_tokens=end)
    streamed = reply(tmp_path, text, stream=True, max_tokens=end)
    assert streamed == whole


def test_an_unoffered_call_stays_in_both_replies(tmp_path):
    _, content, calls = reply(tmp_path, UNOFFERED, stream=True, max_tokens=len(UNOFFERED))
    assert calls == [] and "<function=delete_all>" in content and content.endswith("Then I will stop.")


def test_an_offered_call_streams_once_and_its_envelope_never_shows(tmp_path):
    finish, content, calls = reply(tmp_path, OFFERED, stream=True, max_tokens=len(OFFERED))
    assert finish == "tool_calls" and calls == [{"name": "write_file", "arguments": '{"path":"a.txt"}'}]
    assert "<tool_call>" not in content and content.startswith("Let me.") and content.endswith("Done.")
