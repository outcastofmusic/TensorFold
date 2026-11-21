"""Tool results that answer an assistant's parallel calls render in call order, since Qwen3.8's templates show each
result as an anonymous ``<tool_response>`` (upstream #3775)."""

import pytest

from tensorfold.server.messages import normalize_messages


def _turn(*answered: str, ids: tuple[str, ...] = ("call_a", "call_b")) -> list[dict]:
    calls = [{"id": i, "type": "function", "function": {"name": "read_file", "arguments": {"path": i}}} for i in ids]
    return [{"role": "user", "content": "Which ports?"},
            {"role": "assistant", "content": "", "tool_calls": calls},
            *({"role": "tool", "tool_call_id": a, "content": f"port of {a}"} for a in answered)]


def test_results_sent_back_out_of_order_render_in_call_order():
    out = normalize_messages(_turn("call_b", "call_a"))
    assert [m["content"] for m in out[2:]] == ["port of call_a", "port of call_b"]


def test_results_already_in_order_are_the_request_itself():
    messages = _turn("call_a", "call_b")
    assert normalize_messages(messages) is messages


@pytest.mark.parametrize("answered", [("call_b",), ("call_b", "call_b"), ("call_b", "call_x"), ("call_b", "call_a", "call_a")],
                         ids=["missing", "repeated", "unknown", "extra"])
def test_a_run_that_is_not_one_result_per_call_keeps_arrival_order(answered):
    messages = _turn(*answered)
    assert normalize_messages(messages) is messages


def test_calls_without_ids_or_with_a_repeated_id_keep_arrival_order():
    for ids in (("", "call_b"), ("call_a", "call_a")):
        messages = _turn("call_b", "call_a", ids=ids)
        assert [m["content"] for m in normalize_messages(messages)[2:]] == ["port of call_b", "port of call_a"]


def test_an_anonymous_template_shows_the_answers_in_call_order():
    jinja2 = pytest.importorskip("jinja2")
    template = jinja2.Template("{% for m in messages %}{% if m.role == 'assistant' %}{% for c in m.tool_calls %}"
                               "[{{ c.function.arguments.path }}]{% endfor %}{% elif m.role == 'tool' %}"
                               "<tool_response>{{ m.content }}</tool_response>{% endif %}{% endfor %}")
    text = template.render(messages=normalize_messages(_turn("call_b", "call_a")))
    assert text == "[call_a][call_b]<tool_response>port of call_a</tool_response><tool_response>port of call_b</tool_response>"
