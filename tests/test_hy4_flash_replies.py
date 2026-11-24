"""Hy4-flash replies through the served path: Hy4App over the checkpoint's text view, and Hy4Engine.generate with its
prefill and decode replaced by a recorded reply (CPU only, no weights).

The reply is the model's own form, rendered by its template from an assistant turn that calls two tools: reasoning,
``</think>``, then ``<tool_calls>`` holding two ``<tool_call>NAME<arg_key>..`` blocks, ``</tool_calls>`` and the end
token. Streamed and whole replies carry the same calls in order, ``parallel_tool_calls: false`` keeps the first, and
no control spelling reaches a client (playbook 8.8 and 8.12). The checkpoint's small files come from $MODELS; a
missing one is skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("jinja2")
pytest.importorskip("tokenizers")

MODELS = Path(os.environ.get("MODELS", str(Path.home() / "models")))
WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather",
                                             "parameters": {"type": "object", "required": ["city"],
                                                            "properties": {"city": {"type": "string"}}}}}]
CITIES = ("Oslo", "Bergen")
REASONING = "Two cities, so two calls."
LEAKS = ("<tool_call", "</tool_call", "<arg_key>", "<arg_value>", "<tool_calls>", "</tool_calls>", "hy_", "opensource",
         "<think>", "</think>")


@pytest.fixture(scope="module")
def folder():
    found = Path(os.environ.get("TF_HY4_MODEL") or MODELS / "Hy4-flash-NVFP4")
    if not (found / "tokenizer.json").is_file():
        pytest.skip(f"needs the Hy4-flash tokenizer and chat template ({found})")
    return found


def recorded_reply(view: Path) -> str:
    """The text the model writes for two calls with thinking on: its template's rendering of that assistant turn."""

    from tensorfold.cuda.chat_template import ChatTemplate

    calls = [{"id": f"call_{i}", "type": "function", "function": {"name": "get_weather", "arguments": {"city": c}}}
             for i, c in enumerate(CITIES)]
    turns = [{"role": "user", "content": "Weather in Oslo and Bergen?"}]
    head = ChatTemplate(view).render(turns, tools=WEATHER, enable_thinking=True,
                                     extra={"reasoning_effort": "high"})
    whole = ChatTemplate(view).render(
        [*turns, {"role": "assistant", "content": "", "reasoning_content": REASONING, "tool_calls": calls},
         {"role": "user", "content": "Thanks"}],
        tools=WEATHER, enable_thinking=True, extra={"reasoning_effort": "high", "add_generation_prompt": False})
    assert whole.startswith(head), "the template no longer renders history as its generation prompt"
    reply = whole[len(head):]
    return reply[:reply.index("<｜hy_end｜>") + len("<｜hy_end｜>")]


def engine_for(view: Path, reply_ids: list[int]):
    """Hy4Engine with its weights left out: generate's prompt states off, its prefill and decode the recording."""

    from tensorfold.cuda.kept_states import Store
    from tensorfold.families.hy4_flash.cuda.app import Hy4Engine

    e = Hy4Engine.__new__(Hy4Engine)
    e.max_len, e.drafts, e.drafter, e.e = 1 << 16, False, None, None
    e.store, e.marks, e.eos, e.tp, e.rank, e.hide = Store(0, 0), None, (120025,), 1, 0, set()
    e.reply = reply_ids
    return e


@pytest.fixture
def app(folder, monkeypatch):
    from tokenizers import Tokenizer

    from tensorfold.families.hy4_flash.cuda import model
    from tensorfold.families.hy4_flash.cuda.app import Hy4App, text_view

    view = text_view(folder)
    ids = Tokenizer.from_file(str(view / "tokenizer.json")).encode(recorded_reply(view), add_special_tokens=False).ids
    engine = engine_for(view, list(ids))

    class Done:
        seconds, rounds, tokens_per_second = 0.01, 1, 1.0

    def prefill(e, prompt, sampling, **kw):
        return engine.reply[0]

    def serial_decode(e, pending, count, sampling, *, stop_eos=True, on_tokens=None):
        for t in engine.reply[1:count]:
            if on_tokens([t]) or (stop_eos and t in engine.eos):
                break
        return Done()

    monkeypatch.setattr(model, "prefill", prefill)
    monkeypatch.setattr(model, "serial_decode", serial_decode)
    return Hy4App(engine, folder, "hy4", default_thinking=True)


def run(app, *, stream: bool, **extra):
    """(reasoning, content, calls [(name, arguments)], every streamed text) as a client puts them together."""

    body = {"messages": [{"role": "user", "content": "Weather in Oslo and Bergen?"}], "tools": WEATHER,
            "max_tokens": 512, "stream": stream, **extra}
    deltas: list[dict] = []
    result = app.run(body, True, lambda d: deltas.append(d) or True, prepared=app.prepare(body, True))
    reasoning = "".join(d.get("reasoning_content") or "" for d in deltas)
    content = "".join(d.get("content") or "" for d in deltas)
    calls: dict[int, dict] = {}
    for d in deltas:
        for part in d.get("tool_calls") or []:
            call = calls.setdefault(part["index"], {"name": "", "arguments": ""})
            call["name"] += part.get("function", {}).get("name") or ""
            call["arguments"] += part.get("function", {}).get("arguments") or ""
    if stream:
        content += result["final"].get("content") or ""
        reasoning += result["final"].get("reasoning_content") or ""
    else:
        content, reasoning = result["content"] or "", result.get("reasoning_content") or reasoning
    for i, call in enumerate(result["calls"] or []):
        if not stream or i >= result["calls_streamed"]:
            calls[i] = {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
    shown = [reasoning, content, *(c["name"] + c["arguments"] for c in calls.values())]
    return reasoning, content, [(c["name"], json.loads(c["arguments"])) for _, c in sorted(calls.items())], shown


@pytest.mark.parametrize("stream", [False, True])
def test_two_calls_in_one_turn_come_back_in_order(app, stream):
    reasoning, content, calls, shown = run(app, stream=stream)
    assert calls == [("get_weather", {"city": c}) for c in CITIES]
    assert reasoning.strip() == REASONING and content.strip() == ""
    assert not [leak for text in shown for leak in LEAKS if leak in text]


@pytest.mark.parametrize("stream", [False, True])
def test_one_call_requests_get_the_first_call(app, stream):
    _, _, calls, shown = run(app, stream=stream, parallel_tool_calls=False)
    assert calls == [("get_weather", {"city": CITIES[0]})]
    assert not [leak for text in shown for leak in LEAKS if leak in text]


def test_the_wrapper_ids_never_reach_the_server(app):
    """Hy4Engine.generate drops <tool_calls> and </tool_calls> before on_tokens: the model reads them, GLM's parser
    reads the calls inside."""

    wrappers = {app.tok.token_to_id(t) for t in ("<tool_calls>", "</tool_calls>")}
    assert len(wrappers) == 2 and wrappers <= set(app.engine.reply)
    seen: list[int] = []
    app.engine.generate([1], 512, None, lambda ts: seen.extend(ts))
    assert seen and not wrappers & set(seen) and seen[-1] == 120025
