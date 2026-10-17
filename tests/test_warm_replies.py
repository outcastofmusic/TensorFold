"""After a chat reply, the next turn's shared prompt is prefilled in the background (``tensorfold.cuda.warm``)."""

from __future__ import annotations

import importlib
import threading
import time
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda import warm

ROLE = {"system": 1, "user": 2, "assistant": 3, "tool": 4}


def render(messages):
    """A toy template: a role id, then one id per character; an assistant's reasoning only before a tool result."""

    ids = []
    for i, m in enumerate(messages):
        ids.append(ROLE[m["role"]])
        nxt = messages[i + 1]["role"] if i + 1 < len(messages) else None
        if m.get("reasoning_content") and nxt == "tool":
            ids += [ord(c) for c in m["reasoning_content"]]
        ids += [ord(c) for c in (m.get("content") or "")]
        ids += [ord(c) for call in m.get("tool_calls") or [] for c in call["function"]["name"]]
    return ids + [ROLE["assistant"]]


CALL = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}


def test_the_prefix_ends_where_a_user_and_a_tool_turn_part():
    history = [{"role": "user", "content": "hi"}]
    reply = {"role": "assistant", "content": "ok", "tool_calls": [CALL]}
    prefix = warm.next_turn_prefix(render, history, reply)
    assert prefix == render([*history, reply])[:-1]          # the reply in full, before the next role


def test_a_reply_rendered_differently_by_what_follows_is_cut_where_they_part():
    history = [{"role": "user", "content": "hi"}]
    reply = {"role": "assistant", "content": "ok", "reasoning_content": "why", "tool_calls": [CALL]}
    prefix = warm.next_turn_prefix(render, history, reply)
    assert prefix == [ROLE["user"], ord("h"), ord("i"), ROLE["assistant"]]


def test_without_a_call_only_user_turns_follow_and_the_user_role_is_shared_too():
    history = [{"role": "user", "content": "hi"}]
    reply = {"role": "assistant", "content": "ok"}
    assert warm.next_turn_prefix(render, history, reply) == [*render([*history, reply])[:-1], ROLE["user"]]


def test_the_reply_message_is_what_the_chat_route_sends():
    result = {"content": "", "reasoning": "r", "calls": [CALL]}
    assert warm.reply_message(result) == {"role": "assistant", "content": None, "reasoning_content": "r",
                                          "tool_calls": [CALL]}
    assert warm.reply_message({"content": "x", "reasoning": "", "calls": None}) == {"role": "assistant",
                                                                                    "content": "x"}


class Turns:
    def __init__(self):
        self.lock = threading.Lock()
        self.waiting = False
        self.taken = 0

    def take(self):
        self.lock.acquire()
        self.taken += 1

    def give(self):
        self.lock.release()

    def wanted(self):
        return self.waiting


def make_warmer(run, turns, delay=0.05):
    return warm.Warmer(run, turns.take, turns.give, turns.wanted, delay=delay)


def wait_for(cond, seconds=2.0):
    end = time.monotonic() + seconds
    while not cond() and time.monotonic() < end:
        time.sleep(0.01)
    return cond()


def test_a_queued_pass_runs_after_the_delay_and_holds_the_turns():
    turns, ran = Turns(), []

    def run(prompt, born, stop):
        assert turns.lock.locked()
        ran.append((prompt, born, stop()))
        return len(prompt)

    w = make_warmer(run, turns)
    w.queue(lambda: ([1, 2, 3], 1))
    assert wait_for(lambda: w.passes == [3])
    assert ran == [([1, 2, 3], 1, False)] and not turns.lock.locked()


def test_a_request_that_finishes_before_the_pass_starts_makes_it_stale():
    turns, ran = Turns(), []
    w = make_warmer(lambda p, b, s: ran.append(p), turns, delay=0.2)
    w.queue(lambda: ([1, 2], 0))
    time.sleep(0.05)
    w.finished()
    time.sleep(0.4)
    assert ran == [] and w.passes == []


def test_a_newer_reply_replaces_the_pass_still_waiting():
    turns, ran, gate, inside = Turns(), [], threading.Event(), threading.Event()

    def slow():
        inside.set()
        gate.wait()
        return ([1], 0)

    w = make_warmer(lambda p, b, s: ran.append(p) or len(p), turns)
    w.queue(slow)
    assert inside.wait(2)                           # the worker is rendering this one
    w.queue(lambda: ([2, 2], 0))
    w.queue(lambda: ([3, 3, 3], 0))                 # replaces [2, 2] before the worker takes it
    gate.set()
    assert wait_for(lambda: len(w.passes) == 2)
    assert ran == [[1], [3, 3, 3]]


def test_a_waiting_foreground_request_stops_the_pass_and_none_skips_it():
    turns, seen = Turns(), []

    def run(prompt, born, stop):
        turns.waiting = True
        seen.append(stop())
        return 0

    w = make_warmer(run, turns)
    w.queue(lambda: None)
    w.queue(lambda: ([1], 0))
    assert wait_for(lambda: w.passes == [0])
    assert seen == [True]


def test_off_by_environment(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_WARM_REPLIES", "0")
    assert not warm.enabled()
    monkeypatch.delenv("TENSORFOLD_WARM_REPLIES")
    assert warm.enabled()


@pytest.mark.torch
def test_a_background_state_supersedes_none_of_its_turns_own_states():
    engine = importlib.import_module("tensorfold.families.glm5_next.cuda.engine")
    user = 9

    def snap(ids, born=-1):
        return NS(ids=list(ids), born=born, branched=False)

    turn = snap([user, 1, 2])                        # the turn's own prompt state, which a regenerate resumes from
    warmed = snap([user, 1, 2, 3, 4, 5], born=3)     # the background pass after that turn
    host = NS(cache=[turn, warmed], user_id=user)
    assert engine.GlmEngine._superseded(host) is None
    later = snap([user, 1, 2, 3, 4, 5, 6, 7])         # the next turn's own state supersedes both
    host.cache = [turn, warmed, later]
    assert engine.GlmEngine._superseded(host) is turn
    host.cache = [warmed, later]
    assert engine.GlmEngine._superseded(host) is warmed


@pytest.mark.torch
def test_glm_next_turn_renders_through_the_request_path():
    app_module = importlib.import_module("tensorfold.families.glm5_next.cuda.app")
    app = app_module.GlmApp.__new__(app_module.GlmApp)
    bodies = []

    def prepare(body, chat):
        bodies.append(body)
        return NS(prompt=render(body["messages"]))

    app._prepare = prepare
    body = {"model": "glm", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]}
    result = {"content": "ok", "reasoning": "", "calls": [CALL], "prompt_tokens": 4}
    prompt, born = app._next_turn(body, result)
    assert prompt == render([*body["messages"], warm.reply_message(result)])[:-1] and born == 3
    assert all(b["tools"] == body["tools"] and b["model"] == "glm" for b in bodies) and len(bodies) == 3
    assert app._next_turn(body, {**result, "prompt_tokens": 50}) is None      # nothing past this turn's own state


@pytest.mark.torch
@pytest.mark.parametrize("finish, warms", [("tool_calls", True), ("length", False)])
def test_every_glm_request_queues_its_disk_write_and_only_a_finished_chat_reply_a_pass(finish, warms):
    app_module = importlib.import_module("tensorfold.families.glm5_next.cuda.app")
    from tensorfold.cuda.server import App

    app = app_module.GlmApp.__new__(app_module.GlmApp)
    queued = []
    app.engine = NS(request=NS())
    app.warmer = NS(finished=lambda: None, queue=queued.append)
    app.warming = True
    app._next_turn = lambda body, result: ("pass",)
    result = {"finish": finish}
    original = App.run
    try:
        App.run = lambda self, body, chat, emit, prepared=None, cancelled=None: result
        assert app.run({"model": "glm", "messages": [{"role": "user", "content": "hi"}]}, True, lambda d: True) is result
    finally:
        App.run = original
    assert len(queued) == 1 and queued[0]() == (("pass",) if warms else None)
