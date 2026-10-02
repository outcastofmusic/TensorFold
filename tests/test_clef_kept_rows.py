"""Decision sessions: a prompt resumed from the kept prefill of an earlier one gives a fresh prefill's rows, bit for bit,
and a SystemOne session keeps one prefill per conversation."""

from __future__ import annotations

import pytest

from tests.test_joint_schema import QUESTIONS, _ranked_head, chars
from tests.test_qwen27_prompt_end_cache_host import _bits_equal, _prompt, cpu, cuda_modules  # noqa: F401 - fixtures

torch = pytest.importorskip("torch")


def engine_for(cpu):
    from tensorfold.families.qwen3_5.cuda import engine as engine_mod

    eng = engine_mod.Qwen27Engine.__new__(engine_mod.Qwen27Engine)
    eng.w, eng.torch, eng.tp, eng.concurrent, eng.room, eng.context_window = cpu.w, cpu.torch, 1, False, None, 10**6
    return eng


PREFIX, SCHEMA = _prompt(10, 1), _prompt(20, 2)
STATE = _prompt(30, 3)
LONGER = STATE + _prompt(25, 4)                 # the state grown by a paragraph


def decision(state):
    return PREFIX + state + SCHEMA, len(PREFIX + state) - 4      # (prompt, where the next call resumes)


def test_a_resumed_prompt_gives_a_fresh_prefills_rows(cpu):
    eng = engine_for(cpu)
    first, keep = decision(STATE)
    rows, kept, cached = eng.hidden_rows_from(first, keep)
    assert cached == 0 and kept.ids == tuple(first[:keep]) and kept.rows.shape[0] == keep
    assert _bits_equal(torch, rows, eng.hidden_rows(first))
    second, keep2 = decision(LONGER)
    passes = []
    prefill_chunk = cpu.m.prefill.prefill_chunk
    cpu.m.prefill.prefill_chunk = lambda *a, **k: passes.append(k.get("cut", 0)) or prefill_chunk(*a, **k)
    try:
        rows, kept2, cached = eng.hidden_rows_from(second, keep2, kept)
    finally:
        cpu.m.prefill.prefill_chunk = prefill_chunk
    assert cached == keep
    if cpu.size >= len(second):                    # one chunk: one weight pass, the kept point cut inside it
        assert passes == [keep2 - keep]
    assert _bits_equal(torch, rows, eng.hidden_rows(second))
    assert kept2.ids == tuple(second[:keep2])


def test_one_kept_point_serves_two_continuations(cpu):
    """The tail after the kept point writes past it, so a second resume from the same point is still exact."""

    eng = engine_for(cpu)
    first, keep = decision(STATE)
    _, kept, _ = eng.hidden_rows_from(first, keep)
    for extra in (_prompt(25, 4), _prompt(7, 5)):
        prompt, at = decision(STATE + extra)
        rows, _, cached = eng.hidden_rows_from(prompt, at, kept)
        assert cached == keep and _bits_equal(torch, rows, eng.hidden_rows(prompt))


@pytest.mark.parametrize("other", ["another state", "a shorter keep"])
def test_a_kept_point_that_does_not_start_the_prompt_is_not_used(cpu, other):
    eng = engine_for(cpu)
    first, keep = decision(STATE)
    _, kept, _ = eng.hidden_rows_from(first, keep)
    prompt, at = decision(_prompt(40, 9)) if other == "another state" else (first, keep - 1)
    rows, _, cached = eng.hidden_rows_from(prompt, at, kept)
    assert cached == 0 and _bits_equal(torch, rows, eng.hidden_rows(prompt))


# --- the joint schema's sessions, over an engine that records what it resumed ---------------------------------

class _SessionEngine:
    def __init__(self):
        self.resumed = []

    def hidden_rows(self, prompt):
        self.resumed.append(None)
        return torch.zeros((len(prompt), 4))

    def hidden_rows_from(self, prompt, keep_at, kept=None):
        usable = kept is not None and len(kept) <= keep_at and tuple(prompt[:len(kept)]) == kept
        self.resumed.append(len(kept) if usable else 0)
        return torch.zeros((len(prompt), 4)), tuple(prompt[:keep_at]), len(kept) if usable else 0

    def head_rows(self, ids):
        return torch.zeros((len(ids), 4))


def _joint():
    from tensorfold.server.joint_schema import JointSchema

    return JointSchema(_ranked_head, _SessionEngine(), chars)


def ask(joint, state, session="reply-1"):
    body = {"model": "clef", "state": state, "questions": QUESTIONS}
    return joint.systemone(body if session is None else {**body, "session": session})


def test_a_session_resumes_from_its_last_state_and_reports_the_reuse():
    joint = _joint()
    first = "First, list the small primes.\n\n"            # longer than the margin, so the kept point is in it
    assert ask(joint, first)["usage"]["cached_tokens"] == 0
    usage = ask(joint, first + "Then try each one.\n\n")["usage"]
    assert usage["cached_tokens"] > 0 and usage["cached_tokens"] == joint.engine.resumed[-1]
    assert ask(joint, "Something else entirely, at length.")["usage"]["cached_tokens"] == 0


def test_without_a_session_nothing_is_kept():
    joint = _joint()
    assert ask(joint, "One.", session=None)["usage"]["cached_tokens"] == 0
    assert joint.engine.resumed == [None] and not joint.sessions


def test_sessions_are_capped_and_expire(monkeypatch):
    from tensorfold.server import joint_schema

    clock = [0.0]
    monkeypatch.setattr(joint_schema.time, "monotonic", lambda: clock[0])
    joint = _joint()
    for n in range(joint_schema.MAX_SESSIONS + 1):
        ask(joint, "One.", session=f"s{n}")
    assert list(joint.sessions) == [f"s{n}" for n in range(1, joint_schema.MAX_SESSIONS + 1)]   # s0 went first
    clock[0] = joint_schema.SESSION_IDLE + 1
    assert ask(joint, "One. Two.", session="s1")["usage"]["cached_tokens"] == 0       # expired while idle
    assert list(joint.sessions) == ["s1"]


@pytest.mark.parametrize("session", ["", 7, "x" * 201])
def test_a_bad_session_is_refused(session):
    from tensorfold.server.decisions import DecisionError

    with pytest.raises(DecisionError, match="session must be"):
        ask(_joint(), "One.", session=session)
