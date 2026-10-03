"""Decision sessions: a prompt resumed from the kept prefill of an earlier one gives a fresh prefill's rows, bit for bit,
and a SystemOne session keeps one prefill per conversation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_joint_schema import QUESTIONS, _ranked_head, chars
from tests.test_qwen27_prompt_end_cache_host import _bits_equal, _prompt, cpu, cuda_modules  # noqa: F401 - fixtures

torch = pytest.importorskip("torch")


def engine_for(cpu):  # noqa: F811 - the imported fixture
    from tensorfold.families.qwen3_5.cuda import engine as engine_mod

    eng = engine_mod.Qwen27Engine.__new__(engine_mod.Qwen27Engine)
    eng.w, eng.torch, eng.tp, eng.concurrent, eng.room, eng.context_window = cpu.w, cpu.torch, 1, False, None, 10**6
    return eng


PREFIX, SCHEMA = _prompt(10, 1), _prompt(20, 2)
STATE = _prompt(30, 3)
LONGER = STATE + _prompt(25, 4)                 # the state grown by a paragraph


def decision(state):
    return PREFIX + state + SCHEMA, len(PREFIX + state) - 4      # (prompt, where the next call resumes)


def test_a_resumed_prompt_gives_a_fresh_prefills_rows(cpu):  # noqa: F811 - the imported fixture
    eng = engine_for(cpu)
    first, keep = decision(STATE)
    rows, kept, cached = eng.hidden_rows_from(first, keep)
    assert cached == 0 and kept.ids == tuple(first[:keep]) and kept.rows.shape[0] >= len(first)
    assert _bits_equal(torch, rows, eng.hidden_rows(first))
    second, keep2 = decision(LONGER)
    passes, flags = [], []
    prefill_chunk = cpu.m.prefill.prefill_chunk

    def spy(*args, **kwargs):
        passes.append(kwargs.get("cut", 0))
        flags.append(cpu.m.prefill.DECISION_ROWS.get())
        return prefill_chunk(*args, **kwargs)

    cpu.m.prefill.prefill_chunk = spy
    try:
        rows, kept2, cached = eng.hidden_rows_from(second, keep2, kept)
    finally:
        cpu.m.prefill.prefill_chunk = prefill_chunk
    assert cached == keep
    assert _bits_equal(torch, rows, eng.hidden_rows(second))
    assert kept2.ids == tuple(second[:keep2])
    # one weight pass a chunk from the kept point on, the next kept point cut inside its chunk, on the decision path
    spans = cpu.m.prefill.chunks(keep, len(second))
    assert passes == [keep2 - a if a < keep2 < b else 0 for a, b in spans] and all(flags)


def test_the_kept_rows_are_written_in_place_as_the_session_grows(cpu):  # noqa: F811 - the imported fixture
    eng = engine_for(cpu)
    first, keep = decision(STATE)
    _, kept, _ = eng.hidden_rows_from(first, keep)
    second, keep2 = decision(LONGER)
    rows, kept2, _ = eng.hidden_rows_from(second, keep2, kept)
    assert kept2.rows.data_ptr() == kept.rows.data_ptr()           # the prefix's rows are not copied again
    assert rows.data_ptr() == kept.rows.data_ptr()


def test_without_a_session_the_buffers_hold_only_the_prompt(cpu):  # noqa: F811 - the imported fixture
    eng = engine_for(cpu)
    first, _ = decision(STATE)
    _, kept, _ = eng.hidden_rows_from(first, len(first), headroom=0)
    assert kept.rows.shape[0] == len(first) and kept.state.limit == len(first)


def test_one_kept_point_serves_two_continuations(cpu):  # noqa: F811 - the imported fixture
    """The tail after the kept point writes past it, so a second resume from the same point is still exact."""

    eng = engine_for(cpu)
    first, keep = decision(STATE)
    _, kept, _ = eng.hidden_rows_from(first, keep)
    for extra in (_prompt(25, 4), _prompt(7, 5)):
        prompt, at = decision(STATE + extra)
        rows, _, cached = eng.hidden_rows_from(prompt, at, kept)
        assert cached == keep and _bits_equal(torch, rows, eng.hidden_rows(prompt))


@pytest.mark.parametrize("other", ["another state", "a shorter keep"])
def test_a_kept_point_that_does_not_start_the_prompt_is_not_used(cpu, other):  # noqa: F811 - the imported fixture
    eng = engine_for(cpu)
    first, keep = decision(STATE)
    _, kept, _ = eng.hidden_rows_from(first, keep)
    prompt, at = decision(_prompt(40, 9)) if other == "another state" else (first, keep - 1)
    rows, _, cached = eng.hidden_rows_from(prompt, at, kept)
    assert cached == 0 and _bits_equal(torch, rows, eng.hidden_rows(prompt))


# --- the joint schema's sessions, over an engine that records what it resumed ---------------------------------

class _SessionEngine:
    def __init__(self):
        self.resumed, self.fail = [], False
        self.room = SimpleNamespace(release=None)

    def hidden_rows(self, prompt):
        self.resumed.append(None)
        return torch.zeros((len(prompt), 4))

    def hidden_rows_from(self, prompt, keep_at, kept=None):
        if self.fail:
            raise ValueError("a decision prompt does not fit")
        usable = kept is not None and len(kept.ids) <= keep_at and tuple(prompt[:len(kept.ids)]) == kept.ids
        self.resumed.append(len(kept.ids) if usable else 0)
        made = SimpleNamespace(ids=tuple(prompt[:keep_at]), state=SimpleNamespace(kv=[object()]))
        return torch.zeros((len(prompt), 4)), made, len(kept.ids) if usable else 0

    def head_rows(self, ids):
        return torch.zeros((len(ids), 4))


def _joint():
    from tensorfold.server.joint_schema import JointSchema

    return JointSchema(_ranked_head, _SessionEngine(), chars)


def ask(joint, state, session=True):
    body = {"model": "clef", "state": state, "questions": QUESTIONS}
    return joint.systemone(body if session is None else {**body, "session": session})


FIRST = "First, list the small primes.\n\n"          # longer than the margin, so the kept point is in it


def test_a_session_resumes_from_its_last_state_and_reports_the_reuse():
    joint = _joint()
    opened = ask(joint, FIRST)
    assert opened["usage"]["cached_tokens"] == 0 and len(opened["session"]) == 32
    resumed = ask(joint, FIRST + "Then try each one.\n\n", opened["session"])
    assert resumed["session"] == opened["session"]
    assert resumed["usage"]["cached_tokens"] > 0 and resumed["usage"]["cached_tokens"] == joint.engine.resumed[-1]


def test_a_guessed_id_finds_no_ones_prefill_and_evicts_nothing():
    joint = _joint()
    theirs = ask(joint, FIRST)["session"]
    guessed = ask(joint, FIRST + "A guess at their text.", "reply-1")
    assert guessed["usage"]["cached_tokens"] == 0 and guessed["session"] not in (theirs, "reply-1")
    assert theirs in joint.sessions


def test_without_a_session_nothing_is_kept():
    joint = _joint()
    reply = ask(joint, "One.", session=None)
    assert reply["usage"]["cached_tokens"] == 0 and "session" not in reply
    assert joint.engine.resumed == [None] and not joint.sessions


def test_a_failed_request_leaves_the_session_as_it_was():
    joint = _joint()
    name = ask(joint, FIRST)["session"]
    kept = joint.sessions[name][0]
    joint.engine.fail = True
    with pytest.raises(ValueError):
        ask(joint, FIRST + "More.", name)
    assert joint.sessions[name][0] is kept


def test_sessions_are_capped_and_expire_on_any_request(monkeypatch):
    from tensorfold.server import joint_schema

    clock = [0.0]
    monkeypatch.setattr(joint_schema.time, "monotonic", lambda: clock[0])
    joint = _joint()
    names = [ask(joint, FIRST)["session"] for _ in range(joint_schema.MAX_SESSIONS + 1)]
    assert list(joint.sessions) == names[1:]                         # the oldest went
    clock[0] = joint_schema.SESSION_IDLE + 1
    ask(joint, "One.", session=None)                                 # a request without a session purges too
    assert not joint.sessions


def test_the_engines_room_counts_and_releases_the_sessions():
    joint = _joint()
    first, second = ask(joint, FIRST)["session"], ask(joint, FIRST)["session"]
    room = joint.engine.room
    assert room.others() == [joint.sessions[first][0].state.kv, joint.sessions[second][0].state.kv]
    assert room.release() and list(joint.sessions) == [second]       # the least recently used goes first
    assert room.release() and not room.release()


@pytest.mark.parametrize("session", ["", 7, False, "x" * 201])
def test_a_bad_session_is_refused(session):
    from tensorfold.server.decisions import DecisionError

    with pytest.raises(DecisionError, match="session must be"):
        ask(_joint(), "One.", session=session)
