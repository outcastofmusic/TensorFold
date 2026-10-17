"""GLM's kept conversations on CUDA, without a GPU: saved rows never exceed the budget at any moment, a dropped
snapshot frees its rows at once, and the resume point is never dropped to make room."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tensorfold.families.glm5_next.cuda.engine import GlmEngine

pytestmark = pytest.mark.torch


class Snap:
    def __init__(self, ids, need, states=5):
        self.ids, self.need, self.states, self.rows, self.nbytes = list(ids), need, states, None, 0
        self.born = -1                     # decode.Snapshot's: a request's own state
        self.branched = False


@pytest.fixture
def kept(monkeypatch):
    """An engine with only its kept-conversation state; save_rows records the saved bytes of every snapshot."""
    made, peaks = [], []

    def save_rows(e, snap):
        snap.rows, snap.nbytes = object(), snap.need
        peaks.append(sum(s.nbytes for s in made if s.rows is not None) + sum(s.states for s in made if s in e.cache))

    decode = SimpleNamespace(row_bytes=lambda e, s: s.need, save_rows=save_rows,
                             snapshot_bytes=lambda s: s.states + (s.nbytes if s.rows is not None else 0))
    monkeypatch.setitem(sys.modules, "tensorfold.families.glm5_next.cuda.decode", decode)
    engine = GlmEngine.__new__(GlmEngine)
    engine.cache, engine.live, engine.cache_bytes, engine.cache_entries, engine.user_id = [], [], 100, 8, -1
    engine.e = engine

    def snap(ids, need, states=5):
        s = Snap(ids, need, states)
        made.append(s)
        return s
    return engine, snap, peaks


def test_a_dropped_snapshot_frees_its_rows_before_the_next_is_saved(kept):
    engine, snap, peaks = kept
    first = snap(range(50), 60)
    follow = snap(range(52), 60)                     # the same conversation's next prompt, whose rows are live
    engine.cache, engine.live = [first, follow], list(range(52))
    engine._take_over([])                             # another conversation takes the caches
    assert first not in engine.cache and first.rows is None and follow.rows is not None
    assert max(peaks) <= engine.cache_bytes           # 0.3.6.1's #54 held both (120 of 100) at the second save


def test_the_resume_point_is_never_dropped_to_save_another_conversation(kept):
    engine, snap, peaks = kept
    hit = snap(range(40), 60)
    hit.rows, hit.nbytes = object(), 60               # saved earlier; the next prompt resumes from it
    live = snap(range(100, 150), 60)
    engine.cache, engine.live = [hit, live], list(range(100, 150))
    engine._take_over(list(range(40)) + [7, 8])
    assert hit in engine.cache and hit.rows is not None
    assert live not in engine.cache and live.rows is None and not peaks    # no room: not saved, dropped


def test_remember_keeps_the_newest_and_frees_what_it_drops(kept):
    engine, snap, _ = kept
    old = snap(range(10), 80)
    old.rows, old.nbytes = object(), 80
    engine.cache = [old]
    new = snap(range(20, 30), 0, states=30)
    engine._remember(new)
    assert engine.cache == [new] and old.rows is None
    again = snap(range(20, 30), 0)
    engine._remember(again)                           # the same prompt again replaces its entry
    assert engine.cache == [again] and new.rows is None


def test_a_prompt_keeps_a_state_where_its_system_block_ends(kept):
    engine, _, _ = kept
    engine.user_id = 9
    system = list(range(100, 500))
    prompt = system + [9, 1, 2, 3]
    assert engine._system_end(prompt, 0) == [400]     # the first user message's start
    assert engine._system_end(prompt, 400) == []      # resumed from there already
    assert engine._system_end(prompt, 300) == []      # resumed within MIN_GAP of it: the state would save too little
    assert engine._system_end(system + [1, 2], 0) == []
    assert engine._system_end([9, 1, 2, 3], 0) == []  # no system block


def test_a_system_blocks_state_outlives_its_conversations_turns(kept):
    engine, snap, _ = kept
    engine.user_id = 9
    system = list(range(100, 500))
    shared = snap(system, 0)
    engine._remember(shared)
    prompt = system + [9]
    for turn in range(2 * engine.cache_entries):      # each turn keeps its prompt end
        prompt = prompt + [1, 2]
        engine._touch_system_end(prompt)
        engine._remember(snap(prompt[:-1], 0))
    assert shared in engine.cache and len(engine.cache) == engine.cache_entries


def test_earlier_turn_ends_go_before_another_conversations_states(kept):
    engine, snap, _ = kept
    engine.user_id = 9
    newest = []
    for base in (100, 1000, 2000):                    # three conversations, six turns each, eight entries
        system = list(range(base, base + 400))
        shared = snap(system, 0)
        engine._remember(shared)
        prompt = system + [9]
        for turn in range(6):
            prompt = prompt + [1, 2]
            engine._touch_system_end(prompt)
            end = snap(prompt[:-1], 0)
            engine._remember(end)
        newest += [shared, end]
    assert all(s in engine.cache for s in newest) and len(engine.cache) == engine.cache_entries


def test_a_turn_resumed_after_its_later_turn_is_not_the_superseded_one(kept):
    """Back to turn one after turn two: turn one is a branch point, and stale turn two goes first."""
    engine, snap, _ = kept
    engine.user_id = 7
    one, two = snap([1, 7, 2], 10), snap([1, 7, 2, 7, 3], 10)
    engine.cache = [one, two]
    assert engine._superseded() is one                   # a linear conversation: turn one is behind turn two
    engine._use(one)                                     # turn one resumed again
    assert engine.cache == [two, one] and one.branched and engine._superseded() is None


def test_a_history_every_continuation_resumes_stays_a_branch_point(kept):
    """An agent restarting each step from one history: each step keeps a longer state past it (upstream #735)."""
    engine, snap, _ = kept
    engine.user_id = 7
    history = snap([1, 7, 2], 10)
    engine.cache = [history, snap([1, 7, 2, 7, 3], 10)]  # the first step's state extends the history
    for step in (4, 5, 6):
        engine._use(history)
        engine.cache.append(snap([1, 7, 2, 7, step], 10))
        assert engine._superseded() is not history       # each step's own state goes first, never the shared history
    assert history.branched


def test_a_conversation_that_only_grows_still_frees_its_old_turns_first(kept):
    engine, snap, _ = kept
    engine.user_id = 7
    one = snap([1, 7, 2], 10)
    engine.cache = [one]
    engine._use(one)                                     # turn two resumes turn one: nothing extends it yet
    two = snap([1, 7, 2, 7, 3], 10)
    engine.cache.append(two)
    engine._use(two)                                     # turn three resumes turn two
    engine.cache.append(snap([1, 7, 2, 7, 3, 7, 4], 10))
    assert not one.branched and not two.branched and engine._superseded() is one

def test_a_prompt_keeps_a_state_where_its_last_user_message_starts(kept):
    engine, _, _ = kept
    engine.user_id = 9
    system, history, question = list(range(100, 500)), [9] + list(range(1000, 1300)), [9, 1, 2, 3]
    prompt = system + history + question
    last = len(system) + len(history)
    assert engine._stops(prompt, 0) == [400, last]
    assert engine._stops(prompt, 400) == [last]               # resumed from the system block's end
    assert engine._stops(prompt, last - 100) == []            # resumed within MIN_GAP of it
    assert engine._stops(prompt, last) == []                  # a branch resumed from it keeps no second copy
    assert engine._stops(system + [9, 1, 2, 3], 0) == [400]   # the first user message: the system block's end
    assert engine._stops(system + [9] + list(range(1000, 1100)) + question, 0) == [400]   # within MIN_GAP of it
    assert engine._stops(history + question, 0) == [len(history)]                         # no system block
    assert engine._stops(system + history + [9], 0) == [400]  # the prompt's own end keeps that state
    engine.user_id = -1
    assert engine._stops(prompt, 0) == []


def test_branches_of_one_history_all_resume_where_the_last_question_starts(kept):
    """Five questions after one history: the first keeps the history's state, which the next four resume, and it
    stays kept while each branch's own states fill the entry cap."""
    engine, snap, _ = kept
    engine.user_id = 9
    history = list(range(100, 500)) + [9] + list(range(1000, 1300))
    engine.serial_only = True                                     # _resume: no draft caches to fit
    other = [snap([9, 2, i], 0) for i in range(3)]                # other conversations' states
    for s in other:
        engine._remember(s)
    shared = snap(history, 0)
    engine._remember(shared)                                      # the first question's prefill passes it
    for q in range(5):
        prompt = history + [9, 50 + q, 51, 52]
        if q:
            assert engine._resume(prompt, [0, 0, 0, 0]) is shared
            engine._use(shared)
            assert shared.branched
        end = snap(prompt[:-1], 0)
        engine._remember(end)
        warm = snap(prompt + [60, 61, 9], 0)
        warm.born = len(prompt) - 1
        engine._remember(warm)
        assert shared in engine.cache and len(engine.cache) <= engine.cache_entries


def test_an_unresumed_history_state_goes_before_another_conversations(kept):
    """A history's state nobody resumed yet is the first to go when the cache is full: its turn's end extends it."""
    engine, snap, _ = kept
    engine.user_id = 9
    others = [snap([7, 9, i], 0) for i in range(6)]
    for s in others:
        engine._remember(s)
    history = snap(list(range(100, 400)) + [9, 1, 2], 0)
    engine._remember(history)
    engine._remember(snap(history.ids + [9, 5, 6], 0))           # its turn's end: the eighth entry
    engine._remember(snap([8, 9, 1], 0))                         # a ninth: the unresumed history state goes
    assert history not in engine.cache and all(s in engine.cache for s in others)


def test_prefill_keeps_every_point_as_a_cut_without_splitting_a_chunk(monkeypatch):
    """Points inside a chunk become cuts of that chunk's one pass, a point on a chunk's end needs none, and each kept
    state holds its prefix; the prompt's end comes last."""
    import torch

    from tensorfold.families.glm5_next.cuda import decode

    passes = []

    def compute(w, st, b, R, *, nch, host_pos, cuts):
        passes.append((st.pos, R, [c.point for c in cuts]))
        return torch.zeros(1)

    def commit(w, st, b, R, keep):
        st.pos += R

    monkeypatch.setattr(decode, "stage", lambda w, st, b, chunk: len(chunk))
    monkeypatch.setattr(decode, "compute", compute)
    monkeypatch.setattr(decode, "commit", commit)
    monkeypatch.setattr(decode, "chunks_for", lambda st, R: 1)
    st = SimpleNamespace(rec=torch.zeros(2, 3), conv=torch.zeros(3), cur=[0], pos=0)
    e = SimpleNamespace(w=SimpleNamespace(mtp=None, cfg=SimpleNamespace(mtp_layers=0)), st=st,
                        pbuf=SimpleNamespace(fnormed=torch.zeros(16, 1)), prefill_rows=16, constraint=None,
                        reset=lambda: setattr(st, "pos", 0), sample=lambda last, at, s: [5], follow=lambda t: None)
    prompt = list(range(40))
    kept = []
    assert decode.prefill(e, prompt, None, keep_at=39, stops=[3, 9, 16, 30], keep=kept.append) == 5
    assert passes == [(0, 16, [3, 9]), (16, 16, [14]), (32, 8, [7])]   # 16 ends the first chunk: no cut
    assert [len(s.ids) for s in kept] == [3, 9, 16, 30, 39]
    assert all(s.ids == prompt[:len(s.ids)] for s in kept)
    passes.clear()
    kept.clear()
    decode.prefill(e, prompt, None, keep_at=40, keep=kept.append, rows=8)   # a background pass's state at the end
    assert passes == [(p, 8, []) for p in range(0, 40, 8)] and [len(s.ids) for s in kept] == [40]
