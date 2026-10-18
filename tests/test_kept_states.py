"""The kept-state store (``tensorfold.cuda.kept_states``) on CPU: where a prefill keeps states, which kept prefix a
prompt resumes from, and which entry goes first past its caps."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.kept_states import Store, keep_points
from tensorfold.cuda.markers import MIN_GAP

OPENER = 1                              # the fake chat template's message start


def marks(ids) -> list[int]:
    return [i for i, t in enumerate(ids) if t == OPENER]


# -- where states are kept ----------------------------------------------------------------------------------------
def test_keep_points_take_message_starts_and_the_prompt_end():
    p = [5] * 2000
    assert keep_points(p, 0, None) == [1999]
    p[300] = p[900] = p[1990] = OPENER
    # the end gives way to a message start within MIN_GAP before it
    assert keep_points(p, 0, marks) == [300, 900, 1990]
    # nothing within MIN_GAP of the resume point, and the end still counts
    assert keep_points(p[:1500], 100, marks) == [900, 1499]
    assert keep_points(p[:1500], 300 - MIN_GAP + 1, marks) == [900, 1499]
    assert keep_points(p[:1500], 1499, marks) == []                  # resumed at the end: kept already
    assert keep_points([7], 0, None) == [1]


# -- the store ----------------------------------------------------------------------------------------------------
def snap(n: int) -> dict:
    return {"x": torch.zeros(n, dtype=torch.uint8)}


def test_store_resumes_the_longest_prefix_with_a_drafter_context():
    s = Store(8, 1 << 20)
    s.add([1, 2], snap(10), None)
    s.add([1, 2, 3], snap(10), snap(5))
    s.add([1, 2, 3, 4], snap(10), None)
    s.add([9, 9], snap(10), snap(5))
    assert s.longest([1, 2, 3, 4, 5], False).ids == (1, 2, 3, 4)
    assert s.longest([1, 2, 3, 4, 5], True).ids == (1, 2, 3)          # a drafting prompt needs the drafter's context
    assert s.kept[-1].ids == (1, 2, 3)                               # a hit is the newest entry now
    assert s.longest([1, 2, 3], False).ids == (1, 2)                 # one token is always left to prefill
    assert s.longest([8], False) is None
    assert s.held == 4 * 10 + 2 * 5


def test_store_gives_up_a_conversations_earlier_turns_first():
    s = Store(3, 1 << 20)
    system = [7] * 4
    s.add(system, snap(1), None)                                     # a system block's end
    s.add(system + [1], snap(1), None)                               # turn 1
    s.add(system + [1, 2], snap(1), None)                            # turn 2 supersedes turn 1
    s.add([8, 8], snap(1), None)                                     # another conversation
    assert [k.ids for k in s.kept] == [tuple(system), (7, 7, 7, 7, 1, 2), (8, 8)]
    s.add([8, 8, 8], snap(1), None)                                  # nothing between two entries: the oldest goes
    assert [k.ids for k in s.kept] == [(7, 7, 7, 7, 1, 2), (8, 8), (8, 8, 8)]


def test_store_keeps_within_its_bytes():
    s = Store(8, 100)
    s.add([1], snap(40), None)
    s.add([2], snap(40), None)
    s.add([3], snap(40), None)
    assert [k.ids for k in s.kept] == [(2,), (3,)] and s.held == 80
    s.add([4], snap(101), None)                                      # larger than the budget: not kept
    assert [k.ids for k in s.kept] == [(2,), (3,)]
    s.add([3], snap(30), snap(10))                                   # the same ids replace their entry
    assert [k.ids for k in s.kept] == [(2,), (3,)] and s.held == 80
    assert not Store(0, 100).on and not Store(8, 0).on
