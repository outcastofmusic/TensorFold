"""Copy drafts: the latest earlier copy's continuation, at least COPY_MATCH long, never the needle; the env switch."""

import os

from tensorfold.families.qwen4_exp.cuda.copy_drafts import COPY_MATCH, CopyIndex, enabled

A = list(range(100, 140))                 # 40 distinct tokens
B = list(range(200, 212))                 # 12 more


def test_no_copy_no_chain():
    assert CopyIndex(A).chain(15) == []
    assert CopyIndex(A + A[:COPY_MATCH - 1]).chain(15) == []      # the tail repeats fewer than COPY_MATCH tokens


def test_a_repeated_tail_drafts_its_continuation():
    ctx = A + B + A[:COPY_MATCH]                                    # the tail is A's first 8 tokens again
    assert CopyIndex(ctx).chain(15) == A[COPY_MATCH:COPY_MATCH + 15]
    assert CopyIndex(ctx).chain(40) == A[COPY_MATCH:] + B[:8]       # up to max_nodes, across what followed


def test_a_round_with_fewer_slots_than_the_match_still_copies():
    ctx = A + B + A[:COPY_MATCH]
    assert CopyIndex(ctx).chain(6) == A[COPY_MATCH:COPY_MATCH + 6]  # Flash Next's 6 drafts a round
    assert CopyIndex(ctx).chain(COPY_MATCH) == A[COPY_MATCH:2 * COPY_MATCH]
    assert CopyIndex(ctx).chain(0) == []
    assert CopyIndex(A[:COPY_MATCH] + B[:5] + A[:COPY_MATCH]).chain(6) == B[:5] + A[:1]


def test_the_latest_copy_wins_and_the_needle_is_not_a_copy():
    first = A[:COPY_MATCH] + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    second = A[:COPY_MATCH] + [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ctx = first + second + A[:COPY_MATCH]
    assert CopyIndex(ctx).chain(10) == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # a tail repeated back to back: the earlier copy is followed by the needle, which is drafted (the pattern repeats)
    assert CopyIndex(A[:COPY_MATCH] + A[:COPY_MATCH]).chain(15) == A[:COPY_MATCH]


def test_extend_is_the_same_as_rebuilding():
    ctx = A + B + A[:COPY_MATCH + 4]
    grown = CopyIndex(ctx[:30])
    for t in ctx[30:]:
        grown.extend([t])
    assert grown.chain(15) == CopyIndex(ctx).chain(15) == A[COPY_MATCH + 4:COPY_MATCH + 4 + 15]
    assert grown.ctx == ctx


def test_switch(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_COPY_DRAFTS", raising=False)
    assert enabled() and not enabled(False)
    for off in ("0", "off", "False", " no "):
        monkeypatch.setenv("TENSORFOLD_COPY_DRAFTS", off)
        assert not enabled()
    monkeypatch.setenv("TENSORFOLD_COPY_DRAFTS", "1")
    assert enabled(False)


def test_lone_graphs_switch(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.engine import lone_graphs

    monkeypatch.delenv("TENSORFOLD_LONE_GRAPHS", raising=False)
    assert not lone_graphs()
    for on in ("1", "on", " True "):
        monkeypatch.setenv("TENSORFOLD_LONE_GRAPHS", on)
        assert lone_graphs()
    monkeypatch.setenv("TENSORFOLD_LONE_GRAPHS", "0")
    assert not lone_graphs()
