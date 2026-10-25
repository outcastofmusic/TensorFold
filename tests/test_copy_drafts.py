"""Copy drafts: the latest earlier copy's continuation, at least COPY_MATCH long, never the needle; the env switch."""

import os

from tensorfold.families.qwen4_exp.cuda.copy_drafts import COPY_MATCH, CopyIndex, enabled, most_rows, next_rows

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


def test_window_doubles_after_a_whole_copy_and_halves_after_a_broken_one():
    assert [next_rows(r, True, 7, 32) for r in (7, 14, 28, 32)] == [14, 28, 32, 32]
    assert [next_rows(r, False, 7, 32) for r in (32, 16, 8, 7)] == [16, 8, 7, 7]
    assert next_rows(7, True, 7, 7) == 7                            # TENSORFOLD_COPY_ROWS=0: the chain's window


def test_a_window_caps_the_drafts_and_follows_the_outcomes():
    ctx = A + B + A[:COPY_MATCH]
    index = CopyIndex(ctx, rows=7, most=32)
    assert index.chain(40) == A[COPY_MATCH:COPY_MATCH + 6]
    index.verified(6, 6)
    assert index.chain(40) == A[COPY_MATCH:COPY_MATCH + 13]
    index.verified(13, 13)
    assert len(index.chain(40)) == 27 and len(index.chain(5)) == 5    # the reply's room caps it too
    index.verified(27, 3)
    assert len(index.chain(40)) == 13
    assert CopyIndex(ctx, rows=7).most == 7


def test_rows_switch(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_COPY_ROWS", raising=False)
    assert most_rows(6) == 32
    monkeypatch.setenv("TENSORFOLD_COPY_ROWS", "0")
    assert most_rows(6) == 7                                        # the MTP chain's window, as before 2026-10-20
    monkeypatch.setenv("TENSORFOLD_COPY_ROWS", "32")
    assert most_rows(6) == 32
    monkeypatch.setenv("TENSORFOLD_COPY_ROWS", "4")
    assert most_rows(6) == 7                                        # never narrower than the MTP chain


def test_a_templated_list_copies_the_line_that_matches_furthest_back():
    def line(i):
        return [1000 + i, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]       # an item number, then a shared template
    prompt = [5] + [t for i in range(1, 6) for t in line(i)]
    reply = line(1) + line(2)                                        # the reply repeats the prompt's list
    index = CopyIndex(prompt + reply)
    assert index.chain(4) == [1003, 7, 8, 9]                          # the prompt's line 3, not the reply's line 1
    assert CopyIndex(prompt + reply, min_back=12).chain(4) == [1003, 7, 8, 9]   # its match runs 14 further
    template = [5] + line(1) + line(2)[:9]                            # only the template repeats: 1 token further
    assert CopyIndex(template).chain(4) and CopyIndex(template, min_back=4).chain(4) == []


def test_min_back_takes_the_occurrence_that_reaches_far_enough_over_one_that_fills_the_window():
    needle, before = list(range(1, 9)), list(range(100, 116))     # the tail's 8 tokens and the 16 before them
    short = [999] + needle + list(range(300, 380))                  # matches 0 further and fills any window
    far = before + needle + list(range(60, 70))                     # matches 16 further, 34 tokens follow it
    context = [5] + short + far + before + needle
    assert CopyIndex(context).chain(60)[:3] == [300, 301, 302]       # the full window wins without min_back
    assert CopyIndex(context, min_back=8).chain(60)[:3] == [60, 61, 62]


def test_a_tie_goes_to_the_latest_occurrence():
    ctx = [1] + A[:COPY_MATCH] + [2, 3] * 5 + [1] + A[:COPY_MATCH] + [4, 5] * 5 + [1] + A[:COPY_MATCH]
    assert CopyIndex(ctx).chain(4) == [4, 5, 4, 5]
