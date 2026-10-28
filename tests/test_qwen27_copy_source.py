"""CopyIndex's sources: "longest" continues the occurrence whose earlier tokens match furthest; ``unsure`` and ``reach``
describe the occurrences of the last gram."""

import importlib

import pytest


@pytest.fixture
def decode(monkeypatch):
    from tensorfold.families.qwen3_5.cuda import decode as module

    yield module
    importlib.reload(module)


def _lines(n):
    """Lines that all end alike, so a gram at a line's end recurs on every line before it."""

    out = []
    for i in range(n):
        out += [100 + i, 200 + i % 7, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return out


def test_longest_continues_the_line_being_repeated(decode, monkeypatch):
    listing = _lines(12)
    context = listing + [0] + listing[:5 * 11]          # the copy is five lines into repeating the list
    monkeypatch.setattr(decode, "COPY_SOURCE", "latest")
    latest = decode.CopyIndex().propose(context, 11)
    monkeypatch.setattr(decode, "COPY_SOURCE", "longest")
    index = decode.CopyIndex()
    longest = index.propose(context, 11)
    assert longest == listing[5 * 11:6 * 11]            # the sixth line, as the list goes on
    assert latest != longest                            # today's rule takes the line just written
    assert index.unsure and index.reach >= 5 * 11 - 8   # the lines disagree on what follows the gram


def test_reach_counts_tokens_before_the_gram_past_whole_slices(decode):
    body = list(range(1000, 1100))
    context = [5] + body + [7] + body                   # the gram's earlier occurrence follows 100 matching tokens
    assert decode.CopyIndex()._reach(context, 1 + len(body) - 8) == len(body) - 8


@pytest.mark.parametrize("seed", range(20))
def test_reach_matches_a_token_by_token_count(decode, seed):
    import random

    rng = random.Random(seed)
    unit = [rng.randrange(4) for _ in range(rng.randrange(10, 200))]
    context = (unit * 4)[:rng.randrange(len(unit) + 20, 4 * len(unit))]
    index = decode.CopyIndex()
    here = len(context) - index.n
    for start in range(here):
        k = 0
        while k < min(decode.REACH, start) and context[start - 1 - k] == context[here - 1 - k]:
            k += 1
        assert index._reach(context, start) == k


def test_a_unique_continuation_is_sure(decode):
    context = list(range(20, 60)) + list(range(20, 30))
    index = decode.CopyIndex()
    assert index.propose(context, 8) == list(range(30, 38)) and not index.unsure and index.reach == 2


def test_repeating_text_copies_a_whole_window_rather_than_the_latest_short_one(decode, monkeypatch):
    monkeypatch.setattr(decode, "COPY_SOURCE", "longest")
    context = [3, 1, 4, 1, 5, 9, 2, 6] * 12
    assert decode.CopyIndex().propose(context, 32) == ([3, 1, 4, 1, 5, 9, 2, 6] * 4)
