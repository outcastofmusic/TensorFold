"""GLM decode loops end the round a stop is asked for on one rank: a thinking budget's cut and a client that left stop
the decoding, where the loops decoded on to an end token the reply never reads (2026-10-16: 4,139 tokens kept of
about 31,000 decoded, 923 s)."""

from __future__ import annotations

import importlib
from types import SimpleNamespace as NS

import pytest

pytestmark = pytest.mark.torch

NEVER_ENDS = 7                    # the fake model's every token: never an end token, so only a stop ends a loop
COUNT = 50


@pytest.fixture
def loops(monkeypatch):
    import torch

    decode = importlib.import_module("tensorfold.families.glm5_next.cuda.decode")
    choice = importlib.import_module("tensorfold.families.glm5_next.cuda.drafter_choice")
    for module in (decode, choice):
        monkeypatch.setattr(module, "commit", lambda *args: None)
        monkeypatch.setattr(module, "_sync", lambda w: None)
        monkeypatch.setattr(module, "draft", lambda *args: [])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return decode, choice


class FakeEngine:
    def __init__(self, comm=None):
        import torch

        self.w = NS(cfg=NS(eos=(99,), hidden=4), comm=comm, device="cpu")
        self.st = NS(pos=10, mtp_drafted=0)
        self.buf = NS(taps=[])
        self.last_hidden = torch.zeros(1, 4, dtype=torch.bfloat16)

    def verify_window(self, tokens):
        return list(tokens)

    def forward(self, tokens):
        import torch

        return torch.zeros(len(tokens), 8)

    def sample(self, logits, positions, sampling):
        return [NEVER_ENDS] * len(positions)

    def follow(self, tokens):
        pass

    def main_hidden(self, rows):
        import torch

        return torch.zeros(rows.stop - rows.start, 4, dtype=torch.bfloat16)

    def tap_rows(self, keep):
        import torch

        return torch.zeros(keep, 0, dtype=torch.bfloat16)


class Drafter:
    def propose(self, last, depth, sampling, confidence):
        return []

    def add_taps(self, rows):
        pass


def stop_after(n):
    """A feed that asks to stop once it has seen ``n`` tokens, and records every token it was given."""

    seen = []

    def feed(tokens):
        seen.extend(tokens)
        return len(seen) >= n

    return feed, seen


def run(loops, path, engine, feed):
    decode, choice = loops
    if path == "serial":
        return decode.serial_decode(engine, 1, COUNT, None, on_tokens=feed)
    if path == "mtp":
        return decode.mtp_decode(engine, 1, COUNT, None, policy=decode.DepthPolicy(1, fixed=True), on_tokens=feed)
    if path == "dflash":
        return decode.dflash_decode(engine, Drafter(), 1, COUNT, None, policy=decode.DepthPolicy(1, fixed=True),
                                    on_tokens=feed)
    return choice.auto_decode(engine, None, 1, COUNT, None, choice=None, m_policy=decode.DepthPolicy(1, fixed=True),
                              f_policy=decode.DepthPolicy(1, fixed=True), on_tokens=feed)


@pytest.mark.parametrize("path", ["serial", "mtp", "dflash", "auto"])
def test_one_rank_stops_the_round_the_feed_asks(loops, path):
    feed, seen = stop_after(4)
    result = run(loops, path, FakeEngine(), feed)
    assert len(seen) == 4
    assert len(result.tokens) == 5                       # the first token, then the four the feed took


@pytest.mark.parametrize("path", ["serial", "mtp", "dflash", "auto"])
def test_two_ranks_decode_to_the_count_since_rank_one_cannot_hear_the_stop(loops, path):
    feed, seen = stop_after(4)
    result = run(loops, path, FakeEngine(comm=object()), feed)
    assert len(result.tokens) == COUNT
