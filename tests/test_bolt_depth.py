"""Nemotron 3.5 Bolt's adaptive draft depth on CPU: the cost model's choices, and drafted replies that equal serial ones
whatever depth each round takes."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bolt_lilicorr_fakes import FakeTarget, random_tensors, serial, tiny_config

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.nemotron_bolt.cuda.depth import Costs, DepthPolicy
from tensorfold.families.nemotron_bolt.cuda.lilicorr import DraftConfig, Drafter, expected_shapes, lilicorr_decode

SAMPLINGS = [None, Sampling(seed=77, temperature=1.0, top_k=20, top_p=0.95)]


def gb10(rows: int) -> float:
    """Verify ms by window rows as the GB10 runs the full model (2026-10-17): 34.5 ms a serial step with its host
    work, 141 ms a round of 7 drafts; the MoE layers' distinct experts grow with the rows."""

    return 21.0 + 13.5 * rows


def measured(*, block: float = 12.0, rest: float = 0.0) -> Costs:
    c = Costs(8)
    for rows in (1, 8):
        c.record(rows, gb10(rows), block if rows > 1 else None, rest)
    return c


def at_rate(a: float, costs: Costs) -> DepthPolicy:
    p = DepthPolicy(7, costs)
    p.hits, p.misses = a, 1.0 - a
    return p


def test_costs_fill_unmeasured_windows_by_a_line():
    c = Costs(8)
    assert c.window(3) is None
    c.record(1, 30.0, None, 2.0)
    assert c.window(1) == 30.0 and c.window(3) is None
    c.record(8, 100.0, 12.0, 4.0)
    assert c.window(4) == pytest.approx(30.0 + 70.0 * 3 / 7)
    c.record(8, 110.0, 14.0, 4.0)                      # an exponential average of each part
    assert c.window(8) == pytest.approx(102.0) and c.block == pytest.approx(12.4) and c.rest == pytest.approx(2.72)


def test_depth_follows_acceptance_through_the_cost_model():
    c = measured()
    assert at_rate(0.98, c).next(100) == 7             # tools: 6.7-7.1 tokens a round of 7
    assert at_rate(0.30, c).next(100) == 0             # serial rounds once drafting costs more than it saves
    assert 0 <= at_rate(0.68, c).next(100) <= 3        # prose: about 2.9 tokens a round of 7, near serial's rate
    mid = at_rate(0.9, c)                              # code: shallower than 7, deeper than 1
    assert 3 <= mid.next(100) < 7
    assert at_rate(0.98, c).next(3) == 3               # never past the room the reply leaves
    assert at_rate(0.98, c).next(0) == 0


def test_the_first_rounds_measure_both_ends():
    p = DepthPolicy(7, Costs(8))
    assert p.next(100) == 7                            # nothing measured: draft as deep as allowed
    p.done(7, 2, gb10(8), 12.0, 0.5)
    assert p.next(100) == 0                            # then one serial round
    p.done(0, 0, gb10(1), 0.0, 0.5)
    assert p.costs.window(1) == gb10(1) and p.next(100) in range(8)


def test_a_serial_stretch_probes_the_drafter():
    p = at_rate(0.2, measured())
    picks = [p.next(100) for _ in range(40)]
    assert picks.count(0) == 38 and picks[15] == picks[31] == 1      # every 16th round drafts its best depth


def test_acceptance_moves_with_the_rounds():
    p = DepthPolicy(7, measured())
    for _ in range(16):
        p.done(7, 0, gb10(8), 12.0, 0.0)
    assert p.rate < 0.1 and p.next(100) == 0
    for _ in range(12):
        p.done(3, 3, gb10(4), 12.0, 0.0)
    assert p.rate > 0.9 and p.next(100) == 7


class Scripted(DepthPolicy):
    """The policy on the GB10's costs instead of this CPU's clock, so its choices do not depend on the machine."""

    def done(self, drafts, accepted, verify_ms, block_ms, rest_ms):
        super().done(drafts, accepted, gb10(1 + drafts), 12.0, 0.0)


class Oracle:
    """Right for ``good`` drafts a round, then wrong (the serial reply known in advance)."""

    def __init__(self, reply, prompt_len, good, block=8):
        self.reply, self.base, self.good = reply, prompt_len, good
        self.context_end = prompt_len
        self.c = type("C", (), {"block": block})()
        self.calls: list[int] = []

    def propose(self, pending, depth, sampling):
        self.calls.append(depth)
        at = self.context_end - self.base + 1
        right = self.reply[at:at + depth]
        return (right[:self.good] + [(t + 1) % 300 for t in right[self.good:self.good + 1]])[:depth]

    def add_taps(self, taps):
        self.context_end += taps.shape[0]


def decode(prompt, count, sampling, drafter_for, policy):
    t = FakeTarget(prompt)
    first = t.sample(t.prompt_logits, [t.pos], sampling)[0]
    d = drafter_for(t)
    if isinstance(d, Drafter):
        d.add_taps(t.taps_of(0, list(prompt)))
    return lilicorr_decode(t, d, first, count, sampling, policy=policy), t, d


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "topk"])
@pytest.mark.parametrize("good", [0, 1, 7])
def test_adaptive_rounds_equal_serial(sampling, good):
    prompt = [4, 9, 2, 6]
    want = serial(prompt, 120, sampling)
    res, t, d = decode(prompt, 120, sampling, lambda t: Oracle(want, len(prompt), good), Scripted(7, Costs(8)))
    assert res.tokens == want
    assert d.context_end == t.pos == len(prompt) + len(want) - 1       # the owed taps reach the drafter at the end
    assert res.widths[:2] == [min(8, good + 2), 1]                     # measure both ends first (the oracle
    # drafts at most good + 1)
    serial_rounds = res.widths[2:].count(1)
    if good == 7:
        assert serial_rounds == 0 and res.widths[-6:-1] == [8] * 5      # deepest once acceptance has shown
    elif good == 0:
        assert serial_rounds >= 0.85 * (len(res.widths) - 2)          # probes aside, a useless drafter is not run
    assert len(d.calls) == len(res.widths) - res.widths.count(1)       # serial rounds run no drafter


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "topk"])
def test_the_real_drafter_with_adaptive_depth_equals_serial(sampling):
    prompt = [4, 9, 2, 6, 8, 1, 3]
    want = serial(prompt, 60, sampling)
    cfg = DraftConfig.from_json(tiny_config(window=6))

    def real(t):
        return Drafter((cfg, random_tensors(expected_shapes(cfg), 0)), embed=t.embed, lm_head=t.lm_head,
                       device="cpu", dtype=torch.float32)

    res, t, d = decode(prompt, 60, sampling, real, Scripted(7, Costs(8)))
    assert res.tokens == want and d.context_end == t.pos


def test_costs_persist_across_requests():
    costs = Costs(8)
    prompt = [4, 9, 2]
    want = serial(prompt, 30, None)
    decode(prompt, 30, None, lambda t: Oracle(want, 3, 7), Scripted(7, costs))
    res, _, _ = decode(prompt, 30, None, lambda t: Oracle(want, 3, 7), Scripted(7, costs))
    assert costs.window(1) is not None and 1 not in res.widths[:-1]   # no measuring round in the second request
