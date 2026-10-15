"""StairCut and the plain gate: which MTP drafts a Flash Next round verifies, and when a context class drafts at all."""

import pytest

from tensorfold.families.qwen4_exp.cuda.staircut import (Calibration, PlainGate, Prices, Speculation, StairCut,
                                                         context_class)

# a verify that costs 20 ms for one row and 3 ms for each row after it
PRIOR = {0: {r: 20.0 + 3.0 * (r - 1) for r in range(1, 8)}}


def test_context_classes_follow_the_graph_buckets():
    assert [context_class(n) for n in (1, 8192, 8193, 16384, 16385, 200_000)] == [0, 0, 16384, 16384, 32768, 262144]


def test_prices_follow_rounds_by_ema_and_never_fall_with_rows():
    p = Prices(PRIOR)
    assert p.table(3, 0) == [0.0, 20.0, 23.0, 26.0]
    p.observe(2, 33.0, 0)                                   # 10 ms over the prior: EMA 0.1 of the price and the shift
    assert p.learned[0][2] == pytest.approx(24.0) and p.shift[0] == pytest.approx(1.0)
    assert p.table(3, 0)[3] == pytest.approx(27.0)           # an unseen count follows the prior plus the shift
    p.observe(1, 30.0, 0)                                   # one row measured dearer than two: the table stays monotone
    t = p.table(3, 0)
    assert t[1] <= t[2] <= t[3]
    p.observe(2, 1000.0, 0)                                  # a stall over 4x the price teaches nothing
    assert p.learned[0][2] == pytest.approx(24.0)


def test_a_class_without_its_own_prior_takes_the_nearest_lower_one_and_learns_its_shift():
    p = Prices(PRIOR)
    assert p.table(2, 65536) == [0.0, 20.0, 23.0]
    for _ in range(50):
        p.observe(4, 46.0, 65536)                          # the long context costs 17 ms more a round
    assert p.table(7, 65536)[7] == pytest.approx(38.0 + 17.0, abs=0.2)
    assert p.table(7, 0)[7] == 38.0                          # class 0 untouched


def test_without_a_prior_an_unmeasured_count_costs_the_nearest_measured_one():
    p = Prices()
    assert not p.known(0) and p.table(2, 0) == [0.0, 0.0, 0.0]
    p.observe(5, 40.0, 0)
    assert p.table(6, 0) == [0.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0]


def test_calibration_starts_at_the_probability_and_learns_the_landing_rate():
    c = Calibration()
    assert c.q(0.85) == pytest.approx(0.85)
    for _ in range(36):
        c.learn(0.85, False)                                # drafts of 0.8-0.9 never land
    assert c.q(0.85) == pytest.approx(4 * 0.85 / 40)
    for _ in range(1000):
        c.learn(0.85, True)
    n = c.bins[8][1]
    assert n <= 400 and c.q(0.85) > 0.95                     # old evidence halves past 400


def test_the_cut_keeps_the_prefix_with_most_tokens_a_millisecond():
    cut = StairCut(Prices(PRIOR))
    # sure drafts: every row pays (1 + 6) / 38 beats any shorter prefix
    assert cut.cut([0.99] * 6, 0) == 6
    # a confident head with one weak draft in the middle: the weak draft and everything under it go
    assert cut.cut([0.95, 0.95, 0.1, 0.95, 0.95, 0.95], 0) == 2
    # a uniformly weak chain is cut early: (1 + 0.3) / 23 > (1 + 0.3 + 0.09) / 26
    assert cut.cut([0.3, 0.3, 0.3], 0) == 1
    # an MTP step's cost moves the cut: dear steps shorten a good chain to its first draft
    assert cut.cut([0.9] * 4, 0) == 4
    cut.step[0] = 30.0
    assert cut.cut([0.9] * 4, 0) == 1
    # an anchor-only round still absorbs (one step), so a step's price alone never makes the plain round win
    assert cut.cut([0.3, 0.3, 0.3], 0) == 1


def test_the_cut_is_score_maximal_by_brute_force():
    import itertools

    cut = StairCut(Prices(PRIOR))
    cut.step[0] = 1.5
    for probs in itertools.product((0.05, 0.4, 0.75, 0.97), repeat=4):
        price = cut.prices.table(5, 0)
        reach, run = [], 1.0
        for p in probs:
            run *= p
            reach.append(run)
        scores = [(1 + sum(reach[:n])) / (price[n + 1] + 1.5 * max(n, 1)) for n in range(5)]
        assert cut.cut(list(probs), 0) == max(range(5), key=lambda n: (scores[n], -n))


def test_the_chain_stops_drafting_once_deeper_drafts_are_not_expected_to_pay():
    cut = StairCut(Prices(PRIOR))
    assert cut.more([0.99], 0, 6)                              # sure drafts: go on
    assert not cut.more([0.99] * 6, 0, 6)                      # the deepest level
    assert not cut.more([0.05], 0, 6)                          # drafts under a 5% one are not expected to pay a row
    assert cut.more([0.9, 0.9], 0, 6) and not cut.more([0.9, 0.3], 0, 6)
    # with every row priced, the chain's expected gain is what the cut would take: more() never stops a chain that
    # the same probabilities continued would be cut deeper
    for probs in ([0.95, 0.95], [0.9, 0.9, 0.9], [0.99, 0.99, 0.99, 0.99]):
        assert cut.more(probs, 0, 6) == (cut.cut(probs + [probs[-1]] * (6 - len(probs)), 0) > len(probs))


def test_learning_counts_drafts_up_to_the_first_miss():
    cut = StairCut(Prices(PRIOR))
    cut.learn([0.95, 0.85, 0.75, 0.65], accepted=1)
    assert cut.cal.bins[9] == [1.0, 1.0] and cut.cal.bins[8] == [0.0, 1.0]
    assert cut.cal.bins[7] == [0.0, 0.0] and cut.cal.bins[6] == [0.0, 0.0]     # never reached


def test_plain_gate_takes_plain_rounds_while_drafting_loses_and_probes_with_a_doubling_interval():
    gate = PlainGate()
    gate.plain[0] = 20.0                                       # a measured plain step
    assert not gate.skip(0)                                    # nothing drafted yet: draft
    for _ in range(PlainGate.FIRST - 1):
        gate.drafted(60.0, 2, 0)                               # 30 ms a token against 20 plain
        assert not gate.skip(0)                                # the class's first rounds are only counted
    gate.drafted(60.0, 2, 0)
    pattern = [gate.skip(0) for _ in range(16 + 32 + 1)]
    probes = [i for i, s in enumerate(pattern) if not s]
    assert probes == [15, 47]                                  # one drafted probe after 16, then after 32
    for _ in range(100):
        gate.drafted(30.0, 3, 0)                               # drafting wins again: 10 ms a token
    assert not gate.skip(0) and gate.probe[0] == PlainGate.PROBE


def test_plain_gate_spec_cost_sums_the_first_rounds_then_is_a_ratio_of_emas():
    gate = PlainGate()
    gate.drafted(45.0, 1, 0)                                   # a slow first round weighs as one of sixteen
    for _ in range(PlainGate.FIRST - 1):
        gate.drafted(30.0, 3, 0)
    assert gate.spec_ms(0) == pytest.approx((45.0 + 15 * 30.0) / (1 + 15 * 3))
    gate.drafted(40.0, 4, 0)
    ms, tokens, _ = gate.spec[0]
    assert gate.spec_ms(0) == pytest.approx(ms / tokens)
    assert ms == pytest.approx(0.95 * 495.0 / 16 + 0.05 * 40) and tokens == pytest.approx(0.95 * 46 / 16 + 0.05 * 4)


def test_speculation_is_off_unless_a_flag_is_on(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_STAIRCUT", raising=False)
    monkeypatch.delenv("TENSORFOLD_PLAIN_GATE", raising=False)
    assert Speculation.from_env(confidence=0.7, depth=6) is None
    monkeypatch.setenv("TENSORFOLD_STAIRCUT", "1")
    spec = Speculation.from_env(confidence=0.7, depth=6)
    assert spec is not None and spec.cut is not None and spec.gate is None
    assert Speculation.from_env(confidence=0.7, depth=0) is None
    assert Speculation.from_env(confidence=0.7, depth=6, world=2, log=lambda *_: None) is None
    monkeypatch.setenv("TENSORFOLD_STAIRCUT", "0")
    monkeypatch.setenv("TENSORFOLD_PLAIN_GATE", "on")
    spec = Speculation.from_env(confidence=0.7, depth=6)
    assert spec.cut is None and spec.gate is not None


def test_without_staircut_the_chain_keeps_the_fixed_confidence_rule():
    spec = Speculation(cut=False, gate=True, confidence=0.7, depth=6)
    assert spec.more(0, [0.9]) and not spec.more(0, [0.5])
    assert spec.keep(0, [0.5]) == 1                             # the first draft is always verified
    assert spec.keep(0, [0.9, 0.8, 0.6]) == 2                   # stop before the first later draft under 0.7
    assert not spec.more(0, [0.9] * 6)


def test_plain_gate_measures_the_plain_step_itself():
    gate = PlainGate(lambda cls: 99.0)                         # a prior far off: the fixed rule never verifies 1 row
    for _ in range(PlainGate.FIRST):
        assert not gate.skip(0)
        gate.drafted(40.0, 2, 0)
    assert gate.skip(0)                                        # one plain round to measure the plain step
    gate.plain_round(25.0, 0)
    assert gate.plain_ms(0) == 25.0                            # the first measurement replaces the prior
    calls = [gate.skip(0) for _ in range(PlainGate.PROBE_MAX + 1)]
    assert calls[:-1] == [False] * PlainGate.PROBE_MAX and calls[-1]       # and again every PROBE_MAX rounds


def test_plain_gate_measures_the_plain_step_only_on_a_round_it_can_time():
    gate = PlainGate()                                         # no prior: drafting stands until a plain step is timed
    for _ in range(PlainGate.FIRST):
        gate.drafted(40.0, 2, 0)
    # streams share every round: a forced plain round would go untimed, and the gate would force the next one too
    assert [gate.skip(0, measure=False) for _ in range(1000)] == [False] * 1000
    assert gate.skip(0)                                        # one stream alone: now the plain step is measured
    gate.plain_round(25.0, 0)
    assert not gate.skip(0, measure=False)                     # 20 ms a token drafted against 25 plain: draft


def test_speculation_passes_the_timed_round_to_its_gate():
    spec = Speculation(cut=False, gate=True, confidence=0.7, depth=6)
    for _ in range(PlainGate.FIRST):
        spec.gate.drafted(40.0, 2, 0)
    assert not spec.plain(100, measure=False) and spec.plain(100)
