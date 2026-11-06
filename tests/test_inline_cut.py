"""A gate's cut inside a live stream: the fix is written as forced drafts and the reply is not prefilled again."""

from tensorfold.cuda.streams import Stream
from tensorfold.engine.call_gate import ThinkBudget, ThinkEnd, generate_gated

END = 99


def _engine(rounds, runs):
    """An engine that writes ``rounds`` and applies ``splice`` as a shared round does: the cut's token becomes the fix."""

    def inline(prompt, count, feed, splice):
        runs.append(list(prompt))
        for new in rounds:
            hit = splice(list(new))
            feed(list(new) if hit is None else [*new[:hit[0]], *hit[1]])
            if hit is not None:
                break
        return {"rounds": len(rounds)}

    return inline


def test_an_inline_cut_writes_the_fix_in_one_run_and_the_gate_sees_each_token_once():
    budget, sent, runs = ThinkBudget(5, [7, END, 8], END), [], []
    stats = generate_gated(None, [1, 2], 50, [budget], lambda new: sent.extend(new) or False,
                           _engine([[10, 11, 12], [13, 14, 15]], runs))
    assert sent == [10, 11, 12, 13, 7, END, 8]           # the fifth token is the close's first
    assert runs == [[1, 2]] and stats == {"rounds": 2}
    assert budget.count == 7 and not budget.open


def test_a_replay_gate_keeps_the_separate_runs():
    class Replay:
        replay = True

        def cut(self, tokens):
            return None

        def observe(self, token):
            pass

    runs = []

    def generate(prompt, count, feed):
        runs.append(list(prompt))
        feed([3])

    generate_gated(generate, [1], 5, [Replay()], lambda new: False, _engine([], []))
    assert runs == [[1]]


def _spliced(s, tokens, rows, path, end):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    return MultiDecoder._spliced(s, tokens, rows, path, end)


def test_a_cut_ends_the_path_at_its_token_and_the_fix_waits_as_drafts():
    s = Stream([1], 100, splice=lambda new: (1, [7, END, 8]) if new == [21, 22, 23] else None)
    s.out = [20]
    # the window [20, 21, 22]: both drafts kept, 23 sampled after them; the gate cuts the round's second new token
    assert _spliced(s, [20, 21, 22], [21, 22, 23], [0, 1, 2], 23) == ([0, 1], 7)
    assert s.forced == [END, 8]


def test_forced_drafts_are_kept_whatever_the_rows_sampled():
    seen = []
    s = Stream([1], 100, splice=lambda new: seen.append(new))
    s.out, s.forcing = [20, 7], 2
    assert _spliced(s, [7, END, 8], [50, 51, 52], [0], 50) == ([0, 1, 2], 52)
    assert seen == [[52]] and s.forcing == 0             # the gate reads only the token sampled after the fix


def test_a_long_fix_goes_on_in_the_next_window_and_a_count_ends_it():
    s = Stream([1], 100, splice=lambda new: None)
    s.out, s.forcing, s.forced = [20, 7], 2, [9]
    assert _spliced(s, [7, END, 8], [50, 51, 52], [0], 50) == ([0, 1, 2], 9) and s.forced == []
    s = Stream([1], 4, splice=lambda new: None)
    s.out, s.forcing = [20, 21, 7], 2                    # one token of room: the root's row and the fix's next token
    assert _spliced(s, [7, END, 8], [50, 51, 52], [0], 50) == ([0], END)


def test_a_stream_without_gates_is_untouched():
    s = Stream([1], 100)
    assert _spliced(s, [20, 21], [21, 30], [0, 1], 30) == ([0, 1], 30)


EOS = 0


def test_an_end_token_sampled_in_the_think_block_becomes_the_close_in_the_live_stream():
    closing, sent, runs = ThinkEnd([EOS], [7, END, 8], END), [], []
    generate_gated(None, [1, 2], 50, [closing], lambda new: sent.extend(new) or False,
                   _engine([[10, 11, 12], [13, EOS]], runs))
    assert sent == [10, 11, 12, 13, 7, END, 8] and runs == [[1, 2]]
    assert closing.fired and not closing.open


def test_the_sampled_end_token_a_round_ends_on_is_cut_too():
    closing = ThinkEnd([EOS], [7, END, 8], END)
    s = Stream([1], 100, splice=lambda new: _splice([closing], new))
    s.out = [20]
    # the window [20, 21]: the draft kept, then the end token sampled after it, which the round passes as its end
    assert _spliced(s, [20, 21], [21, EOS], [0, 1], EOS) == ([0, 1], 7)
    assert s.forced == [END, 8]


def test_after_the_block_closes_the_end_token_ends_the_reply():
    closing = ThinkEnd([EOS], [7, END, 8], END)
    assert closing.cut([10, END, 11, EOS]) is None
    for token in (10, END, 11):
        closing.observe(token)
    assert closing.cut([EOS]) is None and not closing.fired


def test_an_earlier_gates_cut_wins_and_the_end_close_did_not_fire():
    budget, closing, sent = ThinkBudget(2, [7, END, 8], END), ThinkEnd([EOS], [7, END, 8], END), []
    generate_gated(None, [1], 50, [budget, closing], lambda new: sent.extend(new) or False,
                   _engine([[10, 11, EOS]], []))
    assert sent == [10, 7, END, 8] and not closing.fired


def _splice(gates, new):
    hits = [hit for gate in gates if (hit := gate.cut(new)) is not None]
    at, fix = min(hits, key=lambda h: h[0]) if hits else (len(new), [])
    for token in [*new[:at], *fix]:
        for gate in gates:
            gate.observe(token)
    return (at, list(fix)) if hits else None


def test_prefill_open_turns_only_a_first_think_end_into_its_blank_line():
    from tensorfold.engine.call_gate import PrefillOpen

    gate = PrefillOpen(9, [5])
    assert gate.cut([9, 1]) == (0, [5])
    gate.observe(5)
    assert gate.fired
    assert gate.cut([9]) is None


def test_prefill_open_leaves_a_later_think_end_alone():
    from tensorfold.engine.call_gate import PrefillOpen

    gate = PrefillOpen(9, [5])
    assert gate.cut([1, 9]) is None
    gate.observe(1)
    gate.observe(9)
    assert not gate.fired and gate.cut([9]) is None
