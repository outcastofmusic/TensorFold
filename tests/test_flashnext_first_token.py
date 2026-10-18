"""A prompt that ends under --parallel sends its first token before its first drafts, and its prefill time covers
the last pass's GPU work: the drafts, the copy index and the reply stay the same either way."""

import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def multi_fill():
    from tensorfold.families.qwen4_exp.cuda import multi_fill

    return multi_fill


@pytest.fixture
def engine():
    from tensorfold.families.qwen4_exp.cuda import engine

    return engine


class Slot:
    """The engine a joined stream drafts on: its sample is the first token; ``events`` records the order."""

    def __init__(self, first, events):
        self.first, self.events, self.last_streams = first, events, None

    def sample(self, logits, positions, sampling):
        self.events.append("sample")
        return [self.first]


def decoder(prompt, count, first, events, *, copy=True, mtp=True, emit=None, error=None, early=True):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    d = object.__new__(MultiDecoder)
    d.w, d.depth, d.confidence, d.eos, d.copy = SimpleNamespace(comm=None, meta={}), 6, 0.7, (2,), copy
    d.first_early = early
    d.follower = d.link = None
    d.arrived, d.streams = (lambda: False), {}
    d._remember = lambda *args: events.append("remember")
    s = Stream(list(prompt), count, emit=emit if emit is not None else (lambda new: events.append(("emit", new))))
    s.sid, s.st, s.error = 0, SimpleNamespace(pos=len(prompt), image_rows=(), image_features=None), error
    e = Slot(first, events)
    d.filling, d.fills = [s], {0: [e, mtp, len(prompt) - 3, None]}
    return d, s


def join(d, s, spent=0.25):
    pieces = [(s, len(s.prompt) - 3, 3)]
    return d._joined(pieces, torch.zeros((1, 4)), [torch.zeros((1, 4))], spent)


@pytest.fixture
def drafts(monkeypatch, multi_fill):
    """``draft`` and ``absorb`` as the MTP head would run them: the same drafts whenever they run."""

    calls = []

    def fake(e, last, next_tokens, position, count, sampling, confidence):
        e.events.append("draft")
        calls.append((list(next_tokens), position, count))
        return [70, 71, 72][:count]

    monkeypatch.setattr(multi_fill, "draft", fake)
    monkeypatch.setattr(multi_fill, "absorb", lambda e, last, next_tokens: e.events.append("absorb"))
    return calls


def test_the_first_token_goes_out_before_the_drafts(drafts):
    events = []
    d, s = decoder(list(range(100, 140)), 8, 7, events)
    assert join(d, s) == []
    assert events == ["sample", ("emit", [7]), "draft"]
    assert s.out == [7] and s.drafts == [70, 71, 72] and drafts == [([7], 41, 6)]
    assert d.streams == {0: s} and d.filling == [] and 0 not in d.fills


def test_the_old_order_drafts_first_and_both_orders_leave_the_same_stream(drafts, monkeypatch, engine):
    got = {}
    for value in ("1", "0"):
        monkeypatch.setenv(engine.FIRST_EARLY, value)
        events = []
        prompt = [5, 6, 7, 8, 9, 10, 11, 12] * 3           # a repeated tail: the copy index drafts nothing yet
        d, s = decoder(prompt, 8, 13, events, early=engine.first_early())
        join(d, s)
        got[value] = (events, s.out, s.drafts, s.drafter, s.copies.ctx, s.copies.chain(6), s.context, s.done)
    assert got["1"][0] == ["sample", ("emit", [13]), "draft"]
    assert got["0"][0] == ["sample", "draft", ("emit", [13])]
    assert got["1"][1:] == got["0"][1:]


def test_the_decoder_keeps_the_flag_it_was_started_with(drafts, monkeypatch, engine):
    events = []
    monkeypatch.setenv(engine.FIRST_EARLY, "0")
    d, s = decoder(list(range(100, 140)), 8, 7, events, early=True)
    monkeypatch.setenv(engine.FIRST_EARLY, "ture")          # read at startup only: a later value is not seen
    join(d, s)
    assert events == ["sample", ("emit", [7]), "draft"]


def test_a_flag_typo_stops_the_engine_before_it_reads_the_model(monkeypatch, tmp_path, engine):
    monkeypatch.setenv(engine.FIRST_EARLY, "ture")
    with pytest.raises(ValueError, match=engine.FIRST_EARLY):
        engine.FlashNextEngine(tmp_path / "missing")          # no checkpoint there: the flag fails first


def test_a_copy_chain_is_drafted_after_the_first_token_too(drafts):
    events = []
    prompt = list(range(20, 40)) + list(range(20, 27))      # the first token 27 repeats an 8-token run
    d, s = decoder(prompt, 8, 27, events)
    join(d, s)
    assert events == ["sample", ("emit", [27]), "absorb"]
    assert s.drafter == "copy" and s.drafts == list(range(28, 34)) and drafts == []


def test_a_stream_ended_by_its_first_token_still_drafts_as_every_rank_does(drafts):
    events = []
    d, s = decoder(list(range(50, 60)), 8, 2, events)       # 2 is an end token
    assert join(d, s) == [s]
    assert s.done and s.out == [2]
    assert events == ["sample", ("emit", [2]), "draft"]


def test_a_failed_gate_ends_the_stream_without_emitting(drafts):
    events = []
    d, s = decoder(list(range(50, 60)), 8, 9, events, error=RuntimeError("gate"))
    assert join(d, s) == [s]
    assert s.done and s.out == [] and ("emit", [9]) not in events


def test_one_token_requests_do_not_draft(drafts):
    events = []
    d, s = decoder(list(range(50, 60)), 1, 9, events)
    assert join(d, s) == [s]
    assert events == ["sample", ("emit", [9])] and s.drafts == []


def test_prefill_time_covers_the_last_pass_s_gpu_work(drafts, monkeypatch):
    waits = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: (waits.append(1), time.sleep(0.05)))
    events = []
    d, s = decoder(list(range(100, 140)), 8, 7, events)
    join(d, s, spent=0.25)
    assert waits == [1]
    assert 0.3 <= s.prefill_s < 0.5


def test_a_pass_that_ends_no_prompt_does_not_wait(drafts, monkeypatch):
    waits = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: waits.append(1))
    events = []
    d, s = decoder(list(range(100, 140)), 8, 7, events)
    pieces = [(s, 0, 20)]                                  # the prompt's first 20 of 40 rows
    assert d._joined(pieces, None, [torch.zeros((1, 4))], 0.25) == []
    assert waits == [] and s.prefill_s == 0.25 and events == [] and d.filling == [s]
