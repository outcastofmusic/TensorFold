"""Flash Next's idle prompt passes read the n-gram rows of the next two passes while the current one runs: every
later pass's lookup takes rows read on the read-ahead thread, in pass order, and a wrong guess changes no bytes."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.cuda.ngram import NGram
from tensorfold.families.qwen4_exp.host_table import ReadAhead

pytestmark = pytest.mark.torch


class Table:
    rows = 1 << 20

    def __init__(self):
        self.calls = []                         # (thread name, ids) of each gather

    def gather(self, ids):
        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        self.calls.append((threading.current_thread().name, flat.copy()))
        return (flat * 3).astype(np.uint32)[:, None], flat.astype(np.uint16)[:, None], (flat + 1).astype(np.uint16)[:, None]


class Ahead(ReadAhead):
    def __init__(self, table):
        super().__init__(table, depth=8)
        self.asked = []

    def read_ahead(self, ids):
        self.asked.append(np.asarray(ids).reshape(-1).copy())
        super().read_ahead(ids)


def _decoder(prompts, histories=None, live=False):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    ngram = NGram(vocab=1000, ngram_size=3, heads_per_ngram=2, vocab_base=500, divisor=1, shards=1, seed=5, eos=1,
                  embed_dim=16)
    table = Ahead(Table())
    m = object.__new__(MultiDecoder)
    m.w = SimpleNamespace(cfg=SimpleNamespace(ngram_size=3),
                          layers=[SimpleNamespace(ple=None), SimpleNamespace(ple=SimpleNamespace(ngram=ngram, table=table))])
    rng = np.random.default_rng(len(prompts))
    histories = histories or [ngram.initial_history()] * len(prompts)
    m.filling = [SimpleNamespace(sid=i, prompt=rng.integers(0, 1000, n).tolist(), background=False,
                                 st=SimpleNamespace(ple_history=h.copy(), ple_last=None))
                 for i, (n, h) in enumerate(zip(prompts, histories))]
    m.fills = {s.sid: [SimpleNamespace(stops=[]), False, 0, None] for s in m.filling}
    m.passed, m.pass_plan, m.pass_index = {}, None, 0
    m.prefill_rows, m.share, m.round_s, m.row_s = 4_096, 0.0, None, None
    m.streams = {99: SimpleNamespace(done=not live)}
    m.slots = [None] * 4
    return m, ngram, table


def _fill(m, ngram, table):
    """The passes ``_pass`` runs, as far as the n-gram rows go: pieces, stage's lookups, read-ahead, commit."""

    passes = []
    while m.filling:
        if m.pass_plan is not None:
            by_sid = {s.sid: s for s in m.filling}
            pieces = [(by_sid[sid], a, n) for sid, a, n in m.pass_plan[m.pass_index]]
            m.pass_index += 1
        else:
            pieces = m._pieces()
        m._note_passed(pieces)
        names = []
        for s, a, n in pieces:                                   # forward.stage
            toks = np.asarray(s.prompt[a:a + n], dtype=np.int64)
            ids = ngram.ids(s.st.ple_history, toks)
            s.st.ple_last = (s.st.ple_history, toks)
            got = table.gather(ids)
            assert all(np.array_equal(g, w) for g, w in zip(got, Table().gather(ids)))
            readers = [name for name, flat in table.table.calls if np.array_equal(flat, ids.reshape(-1))]
            assert len(readers) == 1                              # read once, ahead or here
            names += readers
        m._read_ahead(pieces)
        passes.append(([(s.sid, a, n) for s, a, n in pieces], names))
        for s, a, n in pieces:                                   # commit, then _joined
            history, toks = s.st.ple_last
            s.st.ple_history, s.st.ple_last = np.concatenate([history, toks])[-2:], None
            m.fills[s.sid][2] = a + n
            if a + n == len(s.prompt):
                m.filling.remove(s)
    table._pool.shutdown(wait=True)
    return passes


def test_a_lone_prompts_later_passes_take_rows_read_ahead_in_pass_order():
    m, ngram, table = _decoder([10_000])
    passes = _fill(m, ngram, table)
    assert [p for p, _ in passes] == [[(0, 0, 4096)], [(0, 4096, 4096)], [(0, 8192, 1808)]]
    assert passes[0][1] == ["MainThread"]                                    # nothing was read ahead for it
    assert len(table.table.calls) == 3 and all(n.startswith("ngram-read-ahead") for _, (n,) in passes[1:])
    asked = [len(ids) // 4 for ids in table.asked]                           # rows a lookup asked: 4 heads a row
    assert asked == [4096, 1808, 1808]                                       # passes 2 and 3, then 3 again (kept)


def test_two_prompts_and_a_resumed_history_are_read_ahead_exactly():
    hist = np.array([7, 8])                                                  # a prompt resumed after a kept state
    m, ngram, table = _decoder([5_000, 3_000, 700], histories=[np.array([1, 1]), hist, np.array([1, 1])])
    passes = _fill(m, ngram, table)
    assert len(passes) > 2
    assert all(n.startswith("ngram-read-ahead") for _, names in passes[1:] for n in names)
    assert len(table.table.calls) == sum(len(p) for p, _ in passes)        # no guess went unused


def test_the_agreed_plan_of_two_ranks_is_read_ahead():
    m, ngram, table = _decoder([6_000])
    m.pass_plan = [[[0, 0, 2048]], [[0, 2048, 2048]], [[0, 4096, 1904]]]
    passes = _fill(m, ngram, table)
    assert [p for p, _ in passes] == [[tuple(x) for x in plan] for plan in m.pass_plan]
    assert all(n.startswith("ngram-read-ahead") for _, names in passes[1:] for n in names)


def test_nothing_is_read_ahead_while_replies_decode_or_when_switched_off(monkeypatch):
    m, ngram, table = _decoder([10_000], live=True)
    _fill(m, ngram, table)
    assert table.asked == [] and {n for n, _ in table.table.calls} == {"MainThread"}
    monkeypatch.setenv("TF_NGRAM_READ_AHEAD", "0")
    m, ngram, table = _decoder([10_000])
    _fill(m, ngram, table)
    assert table.asked == [] and len(table.table.calls) == 3


def test_a_wrong_guess_reads_rows_no_lookup_takes_and_the_lookup_reads_its_own():
    m, ngram, table = _decoder([10_000])
    first = m._pieces()
    m._note_passed(first)
    s = first[0][0]
    toks = np.asarray(s.prompt[:4096], dtype=np.int64)
    s.st.ple_last = (s.st.ple_history, toks)
    m._read_ahead(first)
    s.prompt[5000] = (s.prompt[5000] + 1) % 1000                             # the prompt the guess read changed
    ids = ngram.ids(np.asarray(s.prompt[4094:4096]), np.asarray(s.prompt[4096:8192], dtype=np.int64))
    got = table.gather(ids)
    assert all(np.array_equal(g, w) for g, w in zip(got, Table().gather(ids)))
    table._pool.shutdown(wait=True)
