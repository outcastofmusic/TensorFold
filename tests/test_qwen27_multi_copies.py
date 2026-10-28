"""The 27B's concurrent decoder widens copy windows past 16 rows: a lone stream's as the one-stream decode's
(16 -> 128 while they land whole), a shared round's to 32 within the row budget, an unsure copy with the tree beside."""

import importlib
from types import SimpleNamespace

import pytest

from tensorfold.cuda.streams import PrefixCache, Stream
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch


def _multi():
    return importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")


class Drafter:
    """A chain of ``depth`` tree nodes (tokens 1..depth) for every stream."""

    def __init__(self, depth=3):
        self.depth = depth

    def finish_tree(self, launched, length, max_nodes, sampling):
        d = self.depth
        return list(range(1, d + 1)), list(range(-1, d - 1)), [0.01 * (i + 1) for i in range(d)]


def copying_decoder(multi, scripts, *, copy_rows=128, streams=8):
    """A ``MultiDecoder`` whose model is ``scripts`` (prompt -> reply): every row samples its reply's token at its
    depth. Copies, window widths, acceptance and ``take`` are the decoder's own; ``widths`` logs each round's."""

    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.w, dec.draft, dec.max_rows, dec.allow_copy = None, None, 16, True
    dec.copy_rows = dec.copy_cap = copy_rows
    dec.budget = multi.stream_rows(streams, copy_rows)
    dec.context, dec.eos, dec.rank, dec.world, dec.device = 0, (0,), 0, 1, None
    dec.split, dec.drafts, dec.streams, dec.cache = False, False, {}, PrefixCache(8)
    dec.next_id, dec.broken, dec.costs, dec.filling, dec.points = 0, None, None, [], None
    dec.widths = []

    def _queue(s, hit):
        s.st, s.snap, s.stops = SimpleNamespace(pos=0), None, []
        dec.filling.append(s)

    def _step(s, stop):
        s.copies = multi.CopyIndex()
        dec._copy_start(s)
        s.context = list(s.prompt)
        dec.filling = [x for x in dec.filling if x is not s]
        dec.streams[s.sid] = s
        return scripts[tuple(s.prompt)][0]

    def _verify(plan, copied=None):
        wins = dec._windows(plan, copied, {})
        sampled = []
        for (sid, *_), (tokens, parents) in zip(plan, wins):
            s = dec.streams[sid]
            reply, n = scripts[tuple(s.prompt)], len(s.out)
            sampled.append([reply[n + d] if n + d < len(reply) else 99 for d in multi._paths(parents)[0]])
        dec.widths.append({sid: (mode, len(t)) for (sid, mode, *_), (t, _) in zip(plan, wins)})
        return wins, None, None, None, sampled

    dec._queue, dec._step, dec._verify = _queue, _step, _verify
    dec._steps = lambda batch: [_step(s, stop) for s, stop in batch]
    dec._commit = lambda plan, wins, record, taps, starts, paths: [
        dec.streams[item[0]].counted(len(w[0])) for item, w in zip(plan, wins)]
    return dec


def copy_widths(dec, sid):
    return [r[sid][1] for r in dec.widths if sid in r and r[sid][0] == _multi().COPY]


PROMPT = list(range(1, 401))           # distinct tokens: a reply that repeats the prompt copies from its one source


def test_a_lone_stream_doubles_its_copy_window_to_128_rows(allocations):  # noqa: F811
    multi = _multi()
    dec = copying_decoder(multi, {tuple(PROMPT): PROMPT})
    s = Stream(PROMPT, 300, draft=True)
    dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert s.out == PROMPT[:300]
    assert copy_widths(dec, s.sid)[:5] == [16, 32, 64, 128, 128]     # as draft_decode's next_copy_rows
    assert s.rounds == 7 + 5                                         # 7 one-row rounds until the gram repeats


def test_sixteen_copy_rows_keep_the_old_windows(allocations):  # noqa: F811
    multi = _multi()
    dec = copying_decoder(multi, {tuple(PROMPT): PROMPT}, copy_rows=16)
    s = Stream(PROMPT, 120, draft=True)
    dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert s.out == PROMPT[:120] and set(copy_widths(dec, s.sid)) == {16}


def test_shared_rounds_cap_copies_and_a_stream_left_alone_widens(allocations):  # noqa: F811
    multi = _multi()
    other = list(range(1001, 1401))
    dec = copying_decoder(multi, {tuple(PROMPT): PROMPT, tuple(other): other})
    short, long = Stream(other, 100, draft=True), Stream(PROMPT, 400, draft=True)
    for s in (short, long):
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert short.out == other[:100] and long.out == PROMPT[:400]
    shared = [r for r in dec.widths if len(r) == 2]
    assert shared and max(w for r in shared for _, w in r.values()) == 32
    alone = [r[long.sid][1] for r in dec.widths if len(r) == 1]
    assert 64 in alone and 128 in alone                         # from 32 it doubles again once alone


def test_copy_caps_by_streams_and_budget(allocations):  # noqa: F811
    multi = _multi()
    dec = copying_decoder(multi, {}, streams=8)
    assert [dec._copy_cap(n) for n in (1, 2, 7, 8)] == [128, 32, 32, 16]    # 8 streams: 7 trees leave 16 rows
    dec = copying_decoder(multi, {}, streams=2)
    assert dec.budget == 128 and dec._copy_cap(2) == 32
    dec = copying_decoder(multi, {}, copy_rows=16)
    assert {dec._copy_cap(n) for n in (1, 2, 8)} == {16}
    assert multi.calibration_rows(8, True, 128)[-1] == 128 and multi.calibration_rows(2, True, 128)[-1] == 128
    assert multi.calibration_rows(2) == [1, 2, 4, 8, 12, 16, 17, 24, 32]


class Unsure:
    """A copy index whose proposals are tokens 100.. and whose sources disagree (``unsure``) as told."""

    n = 8

    def __init__(self, unsure):
        self.unsure, self.reach = unsure, 40

    def propose(self, context, max_nodes):
        return list(range(100, 100 + max_nodes))


@pytest.mark.parametrize("gate", ["unsure", "none"])
def test_an_unsure_copy_hangs_the_tree_beside_it(allocations, monkeypatch, gate):  # noqa: F811
    multi = _multi()
    monkeypatch.setattr(multi, "COPY_TREE_GATE", gate)
    monkeypatch.setattr(importlib.import_module("tensorfold.families.qwen3_5.cuda.decode"), "COPY_TREE_GATE", gate)
    dec = copying_decoder(multi, {})
    dec.draft = Drafter(3)
    s = SimpleNamespace(sid=0, draft=True, copies=Unsure(True), context=[], sampling=None)
    dec._copy_start(s)
    dec.streams = {0: s}
    s.copy_rows, copied = 64, {}
    assert dec._mode(s, copied) == multi.BESIDE and copied[0] == list(range(100, 148))   # 64 rows less a tree's 16
    plan = [(0, multi.BESIDE, 7, 10)]
    [(tokens, parents)] = dec._windows(plan, copied, {0: 0})
    assert tokens == [7, *range(100, 148), 1, 2, 3] and parents[49:] == [0, 49, 50] and dec.chains == {0: 48}
    dec.copy_cap = 64
    dec._landed(s, tokens, list(range(49)), 48)                  # the copy landed whole: twice as wide
    assert s.copy_rows == 128 and s.copy_missed == (gate == "none")
    dec._landed(s, tokens, [0, 49, 50], 48)                       # the tree's chain held instead: half
    assert s.copy_rows == 32 and s.copy_missed
    s.copies = Unsure(False)
    assert dec._mode(s, copied) == (multi.BESIDE if gate == "none" else multi.COPY)
    dec.copy_cap = 16                                             # no room beside a 16-row copy: a plain copy
    s.copies = Unsure(True)
    assert dec._mode(s, copied) == multi.COPY and len(copied[0]) == 15
