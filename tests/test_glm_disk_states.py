"""GLM's kept prompt states on disk (``tensorfold.cuda.disk_states``): written in blocks, shared blocks stored once, every
piece checked on its way back, the budget kept least recently used first."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.torch


@pytest.fixture
def disk():
    return importlib.import_module("tensorfold.cuda.disk_states")


def rows(n, seed=0, dtype=None):
    """A state's views as the engine's: attention rows of n positions, pools of n // 4 + 1, one 3-D view."""

    import torch

    g = torch.Generator().manual_seed(seed)
    dtype = dtype or torch.bfloat16
    return [torch.randn(n, 8, generator=g).to(dtype), torch.randn(n, 2, 4, generator=g).to(dtype),
            torch.randn(n // 4 + 1, 3, generator=g).to(dtype)]


def extras(seed=0):
    import torch

    g = torch.Generator().manual_seed(seed)
    return {"rec": torch.randn(2, 4, 4, generator=g), "conv": torch.randn(3, 5, generator=g).to(torch.bfloat16),
            "pending": None}


def write(store, ids, views, *, stop=lambda: False, ring=None):
    return store.write(ids, lambda: views, extras(len(ids)), ring, mtp_len=len(ids) - 1, drafter_end=len(ids),
                       born=-1, stop=stop)


def test_a_state_reads_back_as_it_was_written(disk, tmp_path):
    import torch

    store = disk.DiskStates(tmp_path, "k", 1 << 30)
    ids = list(range(700))
    views = rows(700)
    assert write(store, ids, views)
    again = disk.DiskStates(tmp_path, "k", 1 << 30)                 # a restart finds it
    entry = again.find([*ids, 5, 6], 0, lambda e: True)
    assert entry is not None and entry.ids == ids and entry.mtp_len == 699 and entry.drafter_end == 700
    dst = [torch.zeros_like(v) for v in views]
    got = again.load(entry, dst, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(dst, views))
    assert torch.equal(got["rec"], extras(700)["rec"]) and torch.equal(got["conv"], extras(700)["conv"])
    assert "pending" not in got and got["ring"] is None


def test_only_a_strict_prefix_longer_than_memorys_and_fitting_is_found(disk, tmp_path):
    store = disk.DiskStates(tmp_path, "k", 1 << 30)
    write(store, list(range(300)), rows(300))
    write(store, list(range(600)), rows(600))
    prompt = list(range(900))
    assert len(store.find(prompt, 0, lambda e: True).ids) == 600
    assert len(store.find(prompt, 0, lambda e: len(e.ids) < 500).ids) == 300
    assert store.find(prompt, 600, lambda e: True) is None            # memory already holds as much
    assert store.find(list(range(600)), 0, lambda e: True).ids == list(range(300))   # a strict prefix only
    assert store.find([9, *range(1, 900)], 0, lambda e: True) is None


def test_blocks_two_states_share_are_stored_once(disk, tmp_path):
    import torch

    store = disk.DiskStates(tmp_path, "k", 1 << 30)
    long = rows(2 * disk.BLOCK + 40)
    write(store, list(range(2 * disk.BLOCK + 40)), long)
    pieces = len(list(store.pieces.iterdir()))
    short = [v[:disk.BLOCK + 10].clone() for v in long[:2]] + [long[2][:(disk.BLOCK + 10) // 4 + 1].clone()]
    write(store, list(range(disk.BLOCK + 10)), short)
    # the short state's first blocks are the long state's: only its partial blocks and small tensors are new
    assert len(list(store.pieces.iterdir())) - pieces <= 3 + 2
    entry = store.find(list(range(disk.BLOCK + 11)), 0, lambda e: True)
    dst = [torch.zeros_like(v) for v in short]
    store.load(entry, dst, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(dst, short))


def test_a_write_stopped_between_blocks_leaves_no_state_and_keeps_its_pieces(disk, tmp_path):
    store = disk.DiskStates(tmp_path, "k", 1 << 30)
    asked = iter([False, False, True])
    ids = list(range(3 * disk.BLOCK))
    assert not write(store, ids, rows(3 * disk.BLOCK), stop=lambda: next(asked, True))
    assert store.entries == [] and not list(store.states.glob("*.json"))
    kept = len(list(store.pieces.iterdir()))
    assert kept > 0
    assert write(store, ids, rows(3 * disk.BLOCK))                   # the next try writes only the rest
    assert len(store.entries) == 1


def test_rows_gone_mid_write_leave_no_state(disk, tmp_path):
    store = disk.DiskStates(tmp_path, "k", 1 << 30)
    views, calls = rows(3 * disk.BLOCK), []

    def where():
        calls.append(1)
        return views if len(calls) < 3 else None

    assert not store.write(list(range(3 * disk.BLOCK)), where, extras(), None, mtp_len=-1, drafter_end=-1, born=-1,
                           stop=lambda: False)
    assert store.entries == []


def test_a_piece_that_fails_its_hash_refuses_the_state_and_forgets_it(disk, tmp_path):
    import torch

    store = disk.DiskStates(tmp_path, "k", 1 << 30)
    ids = list(range(400))
    write(store, ids, rows(400))
    entry = store.entries[0]
    victim = store.pieces / entry.meta["views"][0]["pieces"][0]
    victim.write_bytes(b"\0" + victim.read_bytes()[1:])
    with pytest.raises(disk.StateError, match="hash"):
        store.load(entry, [torch.zeros_like(v) for v in rows(400)], "cpu")
    assert store.entries == [] and store.find([*ids, 1], 0, lambda e: True) is None


def test_views_of_another_shape_are_refused(disk, tmp_path):
    import torch

    store = disk.DiskStates(tmp_path, "k", 1 << 30)
    write(store, list(range(300)), rows(300))
    wrong = [torch.zeros(300, 9, dtype=torch.bfloat16), *rows(300)[1:]]
    with pytest.raises(disk.StateError, match="shapes"):
        store.load(store.entries[0], wrong, "cpu")


def test_states_past_the_budget_go_least_recently_used_first(disk, tmp_path):
    store = disk.DiskStates(tmp_path, "k", 1 << 30)
    write(store, [1, *range(1000, 1500)], rows(501, seed=1))
    size = sum(p.stat().st_size for p in store.pieces.iterdir())
    store.budget = int(size * 1.5)                                    # room for one such state, not two
    write(store, [2, *range(1000, 1500)], rows(501, seed=2))
    assert [e.ids[0] for e in store.entries] == [2]
    names = {n for e in store.entries for n in store._names(e.meta)}
    assert {p.name for p in store.pieces.iterdir()} >= names


def test_fingerprints_keep_models_apart(disk, tmp_path):
    a = disk.fingerprint({"model": "a", "views": [[8], "torch.bfloat16"]})
    b = disk.fingerprint({"model": "b", "views": [[8], "torch.bfloat16"]})
    assert a != b and len(a) == 16
    write(disk.DiskStates(tmp_path, a, 1 << 30), list(range(300)), rows(300))
    assert disk.DiskStates(tmp_path, b, 1 << 30).entries == []
