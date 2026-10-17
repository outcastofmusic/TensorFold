"""Pruning kept states on disk (``tensorfold.cuda.disk_states``): a TTL on states unused, old fingerprints' folders
gone past it, and one budget for the whole folder, other fingerprints' folders least recently used first."""

from __future__ import annotations

import importlib
import json
import os
import time

import pytest

from tests.test_glm_disk_states import extras, rows, write

pytestmark = pytest.mark.torch
HOUR = 3600


@pytest.fixture
def disk():
    return importlib.import_module("tensorfold.cuda.disk_states")


def age(store, entry, hours):
    """Make ``entry`` look last used ``hours`` ago, as its JSON records and its file's time say."""
    entry.meta["used"] = time.time() - hours * HOUR
    entry.path.write_text(json.dumps(entry.meta))
    os.utime(entry.path, (entry.meta["used"], entry.meta["used"]))


def age_folder(path, hours):
    t = time.time() - hours * HOUR
    for f in path.rglob("*"):
        os.utime(f, (t, t))


def folder_with_state(disk, root, key, n=600, seed=0):
    store = disk.DiskStates(root, key, 1 << 30)
    assert write(store, list(range(n)), rows(n, seed))
    return store


def test_a_state_unused_past_the_ttl_goes_at_the_next_start(disk, tmp_path):
    store = folder_with_state(disk, tmp_path, "a" * 16)
    assert write(store, list(range(1000, 1700)), rows(700, 1))
    old = next(e for e in store.entries if e.ids[0] == 0)
    age(store, old, 80)
    again = disk.DiskStates(tmp_path, "a" * 16, 1 << 30, ttl=72 * HOUR)
    assert [e.ids[0] for e in again.entries] == [1000]
    assert not old.path.exists()


def test_another_fingerprints_folder_past_the_ttl_goes_and_a_recent_one_stays(disk, tmp_path):
    folder_with_state(disk, tmp_path, "b" * 16)
    folder_with_state(disk, tmp_path, "c" * 16)
    age_folder(tmp_path / ("b" * 16), 100)
    disk.DiskStates(tmp_path, "d" * 16, 1 << 30, ttl=72 * HOUR)
    assert not (tmp_path / ("b" * 16)).exists()
    assert (tmp_path / ("c" * 16)).exists()


def test_over_the_budget_other_folders_go_least_recently_used_first(disk, tmp_path):
    for key, hours in (("1" * 16, 5), ("2" * 16, 3), ("3" * 16, 2)):
        folder_with_state(disk, tmp_path, key, seed=hours)
        age_folder(tmp_path / key, hours)
    size = sum(f.stat().st_size for f in (tmp_path / ("1" * 16)).rglob("*") if f.is_file())
    store = disk.DiskStates(tmp_path, "f" * 16, int(size * 2.5), ttl=72 * HOUR)    # room for two folders' worth
    assert write(store, list(range(600)), rows(600, 9))
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["3" * 16, "f" * 16]                                         # the two oldest went, oldest first


def test_a_folder_used_within_the_hour_stays_whatever_the_budget(disk, tmp_path):
    folder_with_state(disk, tmp_path, "e" * 16)                                  # used just now
    store = disk.DiskStates(tmp_path, "f" * 16, 1, ttl=72 * HOUR)               # a budget of one byte
    assert write(store, list(range(600)), rows(600, 3))
    assert (tmp_path / ("e" * 16)).exists()


def test_folders_that_are_not_fingerprints_are_never_touched(disk, tmp_path):
    other = tmp_path / "qwen"                                                    # another family's root, inside this one
    folder_with_state(disk, other, "a" * 16)
    age_folder(other, 500)
    (tmp_path / "notes").mkdir()
    disk.DiskStates(tmp_path, "f" * 16, 1, ttl=1)
    assert (other / ("a" * 16)).exists() and (tmp_path / "notes").exists()


def test_from_env_reads_the_ttl(disk, tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_QWEN_STATES", str(tmp_path))
    monkeypatch.setenv("TENSORFOLD_QWEN_STATES_TTL_H", "5")
    store = disk.from_env("QWEN", {"model": "x"})
    assert store.ttl == 5 * HOUR


def test_from_env_refuses_a_ttl_that_keeps_nothing(disk, tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_QWEN_STATES", str(tmp_path))
    for hours in ("0", "-2", "nan"):
        monkeypatch.setenv("TENSORFOLD_QWEN_STATES_TTL_H", hours)
        with pytest.raises(ValueError, match="above 0"):
            disk.from_env("QWEN", {"model": "x"})


# -- a write a request waits for -------------------------------------------------------------------------------
def test_a_request_waiting_before_the_first_block_writes_nothing(disk, tmp_path):
    store = disk.DiskStates(tmp_path, "a" * 16, 1 << 30)
    assert not write(store, list(range(600)), rows(600), stop=lambda: True)
    assert not list(store.pieces.iterdir()) and store.entries == []


def test_small_tensors_go_after_the_rows_in_pieces_a_stop_can_end_between(disk, tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(disk, "PIECE_BYTES", 40)                     # extras(): rec is 128 bytes, conv 30
    store = disk.DiskStates(tmp_path, "a" * 16, 1 << 30)
    ids, views = list(range(600)), rows(600)
    asked = []

    def stop():
        asked.append(len(list(store.pieces.iterdir())))
        return len(asked) > 4                                         # 3 row blocks, then one piece of rec

    assert not write(store, ids, views, stop=stop)
    assert asked[:5] == [0, 3, 5, 7, 8] and store.entries == []      # the pool view (151 rows) has one block
    assert len(list(store.pieces.iterdir())) == 8                     # nothing written once the stop held
    assert write(store, ids, views)                                   # the next try writes the rest
    entry = store.entries[0]
    assert len(entry.meta["extras"]["rec"]["pieces"]) == 4 and len(entry.meta["extras"]["conv"]["pieces"]) == 1
    got = store.load(entry, [torch.zeros_like(v) for v in views], "cpu")
    assert torch.equal(got["rec"], extras(600)["rec"]) and torch.equal(got["conv"], extras(600)["conv"])


def test_pruning_waits_while_a_request_does(disk, tmp_path, monkeypatch):
    store = disk.DiskStates(tmp_path, "a" * 16, 1 << 30)
    trims = []
    monkeypatch.setattr(store, "_trim", lambda: trims.append(1))
    assert write(store, list(range(600)), rows(600), stop=lambda: any(store.states.glob("*.json")))
    assert trims == [] and len(store.entries) == 1                    # the state is whole; pruning waits
    assert write(store, list(range(700)), rows(700))
    assert trims == [1]


# -- pruning without walking the folders -----------------------------------------------------------------------
def test_other_folders_are_scanned_at_start_and_then_every_scan_s(disk, tmp_path, monkeypatch):
    folder_with_state(disk, tmp_path, "b" * 16)
    scans = []
    real = disk.DiskStates._others
    monkeypatch.setattr(disk.DiskStates, "_others", lambda self: scans.append(1) or real(self))
    store = disk.DiskStates(tmp_path, "a" * 16, 1 << 30)
    assert write(store, list(range(600)), rows(600)) and write(store, list(range(700)), rows(700))
    assert len(scans) == 1 and [p.name for _, _, p in store.seen] == ["b" * 16]
    store.seen_at -= disk.SCAN_S
    assert write(store, list(range(800)), rows(800))
    assert len(scans) == 2


def test_a_folder_the_scan_found_old_is_looked_at_again_before_it_goes(disk, tmp_path):
    folder_with_state(disk, tmp_path, "b" * 16)
    age_folder(tmp_path / ("b" * 16), 5)
    store = disk.DiskStates(tmp_path, "a" * 16, 1 << 30)
    assert [p.name for _, _, p in store.seen] == ["b" * 16]
    age_folder(tmp_path / ("b" * 16), 0)                              # another server used it since the scan
    store.budget = 1
    assert write(store, list(range(600)), rows(600))
    assert (tmp_path / ("b" * 16)).exists()


def test_piece_sizes_are_known_without_a_stat(disk, tmp_path):
    store = disk.DiskStates(tmp_path, "a" * 16, 1 << 30)
    write(store, [1, *range(1000, 1500)], rows(501, seed=1))
    write(store, [2, *range(1000, 1500)], rows(501, seed=2))
    on_disk = {p.name: p.stat().st_size for p in store.pieces.iterdir()}
    assert store.sizes == on_disk
    assert disk.DiskStates(tmp_path, "a" * 16, 1 << 30).sizes == on_disk
    for p in store.pieces.iterdir():                                  # old enough to go once no state names them
        os.utime(p, (time.time() - 2 * HOUR,) * 2)
    store.budget = sum(on_disk.values()) * 2 // 3                     # room for one state
    write(store, [3, *range(1000, 1500)], rows(501, seed=3))
    assert store.sizes == {p.name: p.stat().st_size for p in store.pieces.iterdir()} != on_disk


def test_a_killed_writes_leftovers_go_once_an_hour_old(disk, tmp_path):
    store = disk.DiskStates(tmp_path, "a" * 16, 1 << 30)
    old = [store.pieces / ("0" * 64 + ".tmp41"), store.states / ("1" * 24 + ".tmp")]
    new = [store.pieces / ("2" * 64 + ".tmp42"), store.states / ("3" * 24 + ".tmp")]
    for path in old + new:
        path.write_bytes(b"half")
    for path in old:
        os.utime(path, (time.time() - 2 * HOUR,) * 2)
    again = disk.DiskStates(tmp_path, "a" * 16, 1 << 30)
    assert not any(p.exists() for p in old) and all(p.exists() for p in new)
    assert again.sizes == {}


def test_a_write_after_another_server_pruned_the_folder(disk, tmp_path):
    import shutil

    import torch

    store = disk.DiskStates(tmp_path, "a" * 16, 1 << 30)
    shutil.rmtree(store.dir)
    views = rows(600)
    assert write(store, list(range(600)), views)
    dst = [torch.zeros_like(v) for v in views]
    store.load(disk.DiskStates(tmp_path, "a" * 16, 1 << 30).entries[0], dst, "cpu")
    assert all(torch.equal(a, b) for a, b in zip(dst, views))


# -- finding by name -------------------------------------------------------------------------------------------
def test_a_prompts_prefix_keys_are_the_states_names(disk, tmp_path):
    import numpy as np

    prompt = [5, 123456, 0, 7, *range(300)]
    lengths = [0, 1, 2, 4, 50, 304]
    assert disk._keys(prompt, lengths) == {n: disk._key(prompt[:n]) for n in lengths}
    assert disk._keys(np.array(prompt), [50]) == {50: disk._key(prompt[:50])}
    store = disk.DiskStates(tmp_path, "a" * 16, 1 << 30)
    write(store, prompt[:50], rows(50))
    write(store, prompt[:200], rows(200))
    assert store.entries[0].path.stem == disk._key(prompt[:50])
    assert store.has(prompt[:50]) and store.has(np.array(prompt[:200])) and not store.has(prompt[:51])
    assert store.find(prompt, 0, lambda e: True).ids == prompt[:200]
    assert store.find(prompt, 0, lambda e: len(e.ids) < 100).ids == prompt[:50]
    store.forget(store.entries[-1])
    assert not store.has(prompt[:200]) and store.find(prompt, 0, lambda e: True).ids == prompt[:50]
