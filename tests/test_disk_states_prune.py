"""Pruning kept states on disk (``tensorfold.cuda.disk_states``): a TTL on states unused, old fingerprints' folders
gone past it, and one budget for the whole folder, other fingerprints' folders least recently used first."""

from __future__ import annotations

import importlib
import json
import os
import time

import pytest

from tests.test_glm_disk_states import rows, write

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
