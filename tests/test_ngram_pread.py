"""The pread gather (TF_NGRAM_GATHER=pread) gives every n-gram table the bytes its memory map's copy gives,
reads each distinct row once and rows sharing a page in one read."""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.families.qwen4_exp import host_table, ssd_table
from tensorfold.families.qwen4_exp.host_table import HostTable, NVFP4Table, read_header
from tests.test_ngram_table_threads import _tables
from tests.test_ple_ssd import _checkpoint, _same

ASKS = [np.array([], dtype=np.int64), np.array([0]), np.array([3, 3, 1, 3])]     # none, one, repeated


def _asks(rows: int) -> list[np.ndarray]:
    rng = np.random.default_rng(29)
    return [*ASKS, rng.integers(0, rows, (40, 16)), np.arange(rows)[::-1], rng.integers(0, rows, 7)]


def _both(make, monkeypatch):
    """The table made under each switch: (mmap, pread)."""

    monkeypatch.setenv("TF_NGRAM_GATHER", "mmap")
    mapped = make()
    monkeypatch.setenv("TF_NGRAM_GATHER", "pread")
    read = make()
    assert mapped._reader is None and read._reader is not None
    return mapped, read


def _nvfp4(tmp_path, counts=(3000, 1, 2048)) -> list:
    """NVFP4 shards over two files, many pages each: 80 code bytes and 10 scale bytes a row, as FP4-Local's."""

    rng = np.random.default_rng(31)
    files = []
    for f, shards in enumerate(((0,), (1, 2))):
        header, data, at = {}, b"", 0
        for i in shards:
            for name, dtype, arr in ((f"t.shard_{i}.weight", "U8", rng.integers(0, 256, (counts[i], 80), np.uint8)),
                                     (f"t.shard_{i}.weight_scale", "F8_E4M3",
                                      rng.integers(0x20, 0x48, (counts[i], 10), np.uint8))):
                header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [at, at + arr.nbytes]}
                data, at = data + arr.tobytes(), at + arr.nbytes
        raw = json.dumps(header).encode()
        raw += b" " * (-len(raw) % 8)
        path = tmp_path / f"n{f}.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
        head = read_header(path)
        files += [(path, head[f"t.shard_{i}.weight"], head[f"t.shard_{i}.weight_scale"]) for i in shards]
    return files


def test_mlx_rows_read_with_pread_equal_the_maps(tmp_path, monkeypatch):
    files, _ = _checkpoint(tmp_path)
    mapped, read = _both(lambda: HostTable(files), monkeypatch)
    for ids in _asks(read.rows):
        _same(read.gather(ids), mapped.gather(ids))


@pytest.mark.parametrize("kind", ["bf16", "fp8", "nvfp4"])
def test_bf16_fp8_and_nvfp4_rows_read_with_pread_equal_the_maps(tmp_path, monkeypatch, kind):
    mapped, read = _both(lambda: _tables(tmp_path)[0][kind], monkeypatch)
    for ids in _asks(read.rows):
        got, want = read.gather(ids), mapped.gather(ids)
        assert got.dtype == want.dtype and got.shape == want.shape and got.tobytes() == want.tobytes()


def test_nvfp4_rows_over_many_pages_and_files_equal_the_maps(tmp_path, monkeypatch):
    files = _nvfp4(tmp_path)
    mapped, read = _both(lambda: NVFP4Table(files, 0.0371), monkeypatch)
    rng = np.random.default_rng(37)
    for ids in (rng.integers(0, read.rows, (256, 16)), np.arange(2990, 3010), rng.integers(0, read.rows, 3)):
        assert read.gather(ids).tobytes() == mapped.gather(ids).tobytes()


@pytest.mark.torch
@pytest.mark.parametrize("consolidated", [False, True])
def test_exl3_rows_read_with_pread_equal_the_maps(tmp_path, monkeypatch, consolidated):
    import torch

    from tensorfold.families.qwen4_exp.cuda import exl3_pack

    words, rows = 1 + 160 * 4 // 16, [700, 300]
    data = np.random.default_rng(41).integers(-2 ** 15, 2 ** 15, (sum(rows), words), dtype=np.int16)
    (tmp_path / "ngram.safetensors").write_bytes(b"\0" * 24 + data.tobytes())       # rows after a header
    entries = {f"t.shard_{i}.trellis": ("ngram.safetensors", 24 + 2 * words * sum(rows[:i]),
                                        24 + 2 * words * sum(rows[:i + 1]), "I16", [n, words]) for i, n in enumerate(rows)}
    if consolidated:
        entries = {"t.trellis": ("ngram.safetensors", 24, 24 + data.nbytes, "I16", list(data.shape))}
    tensors = {"t.head_bias": torch.zeros(4), "t.head_offsets": torch.zeros(2, dtype=torch.int64),
               "t.head_vocab_sizes": torch.ones(2, dtype=torch.int64), "t.layer_multipliers": torch.ones(2)}
    pk = SimpleNamespace(dir=tmp_path, entry=entries.__getitem__, get=tensors.__getitem__)
    monkeypatch.setattr(host_table, "GATHER_SPLIT", 8)       # the map's copy on threads from 16 rows
    mapped, read = _both(lambda: exl3_pack.NgramTable(pk, "t.", 2, "cpu"), monkeypatch)
    for ids in _asks(read.rows):
        got = read.gather(ids)
        assert got.dtype == np.int16 and got.tobytes() == mapped.gather(ids).tobytes() == data[ids.reshape(-1)].tobytes()
    with pytest.raises(IndexError, match="outside"):
        read.gather([sum(rows)])


def test_each_distinct_row_is_read_once_and_rows_sharing_a_page_in_one_read(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_NGRAM_GATHER", "pread")
    files = _nvfp4(tmp_path)
    table = NVFP4Table(files, 1.0)
    reads = []
    real = ssd_table._fill
    monkeypatch.setattr(ssd_table, "_fill", lambda batch: reads.extend(batch) or real(batch))
    near = np.array([10, 11, 12, 10, 11, 10])            # three rows in one page of codes and one of scales
    table.gather(near)
    assert len(reads) == 2 and [r[2] for r in reads] == [240, 30]
    reads.clear()
    far = np.array([0, 500, 1000])                       # pages apart (5,000 scale bytes): a read each, both parts
    table.gather(far)
    assert len(reads) == 6
    for fd, offset, size, _, _ in reads:
        assert size in (80, 10)


def test_a_prompt_chunk_reads_on_the_shared_pool_and_a_decode_step_on_the_caller(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_NGRAM_GATHER", "pread")
    files = _nvfp4(tmp_path)
    table = NVFP4Table(files, 1.0)
    pooled = []
    pool = ssd_table._shared_pool()
    real = pool.map
    monkeypatch.setattr(pool, "map", lambda fn, jobs: pooled.append(len(jobs)) or real(fn, jobs))
    table.gather(np.arange(4) * 997)                      # 4 rows: 8 reads, on this thread
    assert pooled == []
    table.gather(np.random.default_rng(43).integers(0, table.rows, (64, 16)))
    assert len(pooled) == 1 and 1 < pooled[0] <= ssd_table.PREAD_WORKERS


def test_a_short_read_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_NGRAM_GATHER", "pread")
    files = _nvfp4(tmp_path, counts=(64, 1, 64))
    table = NVFP4Table(files, 1.0)
    with open(files[0][0], "r+b") as f:                    # the checkpoint shrank under the open table
        f.truncate(4096)
    with pytest.raises(OSError, match="short read"):
        table.gather(np.array([63]))


def test_an_unknown_gather_mode_stops_the_load_and_unset_copies_from_the_map(tmp_path, monkeypatch):
    files, _ = _checkpoint(tmp_path)
    monkeypatch.delenv("TF_NGRAM_GATHER", raising=False)
    assert HostTable(files)._reader is None
    monkeypatch.setenv("TF_NGRAM_GATHER", "mlock")
    with pytest.raises(ValueError, match="TF_NGRAM_GATHER"):
        HostTable(files)


def test_nvfp4_scales_pin_only_when_asked_and_within_the_budget(tmp_path, monkeypatch):
    from tensorfold.cuda import ngram_pages

    files = _nvfp4(tmp_path)
    locked = []

    class Api:
        def mlock(self, at, n):
            locked.append((at, n))
            return 0

        def munlock(self, at, n):
            return 0

    monkeypatch.setattr(ngram_pages, "libc", lambda: Api())
    monkeypatch.delenv("TF_NGRAM_LOCK_SCALES", raising=False)
    table = NVFP4Table(files, 1.0)
    assert table.lock_runs(1 << 30) == 0 and table.pinned_bytes == 0 and not locked
    monkeypatch.setenv("TF_NGRAM_LOCK_SCALES", "1")
    scales = sum(ngram_pages.lock_bytes(SimpleNamespace(values=[a])) for a in table.scales)
    assert table.lock_runs(0) == 0 and not locked
    got = table.lock_runs(1 << 30)
    assert got == table.pinned_bytes == scales == sum(n for _, n in locked)
    codes = [ngram_pages.span(int(a.ctypes.data), int(a.nbytes)) for a in table.values]
    assert all(not (lo < c1 and c0 < lo + n) for lo, n in locked for c0, c1 in codes)       # codes stay unpinned
    assert table.lock_runs(1 << 30) == 0                                                    # nothing new to pin
