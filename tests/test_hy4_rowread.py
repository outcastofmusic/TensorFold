"""The Engram table's native row gather (``rowread.cpp``) against the Python pool: the same bytes, any row order,
repeats and runs of adjacent rows (a tiny bf16 and FP8 table written here, CPU)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.torch


@pytest.mark.parametrize("dtype", ["F8_E4M3", "BF16"])
def test_native_and_python_reads_give_the_same_rows(tmp_path, monkeypatch, dtype):
    import torch
    from safetensors.torch import save_file

    from tensorfold.families.hy4_flash import engram

    rows, dim = 5000, 96
    g = torch.Generator().manual_seed(0)
    w = torch.randn(rows, dim, generator=g)
    t = {"t.weight": w.to(torch.float8_e4m3fn) if dtype == "F8_E4M3" else w.to(torch.bfloat16)}
    if dtype == "F8_E4M3":
        t["t.weight_scale"] = torch.rand(rows, generator=g)
    save_file(t, str(tmp_path / "t.safetensors"))
    reader = engram.RowReader(tmp_path / "t.safetensors", "t", rows, dim)
    rng = np.random.default_rng(1)
    ids = np.concatenate([rng.integers(0, rows, 3000), np.arange(100, 400), [0, rows - 1, 7, 7]]).reshape(-1, 4)
    monkeypatch.setattr(engram, "NATIVE_ROWS", 1 << 30)
    want = reader.gather(ids)
    monkeypatch.setattr(engram, "NATIVE_ROWS", 1)
    got = reader.gather(ids)
    assert np.array_equal(got[0], want[0])
    assert (got[1] is None) == (want[1] is None) and (got[1] is None or np.array_equal(got[1], want[1]))


@pytest.mark.parametrize("dtype", ["F8_E4M3", "BF16"])
def test_inline_reads_give_the_pool_s_rows(tmp_path, monkeypatch, dtype):
    """A decode call's few rows, queued with fadvise and read in the calling thread, against the Python pool."""

    import torch
    from safetensors.torch import save_file

    from tensorfold.families.hy4_flash import engram

    rows, dim = 3000, 96
    g = torch.Generator().manual_seed(2)
    w = torch.randn(rows, dim, generator=g)
    t = {"t.weight": w.to(torch.float8_e4m3fn) if dtype == "F8_E4M3" else w.to(torch.bfloat16)}
    if dtype == "F8_E4M3":
        t["t.weight_scale"] = torch.rand(rows, generator=g)
    save_file(t, str(tmp_path / "t.safetensors"))
    reader = engram.RowReader(tmp_path / "t.safetensors", "t", rows, dim)
    rng = np.random.default_rng(3)
    ids = np.concatenate([rng.integers(0, rows, 52), [10, 11, 12, 12, 0, rows - 1, 5, 5, 6, 7, 8, 9]]).reshape(-1, 16)
    monkeypatch.setattr(engram, "NATIVE_ROWS", 1 << 30)
    monkeypatch.setattr(engram, "INLINE_ROWS", 0)
    want = reader.gather(ids)
    monkeypatch.setattr(engram, "INLINE_ROWS", 256)
    reader.advise(ids)
    got = reader.gather(ids)
    assert np.array_equal(got[0], want[0])
    assert (got[1] is None) == (want[1] is None) and (got[1] is None or np.array_equal(got[1], want[1]))
