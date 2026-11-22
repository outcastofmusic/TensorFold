"""Hy4-flash's Engram memories: the n-gram hashes of transformers PR #50903 and the tables' rows, read from disk.

A position's n-gram holds its token and the two before it (the pad id before the document's start). Each token goes
through the checkpoint's frozen ``lookup_table`` (tokens that differ by case or spacing share an id), is multiplied by
an odd number drawn from NumPy's default generator seeded by the layer, and the products are XORed: orders 2 and 3,
eight heads each, every head taking its own prime modulus. A head's row lives at its hash plus the head's offset in
one table of about 165 million 96-wide rows a layer.

The four tables (127 GB bf16, 66 GB as FP8 with a scale a row) never enter GPU memory: ``RowReader`` reads the rows a
call needs with ``pread`` at each lookup, deduplicated and coalesced, and the engine copies them over. That is 64 rows
(12 KiB) a decoded token. The row counts are checked against the config at load: a table read for the wrong layer stays
inside its count for almost every lookup and addresses rows silently (upstream 1ae2cc7bf).
"""

from __future__ import annotations

import os
import struct
import weakref
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import numpy as np

from tensorfold.families.qwen4_exp.ssd_table import _fill, _no_cache, _release

# reads in flight: a 2,048-row prompt chunk reads 32,768 random rows a table, 910 ms at 16 threads and 445 at 64
# (128: 468), cold, 21 Nov
WORKERS = 64

MAX_READ = 1 << 20            # bytes one read of adjacent rows may cover
# a gather of this many distinct rows or more reads on native threads (``rowread.cpp``) with the GIL released; fewer
# (a decode call's 16 rows a table) stay on the Python pool. TF_HY4_NATIVE_READS=0: always the Python pool
NATIVE_ROWS = 512


@lru_cache(maxsize=1)
def _native():
    from torch.utils import cpp_extension

    return cpp_extension.load(name="tensorfold_hy4_rowread_v1", sources=[str(Path(__file__).with_name("rowread.cpp"))],
                              extra_cflags=["-O2"], verbose=False)


def multipliers(seed: int, layer_id: int, compressed_vocab: int, ngram: int) -> np.ndarray:
    """The odd multipliers of Engram layer ``layer_id`` (one-based, as config.json counts): NumPy's default generator
    seeded per layer, bounded so a product with any compressed id stays inside int64 (``draw_hash_multipliers``)."""

    half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab // 2)
    drawn = np.random.default_rng(seed + 10007 * layer_id).integers(0, half_bound, size=ngram, dtype=np.int64)
    return drawn * 2 + 1


def ngram_tokens(ids: np.ndarray, start: int, previous: np.ndarray, ngram: int, pad: int) -> np.ndarray:
    """[T, ngram] int64: each position's token, then the ``ngram - 1`` before it.

    ``ids`` are the call's tokens at positions start .. start + T - 1 and ``previous`` the ``ngram - 1`` tokens before
    ``start`` (oldest first, fewer near the document's start); a slot before position 0 holds ``pad``."""

    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    previous = np.asarray(previous, dtype=np.int64).reshape(-1)[-(ngram - 1):] if ngram > 1 else np.zeros(0, np.int64)
    full = np.concatenate([previous, ids])
    lead = len(previous)
    out = np.empty((len(ids), ngram), dtype=np.int64)
    for k in range(ngram):
        j = np.arange(len(ids)) + lead - k
        pos = np.arange(len(ids)) + start - k
        out[:, k] = np.where(pos < 0, pad, full[np.clip(j, 0, None)])
    return out


class Hasher:
    """One Engram layer's hash: n-gram tokens [T, ngram] -> table rows [T, heads] (head offsets included)."""

    def __init__(self, lookup: np.ndarray, sizes: tuple[int, ...], *, seed: int, layer_id: int, ngram: int,
                 heads: int, pad: int) -> None:
        self.lookup = np.asarray(lookup, dtype=np.int64)
        self.sizes = np.asarray(sizes, dtype=np.int64)
        if len(self.sizes) != (ngram - 1) * heads:
            raise ValueError(f"Engram layer {layer_id}: {len(self.sizes)} hash heads, the config implies "
                             f"{(ngram - 1) * heads}")
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).astype(np.int64)
        self.rows = int(self.sizes.sum())
        self.mult = multipliers(seed, layer_id, int(self.lookup.max()) + 1, ngram)
        self.ngram, self.heads, self.pad = ngram, heads, pad

    def hashes(self, grams: np.ndarray) -> np.ndarray:
        """[T, ngram] tokens -> [T, heads * (ngram - 1)] hash ids (no offsets), as ``hash_ngrams`` computes them."""

        size = len(self.lookup)
        compressed = np.where(grams < size, self.lookup[np.clip(grams, None, size - 1)], self.lookup[self.pad])
        mixed = compressed * self.mult                      # int64 wraps as torch's does; the bound keeps it inside
        out = []
        for order in range(2, self.ngram + 1):
            mix = mixed[:, 0]
            for k in range(1, order):
                mix = np.bitwise_xor(mix, mixed[:, k])
            first = (order - 2) * self.heads
            out.append(np.mod(mix[:, None], self.sizes[first:first + self.heads]))    # Python's sign, like torch's %
        return np.concatenate(out, axis=1)

    def rows_of(self, grams: np.ndarray) -> np.ndarray:
        return self.hashes(grams) + self.offsets


def _tensor(header: dict, name: str, data: int, size: int) -> tuple[str, list[int], int]:
    entry = header.get(name)
    if entry is None:
        raise ValueError(f"no tensor {name}")
    lo, hi = entry["data_offsets"]
    if data + hi > size:
        raise ValueError(f"{name} passes its file's end")
    return entry["dtype"], entry["shape"], data + lo


class RowReader:
    """One Engram table read row by row from its safetensors file: FP8 codes with an fp32 scale a row, or bf16."""

    ITEM: ClassVar[dict[str, int]] = {"F8_E4M3": 1, "BF16": 2}

    def __init__(self, path: Path, name: str, rows: int, dim: int) -> None:
        self._fds: list[int] = []
        self._pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="engram")
        self._closer = weakref.finalize(self, _release, self._fds, self._pool)
        fd = os.open(path, os.O_RDONLY)
        self._fds.append(fd)
        _no_cache(fd)
        size = os.fstat(fd).st_size
        head = os.pread(fd, 8, 0)
        data = 8 + struct.unpack("<Q", head)[0]
        import json

        header = json.loads(os.pread(fd, data - 8, 8))
        dtype, shape, self.w0 = _tensor(header, name + ".weight", data, size)
        if dtype not in self.ITEM or shape != [rows, dim]:
            raise ValueError(f"{name}: {dtype} {shape}; the config implies {rows} rows of {dim} (FP8 or bf16). A "
                             "table of another layer or a repacked one would address rows silently")
        self.dtype, self.rows, self.dim = dtype, rows, dim
        self.wrow = dim * self.ITEM[dtype]
        self.scales = None
        if dtype == "F8_E4M3":
            sdt, sshape, s0 = _tensor(header, name + ".weight_scale", data, size)
            if sdt != "F32" or sshape != [rows]:
                raise ValueError(f"{name}.weight_scale: {sdt} {sshape}, not F32 [{rows}]")
            # the row scales stay in memory (660 MB a table): a lookup then reads one span a row, not two (a chunk's
            # 32,768 rows 445 -> 232 ms)
            self.scales = np.fromfile(path, dtype=np.float32, count=rows, offset=s0)

    @property
    def fp8(self) -> bool:
        return self.dtype == "F8_E4M3"

    def gather(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Rows ``ids`` -> (bytes [n, row bytes] uint8, fp32 scales [n] or None for bf16), in ``ids``' order."""

        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        if flat.size and (flat.min() < 0 or flat.max() >= self.rows):
            raise ValueError(f"Engram rows must lie in [0, {self.rows})")
        unique, inverse = np.unique(flat, return_inverse=True)
        codes = np.empty((unique.size, self.wrow), dtype=np.uint8)
        if unique.size >= NATIVE_ROWS and os.environ.get("TF_HY4_NATIVE_READS", "1") != "0":
            self._native_reads(self.w0 + unique * self.wrow, self.wrow, codes)
            return codes[inverse], None if self.scales is None else self.scales[flat]
        reads = self._reads(self.w0 + unique * self.wrow, self.wrow, codes)
        batches = min(WORKERS, len(reads))
        if batches > 1:
            list(self._pool.map(_fill, [reads[i::batches] for i in range(batches)]))
        else:
            _fill(reads)
        return codes[inverse], None if self.scales is None else self.scales[flat]

    def nbytes(self) -> int:
        """Host bytes held: the row scales."""

        return 0 if self.scales is None else self.scales.nbytes

    def _native_reads(self, offsets: np.ndarray, width: int, out: np.ndarray) -> None:
        """``_reads``' coalesced spans, read by ``rowread.cpp`` on WORKERS native threads."""

        import torch

        n = offsets.size
        cut = np.ones(n, dtype=bool)
        cut[1:] = offsets[1:] != offsets[:-1] + width
        first = np.flatnonzero(cut)
        cut |= (np.arange(n) - first[np.cumsum(cut) - 1]) % max(1, MAX_READ // width) == 0
        first = np.flatnonzero(cut)
        count = np.diff(first, append=n)
        _native().gather(self._fds[0], torch.from_numpy(offsets[first].astype(np.int64)),
                         torch.from_numpy((count * width).astype(np.int64)),
                         torch.from_numpy((first * width).astype(np.int64)), torch.from_numpy(out.reshape(-1)), WORKERS)

    def _reads(self, offsets: np.ndarray, width: int, out: np.ndarray) -> list[tuple]:
        n = offsets.size
        if not n:
            return []
        cut = np.ones(n, dtype=bool)
        cut[1:] = offsets[1:] != offsets[:-1] + width
        first = np.flatnonzero(cut)
        cut |= (np.arange(n) - first[np.cumsum(cut) - 1]) % max(1, MAX_READ // width) == 0
        first = np.flatnonzero(cut)
        count = np.diff(first, append=n)
        view = memoryview(out.reshape(-1))
        fd = self._fds[0]
        return [(fd, int(o), int(c) * width, view, int(f) * width) for o, c, f in zip(offsets[first], count, first)]

    def close(self) -> None:
        self._closer()


def decode_rows(codes: np.ndarray, scales: np.ndarray | None, dim: int) -> np.ndarray:
    """Gathered rows -> fp32 [n, dim]: FP8 codes times their row's scale, or the bf16 values."""

    if scales is None:
        return (codes.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(-1, dim)
    import torch

    q = torch.from_numpy(np.ascontiguousarray(codes)).view(torch.float8_e4m3fn).float()
    return (q * torch.from_numpy(np.ascontiguousarray(scales))[:, None]).numpy()
