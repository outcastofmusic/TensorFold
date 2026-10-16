"""IQ2_XXS matmuls for Nemotron 3.5 Bolt's routed experts, on the checkpoint's ggml blocks (66 bytes a 256 weights).

``iq2xxs_linear(x, w)`` and ``iq2xxs_grouped(x, w, ids)`` return ``x @ dequantize(w).T`` in bf16, accumulated in fp32
on bf16 weights rounded as llama.cpp's dequantize_row_iq2_xxs and then bf16 give them. Up to ``DECODE_ROWS`` rows a
call runs the decode kernel (one warp a 16-column tile; a grouped call reads each expert's rows once); more rows run
the prefill kernel (64-row tiles of one expert, rows sorted by expert on the GPU, no host sync).
"""

from __future__ import annotations

from functools import cache, lru_cache
from pathlib import Path

import torch

QK = 256                 # weights a block
BLOCK_BYTES = 66         # fp16 d, then 32 uint16 qs
DECODE_ROWS = 64         # the decode kernel's widest call: four m16 tiles
PREFILL_ROWS = 64        # rows a prefill tile

# iq2xxs_grid, copied from llama.cpp ggml/src/ggml-common.h:560 (23f7aabc, 2026-10-16); entry byte j is weight j
GRID = (
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
)
GRID_SHA256 = "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54"   # of the 256 entries, little-endian
MAGNITUDES = (8, 25, 43)  # every byte of GRID is one of these


def ksigns(i: int) -> int:
    """llama.cpp's ksigns_iq2xs[i] (ggml-common.h:513): seven sign bits and their parity as the eighth."""

    return i | (i.bit_count() & 1) << 7


def tables() -> list[int]:
    """The kernels' selector words: two variants of [256][2], each word two 16-bit prmt selectors.

    Selector p of an entry builds the bf16 pair (weights 2p, 2p + 1) from the magnitudes' bytes (8: 0-1, 25: 2-3,
    43: 4-5); word w holds pairs 2w and 2w + 1. Variant 1 lists the pairs in the order 2, 3, 0, 1, for the prefill
    lanes that take an entry's halves the other way round.
    """

    sel = []
    for entry in GRID:
        idx = [MAGNITUDES.index((entry >> (8 * j)) & 255) for j in range(8)]
        sel.append([2 * idx[2 * p] | (2 * idx[2 * p] + 1) << 4 | (2 * idx[2 * p + 1]) << 8
                    | (2 * idx[2 * p + 1] + 1) << 12 for p in range(4)])
    words = []
    for variant in (0, 1):
        for row in sel:
            words += [row[(2 * w) ^ 2 * variant] | row[(2 * w + 1) ^ 2 * variant] << 16 for w in range(2)]
    return words


@cache
def _tables(device: torch.device) -> torch.Tensor:
    words = [v - (1 << 32) if v >= 1 << 31 else v for v in tables()]
    return torch.tensor(words, dtype=torch.int32, device=device)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_bolt_iq2xxs_v17", sources=[str(here / "iq2xxs.cpp"), str(here / "iq2xxs.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def _rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x as the kernels read it: contiguous, 16-byte aligned bf16 [R, K] with K matching w's blocks."""

    if x.dim() != 2 or x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bf16 [R, K], not {x.dtype} {tuple(x.shape)}")
    if x.shape[1] % QK or w.shape[-1] != x.shape[1] // QK * BLOCK_BYTES:
        raise ValueError(f"x [R, {x.shape[1]}] does not match IQ2_XXS rows of {w.shape[-1]} bytes")
    x = x.contiguous()
    return x if x.data_ptr() % 16 == 0 else x.clone()


def iq2xxs_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x bf16 [R, K], w uint8 [N, K / 256 * 66] -> bf16 [R, N]."""

    x = _rows(x, w)
    w3 = w.contiguous().unsqueeze(0)
    rows, n = x.shape[0], w.shape[0]
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=x.device)
    tab = _tables(x.device)
    if rows <= DECODE_ROWS:
        _ext().decode(x, w3, None, max(1, -(-rows // 16)), tab, out)
    else:
        _ext().prefill(x, w3, None, None, None, -(-rows // PREFILL_ROWS), tab, out)
    return out


def iq2xxs_grouped(x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """x bf16 [R, K], w uint8 [E, N, K / 256 * 66], ids int32 [R] in [0, E) -> bf16 [R, N]; row r uses expert ids[r]."""

    x = _rows(x, w)
    w = w.contiguous()
    rows, (experts, n) = x.shape[0], w.shape[:2]
    if ids.shape != (rows,):
        raise ValueError(f"ids {tuple(ids.shape)}: one expert a row of x [{rows}, K]")
    ids = ids.to(device=x.device, dtype=torch.int32).contiguous()
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=x.device)
    tab = _tables(x.device)
    if rows <= DECODE_ROWS:
        _ext().decode(x, w, ids, max(1, -(-rows // 16)), tab, out)
        return out
    order = torch.argsort(ids, stable=True).to(torch.int32)
    counts = torch.bincount(ids, minlength=experts)
    starts = torch.zeros((experts + 1,), dtype=torch.int32, device=x.device)
    starts[1:] = torch.cumsum(counts, 0)
    tile_end = torch.cumsum(torch.div(counts + PREFILL_ROWS - 1, PREFILL_ROWS, rounding_mode="floor"), 0,
                            dtype=torch.int32)
    tiles = min(experts, rows) + rows // PREFILL_ROWS            # a bound on the tiles; the rest exit at once
    _ext().prefill(x, w, order, tile_end, starts, tiles, tab, out)
    return out


__all__ = ["iq2xxs_grouped", "iq2xxs_linear", "tables"]
