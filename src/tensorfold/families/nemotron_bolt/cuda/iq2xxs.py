"""IQ2_XXS matmuls for Nemotron 3.5 Bolt's routed experts, on the checkpoint's ggml blocks (66 bytes a 256 weights).

``iq2xxs_linear(x, w)`` and ``iq2xxs_grouped(x, w, ids)`` return ``x @ dequantize(w).T`` in bf16, accumulated in fp32
on bf16 weights rounded as llama.cpp's dequantize_row_iq2_xxs and then bf16 give them. Up to ``DECODE_ROWS`` rows a
call runs the decode kernel (16-column tiles of the call's distinct experts, each expert's weights read once); a
row's bits there do not depend on the call's other rows or their number, so a verify window of up to 512 rows (56
tokens of 9 experts) equals serial one-token steps bit for bit. More rows run the prefill kernel (64-row tiles of
one expert, rows sorted by expert on the GPU, no host sync), whose sums can differ from the decode kernel's by one
bf16 rounding. A ``prompt`` call runs the prefill kernel at any row count: a prompt row's bits then never depend on
the chunk it came in, so a prompt resumed from a kept state equals a fresh prefill.
"""

from __future__ import annotations

from functools import cache, lru_cache
from pathlib import Path

import torch

from ..quant import IQ2XXS_BLOCK, IQ2XXS_BYTES, IQ2XXS_GRID

DECODE_ROWS = 512        # calls up to this many rows run the decode kernel, bit for bit the same a row at any count
PLAN_ROWS = 64           # a grouped decode call wider than this gets its experts from a plan kernel
PREFILL_ROWS = 64        # rows a prefill tile
MAGNITUDES = (8, 25, 43)  # every byte of IQ2XXS_GRID is one of these


def tables() -> list[int]:
    """The kernels' selector words: two variants of [256][2], each word two 16-bit prmt selectors.

    Selector p of an entry builds the bf16 pair (weights 2p, 2p + 1) from the magnitudes' bytes (8: 0-1, 25: 2-3,
    43: 4-5); word w holds pairs 2w and 2w + 1. Variant 1 lists the pairs in the order 2, 3, 0, 1, for the prefill
    lanes that take an entry's halves the other way round.
    """

    sel = []
    for entry in IQ2XXS_GRID:
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
    return load(name="tensorfold_bolt_iq2xxs_v18", sources=[str(here / "iq2xxs.cpp"), str(here / "iq2xxs.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def _rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x as the kernels read it: contiguous, 16-byte aligned bf16 [R, K] with K matching w's blocks."""

    if x.dim() != 2 or x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bf16 [R, K], not {x.dtype} {tuple(x.shape)}")
    if x.shape[1] % IQ2XXS_BLOCK or w.shape[-1] != x.shape[1] // IQ2XXS_BLOCK * IQ2XXS_BYTES:
        raise ValueError(f"x [R, {x.shape[1]}] does not match IQ2_XXS rows of {w.shape[-1]} bytes")
    x = x.contiguous()
    return x if x.data_ptr() % 16 == 0 else x.clone()


def _blocks(w: torch.Tensor, dims: int) -> torch.Tensor:
    """w as [..., K / 256 * 66] with ``dims`` dims; the checkpoint's [..., K / 256, 66] is the same bytes."""

    if w.dtype != torch.uint8:
        raise ValueError(f"IQ2_XXS weights are uint8, not {w.dtype}")
    if w.dim() == dims + 1 and w.shape[-1] == IQ2XXS_BYTES:
        w = w.flatten(-2)
    if w.dim() != dims:
        raise ValueError(f"IQ2_XXS weights {tuple(w.shape)}: expected {dims} dims of blocks")
    return w.contiguous()


def iq2xxs_linear(x: torch.Tensor, w: torch.Tensor, *, prompt: bool = False) -> torch.Tensor:
    """x bf16 [R, K], w uint8 [N, K / 256 * 66] (or the checkpoint's [N, K / 256, 66]) -> bf16 [R, N]."""

    w = _blocks(w, 2)
    x = _rows(x, w)
    w3 = w.unsqueeze(0)
    rows, n = x.shape[0], w.shape[0]
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=x.device)
    tab = _tables(x.device)
    if rows <= DECODE_ROWS and not prompt:
        _ext().decode(x, w3, None, None, min(4, max(1, -(-rows // 16))), tab, out)
    else:
        _ext().prefill(x, w3, None, None, None, -(-rows // PREFILL_ROWS), tab, out)
    return out


def iq2xxs_grouped(x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor, *, prompt: bool = False) -> torch.Tensor:
    """x bf16 [R, K], w uint8 [E, N, K / 256 * 66] (or [E, N, K / 256, 66]), ids int32 [R] in [0, E) -> bf16 [R, N].

    Row r uses expert ids[r]; an id outside [0, E) reads outside w.
    """

    w = _blocks(w, 3)
    x = _rows(x, w)
    rows, (experts, n) = x.shape[0], w.shape[:2]
    if ids.shape != (rows,):
        raise ValueError(f"ids {tuple(ids.shape)}: one expert a row of x [{rows}, K]")
    ids = ids.to(device=x.device, dtype=torch.int32).contiguous()
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=x.device)
    tab = _tables(x.device)
    if rows <= DECODE_ROWS and not prompt:
        plan = torch.empty((2 + 2 * experts + rows,), dtype=torch.int32, device=x.device) if rows > PLAN_ROWS else None
        _ext().decode(x, w, ids, plan, min(4, max(1, -(-rows // 16))), tab, out)
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
