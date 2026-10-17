"""IQ2_XXS experts until ``iq2xxs.py``'s kernels land: dequantize only the weights a call reads, then the bf16 linear.

The two functions keep ``iq2xxs.py``'s contract signatures, so the engine's call sites stay when the kernels replace
them. ``dequantize_iq2xxs`` follows llama.cpp's ``dequantize_row_iq2_xxs`` (ggml/src/ggml-quants.c, lines 2489-2513)
and ``block_iq2_xxs`` (ggml/src/ggml-common.h, lines 381-385): a block of 256 weights is an fp16 scale d and 32
uint16 words; each run of 32 weights reads two uint32, the first holding four grid indices and the second four 7-bit
sign indices and a 4-bit scale s, and weight j of a run of 8 is d * (0.5 + s) * 0.25 * grid[j] * sign[j] in fp32.
A CUDA tensor dequantizes in a Triton kernel with the same fp32 operations in the same order, so both give the same
bits. Decode rows (up to ``DECODE_ROWS``) run ``b16.matmul``, whose rows never depend on the row count, so a window
row gets the serial step's bits; more rows, and every ``prompt`` call, run ``b16.prompt``.
"""

from __future__ import annotations

from functools import cache

import torch

BLOCK = 256          # weights a block
BLOCK_BYTES = 66     # its fp16 scale and 32 uint16 words
DECODE_ROWS = 64     # rows up to this count take the row-invariant decode linear

# iq2xxs_grid from llama.cpp ggml/src/ggml-common.h (lines 560-625): byte j of entry i is weight j of grid point i
GRID = (
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08, 0x0808080808082b2b,
    0x0808080808190819, 0x0808080808191908, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b2b08,
    0x08080808082b2b2b, 0x0808080819080819, 0x0808080819081908, 0x0808080819190808, 0x0808080819192b08,
    0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808, 0x0808081908191919,
    0x0808081919080808, 0x080808192b081908, 0x080808192b192b08, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b082b082b, 0x0808082b2b08082b, 0x0808190808080819, 0x0808190808081908, 0x0808190808190808,
    0x08081908082b0819, 0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x080819082b2b1908,
    0x0808191908080808, 0x080819190808082b, 0x0808191908082b08, 0x08081919082b0808, 0x080819191908192b,
    0x08081919192b2b19, 0x080819192b080808, 0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b19080808, 0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819, 0x08082b0819081908,
    0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08, 0x08082b1908081908, 0x08082b1919080808,
    0x08082b2b0808082b, 0x08082b2b08191908, 0x0819080808080819, 0x0819080808081908, 0x0819080808190808,
    0x08190808082b0819, 0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808, 0x0819081919190808,
    0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908, 0x0819082b19081919, 0x0819190808080808,
    0x0819190808082b08, 0x08191908082b0808, 0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808,
    0x0819191908192b08, 0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819, 0x08192b1908080808,
    0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819, 0x082b080808080808, 0x082b08080808082b,
    0x082b080808082b2b, 0x082b080819081908, 0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b,
    0x082b0819082b2b19, 0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b, 0x082b191908080808,
    0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808, 0x082b2b0808082b08, 0x082b2b08082b0808,
    0x082b2b082b191908, 0x082b2b2b19081908, 0x1908080808080819, 0x1908080808081908, 0x1908080808190808,
    0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819, 0x190808192b080808, 0x190808192b081919,
    0x1908082b08080819, 0x1908082b08190808, 0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08,
    0x1908190808080808, 0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b1908080808, 0x19082b1919192b08,
    0x19082b19192b0819, 0x19082b192b08082b, 0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808,
    0x1919080808082b08, 0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908, 0x1919082b2b190819,
    0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b, 0x1919192b08080819, 0x1919192b19191908,
    0x19192b0808080808, 0x19192b0808190819, 0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808,
    0x19192b2b08082b08, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808, 0x192b190808080808,
    0x192b190808081919, 0x192b191908190808, 0x192b19190819082b, 0x192b19192b081908, 0x192b2b081908082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b,
    0x2b08081908081908, 0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808,
    0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908, 0x2b08192b08082b19, 0x2b08192b19080808,
    0x2b08192b192b0808, 0x2b082b080808082b, 0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b, 0x2b19190819081908,
    0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808, 0x2b2b08080808082b, 0x2b2b080819190808,
    0x2b2b08082b081919, 0x2b2b081908082b19, 0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808,
    0x2b2b2b1908081908,
)


@cache
def _tables(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(grid bytes [256, 8] uint8, ksigns [128] uint8) on ``device``; ksigns_iq2xs[i] is i with an even-parity bit 7."""

    grid = torch.tensor([[(g >> (8 * j)) & 0xFF for j in range(8)] for g in GRID], dtype=torch.uint8)
    signs = torch.tensor([i | ((i.bit_count() & 1) << 7) for i in range(128)], dtype=torch.uint8)
    return grid.to(device), signs.to(device)


def _check(packed: torch.Tensor) -> int:
    if packed.dtype != torch.uint8 or packed.shape[-1] % BLOCK_BYTES:
        raise ValueError(f"IQ2_XXS rows are uint8 in blocks of {BLOCK_BYTES} bytes, not {packed.dtype} "
                         f"{tuple(packed.shape)}")
    return packed.shape[-1] // BLOCK_BYTES


def dequantize_iq2xxs_torch(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K // 256 * 66] -> bf16 [..., K] with torch ops (any device): the definition the kernel matches."""

    nb = _check(packed)
    lead = packed.shape[:-1]
    b = packed.reshape(-1, nb, BLOCK_BYTES)
    grid, signs = _tables(str(packed.device))
    d = b[..., :2].contiguous().view(torch.float16).float()                           # [M, nb, 1]
    runs = b[..., 2:].reshape(b.shape[0], nb, 8, 8)                                   # 8 runs of 32 weights
    words = runs[..., 4:].contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF   # [M, nb, 8, 1]
    db = d[..., None] * (0.5 + (words >> 28).float()) * 0.25                          # [M, nb, 8, 1]
    sign_idx = (words >> torch.tensor([0, 7, 14, 21], device=packed.device)) & 127   # [M, nb, 8, 4]
    sign = signs[sign_idx].to(torch.int32)[..., None] >> torch.arange(8, device=packed.device, dtype=torch.int32)
    neg = (sign & 1).bool()                                                           # [M, nb, 8, 4, 8]
    values = grid[runs[..., :4].long()].float()                                       # [M, nb, 8, 4, 8]
    y = db[..., None] * values
    y = torch.where(neg, -y, y)
    return y.reshape(*lead, nb * BLOCK).to(torch.bfloat16)


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _dequant(P, GRIDB, SIGNS, OUT, K, NBYTES, BK: tl.constexpr):
        """Program (row, K tile): BK weights of one row from its 66-byte blocks, in the torch definition's order."""

        row = tl.program_id(0)
        k = tl.program_id(1) * BK + tl.arange(0, BK)
        ok = k < K
        base = P + row.to(tl.int64) * NBYTES + (k // 256) * 66
        run = (k % 256) // 32
        part = (k % 32) // 8
        j = k % 8
        lo = tl.load(base, mask=ok, other=0).to(tl.uint16)
        hi = tl.load(base + 1, mask=ok, other=0).to(tl.uint16)
        d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        w = base + 2 + run * 8
        word = (tl.load(w + 4, mask=ok, other=0).to(tl.uint32)
                | (tl.load(w + 5, mask=ok, other=0).to(tl.uint32) << 8)
                | (tl.load(w + 6, mask=ok, other=0).to(tl.uint32) << 16)
                | (tl.load(w + 7, mask=ok, other=0).to(tl.uint32) << 24))
        db = d * (0.5 + (word >> 28).to(tl.float32)) * 0.25
        idx = tl.load(w + part, mask=ok, other=0).to(tl.int32)
        g = tl.load(GRIDB + idx * 8 + j, mask=ok, other=0).to(tl.float32)
        s = tl.load(SIGNS + ((word >> (7 * part).to(tl.uint32)) & 127).to(tl.int32), mask=ok, other=0).to(tl.int32)
        y = db * g
        y = tl.where(((s >> j) & 1) != 0, -y, y)
        tl.store(OUT + row.to(tl.int64) * K + k, y.to(tl.bfloat16), mask=ok)

    return _dequant


_DEQUANT = None


def dequantize_iq2xxs(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K // 256 * 66] -> bf16 [..., K]: the Triton kernel on CUDA, the torch definition elsewhere."""

    global _DEQUANT
    nb = _check(packed)
    if packed.device.type != "cuda":
        return dequantize_iq2xxs_torch(packed)
    import triton

    if _DEQUANT is None:
        _DEQUANT = _kernel()
    rows = packed.reshape(-1, nb * BLOCK_BYTES).contiguous()
    k = nb * BLOCK
    out = torch.empty((rows.shape[0], k), dtype=torch.bfloat16, device=packed.device)
    grid, signs = _tables(str(packed.device))
    bk = 1024
    _DEQUANT[(rows.shape[0], triton.cdiv(k, bk))](rows, grid, signs, out, k, nb * BLOCK_BYTES, BK=bk, num_warps=4)
    return out.view(*packed.shape[:-1], k)


def _linear(x: torch.Tensor, w: torch.Tensor, prompt: bool = False) -> torch.Tensor:
    """x [R, K] @ w [N, K]^T in bf16: the row-invariant linear for decode rows, the prompt mma for more."""

    if x.device.type != "cuda":
        return (x.float() @ w.float().t()).to(torch.bfloat16)
    from tensorfold.families.qwen3_5.cuda import b16

    return b16.matmul(x, w) if x.shape[0] <= DECODE_ROWS and not prompt else b16.prompt(x, w)


def iq2xxs_linear(x: torch.Tensor, w: torch.Tensor, *, prompt: bool = False) -> torch.Tensor:
    """bf16 [R, K] times IQ2_XXS uint8 [N, K // 256 * 66] -> bf16 [R, N]."""

    return _linear(x, dequantize_iq2xxs(w), prompt)


def iq2xxs_grouped(x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor, *, prompt: bool = False) -> torch.Tensor:
    """bf16 [R, K], IQ2_XXS uint8 [E, N, K // 256 * 66], int32 [R] -> bf16 [R, N]: row r times expert ids[r].

    Only the experts ``ids`` names are dequantized, one at a time, so a call never holds a whole table in bf16.
    """

    rows = x.shape[0]
    out = torch.empty((rows, w.shape[1]), dtype=torch.bfloat16, device=x.device)
    if rows == 0:
        return out
    order = torch.argsort(ids.long(), stable=True)
    sorted_ids = ids.long()[order]
    experts, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    start = 0
    for e, n in zip(experts.tolist(), counts.tolist()):
        picked = order[start:start + n]
        out[picked] = _linear(x[picked], dequantize_iq2xxs(w[e]), prompt)
        start += n
    return out
