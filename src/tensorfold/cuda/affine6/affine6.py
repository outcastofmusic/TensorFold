"""MLX affine 6-bit projections (groups of 128) on tensor cores, for dense weights a pack stores that way.

``pack`` reorders a checkpoint's codes into the order the kernel's lanes read them, 6 bits a weight as stored and
nothing rounded; ``matmul`` gives each row sums that depend on its own inputs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

BITS, GS = 6, 128


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_affine6_v1", sources=[str(here / "affine6.cpp"), str(here / "affine6.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


@dataclass
class Affine6:
    """A weight [n, k]: lane-ordered words, and scales and biases a group, group-major."""

    words: torch.Tensor       # int32 [n / 16, k / 32, 32, 3]
    scales: torch.Tensor      # bf16 [k / 128, n]
    biases: torch.Tensor      # bf16 [k / 128, n]
    n: int
    k: int

    def nbytes(self) -> int:
        return self.words.numel() * 4 + 2 * self.scales.numel() * 2


@lru_cache(maxsize=2)
def _lane_order(device) -> torch.Tensor:
    """Where a lane's 16 codes of an output tile and 32 inputs sit in that [16 outputs, 32 inputs] block."""

    lane = torch.arange(32).view(32, 1)
    c = torch.arange(16).view(1, 16)
    kt, half, reg, e = c >> 3, (c >> 2) & 1, (c >> 1) & 1, c & 1
    n = lane // 4 + 8 * half
    k = 16 * kt + 2 * (lane % 4) + 8 * reg + e
    return (n * 32 + k).reshape(-1).to(device)


def codes(weight: torch.Tensor, k: int) -> torch.Tensor:
    """The 6-bit code of every weight, uint8 [n, k], from MLX words int32 or uint32 [n, k * 6 / 32]."""

    w = weight.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bit = torch.arange(k, device=w.device) * BITS
    word, shift = bit // 32, bit % 32
    two = w[:, word] | (w[:, (word + 1).clamp(max=w.shape[1] - 1)] << 32)
    return ((two >> shift) & 63).to(torch.uint8)


def pack(weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, rows: int = 8192) -> Affine6:
    """MLX affine 6-bit group-128 tensors (weight [n, k * 6 / 32], scales and biases [n, k / 128]) for the kernel."""

    n, groups = scales.shape
    k = groups * GS
    if tuple(weight.shape) != (n, k * BITS // 32) or tuple(biases.shape) != (n, groups) or n % 16:
        raise ValueError(f"not an affine 6-bit group-128 weight: {tuple(weight.shape)}, scales {tuple(scales.shape)}")
    dev = weight.device
    order = _lane_order(dev)
    shifts = torch.arange(16, device=dev, dtype=torch.int64) * BITS
    words = torch.empty((n // 16, k // 32, 32, 3), dtype=torch.int32, device=dev)
    for start in range(0, n, rows):
        c = codes(weight[start:start + rows], k)
        tiles = c.shape[0] // 16
        block = c.view(tiles, 16, k // 32, 32).permute(0, 2, 1, 3).reshape(tiles, k // 32, 512)
        lanes = block[:, :, order].view(tiles, k // 32, 32, 16).to(torch.int64)
        lo = (lanes[..., :10] << shifts[:10]).sum(-1)             # codes 0..9: bits 0..59
        hi = (lanes[..., 10:] << shifts[:6]).sum(-1)              # codes 10..15: bits 60..95, from bit 0 here
        w0 = lo & 0xFFFFFFFF
        w1 = ((lo >> 32) | (hi << 28)) & 0xFFFFFFFF
        w2 = (hi >> 4) & 0xFFFFFFFF
        out = torch.stack((w0, w1, w2), -1)
        words[start // 16:start // 16 + tiles] = torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32)
    return Affine6(words, scales.to(torch.bfloat16).t().contiguous(), biases.to(torch.bfloat16).t().contiguous(), n, k)


def tiles(q: Affine6) -> tuple[int, int]:
    """(output tiles a block, warps): set by the weight's shape, so every call adds a row's groups in one order."""

    groups = q.k // GS
    warps = 4 if groups % 4 == 0 else 2 if groups % 2 == 0 else 1
    nt = 4 if q.n % 64 == 0 else 2 if q.n % 32 == 0 else 1
    return nt, warps


def matmul(x: torch.Tensor, q: Affine6, out: torch.Tensor | None = None, f32: bool = False) -> torch.Tensor:
    """y [M, n] = x [M, k] @ W.T + the groups' bias terms; x bf16 (rows may be strided), out bf16 or fp32."""

    from tensorfold.cuda.kernels.qmm import group_sums

    if x.ndim != 2 or x.shape[1] != q.k or x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bf16 [M, {q.k}], got {x.dtype} {tuple(x.shape)}")
    if out is None:
        out = torch.empty((x.shape[0], q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    nt, warps = tiles(q)
    _ext().matmul(x, group_sums(x, GS), q.words, q.scales, q.biases, out, nt, warps)
    return out
