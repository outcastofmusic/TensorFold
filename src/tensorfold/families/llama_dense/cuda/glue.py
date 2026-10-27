"""Llama's rotary embedding over whole heads, one program a (row, head): a row's bits never depend on the window."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope(Q, K, POS, INV, QOUT, KOUT, H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr):
    """Heads 0..H-1 are queries, H..H+HKV-1 keys; transformers' rotate-half pairs d with d + D/2."""

    row = tl.program_id(0)
    head = tl.program_id(1)
    half: tl.constexpr = D // 2
    d = tl.arange(0, D)
    low = d < half
    partner = tl.where(low, d + half, d - half)
    is_q = head < H
    if is_q:
        base = (row * H + head) * D
        src = Q
        dst = QOUT
    else:
        base = (row * HKV + head - H) * D
        src = K
        dst = KOUT
    x = tl.load(src + base + d).to(tl.float32)
    xp = tl.load(src + base + partner).to(tl.float32)
    ang = tl.load(POS + row).to(tl.float32) * tl.load(INV + tl.where(low, d, d - half))
    cos = tl.cos(ang)
    sin = tl.sin(ang)
    out = tl.where(low, x * cos - xp * sin, x * cos + xp * sin)
    tl.store(dst + base + d, out.to(tl.bfloat16))


def rope(q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor, *, heads: int, kv_heads: int,
         head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """q (W, heads * D), k (W, kv_heads * D) bf16 rows at int32 positions ``pos`` (W,): rotated (W, H, D), (W, HKV, D)."""

    W = q.shape[0]
    if pos.shape != (W,) or inv_freq.numel() != head_dim // 2:
        raise ValueError("rope takes one position a row and head_dim / 2 frequencies")
    q, k = q.contiguous(), k.contiguous()
    qo = torch.empty((W, heads, head_dim), dtype=torch.bfloat16, device=q.device)
    ko = torch.empty((W, kv_heads, head_dim), dtype=torch.bfloat16, device=q.device)
    _rope[(W, heads + kv_heads)](q, k, pos, inv_freq, qo, ko, H=heads, HKV=kv_heads, D=head_dim, num_warps=2)
    return qo, ko
