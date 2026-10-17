"""The LiLiCorr drafter's own lm_head: the checkpoint's Q8_0 bytes (0.54 GiB) instead of the target's bf16 copy (1 GiB).

A block pass reads the whole head for its top-k candidates, a few rows at a time, so the head's bytes set its cost.
``logits`` dequantizes each 32-weight block as llama.cpp's dequantize_row_q8_0 does (int8 times the fp16 scale in fp32,
then bf16: the bf16 head's values exactly) and sums in fp32 on the tensor cores, in another order than the bf16 head's
kernel. The drafter's logits therefore round differently, which can move a candidate and so acceptance. The verify
window keeps the bf16 head, so every reply token is still the target's own sample.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..quant import Q8_0_BLOCK, Q8_0_BYTES, dequantize_q8_0

BM, BN, BK = 16, 64, 128     # rows, vocabulary rows and inputs a tile; BK spans four Q8_0 blocks


@triton.jit
def _q8_logits(X, W, OUT, n, V, K: tl.constexpr, RB: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr):
    """Program (vocabulary tile, row tile): OUT[r, v] = sum_k X[r, k] * bf16(q[v, k] * d[v, k // 32]), fp32."""

    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    cok, rok = cols < V, rows < n
    w_rows = W + cols.to(tl.int64)[:, None, None] * RB                        # [BN, 1, 1]: a vocabulary row's bytes
    blocks = tl.arange(0, BK // 32)
    within = tl.arange(0, 32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rows.to(tl.int64)[:, None] * K + k0 + tl.arange(0, BK)[None, :], mask=rok[:, None], other=0.0)
        start = (k0 // 32 + blocks) * 34                                      # each block's first byte: its scale
        q = tl.load(w_rows + (start[:, None] + 2 + within[None, :])[None, :, :], mask=cok[:, None, None], other=0)
        lo = tl.load(w_rows + start[None, :, None], mask=cok[:, None, None], other=0)
        hi = tl.load(w_rows + start[None, :, None] + 1, mask=cok[:, None, None], other=0)
        bits = lo.to(tl.uint8, bitcast=True).to(tl.uint16) | (hi.to(tl.uint8, bitcast=True).to(tl.uint16) << 8)
        d = bits.to(tl.float16, bitcast=True).to(tl.float32)                  # [BN, BK // 32, 1]
        w = tl.reshape((q.to(tl.float32) * d).to(tl.bfloat16), (BN, BK))
        acc += tl.dot(x.to(tl.bfloat16), tl.trans(w))
    tl.store(OUT + rows.to(tl.int64)[:, None] * V + cols[None, :], acc, mask=rok[:, None] & cok[None, :])


def logits(h: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    """Rows h [n, D] times the Q8_0 head ``packed`` uint8 [V, D // 32 * 34] -> fp32 logits [n, V]."""

    v, row_bytes = packed.shape
    d = row_bytes // Q8_0_BYTES * Q8_0_BLOCK
    if packed.dtype != torch.uint8 or row_bytes % Q8_0_BYTES or h.shape[-1] != d:
        raise ValueError(f"rows [n, {h.shape[-1]}] and a Q8_0 head {packed.dtype} {tuple(packed.shape)} do not match")
    if not h.is_cuda:
        return h.float() @ dequantize_q8_0(packed).float().T
    if d % BK:
        raise ValueError(f"the Q8_0 head kernel takes widths that are multiples of {BK}, not {d}")
    x = h.to(torch.bfloat16).contiguous()
    out = torch.empty((x.shape[0], v), dtype=torch.float32, device=x.device)
    _q8_logits[(triton.cdiv(v, BN), triton.cdiv(x.shape[0], BM))](x, packed.view(torch.int8), out, x.shape[0], v,
                                                                   K=d, RB=row_bytes, BM=BM, BN=BN, BK=BK,
                                                                   num_warps=4)
    return out


def head_bytes(c) -> int:
    """The packed head's bytes for a model of ``c``'s vocabulary and width."""

    return c.vocab * (c.hidden // Q8_0_BLOCK) * Q8_0_BYTES
