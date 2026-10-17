"""LFM2-MoE's routed experts: picks from sigmoid scores, then SwiGLU experts in their stored block FP8 layout.

``pairs`` runs a few rows (decode) in two Triton kernels, a program a (row, pick) pair and a slice of outputs: it reads
the picked experts' e4m3 bytes once, sums each 128-input block in fp32 and scales the block's sum by its stored scale,
so no expert is dequantized and nothing waits on the host. ``grouped`` runs prompt chunks (and every call on the CPU):
it orders the pairs by expert, decodes each used expert to fp32 once a call and multiplies its rows in fp32. Both
round as transformers' bf16 forward does: gate and up to bf16, silu(gate) to bf16, its product with up to bf16, the
down projection to bf16, each pick's output times its weight to bf16; the picks then add in fp32 in pick order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .weights import Experts, Moe

BN = 32             # outputs a program
BK = 128            # inputs a step: one FP8 block's width


def route(x: torch.Tensor, moe: Moe, top_k: int, scaling: float) -> tuple[torch.Tensor, torch.Tensor]:
    """x [R, D] bf16 -> (expert ids [R, k] int64, fp32 weights [R, k]): picks by sigmoid + bias, weights by sigmoid."""

    gates = torch.sigmoid(x.float() @ moe.router.t())
    picks = torch.topk(gates + moe.bias, top_k, dim=-1).indices
    chosen = gates.gather(1, picks)
    return picks, chosen / (chosen.sum(-1, keepdim=True) + 1e-6) * scaling


@triton.jit
def _gate_up(X, IDS, W, S, ACT, x_stride, TOPK: tl.constexpr, D: tl.constexpr, I: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr, FP8: tl.constexpr):
    """Program (pair, output block): ACT[pair, n] = bf16(bf16(silu(gate)) * up) over expert IDS[pair]'s w1 and w3."""

    p = tl.program_id(0)
    nb = tl.program_id(1)
    r = p // TOPK
    e = tl.load(IDS + p).to(tl.int64)
    n = nb * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate_rows = (e * 2 * I + n).to(tl.int64)
    up_rows = gate_rows + I
    acc_g = tl.zeros((BN,), tl.float32)
    acc_u = tl.zeros((BN,), tl.float32)
    for k0 in range(0, D, BK):
        x = tl.load(X + r * x_stride + k0 + kk).to(tl.float32)
        wg = tl.load(W + gate_rows[:, None] * D + k0 + kk[None, :])
        wu = tl.load(W + up_rows[:, None] * D + k0 + kk[None, :])
        if FP8:
            sg = tl.load(S + (e * (2 * I // BK) + (nb * BN) // BK) * (D // BK) + k0 // BK)
            su = tl.load(S + (e * (2 * I // BK) + (I + nb * BN) // BK) * (D // BK) + k0 // BK)
            pg = tl.sum(wg.to(tl.float8e4nv, bitcast=True).to(tl.float32) * x[None, :], axis=1)
            pu = tl.sum(wu.to(tl.float8e4nv, bitcast=True).to(tl.float32) * x[None, :], axis=1)
            acc_g += pg * sg
            acc_u += pu * su
        else:
            acc_g += tl.sum(wg.to(tl.float32) * x[None, :], axis=1)
            acc_u += tl.sum(wu.to(tl.float32) * x[None, :], axis=1)
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    silu = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(ACT + p.to(tl.int64) * I + n, (silu * u).to(tl.bfloat16))


@triton.jit
def _down(ACT, IDS, W, S, OUT, D: tl.constexpr, I: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          FP8: tl.constexpr):
    """Program (pair, output block): OUT[pair, n] = bf16(ACT[pair] . w2[IDS[pair], n])."""

    p = tl.program_id(0)
    nb = tl.program_id(1)
    e = tl.load(IDS + p).to(tl.int64)
    n = nb * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    rows = (e * D + n).to(tl.int64)
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, I, BK):
        a = tl.load(ACT + p.to(tl.int64) * I + k0 + kk).to(tl.float32)
        w = tl.load(W + rows[:, None] * I + k0 + kk[None, :])
        if FP8:
            s = tl.load(S + (e * (D // BK) + (nb * BN) // BK) * (I // BK) + k0 // BK)
            acc += tl.sum(w.to(tl.float8e4nv, bitcast=True).to(tl.float32) * a[None, :], axis=1) * s
        else:
            acc += tl.sum(w.to(tl.float32) * a[None, :], axis=1)
    tl.store(OUT + p.to(tl.int64) * D + n, acc.to(tl.bfloat16))


def pairs(x: torch.Tensor, ids: torch.Tensor, ex: Experts) -> torch.Tensor:
    """x [R, D] bf16, ids [R, k] -> each (row, pick)'s expert output [R * k, D] bf16, on the Triton kernels."""

    rows, top_k = ids.shape
    d, width = x.shape[1], ex.width
    if d % BK or width % BK:
        raise ValueError(f"experts of width {width} over {d} inputs are not whole {BK}-input blocks")
    flat = ids.reshape(-1).to(torch.int32).contiguous()
    act = torch.empty((rows * top_k, width), dtype=torch.bfloat16, device=x.device)
    out = torch.empty((rows * top_k, d), dtype=torch.bfloat16, device=x.device)
    fp8 = ex.fp8
    s13 = ex.s13 if fp8 else act                    # unread without FP8
    s2 = ex.s2 if fp8 else act
    if x.stride(-1) != 1:
        x = x.contiguous()
    _gate_up[(rows * top_k, width // BN)](x, flat, ex.w13, s13, act, x.stride(0), TOPK=top_k, D=d, I=width, BN=BN,
                                          BK=BK, FP8=fp8, num_warps=4)
    _down[(rows * top_k, d // BN)](act, flat, ex.w2, s2, out, D=d, I=width, BN=BN, BK=BK, FP8=fp8, num_warps=4)
    return out


def grouped(x: torch.Tensor, ids: torch.Tensor, ex: Experts) -> torch.Tensor:
    """``pairs``' result by expert: each used expert decoded to fp32 once and its rows multiplied in fp32."""

    rows, top_k = ids.shape
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=ex.count).tolist()          # the one host sync a call
    out = torch.empty((rows * top_k, x.shape[1]), dtype=torch.bfloat16, device=x.device)
    start = 0
    for e, n in enumerate(counts):
        if not n:
            continue
        slots = order[start:start + n]
        start += n
        w13, w2 = ex.dense(e)
        h = x[slots // top_k].float() @ w13.t()
        g, u = h.to(torch.bfloat16).float().chunk(2, dim=-1)
        act = (torch.nn.functional.silu(g).to(torch.bfloat16).float() * u).to(torch.bfloat16)
        out[slots] = (act.float() @ w2.t()).to(torch.bfloat16)
    return out


def combine(outs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """[R * k, D] expert outputs and [R, k] weights -> [R, D] bf16: bf16(out * weight) added in fp32, pick order."""

    rows, top_k = weights.shape
    y = outs.view(rows, top_k, -1)
    w = weights.to(torch.bfloat16)
    acc = (y[:, 0] * w[:, 0:1]).float()
    for j in range(1, top_k):
        acc = acc + (y[:, j] * w[:, j:j + 1]).float()
    return acc.to(torch.bfloat16)


def moe(x: torch.Tensor, m: Moe, top_k: int, scaling: float, *, few: int) -> torch.Tensor:
    """The MoE block's output [R, D] bf16; up to ``few`` rows on CUDA take ``pairs``, the rest ``grouped``."""

    ids, weights = route(x, m, top_k, scaling)
    run = pairs if x.is_cuda and x.shape[0] <= few else grouped
    return combine(run(x, ids, m.experts), weights)
