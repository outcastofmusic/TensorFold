"""LFM2-MoE's routed experts: picks from sigmoid scores, then SwiGLU experts in their stored block FP8 layout.

Three paths compute the same products and differ only in summation order:

- ``pairs`` runs a few rows (decode) in two Triton kernels, a program a (row, pick) pair and a slice of outputs: it
  reads the picked experts' e4m3 bytes, sums each 128-input block in fp32 and scales the block's sum by its stored
  scale, so no expert is dequantized and nothing waits on the host. Each pair reads its expert in full.
- ``dot`` runs prompt chunks on the tensor cores: the pairs sorted by expert in tiles of up to ``BM`` rows (a schedule
  built on the device, ``schedule``), each tile reading its expert's e4m3 bytes once as bf16 (exact) and summing each
  128-input block with ``tl.dot`` in fp32 before the block's scale. A used expert's bytes are read about once a chunk.
- ``grouped`` decodes each used expert's linears to fp32 in turn into one reused buffer (a byte read, a float written:
  ``dense``) and multiplies its rows in fp32: the CPU's path, and the prompt path ``TF_LFM2_PROMPT_KERNEL=fp32`` keeps
  as the reference.

All three round as transformers' bf16 forward does: gate and up to bf16, silu(gate) to bf16, its product with up to
bf16, the down projection to bf16, each pick's output times its weight to bf16; the picks then add in fp32 in pick
order.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .weights import Experts, Moe

BN = 32             # outputs a program
BK = 128            # inputs a step: one FP8 block's width
BR = 32             # rows a ``_decode`` program writes, within one 128-row block
DOT_BN = 64         # outputs a ``dot`` program: within one 128-row scale block
KERNELS = ("auto", "pairs", "dot", "fp32")       # TF_LFM2_PROMPT_KERNEL's values


def route(x: torch.Tensor, moe: Moe, top_k: int, scaling: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    """x [R, D] bf16 -> (expert ids [R, k] int64, fp32 weights [R, k]): picks by sigmoid + bias, weights by sigmoid.
    A ``scaling`` of None skips the multiply, which a scaling of 1.0 makes without changing a value."""

    gates = torch.sigmoid(x.float() @ moe.router.t())
    picks = torch.topk(gates + moe.bias, top_k, dim=-1).indices
    chosen = gates.gather(1, picks)
    weights = chosen / (chosen.sum(-1, keepdim=True) + 1e-6)
    return picks, weights if scaling is None else weights * scaling


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


@triton.jit
def _decode(W, S, OUT, K: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr):
    """Program (BR rows, one 128-input block): OUT = fp32(e4m3 W) * the block's scale, rounded once as in
    ``dequantize``."""

    rows = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    kb = tl.program_id(1)
    cols = kb * BK + tl.arange(0, BK)
    s = tl.load(S + (tl.program_id(0) * BR // BK) * (K // BK) + kb)
    w = tl.load(W + rows[:, None] * K + cols[None, :])
    tl.store(OUT + rows[:, None] * K + cols[None, :], w.to(tl.float8e4nv, bitcast=True).to(tl.float32) * s)


def decode_blocks(w: torch.Tensor, s: torch.Tensor, out: torch.Tensor) -> None:
    """``_decode`` over e4m3 bytes [N, K] and fp32 block scales [N / 128, K / 128] into ``out`` [N, K] fp32."""

    n, k = w.shape
    _decode[(n // BR, k // BK)](w, s, out, K=k, BR=BR, BK=BK, num_warps=4)


def dense(w: torch.Tensor, s: torch.Tensor | None, out: torch.Tensor | None = None) -> torch.Tensor:
    """An expert linear as stored, [N, K] e4m3 bytes with fp32 block scales or bf16 (``s`` None), in fp32: the bits
    ``checkpoint.dequantize`` gives, written to the first N * K floats of ``out`` when given."""

    n, k = w.shape
    out = torch.empty((n, k), dtype=torch.float32, device=w.device) if out is None else out[:n * k].view(n, k)
    if s is None:
        return out.copy_(w)
    if n % BK or k % BK:
        raise ValueError(f"an expert linear [{n}, {k}] is not whole {BK}x{BK} FP8 blocks")
    if w.is_cuda:
        decode_blocks(w, s, out)
        return out
    blocks = out.view(n // BK, BK, k // BK, BK)
    blocks.copy_(w.view(torch.float8_e4m3fn).view(blocks.shape))
    blocks.mul_(s[:, None, :, None])
    return out


def pairs(x: torch.Tensor, ids: torch.Tensor, ex: Experts, fuse: bool = False) -> torch.Tensor:
    """x [R, D] bf16, ids [R, k] -> each (row, pick)'s expert output [R * k, D] bf16, on the Triton kernels.
    ``fuse`` hands the kernels route's int64 ids as they are, without an int32 copy."""

    rows, top_k = ids.shape
    d, width = x.shape[1], ex.width
    if d % BK or width % BK:
        raise ValueError(f"experts of width {width} over {d} inputs are not whole {BK}-input blocks")
    flat = ids.reshape(-1).contiguous() if fuse else ids.reshape(-1).to(torch.int32).contiguous()
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


@triton.jit
def _mma(a, b, WIDE: tl.constexpr):
    """a @ b in fp32 sums; ``WIDE`` (Triton's interpreter, whose tl.dot multiplies bf16 operands' raw bits) widens
    the operands to fp32 first, which holds every bf16 and e4m3 value exactly."""

    if WIDE:
        return tl.dot(a.to(tl.float32), b.to(tl.float32))
    return tl.dot(a, b)


@triton.jit
def _dot_gate_up(X, ORDER, TILE_E, TILE_START, TILE_STOP, W, S, ACT, x_stride, TOPK: tl.constexpr,
                 D: tl.constexpr, I: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 FP8: tl.constexpr, WIDE: tl.constexpr):
    """Program (tile, output block): ``_gate_up``'s ACT rows for a tile of up to BM pairs of one expert, on tl.dot."""

    t = tl.program_id(0)
    nb = tl.program_id(1)
    e = tl.load(TILE_E + t).to(tl.int64)
    if e < 0:                                   # past the chunk's tiles: the grid is sized without a host sync
        return
    m = tl.load(TILE_START + t) + tl.arange(0, BM)
    valid = m < tl.load(TILE_STOP + t)
    p = tl.load(ORDER + m, mask=valid, other=0).to(tl.int64)
    r = p // TOPK
    n = nb * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate_rows = e * 2 * I + n
    up_rows = gate_rows + I
    acc_g = tl.zeros((BM, BN), tl.float32)
    acc_u = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        x = tl.load(X + r[:, None] * x_stride + k0 + kk[None, :], mask=valid[:, None], other=0.0)
        wg = tl.load(W + gate_rows[None, :] * D + k0 + kk[:, None])          # [BK, BN]
        wu = tl.load(W + up_rows[None, :] * D + k0 + kk[:, None])
        if FP8:
            sg = tl.load(S + (e * (2 * I // BK) + (nb * BN) // BK) * (D // BK) + k0 // BK)
            su = tl.load(S + (e * (2 * I // BK) + (I + nb * BN) // BK) * (D // BK) + k0 // BK)
            # e4m3 -> bf16 is exact; each 128-input block sums in fp32 before its scale, as in ``_gate_up``
            acc_g += _mma(x, wg.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16), WIDE) * sg
            acc_u += _mma(x, wu.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16), WIDE) * su
        else:
            acc_g += _mma(x, wg, WIDE)
            acc_u += _mma(x, wu, WIDE)
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    silu = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(ACT + p[:, None] * I + n[None, :], (silu * u).to(tl.bfloat16), mask=valid[:, None])


@triton.jit
def _dot_down(ACT, ORDER, TILE_E, TILE_START, TILE_STOP, W, S, OUT, D: tl.constexpr, I: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FP8: tl.constexpr, WIDE: tl.constexpr):
    """Program (tile, output block): ``_down``'s OUT rows for a tile of up to BM pairs of one expert, on tl.dot."""

    t = tl.program_id(0)
    nb = tl.program_id(1)
    e = tl.load(TILE_E + t).to(tl.int64)
    if e < 0:
        return
    m = tl.load(TILE_START + t) + tl.arange(0, BM)
    valid = m < tl.load(TILE_STOP + t)
    p = tl.load(ORDER + m, mask=valid, other=0).to(tl.int64)
    n = nb * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    rows = e * D + n
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, I, BK):
        a = tl.load(ACT + p[:, None] * I + k0 + kk[None, :], mask=valid[:, None], other=0.0)
        w = tl.load(W + rows[None, :] * I + k0 + kk[:, None])
        if FP8:
            s = tl.load(S + (e * (D // BK) + (nb * BN) // BK) * (I // BK) + k0 // BK)
            acc += _mma(a, w.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16), WIDE) * s
        else:
            acc += _mma(a, w, WIDE)
    tl.store(OUT + p[:, None] * D + n[None, :], acc.to(tl.bfloat16), mask=valid[:, None])


def tile_rows(pairs: int, experts: int) -> int:
    """Rows a ``dot`` tile takes: the pairs an expert gets on average, as 16, 32 or 64 (tl.dot's shapes)."""

    mean = pairs / max(1, experts)
    return 16 if mean <= 16 else 32 if mean <= 32 else 64


def schedule(flat: torch.Tensor, experts: int, bm: int) -> tuple[torch.Tensor, ...]:
    """Tiles of up to ``bm`` pairs of one expert, built on the device with no host sync.

    Returns (order, expert, start, stop): ``order`` the pair indices sorted by expert (stable), and for each of
    ``len(flat) // bm + experts`` tiles (a bound on the tiles any picks need) its expert (-1 past the last tile) and
    its slots ``order[start:stop]``.
    """

    flat = flat.long()
    counts = torch.zeros(experts, dtype=torch.long, device=flat.device).scatter_add_(0, flat, torch.ones_like(flat))
    tiles = (counts + bm - 1) // bm
    tile_end = tiles.cumsum(0)
    seg_end = counts.cumsum(0)
    t = torch.arange(len(flat) // bm + experts, device=flat.device)
    owner = torch.searchsorted(tile_end, t, right=True)
    e = owner.clamp(max=experts - 1)
    start = seg_end[e] - counts[e] + (t - (tile_end[e] - tiles[e])) * bm
    stop = torch.minimum(seg_end[e], start + bm)
    expert = torch.where(owner < experts, owner, torch.full_like(owner, -1)).to(torch.int32)
    order = torch.argsort(flat, stable=True).to(torch.int32)
    return order, expert, start.to(torch.int32), stop.to(torch.int32)


def dot(x: torch.Tensor, ids: torch.Tensor, ex: Experts, bm: int = 0) -> torch.Tensor:
    """``pairs``' result on the tensor cores, a tile of up to ``bm`` pairs of one expert a program (``tile_rows``
    when 0): x [R, D] bf16, ids [R, k] -> [R * k, D] bf16."""

    rows, top_k = ids.shape
    d, width = x.shape[1], ex.width
    if d % BK or width % BK:
        raise ValueError(f"experts of width {width} over {d} inputs are not whole {BK}-input blocks")
    bm = bm or tile_rows(rows * top_k, ex.count)
    order, expert, start, stop = schedule(ids.reshape(-1), ex.count, bm)
    act = torch.empty((rows * top_k, width), dtype=torch.bfloat16, device=x.device)
    out = torch.empty((rows * top_k, d), dtype=torch.bfloat16, device=x.device)
    fp8 = ex.fp8
    s13 = ex.s13 if fp8 else act                    # unread without FP8
    s2 = ex.s2 if fp8 else act
    if x.stride(-1) != 1:
        x = x.contiguous()
    tiles = expert.shape[0]
    wide = os.environ.get("TRITON_INTERPRET") == "1"
    _dot_gate_up[(tiles, width // DOT_BN)](x, order, expert, start, stop, ex.w13, s13, act, x.stride(0), TOPK=top_k,
                                           D=d, I=width, BM=bm, BN=DOT_BN, BK=BK, FP8=fp8, WIDE=wide, num_warps=4,
                                           num_stages=2)
    _dot_down[(tiles, d // DOT_BN)](act, order, expert, start, stop, ex.w2, s2, out, D=d, I=width, BM=bm,
                                    BN=DOT_BN, BK=BK, FP8=fp8, WIDE=wide, num_warps=4, num_stages=2)
    return out


def grouped(x: torch.Tensor, ids: torch.Tensor, ex: Experts, work: torch.Tensor | None = None) -> torch.Tensor:
    """``pairs``' result by expert: each used expert's w13, then its w2, decoded to fp32 in ``work`` (at least
    ``ex.w13[0].numel()`` floats; made when None) and its rows multiplied in fp32."""

    rows, top_k = ids.shape
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=ex.count).tolist()          # the one host sync a call
    out = torch.empty((rows * top_k, x.shape[1]), dtype=torch.bfloat16, device=x.device)
    if work is None:
        work = torch.empty(ex.w13[0].numel(), dtype=torch.float32, device=x.device)
    start = 0
    for e, n in enumerate(counts):
        if not n:
            continue
        slots = order[start:start + n]
        start += n
        h = x[slots // top_k].float() @ dense(ex.w13[e], ex.s13[e] if ex.fp8 else None, work).t()
        g, u = h.to(torch.bfloat16).float().chunk(2, dim=-1)
        act = (torch.nn.functional.silu(g).to(torch.bfloat16).float() * u).to(torch.bfloat16)
        out[slots] = (act.float() @ dense(ex.w2[e], ex.s2[e] if ex.fp8 else None, work).t()).to(torch.bfloat16)
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


def path(kernel: str, rows: int, few: int, pairs_rows: int) -> str:
    """The expert path a CUDA call of ``rows`` rows takes under TF_LFM2_PROMPT_KERNEL=``kernel``: up to ``few`` rows
    (decode) always ``pairs``; ``auto`` keeps ``pairs`` up to ``pairs_rows`` rows, then ``dot``; ``fp32`` runs the
    reference ``grouped``."""

    if kernel not in KERNELS:
        raise ValueError(f"TF_LFM2_PROMPT_KERNEL={kernel!r}: one of {', '.join(KERNELS)}")
    if rows <= few:
        return "pairs"
    if kernel == "auto":
        return "pairs" if rows <= pairs_rows else "dot"
    return "grouped" if kernel == "fp32" else kernel


def moe(x: torch.Tensor, m: Moe, top_k: int, scaling: float, *, few: int, work: torch.Tensor | None = None,
        kernel: str = "auto", pairs_rows: int = 0, fuse: bool = False) -> torch.Tensor:
    """The MoE block's output [R, D] bf16 on ``path``'s kernels (on the CPU always ``grouped``, in ``work``).
    ``fuse`` (TF_LFM2_DECODE_FUSE, CUDA) takes the ``pairs`` path's decode rows with ``fused.combine``: same bits,
    one launch for combine's 3k + 1."""

    ids, weights = route(x, m, top_k, None if fuse and scaling == 1.0 else scaling)
    way = path(kernel, x.shape[0], few, pairs_rows) if x.is_cuda else "grouped"
    if way == "pairs" and fuse:
        from . import fused

        return fused.combine(pairs(x, ids, m.experts, fuse=True), weights)
    if way == "pairs":
        return combine(pairs(x, ids, m.experts), weights)
    if way == "dot":
        return combine(dot(x, ids, m.experts), weights)
    return combine(grouped(x, ids, m.experts, work), weights)
