"""One-kernel forms of the decode step's elementwise chains (``TF_LFM2_DECODE_FUSE``), each giving the torch chain's bits.

A one-row decode step launched about 2,040 kernels, and about 1,000 of them were bf16 or fp32 elementwise ops on 2,048
values: the conv's 15 after its in_proj and the MoE combine's 13 (top 4). Each cost a launch and a near-empty kernel.
``conv_step`` and ``combine`` run those chains as one kernel each and round where torch's kernels round:

- a bf16 x bf16 product is exact in fp32, so it rounds once, to bf16, as torch's bf16 ``mul`` does;
- an fp32 product and an fp32 sum each round to fp32, as two torch kernels do. Triton fuses a multiply and the add
  that follows it into an fma by default, which rounds once, so these kernels launch with ``enable_fp_fusion=False``
  (tests/test_lfm2_moe_decode_fuse.py checks the PTX for sm_121 has no fma);
- fp32 to bf16 rounds to nearest even in both.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BD = 256            # channels a program
TAPS = 3            # LFM2's conv_L_cache: the state holds the last two products B * x


@triton.jit
def _conv_step(BCX, STATE, TAPS_W, Y, D: tl.constexpr, BD: tl.constexpr):
    """One row of the gated short conv over channels c: STATE [2, D] bf16 (the last two B * x, oldest first) and
    BCX [1, 3D] bf16 (B, C, x) -> Y = bf16(C * bf16(conv)), and STATE moves on by B * x."""

    c = tl.program_id(0) * BD + tl.arange(0, BD)
    b = tl.load(BCX + c).to(tl.float32)
    gate = tl.load(BCX + D + c).to(tl.float32)
    v = tl.load(BCX + 2 * D + c).to(tl.float32)
    s0 = tl.load(STATE + c)
    s1 = tl.load(STATE + D + c)
    bv = (b * v).to(tl.bfloat16)                               # torch: b * v in bf16
    t0 = tl.load(TAPS_W + c * 3)
    t1 = tl.load(TAPS_W + c * 3 + 1)
    t2 = tl.load(TAPS_W + c * 3 + 2)
    acc = s0.to(tl.float32) * t0                               # torch: seq[0].float() * taps[:, 0], then + each tap
    acc = acc + s1.to(tl.float32) * t1
    acc = acc + bv.to(tl.float32) * t2
    tl.store(STATE + c, s1)
    tl.store(STATE + D + c, bv)
    tl.store(Y + c, (gate * acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16))


def conv_step(bcx: torch.Tensor, state: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """bcx [1, 3D] bf16 (in_proj's B, C, x), state [2, D] bf16 (updated in place), taps [D, 3] fp32 -> [1, D] bf16."""

    d = state.shape[1]
    y = torch.empty((1, d), dtype=torch.bfloat16, device=bcx.device)
    _conv_step[(triton.cdiv(d, BD),)](bcx, state, taps, y, D=d, BD=BD, num_warps=2, enable_fp_fusion=False)
    return y


def conv_fits(rows: int, state: torch.Tensor, taps: torch.Tensor) -> bool:
    """``conv_step`` takes one row of a 3-tap conv over whole BD-channel blocks, contiguous."""

    return (rows == 1 and taps.shape[1] == TAPS and state.shape[1] % BD == 0 and state.is_contiguous()
            and taps.is_contiguous())


def combine_fits(d: int) -> bool:
    """``combine`` takes whole BD-channel blocks: its loads and stores have no mask."""

    return d % BD == 0


@triton.jit
def _combine(OUTS, WEIGHTS, Y, K: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    """Row r, channels c: Y = bf16(sum over picks j in order of fp32(bf16(OUTS[r K + j] * bf16(WEIGHTS[r, j]))))."""

    r = tl.program_id(0)
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for j in tl.static_range(K):
        w = tl.load(WEIGHTS + r * K + j).to(tl.bfloat16).to(tl.float32)
        y = tl.load(OUTS + (r * K + j).to(tl.int64) * D + c).to(tl.float32)
        part = (y * w).to(tl.bfloat16).to(tl.float32)
        if j == 0:
            acc = part                                         # torch starts from the first pick, not from 0.0
        else:
            acc = acc + part
    tl.store(Y + r.to(tl.int64) * D + c, acc.to(tl.bfloat16))


def combine(outs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``experts.combine`` in one kernel: [R * k, D] bf16 outputs and [R, k] fp32 weights -> [R, D] bf16."""

    rows, top_k = weights.shape
    d = outs.shape[1]
    y = torch.empty((rows, d), dtype=torch.bfloat16, device=outs.device)
    _combine[(rows, triton.cdiv(d, BD))](outs, weights.contiguous(), y, K=top_k, D=d, BD=BD, num_warps=2,
                                         enable_fp_fusion=False)
    return y
