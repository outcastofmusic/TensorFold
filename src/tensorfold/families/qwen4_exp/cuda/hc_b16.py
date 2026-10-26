"""The bf16 hyper-connection's prompt fusions (an NVFP4 checkpoint such as FP4-Local).

Prompt chunks of more than ``forward.FUSED_ROWS`` rows run three fusions with the old kernels' bytes: the write-back
and the norm in one kernel (``hc_check.fuser``, self-checked), the down projection's K slices summed inside each
program (``bf16.matmul(inner_k=True)``), and the up projection with the stream mix (``up_mix``). An EXL3 pack's fp16
HC takes the first of them, which reads no matrix. FP4-Local 12k prefill went from 1,856 to 2,228 tok/s (1.14-1.20x)
and 32k 1.08-1.17x with replies identical 4/4 (tf-runs/gpu-hc); TF_FLASHNEXT_HC_FUSE turned them off until 25 Oct.

``_hc_up_mix`` is ported from MiaAI-Lab's Qwen3.8 Flash Next recipe, patch 0008 ``prompt_mm._hc_up_mix`` (MIT
licence), which builds on the TensorFold contributors' ``bf16._b16mm`` and ``glue._hc_mix``.
"""

from __future__ import annotations

import torch

HAS_TRITON = True
try:
    import triton
    import triton.language as tl

    from .glue import _bsig
except ModuleNotFoundError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _hc_up_mix(ACT, W, NORMED, MIXED, M, D: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
        """Each stream's up = bf16(act @ W[s D + d].T) as ``bf16._b16mm`` computes it at one K slice (BK steps in
        order from a zero accumulator, one bf16 rounding), then ``glue._hc_mix``'s expression in stream order:
        mixed = bf16((sum_s bf16(bsig(up_s) * normed_s)) / S). The up rows are never stored, and the 32-group sums
        of mixed are not made: no reader of a layer's bf16 read-out takes them (``forward.hc_block``)."""

        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rd = tl.program_id(1) * BD + tl.arange(0, BD)
        rk = tl.arange(0, BK)
        m_ok = rm < M
        total = tl.zeros((BM, BD), dtype=tl.float32)
        for s in tl.static_range(S):
            acc = tl.zeros((BM, BD), dtype=tl.float32)
            for i in range(K // BK):
                x = tl.load(ACT + rm[:, None] * K + (i * BK + rk)[None, :], mask=m_ok[:, None], other=0.0)
                w = tl.load(W + (s * D + rd)[:, None] * K + (i * BK + rk)[None, :])
                acc = tl.dot(x, tl.trans(w), acc)
            u = acc.to(tl.bfloat16).to(tl.float32)
            n = tl.load(NORMED + rm[:, None] * (S * D) + s * D + rd[None, :], mask=m_ok[:, None],
                        other=0.0).to(tl.float32)
            total += (_bsig(u) * n).to(tl.bfloat16).to(tl.float32)
        m = (total / S).to(tl.bfloat16)
        tl.store(MIXED + rm[:, None] * D + rd[None, :], m, mask=m_ok[:, None])


def up_mix(act: torch.Tensor, up, normed: torch.Tensor, mixed: torch.Tensor, streams: int, *,
           bm: int = 64, bd: int = 64, bk: int = 64, num_warps: int = 4, num_stages: int = 3) -> None:
    """mixed [M, D] from act [M, low] and the up matrix ``up`` (``B16``-like: weight [S*D, low], n, k): the bytes of
    ``bf16.matmul(act, up)`` then ``glue.hc_mix``, without the [M, S*D] up rows; ``bk`` is ``bf16.BK``."""

    if not HAS_TRITON:
        raise RuntimeError("the HC up-mix needs Triton (the CUDA engine's environment)")
    rows, k = act.shape
    d = up.n // streams
    if k != up.k or k % bk or up.n % streams or d % bd:
        raise ValueError(f"HC up-mix: act {tuple(act.shape)}, up [{up.n}, {up.k}] and {streams} streams do not tile "
                         f"by K {bk} and D {bd}")
    if not (act.is_contiguous() and normed.is_contiguous() and mixed.is_contiguous()):
        raise ValueError("HC up-mix: act, normed and mixed must be contiguous")
    if normed.shape != (rows, up.n) or mixed.shape != (rows, d):
        raise ValueError(f"HC up-mix: normed {tuple(normed.shape)} or mixed {tuple(mixed.shape)} is not "
                         f"({rows}, {up.n}) and ({rows}, {d})")
    _hc_up_mix[(triton.cdiv(rows, bm), d // bd)](act, up.weight, normed, mixed, rows, D=d, S=streams, K=k, BM=bm,
                                                 BD=bd, BK=bk, num_warps=num_warps, num_stages=num_stages)
