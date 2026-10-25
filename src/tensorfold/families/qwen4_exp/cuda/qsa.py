"""The prompt indexer's scores and selection on prompt chunks: ``qsa_scores.cu`` for `_scores`' bits, and the tiled select from 8,192 blocks.

Served on prompt chunks: 4k 1.001x, 32k 1.034x and 131k 1.038x prefill, replies identical (tf-runs/gpu-qsa).
TF_FLASHNEXT_QSA=torch kept the Triton launches on them until 25 Oct. Decode windows and indexers of other shapes keep
the Triton kernels.

The kernel and the tiled-select threshold are ported from MiaAI-Lab's Qwen3.8 Flash Next recipe, patch 0002
(``fn_qsa_scores.cu``) and patch 0005 (``qsa_tiles_from``), MIT licence, Copyright (c) 2026 MiaAI-Lab; both build on
the TensorFold contributors' ``attention._scores`` and ``attention._select_tiles``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

HEADS, DIM = 4, 128          # the index heads and head dim qsa_scores.cu is written for (both shipped checkpoints)
TILES_FROM = 8192            # prompt rows with this many blocks (32,768 keys) or more run ``_select_tiles``


def fits(index_heads: int, index_dim: int) -> bool:
    """Whether qsa_scores.cu covers this indexer; another shape keeps the Triton ``_scores``."""

    return (index_heads, index_dim) == (HEADS, DIM)


def fast(prefill: bool, index_heads: int, index_dim: int) -> bool:
    """Whether a scratch scores on qsa_scores.cu and selects tiled from TILES_FROM: prompt buffers, for an indexer the
    kernel covers."""

    return prefill and fits(index_heads, index_dim)


def tiled(blocks: int, width: int, fast: bool, regs: int) -> bool:
    """``_select_tiles`` (rather than ``_select``) for rows of up to ``blocks`` blocks: when ``_select``'s ``width``
    (the next power of two from ``blocks``) passes its ``regs`` register width always, and from TILES_FROM blocks on a
    ``fast`` (prompt) scratch. Both list the same blocks."""

    return width > regs or (fast and blocks >= TILES_FROM)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_qwen4_exp_qsa", sources=[str(here / "qsa_scores.cpp"), str(here / "qsa_scores.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def scores(iq: torch.Tensor, pooled: torch.Tensor, pos0: torch.Tensor, out: torch.Tensor, rows: int, blocks: int,
           ratio: int, top: int) -> None:
    """``attention._scores`` over ``blocks`` blocks for rows P0 .. P0 + rows - 1 into ``out`` [rows, NB]: the same
    words at the same positions, nothing else written."""

    _ext().scores(iq, pooled, pos0, out, int(rows), int(blocks), int(ratio), int(top))
