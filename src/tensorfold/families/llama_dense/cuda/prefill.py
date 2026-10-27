"""Llama prompt chunks: the bits never depend on the chunk bounds, and the KV cache is the whole state.

So the state after prompt[:p] is the state after the whole prompt with ``pos`` set back to p: a prefill keeps
states at any points by cloning, with no extra pass. Prompt rows take the prompt kernels (``b16.prompt``, the
EXL3 and NVFP4 prompt GEMMs, ``prefill_attention``), whose bits differ from a decode window's, so a kept state
never covers decoded rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tensorfold.cuda.kernels.prefill_attention import attention
from tensorfold.families.qwen3_5.cuda import glue as qglue
from tensorfold.families.qwen3_5.cuda import prefill_bf16
from tensorfold.families.qwen3_5.cuda.decode import clone_state
from tensorfold.families.qwen3_5.cuda.forward import State, grow
from tensorfold.families.qwen3_5.cuda.prefill import _mlp, _mm, _mm_group, chunks

from . import glue
from .weights import Weights


@torch.no_grad()
def prefill_chunk(w: Weights, tokens: torch.Tensor, st: State, *, every: bool = False) -> torch.Tensor:
    """Commit ``tokens`` at [st.pos, st.pos + W) into ``st``; the last row's final normed state (``every``: all rows')."""

    c = w.config
    W = int(tokens.shape[0])
    p0 = st.pos
    dev = tokens.device
    pos = torch.arange(p0, p0 + W, device=dev, dtype=torch.int32)
    x = qglue.embedding(tokens.to(torch.int32), w.embed)
    pending: torch.Tensor | None = None
    scale = c.head_dim ** -0.5
    for i, layer in enumerate(w.layers):
        x, h = prefill_bf16.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        a = layer.attn
        q, key, value = _mm_group(h, [a.q, a.k, a.v])
        q, key = glue.rope(q, key, pos, w.inv_freq, heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim)
        kbuf, vbuf = grow(st, i, p0 + W)
        kbuf[p0:p0 + W] = key
        vbuf[p0:p0 + W] = value.view(W, c.kv_heads, c.head_dim)
        out = attention(q, kbuf, vbuf, p0, scale=scale)
        r = _mm(out.view(W, c.heads * c.head_dim), a.o)
        x, h = prefill_bf16.add_rmsnorm(x, r, layer.post_norm, c.eps)
        pending = _mlp(h, layer, prefill_bf16, False)
    if pending is None:
        raise ValueError("a model needs at least one layer")
    st.pos = p0 + W
    if not every:
        x, pending = x[-1:].contiguous(), pending[-1:].contiguous()
    _, normed, _ = qglue.add_rmsnorm(x, pending, w.norm, c.eps)
    return normed


@torch.no_grad()
def prefill_state(w: Weights, prompt: Sequence[int], st: State, *, size: int | None = None, every: bool = False,
                  keep: Sequence[int] = ()) -> tuple[torch.Tensor, list[State]]:
    """Commit prompt[st.pos:] into ``st`` in even chunks; the last row's normed state (``every``: each new row's) and a
    clone of the state after prompt[:p] for each ``p`` in ``keep`` (all within [st.pos, len(prompt)])."""

    base, n = st.pos, len(prompt)
    if not base < n:
        raise ValueError(f"nothing to prefill: the state is at {base} of a {n}-token prompt")
    if any(not base <= p <= n for p in keep):
        raise ValueError(f"kept points {list(keep)} must lie in the prefilled range [{base}, {n}]")
    ids = torch.tensor(list(prompt[base:]), dtype=torch.int32, device=w.norm.device)
    pieces = [prefill_chunk(w, ids[a - base:b - base], st, every=every)
              for a, b in chunks(base, n, size or w.prompt_rows)]
    kept = []
    for p in keep:
        clone = clone_state(st)          # shares the attention buffers: rows below p are this prompt's
        clone.pos = p
        kept.append(clone)
    return (torch.cat(pieces) if every else pieces[-1]), kept
