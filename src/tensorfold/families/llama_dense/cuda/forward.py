"""A Llama verify window over committed keys, the state read-only until ``commit``; serial decode is a one-row window.

Every kernel here reads its own row: qwen3_5's norm and SwiGLU glue, ``glue.rope``, the shared tree attention, and
projections whose sums never depend on the row count (``b16`` for bf16 weights, the EXL3 linear, the NVFP4 / FP8
lane matmuls). The state is qwen3_5's ``State`` with attention layers only, so its ``commit`` and ``grow`` apply.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tensorfold.cuda.kernels import attention as tree_attention
from tensorfold.families.qwen3_5.cuda import glue as qglue
from tensorfold.families.qwen3_5.cuda.forward import AttentionRecord, State, _cache_offsets, _mm, _mm_group, _paths

from . import glue
from .weights import Weights

__all__ = ["State", "new_state", "tree_forward"]


def new_state(w: Weights) -> State:
    """qwen3_5's ``State`` with every layer attention: empty key and value caches that grow as rows commit."""

    c, dev = w.config, w.norm.device
    st = object.__new__(State)                   # its constructor reads qwen3_5's layer kinds; these are all attention
    st.pos, st.limit = 0, 0
    st.conv, st.rec = [None] * c.layers, [None] * c.layers
    st.kv = [(torch.empty((0, c.kv_heads, c.head_dim), device=dev, dtype=torch.bfloat16),
              torch.empty((0, c.kv_heads, c.head_dim), device=dev, dtype=torch.bfloat16)) for _ in range(c.layers)]
    return st


@torch.no_grad()
def tree_forward(w: Weights, tokens: torch.Tensor, parents: Sequence[int], st: State, *, hidden: bool = False,
                 taps: Sequence[int] = ()):
    """Logits of every window row and the layers' records; a row sees its root-to-row path and the committed keys.
    ``hidden`` adds the final normed rows, ``taps`` the residuals after those layers side by side (a drafter's input)."""

    c = w.config
    parents = [int(p) for p in parents]
    W = len(parents)
    depths, _ = _paths(parents)
    if tokens.shape != (W,) or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("tokens must be a 1-D int tensor matching parents")
    dev = w.norm.device
    pos = torch.tensor([st.pos + d for d in depths], dtype=torch.int32).pin_memory().to(dev, non_blocking=True)
    aplan = tree_attention.plan([parents], [st.pos], c.heads // c.kv_heads, dev)
    aoffs = _cache_offsets([st], list(range(c.layers)), dev)
    x = qglue.embedding(tokens.to(torch.int32), w.embed)
    pending: torch.Tensor | None = None
    record: list[AttentionRecord] = []
    tapped: list[torch.Tensor] = []
    scale = c.head_dim ** -0.5
    for i, layer in enumerate(w.layers):
        x, h, _ = qglue.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        a = layer.attn
        q, key, value = _mm_group(h, [a.q, a.k, a.v])
        q, key = glue.rope(q, key, pos, w.inv_freq, heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim)
        value = value.reshape(W, c.kv_heads, c.head_dim)
        out = tree_attention.attention(q, key, value, aoffs[i], aplan, scale=scale)
        r = _mm(out.reshape(W, c.heads * c.head_dim), a.o)
        record.append(AttentionRecord(key, value))
        x, h, _ = qglue.add_rmsnorm(x, r, layer.post_norm, c.eps)
        act, _ = qglue.swiglu(*_mm_group(h, [layer.gate, layer.up]))
        pending = _mm(act, layer.down)
        if i in taps:
            tapped.append((x.float() + pending.float()).to(torch.bfloat16))
    _, h, _ = qglue.add_rmsnorm(x, pending, w.norm, c.eps)
    logits = _mm(h, w.head)
    return (logits, record) + ((h,) if hidden else ()) + ((torch.cat(tapped, dim=-1),) if taps else ())
