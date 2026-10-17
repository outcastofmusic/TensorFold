"""fp32 PyTorch forward of Bolt from the loaded weights (any device), the yardstick for the engine's bf16 kernels.

It follows modeling_nemotron_h.py of the checkpoint (NemotronHBlock, NemotronHMoE, NemotronHMamba2Mixer's torch path)
with every activation in fp32; NVFP4 projections need ``load(..., raw=True)``. The routing, relu2 and RMSNorm are
``reference.py``'s; this module keeps its own layers because it reads the engine's loaded ``Weights`` (bf16 Q8_0
decodes, stacked IQ2_XXS tables, fused qkv) where ``reference.py`` reads the checkpoint, a layer at a time.
"""

from __future__ import annotations

import torch

from ..quant import dequantize_iq2xxs
from ..reference import relu2, rms, route
from .weights import Attention, Mamba, MoE, Weights


def moe(m: MoE, x: torch.Tensor, scaling: float, norm: bool) -> torch.Tensor:
    idx, wts = route(x @ m.router.float().T, m.bias, m.top_k, scaling, norm)
    latent = x @ m.fc1.float().T
    routed = torch.zeros_like(latent)
    for e in torch.unique(idx).tolist():
        rows, slot = (idx == e).nonzero(as_tuple=True)
        up = latent[rows] @ dequantize_iq2xxs(m.up[e]).float().T
        routed.index_add_(0, rows, wts[rows, slot][:, None] * (relu2(up) @ dequantize_iq2xxs(m.down[e]).float().T))
    shared = relu2(x @ dequantize_iq2xxs(m.shared_up).float().T) @ dequantize_iq2xxs(m.shared_down).float().T
    return routed @ m.fc2.float().T + shared


def mamba(m: Mamba, x: torch.Tensor, c) -> torch.Tensor:
    T = x.shape[0]
    proj = x @ m.in_proj.dequantized().float().T
    z, xbc, dtr = proj[:, :c.xd], proj[:, c.xd:c.xd + c.conv_dim], proj[:, c.xd + c.conv_dim:]
    kc = c.conv_kernel
    padded = torch.cat([torch.zeros((kc - 1, c.conv_dim), device=x.device), xbc])
    conv = torch.nn.functional.silu(m.conv_b + sum(m.conv_w[k] * padded[k:k + T] for k in range(kc)))
    xs = conv[:, :c.xd].reshape(T, c.m_heads, c.m_head_dim)
    gs = c.m_groups * c.m_state
    B = conv[:, c.xd:c.xd + gs].reshape(T, c.m_groups, c.m_state)
    C = conv[:, c.xd + gs:].reshape(T, c.m_groups, c.m_state)
    dt = torch.nn.functional.softplus(dtr + m.dt_bias).clamp(c.dt_min, c.dt_max)
    rep = c.m_heads // c.m_groups
    state = torch.zeros((c.m_heads, c.m_head_dim, c.m_state), device=x.device)
    ys = []
    for t in range(T):
        Bt, Ct = B[t].repeat_interleave(rep, 0), C[t].repeat_interleave(rep, 0)
        state = state * torch.exp(m.a * dt[t])[:, None, None] + (xs[t] * dt[t][:, None])[:, :, None] * Bt[:, None, :]
        ys.append((state * Ct[:, None, :]).sum(-1) + m.d[:, None] * xs[t])
    y = torch.stack(ys).reshape(T, c.xd) * torch.nn.functional.silu(z)
    y = rms(y.reshape(T, c.m_groups, -1), None, c.eps).reshape(T, c.xd) * m.gnorm.float()
    return y @ m.out_proj.dequantized().float().T


def attention(a: Attention, x: torch.Tensor, c) -> torch.Tensor:
    T = x.shape[0]
    qkv = x @ a.qkv.float().T
    qd, kd = c.heads * c.head_dim, a.kv_heads * c.head_dim
    q = qkv[:, :qd].reshape(T, c.heads, c.head_dim)
    k = qkv[:, qd:qd + kd].reshape(T, a.kv_heads, c.head_dim)
    v = qkv[:, qd + kd:].reshape(T, a.kv_heads, c.head_dim)
    rep = c.heads // a.kv_heads
    kf, vf = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = torch.einsum("thd,lhd->htl", q, kf) * c.head_dim ** -0.5
    mask = torch.arange(T, device=x.device)[None, :] > torch.arange(T, device=x.device)[:, None]
    o = torch.einsum("htl,lhd->thd", torch.softmax(s.masked_fill(mask, float("-inf")), -1), vf).reshape(T, qd)
    return o @ a.o.float().T


@torch.no_grad()
def forward(w: Weights, tokens: torch.Tensor, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Logits (T, V) fp32 for ``tokens`` from an empty state, and the stream after each of ``taps``' layers."""

    c = w.config
    x = w.embed[tokens.long()].float()
    streams = []
    for blk in w.blocks:
        h = rms(x, blk.norm, c.eps)
        if blk.kind == "M":
            x = x + mamba(blk.mamba, h, c)
        elif blk.kind == "*":
            x = x + attention(blk.attn, h, c)
        else:
            x = x + moe(blk.moe, h, c.scaling, c.norm_topk)
        streams.append(x)
    return rms(x, w.norm_f, c.eps) @ w.head.float().T, [streams[t] for t in taps]
