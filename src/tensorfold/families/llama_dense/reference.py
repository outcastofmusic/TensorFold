"""A plain fp32 Llama forward over checkpoint tensors by name, for quality checks of the CUDA engine (any device)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def _rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """x (T, H, D): transformers' rotate-half with the default frequencies."""

    half = x.shape[-1] // 2
    inv = theta ** (-torch.arange(0, half, dtype=torch.float64, device=x.device) / half)
    ang = pos.double()[:, None] * inv[None, :]
    cos, sin = torch.cos(ang).float()[:, None, :], torch.sin(ang).float()[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


@torch.no_grad()
def forward(tensors: Mapping[str, torch.Tensor], config, ids: Sequence[int]) -> torch.Tensor:
    """Logits (T, vocab) fp32 for every position of ``ids``; ``config`` is ``cuda.weights.Config``'s fields."""

    c = config
    dev = tensors["model.embed_tokens.weight"].device
    T = len(ids)

    def lin(x: torch.Tensor, name: str) -> torch.Tensor:
        return x @ tensors[name + ".weight"].float().T

    x = tensors["model.embed_tokens.weight"].float()[torch.tensor(list(ids), device=dev)]
    pos = torch.arange(T, device=dev)
    group = c.heads // c.kv_heads
    mask = torch.ones((T, T), dtype=torch.bool, device=dev).tril()
    for i in range(c.layers):
        p = f"model.layers.{i}."
        h = _rms(x, tensors[p + "input_layernorm.weight"], c.eps)
        q = _rope(lin(h, p + "self_attn.q_proj").view(T, c.heads, c.head_dim), pos, c.rope_theta)
        k = _rope(lin(h, p + "self_attn.k_proj").view(T, c.kv_heads, c.head_dim), pos, c.rope_theta)
        v = lin(h, p + "self_attn.v_proj").view(T, c.kv_heads, c.head_dim)
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * c.head_dim ** -0.5
        scores = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        x = x + lin(torch.einsum("hqk,khd->qhd", scores, v).reshape(T, -1), p + "self_attn.o_proj")
        h = _rms(x, tensors[p + "post_attention_layernorm.weight"], c.eps)
        x = x + lin(torch.nn.functional.silu(lin(h, p + "mlp.gate_proj")) * lin(h, p + "mlp.up_proj"),
                    p + "mlp.down_proj")
    return lin(_rms(x, tensors["model.norm.weight"], c.eps), "lm_head")
