"""A slow fp32 forward of LFM2-MoE on the CPU, straight from the checkpoint (FP8 blocks decoded exactly).

The math is transformers' modeling_lfm2_moe.py in fp32 throughout: a layer is ``h += mixer(rmsnorm(h))`` then
``h += ffn(rmsnorm(h))``. The short conv projects to B, C and x, convolves B * x causally over ``conv_L_cache`` taps
(no bias) and multiplies by C. Attention RMS-norms each query and key head, rotates them (RoPE, rotate-half) and
runs causal GQA. The first ``num_dense_layers`` run a SwiGLU; the rest route on sigmoid(router logits) plus
expert_bias, weight the picks by their plain sigmoids normalized to sum 1 (plus 1e-6) times routed_scaling_factor, and
add the picks' SwiGLU experts. The final norm is ``embedding_norm`` and the head is the tied embedding.

A ``State`` carries each conv layer's last inputs and each attention layer's keys and values, so a prompt fed whole,
in chunks or a token at a time gives the same logits up to fp32 summation order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import PREFIX, Tensors, layer
from .config import Config


@dataclass
class State:
    pos: int = 0
    conv: dict[int, torch.Tensor] = field(default_factory=dict)    # layer -> [taps - 1, D], oldest first
    keys: dict[int, torch.Tensor] = field(default_factory=dict)    # layer -> [T, KV, HD], rotated
    values: dict[int, torch.Tensor] = field(default_factory=dict)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """x [R, H, D] rotated at ``positions`` [R]: transformers' default RoPE, halves paired (rotate_half)."""

    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    freqs = positions.float()[:, None] * inv[None, :]
    cos, sin = torch.cat([freqs, freqs], -1).cos()[:, None], torch.cat([freqs, freqs], -1).sin()[:, None]
    half = torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1)
    return x * cos + half * sin


def route(x: torch.Tensor, router: torch.Tensor, bias: torch.Tensor, top_k: int,
          scaling: float) -> tuple[torch.Tensor, torch.Tensor]:
    """x [R, D] -> (expert ids [R, k], weights [R, k]): picks by sigmoid + bias, weights the picks' sigmoids."""

    gates = torch.sigmoid(x @ router.t())
    picks = torch.topk(gates + bias, top_k, dim=-1).indices
    chosen = gates.gather(1, picks)
    return picks, chosen / (chosen.sum(-1, keepdim=True) + 1e-6) * scaling


def swiglu(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    return (F.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()


class Reference:
    """``forward(ids, state)``: fp32 logits of every row, the state advanced past them."""

    def __init__(self, model_dir: str | Path, *, keep_experts: bool = False):
        """``keep_experts`` keeps decoded experts in memory (a tiny checkpoint); else each call decodes its picks."""

        self.c = Config.read(model_dir)
        self.t = Tensors(model_dir)
        self.keep_experts = keep_experts
        self._dense: dict[str, torch.Tensor] = {}
        self._experts: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        if PREFIX + "lm_head.weight" in self.t or "lm_head.weight" in self.t:
            raise ValueError("the checkpoint has its own lm_head; LFM2-MoE ties the head to the embeddings")

    def w(self, name: str) -> torch.Tensor:
        """A non-expert weight (``name.weight``) in fp32, decoded once."""

        if name not in self._dense:
            self._dense[name] = self.t.linear(name)
        return self._dense[name]

    def expert(self, i: int, e: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (i, e)
        if key in self._experts:
            return self._experts[key]
        p = layer(i) + f"feed_forward.experts.{e}."
        found = tuple(self.t.linear(p + name) for name in ("w1", "w3", "w2"))
        if self.keep_experts:
            self._experts[key] = found
        return found

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head()[ids]

    def head(self) -> torch.Tensor:
        return self.w(PREFIX + "embed_tokens")

    def conv(self, i: int, x: torch.Tensor, state: State) -> torch.Tensor:
        c, p = self.c, layer(i) + "conv."
        b, gate, v = (x @ self.w(p + "in_proj").t()).chunk(3, dim=-1)
        bx = b * v
        taps = self.w(p + "conv")[:, 0, :]                                     # [D, taps], oldest input first
        before = state.conv.get(i, torch.zeros(c.conv_taps - 1, c.hidden))
        seq = torch.cat([before, bx])
        rows = x.shape[0]
        out = sum(seq[j:j + rows] * taps[:, j] for j in range(c.conv_taps))
        state.conv[i] = seq[-(c.conv_taps - 1):].clone()
        return (gate * out) @ self.w(p + "out_proj").t()

    def attention(self, i: int, x: torch.Tensor, state: State) -> torch.Tensor:
        c, p = self.c, layer(i) + "self_attn."
        rows = x.shape[0]
        q = (x @ self.w(p + "q_proj").t()).view(rows, c.heads, c.head_dim)
        k = (x @ self.w(p + "k_proj").t()).view(rows, c.kv_heads, c.head_dim)
        v = (x @ self.w(p + "v_proj").t()).view(rows, c.kv_heads, c.head_dim)
        positions = torch.arange(state.pos, state.pos + rows)
        q = rotate(rmsnorm(q, self.w(p + "q_layernorm"), c.eps), positions, c.theta)
        k = rotate(rmsnorm(k, self.w(p + "k_layernorm"), c.eps), positions, c.theta)
        keys = torch.cat([state.keys[i], k]) if i in state.keys else k
        values = torch.cat([state.values[i], v]) if i in state.values else v
        state.keys[i], state.values[i] = keys, values
        group = c.heads // c.kv_heads
        kk, vv = keys.repeat_interleave(group, dim=1), values.repeat_interleave(group, dim=1)   # [T, H, HD]
        scores = torch.einsum("rhd,thd->hrt", q, kk) * c.head_dim ** -0.5
        seen = torch.arange(keys.shape[0])[None, :] <= positions[:, None]       # [R, T]
        scores = scores.masked_fill(~seen[None], float("-inf"))
        out = torch.einsum("hrt,thd->rhd", scores.softmax(-1), vv).reshape(rows, c.heads * c.head_dim)
        return out @ self.w(p + "out_proj").t()

    def moe(self, i: int, x: torch.Tensor) -> torch.Tensor:
        c, p = self.c, layer(i) + "feed_forward."
        picks, weights = route(x, self.w(p + "gate"), self.t.vector(p + "expert_bias"), c.top_k, c.scaling)
        out = torch.zeros_like(x)
        for e in picks.unique().tolist():
            rows, slot = (picks == e).nonzero(as_tuple=True)
            out[rows] += swiglu(x[rows], *self.expert(i, e)) * weights[rows, slot, None]
        return out

    def ffn(self, i: int, x: torch.Tensor) -> torch.Tensor:
        if self.c.moe(i):
            return self.moe(i, x)
        p = layer(i) + "feed_forward."
        return swiglu(x, self.w(p + "w1"), self.w(p + "w3"), self.w(p + "w2"))

    @torch.no_grad()
    def hidden(self, ids: list[int] | torch.Tensor, state: State, *, upto: int | None = None) -> torch.Tensor:
        """The residual stream after the first ``upto`` layers (all, then the final norm, when None)."""

        c = self.c
        ids = torch.as_tensor(ids, dtype=torch.long)
        h = self.embed(ids)
        count = len(c.layers) if upto is None else upto
        for i in range(count):
            p = layer(i)
            x = rmsnorm(h, self.w(p + "operator_norm"), c.eps)
            h = h + (self.attention(i, x, state) if c.layers[i] == "full_attention" else self.conv(i, x, state))
            h = h + self.ffn(i, rmsnorm(h, self.w(p + "ffn_norm"), c.eps))
        state.pos += len(ids)
        return h if upto is not None else rmsnorm(h, self.w(PREFIX + "embedding_norm"), c.eps)

    def forward(self, ids: list[int] | torch.Tensor, state: State) -> torch.Tensor:
        return self.hidden(ids, state) @ self.head().t()

    def greedy(self, prompt: list[int], steps: int, *, chunk: int = 512) -> tuple[list[int], list[torch.Tensor]]:
        """``steps`` greedy tokens after ``prompt`` (fed in chunks), and each step's fp32 logits."""

        state = State()
        for s in range(0, len(prompt), chunk):
            logits = self.forward(prompt[s:s + chunk], state)[-1]
        out, seen = [], [logits]
        for _ in range(steps):
            out.append(int(torch.argmax(seen[-1])))
            if len(out) == steps:
                break
            seen.append(self.forward([out[-1]], state)[-1])
        return out, seen
