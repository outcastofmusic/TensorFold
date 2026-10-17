"""LFM2 in plain torch: gated short convolutions and GQA attention over a tree of packed rows.

A forward reads one trunk (the tokens every question shares: the state and its pictures) and N branches (the
questions) packed back to back on one axis, ``[trunk | branch 0 | branch 1 | ...]``, and returns the final hidden state
at each branch's last token. Token-wise work (norms, projections, the MLP) runs once over the packed axis. A
convolution continues each branch from the trunk's last inputs through a predecessor table. Attention runs the trunk
causally in one call and every branch token over the trunk and its own branch's earlier tokens in a second, masked
call. Each branch therefore reads exactly what the same question's own prompt would read alone, and the state is read
once for every question. Rounding follows transformers' Lfm2 modules (bf16 between ops, RMSNorm in fp32, RoPE in
the model's dtype), which is the reference the answers are checked against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
ROWS = 512           # branch query rows a masked attention call takes


@dataclass(frozen=True)
class Config:
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    layers: tuple[str, ...]
    eps: float
    theta: float
    taps: int
    vocab: int

    @classmethod
    def of(cls, text: dict) -> Config:
        """From a checkpoint's text config (transformers' Lfm2Config fields)."""

        hidden, heads = int(text["hidden_size"]), int(text["num_attention_heads"])
        ffn = int(text["intermediate_size"])
        if text.get("block_auto_adjust_ff_dim"):          # transformers' sizing rule, kept exactly
            ffn = int(2 * ffn / 3)
            if text.get("block_ffn_dim_multiplier") is not None:
                ffn = int(text["block_ffn_dim_multiplier"] * ffn)
                m = int(text["block_multiple_of"])
                ffn = m * ((ffn + m - 1) // m)
        rope = text.get("rope_parameters") or {"rope_theta": text.get("rope_theta", 1e6)}
        return cls(hidden, heads, int(text["num_key_value_heads"]), int(text.get("head_dim") or hidden // heads), ffn,
                   tuple(text["layer_types"]), float(text.get("norm_eps") or text.get("block_norm_eps")),
                   float(rope["rope_theta"]), int(text["conv_L_cache"]), int(text["vocab_size"]))

    def token_bytes(self) -> int:
        """Bytes one packed token's forward holds at once (bf16): the residual stream and its normed copy, the conv's
        or q|k|v's projection, the merged gate|up and its product, and K/V repeated over the query heads."""

        return 2 * (4 * self.hidden + 3 * self.hidden + 3 * self.ffn + 3 * self.heads * self.head_dim)


class Tree:
    """A trunk of ``p`` tokens and branches of ``lengths`` tokens, packed on one axis."""

    def __init__(self, p: int, lengths: list[int], device, taps: int = 3):
        if not lengths or min(lengths) < 1:
            raise ValueError("a tree needs at least one branch, each of at least one token")
        self.p, self.lengths = int(p), [int(n) for n in lengths]
        ln = torch.tensor(self.lengths)
        cu = F.pad(ln.cumsum(0), (1, 0))
        branch = torch.repeat_interleave(torch.arange(len(ln)), ln)
        offset = torch.arange(len(branch)) - cu[branch]
        positions = torch.cat((torch.arange(p), p + offset))
        index = torch.arange(len(positions))[:, None]
        back = positions[:, None] - torch.arange(1, taps)
        # each token's predecessors 1..taps-1 back in its own sequence: its branch, then the trunk; -1 before the start
        prev = torch.where(back < 0, -1, torch.where(back >= p, index - torch.arange(1, taps), back))
        # a branch token sees the whole trunk and its own branch up to itself
        keys = torch.arange(len(positions))
        owner = torch.cat((torch.full((p,), -1), branch))
        place = torch.cat((torch.zeros(p, dtype=torch.long), offset))
        seen = (keys[None] < p) | ((owner[None] == branch[:, None]) & (place[None] <= offset[:, None]))
        self.positions = positions.to(device)
        self.leaves = (p + cu[1:] - 1).to(device)
        self.prev = prev.flip(1).to(device)          # oldest first, as the conv's taps run
        self.mask = seen.to(device)

    @property
    def tokens(self) -> int:
        return self.p + sum(self.lengths)


def rms(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """transformers' Lfm2RMSNorm: normalized in fp32, rounded to the input's dtype, then times the weight."""

    x32 = x.float()
    return weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


@dataclass
class Layer:
    kind: str
    op_norm: Tensor
    ffn_norm: Tensor
    up: Tensor            # the MLP's gate (w1) and up (w3), merged: [2 ffn, hidden]
    down: Tensor
    proj: Tensor          # conv: in_proj [3 hidden, hidden]; attention: q|k|v merged
    out: Tensor
    conv: Tensor | None = None            # [taps, hidden]
    q_norm: Tensor | None = None
    k_norm: Tensor | None = None


@dataclass
class Model:
    cfg: Config
    embed: Tensor                          # [vocab, hidden], also the tied head
    norm: Tensor
    layers: list[Layer] = field(default_factory=list)

    @classmethod
    def from_tensors(cls, cfg: Config, tensors: dict[str, Tensor], device="cuda", prefix: str = "") -> Model:
        """The language model from its checkpoint tensors (transformers' names under ``prefix``), moved to
        ``device``; every tensor the forward needs must be there and nothing else under the prefix."""

        tensors = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}

        def take(name: str) -> Tensor:
            found = tensors.pop(name, None)
            if found is None:
                raise ValueError(f"the checkpoint has no {prefix}{name}, which an LFM2 forward needs")
            return found.to(device)

        model = cls(cfg, take("embed_tokens.weight"), take("embedding_norm.weight"))
        for i, kind in enumerate(cfg.layers):
            p = f"layers.{i}."
            op_norm, ffn_norm = take(p + "operator_norm.weight"), take(p + "ffn_norm.weight")
            up = torch.cat((take(p + "feed_forward.w1.weight"), take(p + "feed_forward.w3.weight")))
            down = take(p + "feed_forward.w2.weight")
            if kind == "conv":
                conv = take(p + "conv.conv.weight")       # (C, 1, taps)
                model.layers.append(Layer(kind, op_norm, ffn_norm, up, down, take(p + "conv.in_proj.weight"),
                                          take(p + "conv.out_proj.weight"), conv=conv[:, 0].t().contiguous()))
            else:
                qkv = torch.cat((take(p + "self_attn.q_proj.weight"), take(p + "self_attn.k_proj.weight"),
                                 take(p + "self_attn.v_proj.weight")))
                model.layers.append(Layer(kind, op_norm, ffn_norm, up, down, qkv, take(p + "self_attn.out_proj.weight"),
                                          q_norm=take(p + "self_attn.q_layernorm.weight"),
                                          k_norm=take(p + "self_attn.k_layernorm.weight")))
        tensors.pop("lm_head.weight", None)              # tied: the head is the embedding
        if tensors:
            raise ValueError(f"{len(tensors)} language tensor(s) the LFM2 forward does not read: "
                             f"{', '.join(sorted(tensors)[:4])}")
        return model

    @classmethod
    def load(cls, model_dir: Path, cfg: Config, device="cuda") -> Model:
        """The checkpoint's bf16 language weights, read straight onto ``device`` one tensor at a time (a GB10's host
        and GPU share one pool, so a host copy of the whole model would double the load's peak)."""

        from safetensors import safe_open

        tensors: dict[str, Tensor] = {}
        for path in sorted(Path(model_dir).glob("*.safetensors")):
            with safe_open(str(path), "pt", device=str(device)) as f:
                for key in f.keys():
                    if not key.startswith("model.language_model."):
                        continue
                    tensors[key] = f.get_tensor(key)
                    if tensors[key].dtype != torch.bfloat16:
                        raise ValueError(f"{key} is stored as {tensors[key].dtype}, and LFM2 d1 reads bf16 weights")
        if not tensors:
            raise ValueError(f"no language-model tensors (model.language_model.*) in {model_dir}")
        return cls.from_tensors(cfg, tensors, device, prefix="model.language_model.")

    @property
    def device(self) -> torch.device:
        return self.embed.device

    def embed_ids(self, ids: list[int]) -> Tensor:
        return self.embed[torch.tensor(ids, device=self.device)]

    def _cos_sin(self, positions: Tensor, dtype) -> tuple[Tensor, Tensor]:
        d = self.cfg.head_dim
        inv = 1.0 / self.cfg.theta ** (torch.arange(0, d, 2, device=positions.device, dtype=torch.float) / d)
        angles = positions.float()[:, None] * inv
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(dtype)[:, None], angles.sin().to(dtype)[:, None]

    def _conv(self, layer: Layer, x: Tensor, tree: Tree) -> Tensor:
        c = self.cfg.hidden
        b, gate, u = F.linear(x, layer.proj).split(c, dim=-1)
        bx = b * u
        padded = torch.cat((bx, bx.new_zeros(1, c)))           # row -1: before the sequence starts
        taps = torch.cat((padded[tree.prev], bx[:, None]), dim=1)      # [T, taps, C], oldest first
        conv = (taps.float() * layer.conv.float()).sum(1).to(x.dtype)
        return F.linear(gate * conv, layer.out)

    def _attention(self, layer: Layer, x: Tensor, tree: Tree, cos: Tensor, sin: Tensor) -> Tensor:
        c = self.cfg
        hq, hk = c.heads * c.head_dim, c.kv_heads * c.head_dim
        qkv = F.linear(x, layer.proj)
        q = rotate(rms(qkv[:, :hq].view(-1, c.heads, c.head_dim), layer.q_norm, c.eps), cos, sin)
        k = rotate(rms(qkv[:, hq:hq + hk].view(-1, c.kv_heads, c.head_dim), layer.k_norm, c.eps), cos, sin)
        v = qkv[:, hq + hk:].view(-1, c.kv_heads, c.head_dim)
        groups = c.heads // c.kv_heads
        # (1, H, T, D); K and V repeated over each query group as transformers' repeat_kv does
        q = q.transpose(0, 1)[None]
        k = k.transpose(0, 1).repeat_interleave(groups, dim=0)[None]
        v = v.transpose(0, 1).repeat_interleave(groups, dim=0)[None]
        p, scale = tree.p, c.head_dim ** -0.5
        # branch rows in blocks: a backend without a masked kernel holds a block's fp32 scores, not every row's
        parts = [F.scaled_dot_product_attention(q[:, :, p + i:p + i + ROWS], k, v, attn_mask=tree.mask[i:i + ROWS],
                                                scale=scale) for i in range(0, q.shape[2] - p, ROWS)]
        if p:
            parts.insert(0, F.scaled_dot_product_attention(q[:, :, :p], k[:, :, :p], v[:, :, :p], is_causal=True,
                                                           scale=scale))
        return F.linear(torch.cat(parts, dim=2)[0].transpose(0, 1).reshape(-1, hq), layer.out)

    @torch.inference_mode()
    def forward(self, embeds: Tensor, tree: Tree) -> Tensor:
        """``embeds`` [T, hidden] over ``tree`` -> the final hidden state at each branch's last token [N, hidden]."""

        c = self.cfg
        cos, sin = self._cos_sin(tree.positions, embeds.dtype)
        h = embeds
        for layer in self.layers:
            x = rms(h, layer.op_norm, c.eps)
            h = h + (self._conv(layer, x, tree) if layer.kind == "conv" else self._attention(layer, x, tree, cos, sin))
            gate, up = F.linear(rms(h, layer.ffn_norm, c.eps), layer.up).chunk(2, dim=-1)
            h = h + F.linear(F.silu(gate) * up, layer.down)
        return rms(h[tree.leaves], self.norm, c.eps)

    def logits(self, hidden: Tensor, ids: list[int] | None = None) -> Tensor:
        """fp32 logits at each row of ``hidden``: the tied head in the model's dtype (as transformers' lm_head), over
        ``ids`` alone when given, so a question reads only its own options' rows of the head."""

        head = self.embed if ids is None else self.embed[torch.tensor(ids, device=hidden.device)]
        return F.linear(hidden, head).float()
