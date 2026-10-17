"""LiLiCorr drafts for Nemotron 3.5 Bolt: a DFlash block over target taps, a candidate lattice walked by pair scores.

The draft model follows vLLM's ``qwen3_dflash.py`` / ``qwen3_dflash2.py`` / ``lilicorr.py`` and the speculator in
``v1/worker/gpu/spec_decode/{dflash,dflash2,lilicorr}``:

- Context. Each committed target position p contributes ``fc(concat of the residual stream after the tap layers)``.
  Every draft layer projects its ``hidden_norm`` to keys and values (k_norm, RoPE at p), and drafts read them through
  a 1024-position sliding window. The last committed row's normed context is the lattice's anchor.
- Block. The pending token (at position ``context_end``) and N mask tokens run through 5 dense GQA layers with
  bidirectional attention inside the block, per-head attention sinks, and a two-tap grouped dynamic convolution
  around each attention and MLP. The mask rows' final hidden states give the target lm_head's top-8 candidates
  at positions ``context_end + 1 .. context_end + N``.
- Lattice. The two-layer LiLiCorr head scores anchor -> slot 0 and slot s -> slot s + 1 candidate pairs. A walk
  then picks one candidate a slot: the argmax for greedy requests, the argmax under the target's keyed Gumbel noise
  for sampled ones.

``lilicorr_decode`` verifies the pending token and the drafts in one target window and keeps the drafts through the
first one that differs from the target's keyed sample, so a drafted reply equals the serial reply token for token.
A ``depth.DepthPolicy`` picks the drafts a round from measured costs and running acceptance, serial rounds included.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tensorfold.engine.exact_sampling import Sampling, uniform_rows

from .engine import DecodeResult

NVFP4_GROUP = 16              # inputs an e4m3 scale covers
LATTICE_FEATURES = 5          # log p, p, log p - max, rank / (k - 1), is-top-1 (vLLM LiLiCorrHead)


def _nvfp4_rows(weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2, dtype: torch.dtype) -> torch.Tensor:
    """``quant.dequantize_nvfp4`` a slice of rows at a time, so its int64 gather indices stay near 128 MiB."""

    from .. import quant

    n, half = weight.shape
    if weight_scale.shape != (n, 2 * half // NVFP4_GROUP):
        raise ValueError(f"NVFP4 scales {tuple(weight_scale.shape)} do not fit codes {tuple(weight.shape)}")
    out = torch.empty((n, 2 * half), dtype=dtype, device=weight.device)
    step = max(1, (1 << 24) // (2 * half))
    for r in range(0, n, step):
        out[r:r + step] = quant.dequantize_nvfp4(weight[r:r + step], weight_scale[r:r + step], weight_scale_2, dtype)
    return out


# -- the checkpoint -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DraftConfig:
    """The draft checkpoint's geometry, from its config.json (``dflash_config`` holds the drafter's own keys)."""

    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    inter: int
    eps: float
    theta: float
    vocab: int
    mask_id: int
    block: int                     # the trained block: the pending token and block - 1 mask tokens
    window: int                    # sliding window: a query at q reads keys at q - window + 1 and later
    tap_layers: tuple[int, ...]    # target layers (0-based) whose output residual stream is a tap
    conv_taps: int
    conv_group: int
    sinks: bool
    top_k: int                     # lattice candidates a slot
    lili_hidden: int
    lili_layers: int
    lili_heads: int
    lili_inter: int
    factor_dim: int
    vector_eps: float
    logit_scale: float

    @classmethod
    def from_json(cls, cfg: Mapping) -> DraftConfig:
        dc = cfg["dflash_config"]
        if not dc.get("lilicorr_enabled", True) or dc.get("projector_type", "lilicorr") != "lilicorr":
            raise ValueError("this draft checkpoint has no LiLiCorr head")
        if dc.get("causal", False):
            raise ValueError("only the bidirectional draft block is ported")
        if int(dc.get("conv_kernel_size", 0)) != 2:
            raise ValueError("only two-tap grouped convolutions are ported")
        window = int(dc.get("swa_window_size") or cfg.get("sliding_window") or 0) if dc.get("use_swa", True) else 0
        if window <= 0:
            raise ValueError("the drafter's sliding window is missing")
        top_k = int(dc["lilicorr_candidate_topk"])
        if top_k <= 0 or top_k > 16 or top_k & (top_k - 1):
            raise ValueError("LiLiCorr candidate_topk must be a power of two <= 16")
        hidden = int(cfg["hidden_size"])
        lili = int(dc.get("lilicorr_hidden_size") or hidden)
        theta = cfg.get("rope_theta") or (cfg.get("rope_parameters") or {}).get("rope_theta") or 1_000_000
        return cls(hidden=hidden, layers=int(cfg["num_hidden_layers"]), heads=int(cfg["num_attention_heads"]),
                   kv_heads=int(cfg["num_key_value_heads"]),
                   head_dim=int(cfg.get("head_dim") or hidden // int(cfg["num_attention_heads"])),
                   inter=int(cfg["intermediate_size"]), eps=float(cfg["rms_norm_eps"]), theta=float(theta),
                   vocab=int(cfg["vocab_size"]), mask_id=int(dc["mask_token_id"]),
                   block=int(cfg.get("block_size") or dc["block_size"]), window=window,
                   tap_layers=tuple(int(i) for i in dc["target_layer_ids"]),
                   conv_taps=int(dc["conv_kernel_size"]), conv_group=int(dc["conv_group_size"]),
                   sinks=bool(dc.get("attention_sink_bias", cfg.get("attention_sink_bias", False))), top_k=top_k,
                   lili_hidden=lili, lili_layers=int(dc["lilicorr_num_layers"]),
                   lili_heads=int(dc["lilicorr_num_heads"]),
                   lili_inter=int(lili * float(dc["lilicorr_mlp_ratio"])),
                   factor_dim=int(dc["lilicorr_factor_dim"]), vector_eps=float(dc["lilicorr_vector_eps"]),
                   logit_scale=float(dc["lilicorr_logit_scale"]))


def expected_shapes(c: DraftConfig) -> dict[str, tuple[int, ...]]:
    """Every dense tensor the drafter reads, by checkpoint name, at its dense (dequantized) shape."""

    D, hd, H, KV, G = c.hidden, c.head_dim, c.heads, c.kv_heads, c.hidden // c.conv_group
    L, I, k = c.lili_hidden, c.lili_inter, c.top_k
    shapes: dict[str, tuple[int, ...]] = {
        "fc.weight": (D, D * len(c.tap_layers)), "hidden_norm.weight": (D,), "norm.weight": (D,)}
    for i in range(c.layers):
        p = f"layers.{i}."
        shapes.update({
            p + "input_layernorm.weight": (D,), p + "post_attention_layernorm.weight": (D,),
            p + "self_attn.q_proj.weight": (H * hd, D), p + "self_attn.k_proj.weight": (KV * hd, D),
            p + "self_attn.v_proj.weight": (KV * hd, D), p + "self_attn.o_proj.weight": (D, H * hd),
            p + "self_attn.q_norm.weight": (hd,), p + "self_attn.k_norm.weight": (hd,),
            p + "mlp.gate_proj.weight": (c.inter, D), p + "mlp.up_proj.weight": (c.inter, D),
            p + "mlp.down_proj.weight": (D, c.inter)})
        if c.sinks:
            shapes[p + "self_attn.attention_sink_bias"] = (H,)
        for conv in ("attention_conv", "mlp_conv"):
            shapes[p + conv + ".base_kernel"] = (2, c.conv_taps, D)
            shapes[p + conv + ".kernel_projection.weight"] = (2 * c.conv_taps * G, D)
    q = "lilicorr."
    for name, shape in {"token_proj": (L, D), "pass_hidden_proj": (L, D), "context_proj": (L, D),
                        "feature_mlp.up_proj": (L, LATTICE_FEATURES), "feature_mlp.down_proj": (L, L),
                        "factor_input_proj": (L, 3 * L), "out_head": (c.factor_dim, L),
                        "in_head": (c.factor_dim, L), "anchor_out_head": (c.factor_dim, L)}.items():
        shapes[q + name + ".weight"] = shape
        shapes[q + name + ".bias"] = (shape[0],)
    shapes.update({q + "feature_norm.weight": (LATTICE_FEATURES,), q + "feature_norm.bias": (LATTICE_FEATURES,),
                   q + "slot_embedding": (1, 1, c.block - 1, 1, L), q + "rank_embedding": (1, 1, 1, k, L),
                   q + "relative_slot_bias": (c.lili_heads, 2 * c.block - 1), q + "same_slot_bias": (c.lili_heads,),
                   q + "output_norm.weight": (L,), q + "anchor_norm.weight": (L,)})
    for i in range(c.lili_layers):
        p = f"{q}layers.{i}."
        shapes.update({p + "attn_norm.weight": (L,), p + "mlp_norm.weight": (L,),
                       p + "attn.in_proj_weight": (3 * L, L), p + "attn.in_proj_bias": (3 * L,),
                       p + "attn.out_proj.weight": (L, L), p + "attn.out_proj.bias": (L,),
                       p + "mlp.up_proj.weight": (I, L), p + "mlp.up_proj.bias": (I,),
                       p + "mlp.down_proj.weight": (L, I), p + "mlp.down_proj.bias": (L,)})
    return shapes


def dense_tensors(raw: Mapping[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Checkpoint tensors -> dense ones by name: NVFP4 triples dequantized, the rest cast to ``dtype``."""

    out: dict[str, torch.Tensor] = {}
    for name, t in raw.items():
        if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
            continue
        base = name[:-len(".weight")] if name.endswith(".weight") else None
        if base is not None and t.dtype == torch.uint8:
            if base + ".weight_scale" not in raw or base + ".weight_scale_2" not in raw:
                raise ValueError(f"{name}: packed NVFP4 codes without their scales")
            out[name] = _nvfp4_rows(t, raw[base + ".weight_scale"], raw[base + ".weight_scale_2"], dtype)
        else:
            out[name] = t.to(dtype)
    return out


def fp4_kept(name: str) -> bool:
    """The backbone's NVFP4 projections (fc and the MLPs, 0.86B of the 1.11B weights) that stay packed on CUDA."""

    return name == "fc.weight" or (name.startswith("layers.") and ".mlp." in name)


def device_bytes(model_dir: str | Path, *, fp4: bool = True) -> int:
    """The bytes a bf16 ``Drafter`` of ``model_dir`` holds (its ``nbytes()``): dense weights, the NVFP4 ones it keeps
    packed with ``fp4`` (CUDA: ``Fp4Linear`` pads rows to 128) and its window's keys and values."""

    from tensorfold.cuda.capacity import headers

    path = Path(model_dir)
    c = DraftConfig.from_json(json.loads((path / "config.json").read_text()))
    stored = headers(path)
    total = 2 * c.layers * c.kv_heads * c.window * c.head_dim * 2
    for name, shape in expected_shapes(c).items():
        if fp4 and fp4_kept(name) and stored.get(name, {}).get("dtype") == "U8":
            n, k = shape
            total += -(-n // 128) * 128 * (k // 2 + k // NVFP4_GROUP)
        else:
            total += math.prod(shape) * 2
    return total


Packed = tuple[torch.Tensor, torch.Tensor, torch.Tensor]       # NVFP4 codes, e4m3 scales, the fp32 tensor scale


def read_checkpoint(model_dir: str | Path, device: torch.device | str, dtype: torch.dtype = torch.bfloat16, *,
                    keep: Callable[[str], bool] | None = None
                    ) -> tuple[DraftConfig, dict[str, torch.Tensor], dict[str, Packed]]:
    """config.json and model.safetensors -> the geometry, dense tensors and the NVFP4 ones ``keep`` names, packed."""

    from contextlib import closing

    from tensorfold.cuda.direct_read import SafeTensors

    path = Path(model_dir)
    cfg = DraftConfig.from_json(json.loads((path / "config.json").read_text()))
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{path}: no safetensors files")
    out: dict[str, torch.Tensor] = {}
    packed: dict[str, Packed] = {}
    with closing(SafeTensors(files)) as f:
        names = set(f.keys())
        for name in sorted(names):
            if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
                continue
            group = {name: f.get(name, device)}
            base = name[:-len(".weight")]
            if group[name].dtype == torch.uint8:
                for extra in (base + ".weight_scale", base + ".weight_scale_2"):
                    if extra in names:
                        group[extra] = f.get(extra, device)
                if keep is not None and keep(name) and len(group) == 3:
                    packed[name] = (group[name], group[base + ".weight_scale"], group[base + ".weight_scale_2"])
                    continue
            out.update(dense_tensors(group, dtype))
    return cfg, out, packed


# -- pieces ---------------------------------------------------------------------------------------------------------
def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """vLLM's RMSNorm: normalize in fp32, round to the model dtype, then scale by the weight."""

    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)
    return y * weight


def _conv(x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, group: int) -> torch.Tensor:
    """DFlash2's two-tap grouped convolution over one block: row r mixes rows r and r - 1, fp32 accumulation.

    x [n, D]; delta [n, taps, D / group] (one side of the kernel projection); base [taps, D].
    """

    n = x.shape[0]
    coef = base.float()[None] + delta.float().repeat_interleave(group, dim=-1)        # [n, taps, D]
    out = coef[:, 0] * x.float()
    if n > 1:
        out[1:] += coef[1:, 1] * x[:-1].float()
    return out.to(x.dtype)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Neox rotary on [..., rows, head_dim] with cos/sin [rows, head_dim / 2]."""

    half = x.shape[-1] // 2
    a, b = x[..., :half].float(), x[..., half:].float()
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1).to(x.dtype)


@dataclass
class Proposal:
    """One propose(): the drafts and what produced them (kept for tests and the acceptance probe)."""

    tokens: list[int]
    candidates: torch.Tensor        # [N, top_k] int64 candidate ids
    log_probs: torch.Tensor         # [N, top_k] fp32 candidate log-probabilities under the target lm_head
    scores: torch.Tensor            # [N, top_k, top_k] fp32 lattice logits (row 0 repeats the anchor scores)


class Drafter:
    """One sequence's LiLiCorr drafter: committed context in a window-sized ring, a block pass per propose().

    ``embed(ids int64 [n]) -> [n, D]`` and ``lm_head(h [n, D]) -> [n, V]`` are the target's own embedding and
    output head (LiLiCorr shares both). ``source`` is the draft checkpoint directory, or (config, dense tensors).
    ``fp4`` (CUDA, bf16, a directory): fc and the MLPs stay NVFP4 on the fork's exact W4A16 matmul, 0.45 GiB a
    block pass reads instead of 1.6 GiB; otherwise every weight is dequantized to bf16 at load.
    """

    def __init__(self, source: str | Path | tuple[DraftConfig, Mapping[str, torch.Tensor]], *,
                 embed: Callable[[torch.Tensor], torch.Tensor], lm_head: Callable[[torch.Tensor], torch.Tensor],
                 device: torch.device | str = "cuda", dtype: torch.dtype = torch.bfloat16, fp4: bool = True) -> None:
        self.dev, self.dt = torch.device(device), dtype
        packed: dict[str, Packed] = {}
        if isinstance(source, (str, Path)):
            fp4 = fp4 and self.dev.type == "cuda" and dtype == torch.bfloat16
            cfg, tensors, packed = read_checkpoint(source, self.dev, dtype, keep=fp4_kept if fp4 else None)
        else:
            cfg, tensors = source[0], {k: v.to(self.dev, dtype) for k, v in source[1].items()}
        shapes = expected_shapes(cfg)
        got = {k: tuple(v.shape) for k, v in tensors.items()}
        got.update({k: (w.shape[0], 2 * w.shape[1]) for k, (w, _, _) in packed.items()})
        missing = sorted(set(shapes) - set(got))
        if missing:
            raise ValueError(f"the draft checkpoint lacks {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        for name, shape in shapes.items():
            if got[name] != shape:
                raise ValueError(f"{name}: shape {got[name]}, expected {shape}")
        self.c, self.p = cfg, {k: tensors[k].contiguous() for k in shapes if k in tensors}
        self.q = {}
        if packed:
            from tensorfold.cuda.nvfp4.linear import Fp4Linear

            self.q = {k: Fp4Linear.from_checkpoint(w, s, float(s2.item())) for k, (w, s, s2) in packed.items()}
            del packed
            torch.cuda.empty_cache()
        self.embed, self.lm_head = embed, lm_head
        self.tap_layers = cfg.tap_layers
        c = cfg
        self.inv_freq = 1.0 / c.theta ** (torch.arange(c.head_dim // 2, device=self.dev, dtype=torch.float32)
                                          * 2 / c.head_dim)
        # every context row a block can read: positions context_end - window + 1 .. context_end - 1 fit in `window`
        self.ring = c.window
        self.kc = torch.zeros((c.layers, c.kv_heads, self.ring, c.head_dim), dtype=dtype, device=self.dev)
        self.vc = torch.zeros_like(self.kc)
        self.slot_pos = torch.full((self.ring,), -1, dtype=torch.int64, device=self.dev)
        self.anchor = torch.zeros((c.hidden,), dtype=dtype, device=self.dev)
        self.context_end = 0
        self._lattice_bias = self._build_lattice_bias()
        self.last: Proposal | None = None

    # -- state ----------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.slot_pos.fill_(-1)
        self.anchor.zero_()
        self.context_end = 0

    def nbytes(self) -> int:
        dense = sum(t.numel() * t.element_size() for t in (*self.p.values(), self.kc, self.vc))
        return dense + sum(q.nbytes() for q in self.q.values())

    def _lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """x [n, K] times the weight ``name``: a packed NVFP4 one on its matmul (prompt rows on the prompt GEMM)."""

        q = self.q.get(name)
        if q is None:
            return F.linear(x, self.p[name])
        return q(x.contiguous()) if x.shape[0] <= 64 else q.prefill(x.contiguous())

    def snapshot(self) -> dict:
        """The context state (window rows, anchor, length), copied: ``restore`` resumes a kept prompt from it."""

        return {"kc": self.kc.clone(), "vc": self.vc.clone(), "slot_pos": self.slot_pos.clone(),
                "anchor": self.anchor.clone(), "context_end": self.context_end}

    def restore(self, snap: dict) -> None:
        self.kc.copy_(snap["kc"])
        self.vc.copy_(snap["vc"])
        self.slot_pos.copy_(snap["slot_pos"])
        self.anchor.copy_(snap["anchor"])
        self.context_end = int(snap["context_end"])

    def _cos_sin(self, start: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(start, start + n, device=self.dev, dtype=torch.float32)
        phase = pos[:, None] * self.inv_freq[None, :]
        return phase.cos(), phase.sin()

    def _kv(self, i: int, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer i's keys (k_norm, RoPE) and values for rows h: [kv_heads, n, head_dim] each."""

        c, p, pre = self.c, self.p, f"layers.{i}.self_attn."
        n = h.shape[0]
        k = F.linear(h, p[pre + "k_proj.weight"]).view(n, c.kv_heads, c.head_dim)
        v = F.linear(h, p[pre + "v_proj.weight"]).view(n, c.kv_heads, c.head_dim)
        k = _rope(_rms(k, p[pre + "k_norm.weight"], c.eps).transpose(0, 1), cos, sin)
        return k, v.transpose(0, 1)

    # -- the context ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def add_taps(self, taps: torch.Tensor) -> None:
        """Committed rows' taps [n, len(tap_layers) * D] at positions context_end .. context_end + n - 1.

        Only the window's last rows are projected: older ones are never read again.
        """

        c = self.c
        n = int(taps.shape[0])
        if n == 0:
            return
        if taps.shape[1] != c.hidden * len(c.tap_layers):
            raise ValueError(f"taps [{n}, {taps.shape[1]}], expected {len(c.tap_layers)} rows of {c.hidden}")
        m = min(n, self.ring)
        start = self.context_end + n - m
        ctx = _rms(self._lin(taps[n - m:].to(self.dev, self.dt), "fc.weight"), self.p["hidden_norm.weight"], c.eps)
        cos, sin = self._cos_sin(start, m)
        slots = torch.arange(start, start + m, device=self.dev) % self.ring
        for i in range(c.layers):
            k, v = self._kv(i, ctx, cos, sin)
            self.kc[i].index_copy_(1, slots, k)
            self.vc[i].index_copy_(1, slots, v)
        self.slot_pos.index_copy_(0, slots, torch.arange(start, start + m, device=self.dev))
        self.anchor.copy_(ctx[-1])
        self.context_end += n

    # -- the block ------------------------------------------------------------------------------------------------
    def _attend(self, i: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, s: int) -> torch.Tensor:
        """Block queries [H, n, hd] over the window's context rows and the block (bidirectional), with sinks."""

        c = self.c
        n = q.shape[1]
        g = c.heads // c.kv_heads
        keys = torch.cat([self.kc[i], k], dim=1).float()                                # [KV, ring + n, hd]
        vals = torch.cat([self.vc[i], v], dim=1).float()
        # FlashAttention's non-causal window is symmetric, (window - 1, window - 1): |q - k| < window. The real
        # window (1024) covers the whole block; a smaller one masks block keys too.
        qpos = torch.arange(s, s + n, device=self.dev)
        ok_ctx = (self.slot_pos[None, :] >= 0) & (self.slot_pos[None, :] > qpos[:, None] - c.window)
        ok_blk = (qpos[:, None] - qpos[None, :]).abs() < c.window
        ok = torch.cat([ok_ctx, ok_blk], dim=1)                                          # [n, ring + n]
        scores = torch.einsum("kgnd,kld->kgnl", q.float().view(c.kv_heads, g, n, c.head_dim), keys)
        scores = (scores * c.head_dim ** -0.5).masked_fill(~ok, float("-inf"))
        if c.sinks:
            sink = self.p[f"layers.{i}.self_attn.attention_sink_bias"].float().view(c.kv_heads, g, 1, 1)
            top = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
            e = torch.exp(scores - top)
            probs = e / (e.sum(dim=-1, keepdim=True) + torch.exp(sink - top))
        else:
            probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("kgnl,kld->kgnd", probs, vals).reshape(c.heads, n, c.head_dim)
        return out.transpose(0, 1).reshape(n, c.heads * c.head_dim).to(self.dt)

    def _layer(self, i: int, x: torch.Tensor, residual: torch.Tensor | None, s: int, cos: torch.Tensor,
               sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c, p, pre = self.c, self.p, f"layers.{i}."
        n = x.shape[0]
        G = c.hidden // c.conv_group
        residual = x if residual is None else (x.float() + residual.float()).to(self.dt)
        h = _rms(residual, p[pre + "input_layernorm.weight"], c.eps)
        coef = F.linear(h, p[pre + "attention_conv.kernel_projection.weight"]).view(n, 2, c.conv_taps, G)
        base = p[pre + "attention_conv.base_kernel"]
        h = _conv(h, coef[:, 0], base[0], c.conv_group)
        q = F.linear(h, p[pre + "self_attn.q_proj.weight"]).view(n, c.heads, c.head_dim)
        q = _rope(_rms(q, p[pre + "self_attn.q_norm.weight"], c.eps).transpose(0, 1), cos, sin)
        k, v = self._kv(i, h, cos, sin)
        h = F.linear(self._attend(i, q, k, v, s), p[pre + "self_attn.o_proj.weight"])
        h = _conv(h, coef[:, 1], base[1], c.conv_group)
        residual = (h.float() + residual.float()).to(self.dt)
        h = _rms(residual, p[pre + "post_attention_layernorm.weight"], c.eps)
        coef = F.linear(h, p[pre + "mlp_conv.kernel_projection.weight"]).view(n, 2, c.conv_taps, G)
        base = p[pre + "mlp_conv.base_kernel"]
        h = _conv(h, coef[:, 0], base[0], c.conv_group)
        h = self._lin(F.silu(self._lin(h, pre + "mlp.gate_proj.weight")) * self._lin(h, pre + "mlp.up_proj.weight"),
                      pre + "mlp.down_proj.weight")
        return _conv(h, coef[:, 1], base[1], c.conv_group), residual

    def block_hidden(self, pending: int, depth: int) -> torch.Tensor:
        """The final-normed hidden rows of the mask tokens [depth, D]: row j drafts position context_end + 1 + j."""

        c, s = self.c, self.context_end
        n = depth + 1
        ids = torch.full((n,), c.mask_id, dtype=torch.int64, device=self.dev)
        ids[0] = int(pending)
        x = self.embed(ids).to(self.dt)
        cos, sin = self._cos_sin(s, n)
        residual = None
        for i in range(c.layers):
            x, residual = self._layer(i, x, residual, s, cos, sin)
        return _rms((x.float() + residual.float()).to(self.dt), self.p["norm.weight"], c.eps)[1:]

    # -- the lattice ----------------------------------------------------------------------------------------------
    def _build_lattice_bias(self) -> torch.Tensor:
        """[heads, (block - 1) * k, (block - 1) * k]: relative-slot plus same-slot bias over the flattened lattice."""

        c, p = self.c, self.p
        slots = torch.arange(c.block - 1, device=self.dev).repeat_interleave(c.top_k)
        rel = slots[:, None] - slots[None, :]
        bias = p["lilicorr.relative_slot_bias"][:, rel + c.block - 1]
        same = (rel == 0).to(bias.dtype)[None] * p["lilicorr.same_slot_bias"].view(-1, 1, 1)
        return (bias + same).contiguous()

    def lattice_scores(self, cand: torch.Tensor, log_probs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """vLLM ``LiLiCorrHead.forward`` for one sequence: [N, k, k] fp32, row 0 the anchor's scores repeated."""

        c, p, q = self.c, self.p, "lilicorr."
        N, k = log_probs.shape
        L = c.lili_hidden

        def lin(x: torch.Tensor, name: str) -> torch.Tensor:
            return F.linear(x, p[q + name + ".weight"], p[q + name + ".bias"])

        lp = log_probs.float()
        rank = torch.arange(k, device=self.dev, dtype=torch.float32) / float(max(k - 1, 1))
        top1 = (torch.arange(k, device=self.dev) == 0).float()
        feats = torch.stack([lp, lp.exp(), lp - lp.amax(dim=-1, keepdim=True), rank.expand_as(lp),
                             top1.expand_as(lp)], dim=-1).to(self.dt)
        feats = F.layer_norm(feats, (LATTICE_FEATURES,), p[q + "feature_norm.weight"], p[q + "feature_norm.bias"])
        x = lin(self.embed(cand.reshape(-1)).to(self.dt), "token_proj").view(N, k, L)
        x = x + lin(hidden.to(self.dt), "pass_hidden_proj")[:, None, :]
        x = x + lin(F.silu(lin(feats, "feature_mlp.up_proj")), "feature_mlp.down_proj")
        x = x + p[q + "slot_embedding"][0, 0, :N] + p[q + "rank_embedding"][0, 0]
        x = x.reshape(1, N * k, L)
        valid = 1.0 if self.context_end > 0 else 0.0
        anchor = lin(self.anchor, "context_proj") * valid
        bias = self._lattice_bias[None, :, :N * k, :N * k]
        hh = c.lili_heads
        for i in range(c.lili_layers):
            pre = f"{q}layers.{i}."
            h = _rms(x, p[pre + "attn_norm.weight"], c.eps)
            qkv = F.linear(h, p[pre + "attn.in_proj_weight"], p[pre + "attn.in_proj_bias"])
            qq, kk, vv = (t.view(1, N * k, hh, L // hh).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
            att = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=bias)
            x = x + F.linear(att.transpose(1, 2).reshape(1, N * k, L), p[pre + "attn.out_proj.weight"],
                             p[pre + "attn.out_proj.bias"])
            h = _rms(x, p[pre + "mlp_norm.weight"], c.eps)
            x = x + F.linear(F.silu(F.linear(h, p[pre + "mlp.up_proj.weight"], p[pre + "mlp.up_proj.bias"])),
                             p[pre + "mlp.down_proj.weight"], p[pre + "mlp.down_proj.bias"])
        x = _rms(x, p[q + "output_norm.weight"], c.eps).view(N, k, L)
        a = _rms(anchor, p[q + "anchor_norm.weight"], c.eps)
        w = p[q + "factor_input_proj.weight"]
        pre_f = (F.linear(x, w[:, :L], p[q + "factor_input_proj.bias"]) + F.linear(a, w[:, L:2 * L])
                 + F.linear(x * a, w[:, 2 * L:]))
        f = F.silu(pre_f)
        out_vec = F.normalize(lin(f, "out_head"), dim=-1, eps=c.vector_eps)
        in_vec = F.normalize(lin(f, "in_head"), dim=-1, eps=c.vector_eps)
        a_out = F.normalize(lin(a, "anchor_out_head"), dim=-1, eps=c.vector_eps)
        start = (a_out[None, :] * in_vec[0]).sum(dim=-1)                                   # [k]
        pairs = torch.matmul(out_vec[:-1], in_vec[1:].transpose(-1, -2))                   # [N - 1, k, k]
        first = start.float()[None, None, :].expand(1, k, k)
        return c.logit_scale * torch.cat([first, pairs.float()], dim=0)

    # -- drafts ---------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def candidates(self, pending: int, depth: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The block pass: top-k candidate ids [N, k], their log-probabilities [N, k], the lattice scores."""

        hidden = self.block_hidden(pending, depth)
        logits = self.lm_head(hidden).float()
        vals, ids = torch.topk(logits, self.c.top_k, dim=-1)
        log_probs = vals - torch.logsumexp(logits, dim=-1, keepdim=True)
        return ids, log_probs, self.lattice_scores(ids, log_probs, hidden)

    def walk(self, cand: np.ndarray, scores: np.ndarray, sampling: Sampling | None) -> list[int]:
        """One candidate a slot along the lattice: argmax of scores[s, previous index], under keyed noise when sampled.

        The noise is the target's (seed, position, token id) draw, so a draft agrees with the target's sample as
        often as the lattice agrees with the target's distribution. Exactness never depends on it.
        """

        N = cand.shape[0]
        noise = None
        if sampling is not None and sampling.temperature > 0:
            positions = self.context_end + 1 + np.arange(N)
            noise = -np.log(-np.log(uniform_rows(sampling.seed, positions, cand.astype(np.int64))))
        temp = float(sampling.temperature) if noise is not None else 1.0
        prev, out = 0, []
        for s in range(N):
            row = scores[s, prev].astype(np.float64)
            pick = row / temp + noise[s] if noise is not None else row
            prev = int(np.argmax(pick))
            out.append(int(cand[s, prev]))
        return out

    @torch.no_grad()
    def propose(self, pending: int, depth: int, sampling: Sampling | None) -> list[int]:
        """Up to ``depth`` (at most block - 1) drafts for the positions after ``pending``, which sits at context_end."""

        depth = min(int(depth), self.c.block - 1)
        if depth < 1 or self.context_end == 0:
            return []
        ids, log_probs, scores = self.candidates(pending, depth)
        cand, sc = ids.cpu().numpy(), scores.cpu().numpy()
        tokens = self.walk(cand, sc, sampling)
        self.last = Proposal(tokens, ids, log_probs, scores)
        return tokens


# -- verification ---------------------------------------------------------------------------------------------------
def _sample(engine, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None) -> list[int]:
    """The window's tokens from the sampler serial decoding uses: the engine's own when it has one.

    An engine that samples inside ``verify`` (``tokens()``, like Nemotron-H's keyed GPU sampler) wins; then an
    engine ``sample(logits, positions, sampling)``; then the shared CUDA rule.
    """

    if hasattr(engine, "tokens"):
        return [int(t) for t in engine.tokens()[:len(positions)]]
    sample = getattr(engine, "sample", None)
    if sample is not None:
        return list(sample(logits, positions, sampling))
    from tensorfold.cuda.sampling import sample_rows

    return sample_rows(logits, positions, sampling)


OWED = 256             # committed tap rows a drafter may owe before it takes them, drafting or not


@torch.no_grad()
def lilicorr_decode(engine, drafter: Drafter, first: int, count: int, sampling: Sampling | None, *,
                    stop_eos: bool = False, on_tokens: Callable[[list[int]], bool | None] | None = None,
                    depth: int | None = None, policy=None) -> DecodeResult:
    """``count`` tokens (the first is ``first``, the prompt's sample) in verify rounds of the pending token + drafts.

    A round keeps the target's keyed samples through the first draft that differs from them (or the first end token
    under ``stop_eos``), so the reply equals serial decoding's. It commits no row past ``count`` tokens and stops
    when ``on_tokens`` returns True. The drafter must hold the prompt's context (``context_end == engine.pos``), and
    holds the reply's by the end. ``policy`` (a ``depth.DepthPolicy``) picks each round's drafts in place of the
    fixed ``depth``; a round of none runs no drafter, and the drafter takes the rows it owes before it next drafts.
    """

    if drafter.context_end != engine.pos:
        raise ValueError(f"the drafter holds {drafter.context_end} positions, the engine {engine.pos}")
    depth = drafter.c.block - 1 if depth is None else max(0, min(int(depth), drafter.c.block - 1))
    eos = set(engine.eos)
    if hasattr(engine, "set_sampling"):
        engine.set_sampling(sampling)
    out = [int(first)]
    res = DecodeResult([], 0.0, 0)
    owed: list[torch.Tensor] = []

    def pay() -> None:
        if owed:
            drafter.add_taps(owed[0] if len(owed) == 1 else torch.cat(owed))
            owed.clear()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in eos):
        t0 = time.perf_counter()
        room = count - len(out) - 1                       # the round may add 1 + n tokens
        n = min(policy.next(room) if policy is not None else depth, room, drafter.c.block - 1)
        drafts = []
        if n > 0:
            pay()
            drafts = drafter.propose(out[-1], n, sampling)
        t1 = time.perf_counter()
        logits = engine.verify([out[-1]] + drafts)
        pos = engine.pos
        sampled = _sample(engine, logits[:1 + len(drafts)], [pos + 1 + r for r in range(1 + len(drafts))], sampling)
        t2 = time.perf_counter()
        keep = 1
        for d, s in zip(drafts, sampled):
            if s != d or (stop_eos and s in eos):
                break
            keep += 1
        engine.commit(keep)
        owed.append(engine.tap_rows(keep))
        if sum(t.shape[0] for t in owed) >= OWED:
            pay()
        new = [int(t) for t in sampled[:keep]]
        out.extend(new)
        res.rounds += 1
        res.drafted += len(drafts)
        res.accepted += keep - 1
        res.widths.append(1 + len(drafts))
        res.keeps.append(keep)
        stop = on_tokens is not None and on_tokens(new)
        if policy is not None:
            policy.done(len(drafts), keep - 1, 1e3 * (t2 - t1), 1e3 * (t1 - t0), 1e3 * (time.perf_counter() - t2))
        if stop:
            break
    pay()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    res.seconds = time.perf_counter() - start
    res.tokens = out
    return res


__all__ = ["DecodeResult", "DraftConfig", "Drafter", "Proposal", "dense_tensors", "device_bytes",
           "expected_shapes", "fp4_kept", "lilicorr_decode", "read_checkpoint"]
