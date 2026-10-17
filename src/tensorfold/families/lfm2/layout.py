"""What the two LFM2 families' checkpoints share: a layer's tensor names and the text config's core fields.

d1 (LFM2-VL, this package) and LFM2.5-24B-A2B (``lfm2_moe``) write their mixer layers alike: a gated short conv or GQA
attention with per-head query and key norms, between an ``operator_norm`` and an ``ffn_norm``. Each family keeps its
own forward, feed-forward and refusals. Names here are relative to ``layers.{i}.`` and leave off ``.weight``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

NORMS = ("operator_norm", "ffn_norm")
CONV = ("conv.in_proj", "conv.out_proj")                    # [3 hidden, hidden] (B, C, x), then [hidden, hidden]
CONV_TAPS = "conv.conv"                                     # [hidden, 1, taps], the oldest input's tap first
QKV = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
ATTENTION_OUT = "self_attn.out_proj"
HEAD_NORMS = ("self_attn.q_layernorm", "self_attn.k_layernorm")
MLP = ("feed_forward.w1", "feed_forward.w3", "feed_forward.w2")     # gate, up, down (the SwiGLU)


@dataclass(frozen=True)
class Core:
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: tuple[str, ...]          # "conv" or "full_attention", by layer
    taps: int                        # conv_L_cache: a short conv's taps
    eps: float
    theta: float
    vocab: int


def core(text: dict[str, Any]) -> Core:
    """The fields both forwards read from a text config (transformers' Lfm2Config and Lfm2MoeConfig names)."""

    hidden, heads = int(text["hidden_size"]), int(text["num_attention_heads"])
    rope = text.get("rope_parameters") or {}
    return Core(hidden=hidden, heads=heads, kv_heads=int(text["num_key_value_heads"]),
                head_dim=int(text.get("head_dim") or hidden // heads), layers=tuple(text["layer_types"]),
                taps=int(text["conv_L_cache"]), eps=float(text.get("norm_eps") or text.get("block_norm_eps") or 1e-5),
                theta=float(rope.get("rope_theta") or text.get("rope_theta") or 1e6), vocab=int(text["vocab_size"]))
