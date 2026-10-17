"""LFM2-MoE's config.json read once, with the reasons a checkpoint is refused (no torch, so ``check`` runs anywhere)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorfold.families.lfm2.layout import core

BLOCK = (128, 128)          # compressed-tensors' FP8 weight blocks (rows, inputs)
KINDS = ("conv", "full_attention")
# compressed-tensors' ``ignore`` patterns of the 24B-A2B FP8 export: what it keeps in bf16
FP8_KEEPS = ("lm_head", "model.embed_tokens", "re:.*\\.gate$", "re:.*\\.(?:conv|short_conv)\\..*")


@dataclass(frozen=True)
class Config:
    hidden: int
    vocab: int
    layers: tuple[str, ...]          # "conv" or "full_attention", by layer
    heads: int
    kv_heads: int
    head_dim: int
    conv_taps: int                   # conv_L_cache: the short conv's taps (a layer keeps taps - 1 inputs)
    dense_layers: int                # the first layers run a dense SwiGLU, the rest the MoE
    intermediate: int                # the dense SwiGLU's width
    experts: int
    top_k: int
    moe_width: int
    scaling: float                   # routed_scaling_factor
    eps: float
    theta: float
    context: int
    fp8: bool                        # compressed-tensors 128x128 block FP8 linears (else every weight bf16)
    eos: tuple[int, ...] = (124900,)
    bos: int = 124894

    @property
    def attention_layers(self) -> tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layers) if kind == "full_attention")

    @property
    def conv_layers(self) -> tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layers) if kind == "conv")

    def moe(self, layer: int) -> bool:
        return layer >= self.dense_layers

    @classmethod
    def read(cls, model_dir: str | Path) -> Config:
        model_dir = Path(model_dir)
        raw = json.loads((model_dir / "config.json").read_text())
        why = refusal(raw)
        if why:
            raise ValueError(f"LFM2-MoE cannot run this checkpoint: {why}")
        return cls.parse(raw, eos_tokens(model_dir, raw))

    @classmethod
    def parse(cls, raw: dict[str, Any], eos: tuple[int, ...] = (124900,)) -> Config:
        c = core(raw)
        return cls(hidden=c.hidden, vocab=c.vocab, layers=c.layers, heads=c.heads, kv_heads=c.kv_heads,
                   head_dim=c.head_dim, conv_taps=c.taps, dense_layers=int(raw["num_dense_layers"]),
                   intermediate=int(raw["intermediate_size"]), experts=int(raw["num_experts"]),
                   top_k=int(raw["num_experts_per_tok"]), moe_width=int(raw["moe_intermediate_size"]),
                   scaling=float(raw.get("routed_scaling_factor") or 1.0), eps=c.eps, theta=c.theta,
                   context=int(raw.get("max_position_embeddings") or 0),
                   fp8=bool(raw.get("quantization_config")), eos=eos, bos=int(raw.get("bos_token_id") or 1))


def eos_tokens(model_dir: Path, raw: dict[str, Any]) -> tuple[int, ...]:
    """generation_config.json's end tokens (<|im_end|> 124900 on the 24B-A2B), else config.json's."""

    found: list[int] = []
    sources = [raw.get("eos_token_id")]
    gen = model_dir / "generation_config.json"
    if gen.is_file():
        sources.insert(0, json.loads(gen.read_text()).get("eos_token_id"))
    for value in sources:
        for t in value if isinstance(value, list) else [value]:
            if isinstance(t, int) and t not in found:
                found.append(t)
    return tuple(found) or (124900,)


def _fp8_refusal(q: dict[str, Any]) -> str | None:
    if str(q.get("quant_method", "")).lower() != "compressed-tensors":
        return f"its weights are {q.get('quant_method')!r}; the engine reads bf16 or compressed-tensors block FP8"
    groups = list((q.get("config_groups") or {}).values())
    if len(groups) != 1:
        return f"it has {len(groups)} quantization groups; the engine reads one (FP8_BLOCK)"
    w = groups[0].get("weights") or {}
    if (w.get("type"), w.get("num_bits"), w.get("strategy"), list(w.get("block_structure") or [])) != \
            ("float", 8, "block", list(BLOCK)):
        return (f"its weights are {w.get('num_bits')}-bit {w.get('type')} by {w.get('strategy')} "
                f"{w.get('block_structure')}; the engine reads 8-bit float in {BLOCK[0]}x{BLOCK[1]} blocks")
    if w.get("dynamic") or not w.get("symmetric", True):
        return "its FP8 weights are dynamic or asymmetric"
    if q.get("quantization_status") not in (None, "compressed"):
        return f"its quantization status is {q.get('quantization_status')!r}, not compressed"
    keeps = [p for p in q.get("ignore") or [] if p not in FP8_KEEPS]
    if keeps:
        return f"it keeps {keeps} unquantized beyond the head, embeddings, routers and convs"
    return None


def refusal(raw: dict[str, Any]) -> str | None:
    """Why the engine cannot run a checkpoint, from config.json alone; None when it can."""

    if raw.get("model_type") != "lfm2_moe":
        return f"its model_type is {raw.get('model_type')!r}, not lfm2_moe"
    for key, want in (("tie_word_embeddings", True), ("conv_bias", False), ("norm_topk_prob", True),
                      ("use_expert_bias", True)):
        if raw.get(key, want) is not want:
            return f"{key} is {raw.get(key)!r}; the engine runs {want}"
    kinds = raw.get("layer_types") or []
    if len(kinds) != int(raw.get("num_hidden_layers") or -1) or set(kinds) - set(KINDS):
        return (f"its layer_types are {sorted(set(kinds))} over {len(kinds)} layers; the engine runs conv and "
                f"full_attention")
    if "full_attention" not in kinds:
        return "it has no attention layer"
    rope = raw.get("rope_parameters") or {"rope_type": "default"}
    if rope.get("rope_type", "default") != "default":
        return f"its RoPE is {rope.get('rope_type')!r}; the engine runs the default rotation"
    if int(raw.get("conv_L_cache") or 0) < 2:
        return f"conv_L_cache is {raw.get('conv_L_cache')}"
    hidden, heads = int(raw["hidden_size"]), int(raw["num_attention_heads"])
    head_dim = int(raw.get("head_dim") or hidden // heads)
    if head_dim not in (64, 128) or heads % int(raw["num_key_value_heads"]) or \
            heads // int(raw["num_key_value_heads"]) > 16:
        return f"its attention is {heads} heads of {head_dim} over {raw['num_key_value_heads']} KV heads"
    if hidden % BLOCK[1] or int(raw["moe_intermediate_size"]) % BLOCK[0] or int(raw["intermediate_size"]) % 64:
        return "its widths are not whole 128-wide FP8 blocks"
    if int(raw["num_experts_per_tok"]) > int(raw["num_experts"]):
        return "it picks more experts than it has"
    q = raw.get("quantization_config")
    return _fp8_refusal(q) if q else None

