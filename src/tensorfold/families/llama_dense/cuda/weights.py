"""A Llama checkpoint as projections the row-invariant kernels read: bf16 as stored, an EXL3 pack or NVFP4 / FP8."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tensorfold.families.qwen3_5.cuda.weights import Plain


@dataclass
class Config:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    eps: float
    rope_theta: float
    eos: tuple[int, ...]

    @classmethod
    def read(cls, model_dir: str | Path) -> Config:
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        t = raw.get("text_config", raw)
        heads = int(t["num_attention_heads"])
        gen = Path(model_dir) / "generation_config.json"
        found: list[int] = []
        for value in (t.get("eos_token_id"), json.loads(gen.read_text()).get("eos_token_id") if gen.exists() else None):
            found += [] if value is None else [int(e) for e in value] if isinstance(value, list) else [int(value)]
        if not found:
            raise ValueError("no eos_token_id in config.json or generation_config.json")
        return cls(hidden=int(t["hidden_size"]), intermediate=int(t["intermediate_size"]),
                   layers=int(t["num_hidden_layers"]), heads=heads,
                   kv_heads=int(t.get("num_key_value_heads") or heads),
                   head_dim=int(t.get("head_dim") or int(t["hidden_size"]) // heads), vocab=int(t["vocab_size"]),
                   eps=float(t.get("rms_norm_eps", 1e-6)), rope_theta=rope_theta(t),
                   eos=tuple(dict.fromkeys(found)))


def rope_theta(t: dict[str, Any]) -> float:
    """The rotation base: top-level ``rope_theta``, or under ``rope_parameters`` where transformers 5 writes it (the
    ModelOpt export of Nanbeige4.3-33B); read as 10,000 there, its long replies turned to junk (tf-runs/round3-1027)."""

    value = t.get("rope_theta", (t.get("rope_parameters") or {}).get("rope_theta"))
    return 10000.0 if value is None else float(value)


@dataclass
class Attention:
    q: Any
    k: Any
    v: Any
    o: Any


@dataclass
class Layer:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    attn: Attention
    gate: Any
    up: Any
    down: Any


@dataclass
class Weights:
    config: Config
    embed: Plain
    layers: list[Layer]
    norm: torch.Tensor
    head: Any
    inv_freq: torch.Tensor
    quant: str = "bf16"                              # "bf16", "exl3" or "nvfp4"
    prompt_rows: int = 4096                          # a prompt chunk's rows: any count gives the same bits
    precision: str = field(default="full")           # "checkpoint": NVFP4 / FP8 layers in their own math

    def nbytes(self) -> int:
        mods = [self.embed, self.head]
        for layer in self.layers:
            a = layer.attn
            mods += [a.q, a.k, a.v, a.o, layer.gate, layer.up, layer.down]
        return sum(m.nbytes() for m in mods)


def inv_freq(cfg: Config, device) -> torch.Tensor:
    """transformers' default rope frequencies over the whole head, in fp64 then fp32."""

    half = cfg.head_dim // 2
    return (cfg.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)).to(torch.float32).to(device)


def build(cfg: Config, linear: Callable[[str], Any], tensor: Callable[[str], torch.Tensor], device,
          quant: str) -> Weights:
    """``linear(name)`` reads a projection, ``tensor(name)`` a stored tensor; each projection's shape is checked."""

    def proj(name: str, n: int, k: int):
        got = linear(name)
        if (got.n, got.k) != (n, k):
            raise ValueError(f"{name} is {got.n} x {got.k}; config.json implies {n} x {k}")
        return got

    d, i, hd = cfg.hidden, cfg.intermediate, cfg.head_dim
    layers = []
    for j in range(cfg.layers):
        p = f"model.layers.{j}."
        attn = Attention(q=proj(p + "self_attn.q_proj", cfg.heads * hd, d),
                         k=proj(p + "self_attn.k_proj", cfg.kv_heads * hd, d),
                         v=proj(p + "self_attn.v_proj", cfg.kv_heads * hd, d),
                         o=proj(p + "self_attn.o_proj", d, cfg.heads * hd))
        layers.append(Layer(input_norm=tensor(p + "input_layernorm.weight"),
                            post_norm=tensor(p + "post_attention_layernorm.weight"), attn=attn,
                            gate=proj(p + "mlp.gate_proj", i, d), up=proj(p + "mlp.up_proj", i, d),
                            down=proj(p + "mlp.down_proj", d, i)))
    embed = Plain(tensor("model.embed_tokens.weight"))
    if (embed.n, embed.k) != (cfg.vocab, d):
        raise ValueError(f"the embedding is {embed.n} x {embed.k}; config.json implies {cfg.vocab} x {d}")
    return Weights(cfg, embed, layers, tensor("model.norm.weight"), proj("lm_head", cfg.vocab, d),
                   inv_freq(cfg, device), quant=quant)


def skipped(name: str) -> bool:
    """Tensors the forward never reads: rope tables older exports saved beside each layer."""

    return name.endswith("rotary_emb.inv_freq")


def load(model_dir: str | Path, device: str = "cuda") -> Weights:
    """bf16 projections as stored, an EXL3 pack (``exl3_load``) or NVFP4 / FP8 (``nvfp4_load``)."""

    from tensorfold.cuda.nvfp4.format import is_quantized
    from tensorfold.families.qwen3_5.cuda.exl3_load import quant_config
    from tensorfold.families.qwen3_5.cuda.weights import _Tensors

    model_dir = Path(model_dir)
    if quant_config(model_dir) is not None:
        from .exl3_load import load_exl3

        return load_exl3(model_dir, device)
    if is_quantized(model_dir):
        from .nvfp4_load import load_nvfp4

        return load_nvfp4(model_dir, device)
    cfg = Config.read(model_dir)
    t = _Tensors(model_dir, device, skip=skipped)

    def tensor(name: str) -> torch.Tensor:
        w = t.pop(name)
        if w.dtype != torch.bfloat16:          # b16's outputs take the weight's dtype; the attention kernels read bf16
            raise ValueError(f"{name} is {w.dtype}: the engine reads bf16 checkpoints (convert fp16 or fp32 first)")
        return w.contiguous()

    w = build(cfg, lambda name: Plain(tensor(name + ".weight")), tensor, device, "bf16")
    left = list(t)
    t.close()
    if left:
        raise ValueError(f"unused checkpoint tensors: {left[:5]} ...")
    torch.cuda.empty_cache()
    return w
