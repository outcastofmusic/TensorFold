"""A tiny random Llama checkpoint for the llama_dense tests: bf16 as released, an EXL3 pack or a ModelOpt NVFP4 export.

Shapes keep every projection a multiple of 128 (the EXL3 Hadamard blocks) and the head at 128 (tree attention)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

LAYERS, HIDDEN, INTER, HEADS, KV, HEAD_DIM, VOCAB = 2, 256, 512, 4, 2, 128, 512


def text_config(**change) -> dict:
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": HIDDEN,
            "intermediate_size": INTER, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
            "num_key_value_heads": KV, "head_dim": HEAD_DIM, "vocab_size": VOCAB, "rms_norm_eps": 1e-5,
            "rope_theta": 1.2e8, "rope_scaling": None, "hidden_act": "silu", "attention_bias": False,
            "mlp_bias": False, "tie_word_embeddings": False, "max_position_embeddings": 4096,
            "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "bfloat16", **change}


PROJECTIONS = {"self_attn.q_proj": (HEADS * HEAD_DIM, HIDDEN), "self_attn.k_proj": (KV * HEAD_DIM, HIDDEN),
               "self_attn.v_proj": (KV * HEAD_DIM, HIDDEN), "self_attn.o_proj": (HIDDEN, HEADS * HEAD_DIM),
               "mlp.gate_proj": (INTER, HIDDEN), "mlp.up_proj": (INTER, HIDDEN), "mlp.down_proj": (HIDDEN, INTER)}


def tensors(seed: int = 0, layers: int = LAYERS) -> dict[str, torch.Tensor]:
    """Every tensor of the bf16 checkpoint by name; norms near 1 (stored as gamma, transformers' Llama layout)."""

    g = torch.Generator().manual_seed(seed)

    def rand(*shape, scale):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    out = {"model.embed_tokens.weight": rand(VOCAB, HIDDEN, scale=1.0), "model.norm.weight": 1 + rand(HIDDEN, scale=0.1),
           "lm_head.weight": rand(VOCAB, HIDDEN, scale=HIDDEN ** -0.5)}
    for i in range(layers):
        p = f"model.layers.{i}."
        out[p + "input_layernorm.weight"] = 1 + rand(HIDDEN, scale=0.1)
        out[p + "post_attention_layernorm.weight"] = 1 + rand(HIDDEN, scale=0.1)
        for name, (n, k) in PROJECTIONS.items():
            out[p + name + ".weight"] = rand(n, k, scale=k ** -0.5)
    return out


def _save(folder: Path, named: dict[str, torch.Tensor], config: dict) -> Path:
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 3]}))
    names = sorted(named)
    half = len(names) // 2                       # two shards, so a reader crosses files
    save_file({n: named[n].contiguous() for n in names[:half]}, str(folder / "model-00001-of-00002.safetensors"))
    save_file({n: named[n].contiguous() for n in names[half:]}, str(folder / "model-00002-of-00002.safetensors"))
    return folder


def write(folder: Path, seed: int = 0, layers: int = LAYERS, **change) -> Path:
    """The bf16 checkpoint as transformers saves it."""

    return _save(Path(folder), tensors(seed, layers), text_config(num_hidden_layers=layers, **change))


def write_exl3(folder: Path, seed: int = 0, bits: int = 3, head: bool = True) -> Path:
    """An EXL3 pack as exllamav3's convert writes it: each projection's trellis (int16 [K/16, N/16, 16 bits]), suh and
    svh (fp16), mul1 markers; embedding and norms as stored, the head quantized unless ``head`` is False."""

    g = torch.Generator().manual_seed(seed + 1)
    plain = tensors(seed)
    named: dict[str, torch.Tensor] = {}

    def group(prefix: str, n: int, k: int) -> None:
        named[prefix + ".trellis"] = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * bits),
                                                   generator=g).to(torch.int16)
        named[prefix + ".suh"] = (torch.randn(k, generator=g) * 0.05).half()
        named[prefix + ".svh"] = (torch.randn(n, generator=g) * 0.05).half()
        named[prefix + ".mul1"] = torch.tensor(0x83DCD12D - (1 << 32), dtype=torch.int32)

    for i in range(LAYERS):
        for name, (n, k) in PROJECTIONS.items():
            group(f"model.layers.{i}.{name}", n, k)
    if head:
        group("lm_head", VOCAB, HIDDEN)
    else:
        named["lm_head.weight"] = plain["lm_head.weight"]
    for name, t in plain.items():
        if "norm" in name or "embed" in name:
            named[name] = t
    config = text_config(quantization_config={"quant_method": "exl3", "version": "0.0.10", "bits": float(bits),
                                              "head_bits": bits, "calibration": {}, "codebook": "mul1"})
    return _save(Path(folder), named, config)


def write_nvfp4(folder: Path, seed: int = 0) -> Path:
    """A ModelOpt NVFP4 export: projections as e2m1 codes (uint8 [N, K/2]), e4m3 scales a 16 inputs, a tensor scale and
    a static input scale; embedding, norms and head bf16 (ModelOpt's default exclusions)."""

    g = torch.Generator().manual_seed(seed + 2)
    plain = tensors(seed)
    named = {name: t for name, t in plain.items() if "_proj." not in name}
    for i in range(LAYERS):
        for name, (n, k) in PROJECTIONS.items():
            p = f"model.layers.{i}.{name}"
            named[p + ".weight"] = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
            named[p + ".weight_scale"] = torch.randint(0x20, 0x40, (n, k // 16), generator=g,
                                                       dtype=torch.uint8).view(torch.float8_e4m3fn)
            named[p + ".weight_scale_2"] = torch.tensor(0.05 / k ** 0.5, dtype=torch.float32)
            named[p + ".input_scale"] = torch.tensor(0.01, dtype=torch.float32)
    config = text_config(quantization_config={"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16,
                                              "exclude_modules": ["lm_head"]})
    return _save(Path(folder), named, config)
