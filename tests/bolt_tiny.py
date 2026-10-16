"""A tiny Nemotron 3.5 Bolt checkpoint in the real layout: omni config with llm_config.block_configs, ModelOpt's
hf_quant_config.json, Q8_0 / IQ2_XXS ggml blocks stored [N, K // block, bytes], NVFP4 Mamba projections, per-expert
tensors, and vision / projector tensors the text engine skips. Shapes are the smallest the CUDA kernels take."""

from __future__ import annotations

import json
from pathlib import Path

import torch

PATTERN = ("mamba", "moe", "attention", "moe", "mamba", "moe")
EXPERTS = (8, 12, 8)
WIDTHS = (256, 512, 256)


def llm_config(*, vocab: int = 1024, hidden: int = 512, top_k: int = 3) -> dict:
    blocks, mi = [], 0
    for kind in PATTERN:
        if kind == "moe":
            blocks.append({"block_type": "moe", "moe_intermediate_size": WIDTHS[mi], "n_routed_experts": EXPERTS[mi],
                           "num_experts_per_tok": top_k})
            mi += 1
        else:
            blocks.append({"block_type": kind})
    return {"model_type": "nemotron_h_puzzle", "block_configs": blocks, "layers_block_type": list(PATTERN),
            "hidden_size": hidden, "vocab_size": vocab, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 64, "mamba_num_heads": 8, "mamba_head_dim": 32, "n_groups": 2, "ssm_state_size": 128,
            "conv_kernel": 4, "moe_latent_size": 256, "moe_shared_expert_intermediate_size": 512,
            "n_shared_experts": 1, "n_routed_experts": 512, "num_experts_per_tok": top_k,
            "moe_intermediate_size": 2688, "routed_scaling_factor": 5.0, "norm_topk_prob": True,
            "layer_norm_epsilon": 1e-5, "time_step_limit": [0.0, {"__float__": "Infinity"}], "eos_token_id": 2,
            "mtp_block_configs": [], "num_nextn_predict_layers": 0}


def q8_0(w: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """fp32 [N, K] -> ggml Q8_0 blocks uint8 [N, K // 32, 34] (absmax / 127 scale, rounded codes)."""

    n, k = w.shape
    g = w.reshape(n, k // 32, 32)
    d = (g.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-8).to(torch.float16)
    q = torch.round(g / d.float()).clamp(-127, 127).to(torch.int8)
    return torch.cat([d.view(torch.uint8).reshape(n, k // 32, 2), q.view(torch.uint8)], dim=-1).contiguous()


def iq2xxs(n: int, k: int, rng: torch.Generator, scale: float = 0.02) -> torch.Tensor:
    """Random ggml IQ2_XXS blocks uint8 [N, K // 256, 66]: fp16 scales near ``scale`` / 100, random words."""

    nb = k // 256
    d = (torch.rand((n, nb, 1), generator=rng) * 0.5 + 0.5) * (scale / 100)
    words = torch.randint(0, 256, (n, nb, 64), generator=rng, dtype=torch.int32).to(torch.uint8)
    return torch.cat([d.to(torch.float16).view(torch.uint8).reshape(n, nb, 2), words], dim=-1).contiguous()


def nvfp4(n: int, k: int, rng: torch.Generator, scale: float = 0.02) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codes = torch.randint(0, 256, (n, k // 2), generator=rng, dtype=torch.int32).to(torch.uint8)
    scales = (torch.rand((n, k // 16), generator=rng) * 1.5 + 0.5).to(torch.float8_e4m3fn)
    return codes, scales, torch.tensor(scale / 6 / 2, dtype=torch.float32)


def write(path: Path, *, seed: int = 0, vocab: int = 1024, hidden: int = 512, top_k: int = 3,
          extra: dict[str, torch.Tensor] | None = None) -> Path:
    from safetensors.torch import save_file

    rng = torch.Generator().manual_seed(seed)
    path.mkdir(parents=True, exist_ok=True)
    t = llm_config(vocab=vocab, hidden=hidden, top_k=top_k)
    quant = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "config_groups": {   # the real one's shape
        "group_0": {"quant_algo": "Q8_0", "group_size": 32, "packing": "ggml", "targets": []},
        "group_1": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}, "targets": []},
        "group_2": {"quant_algo": "IQ2_XXS", "group_size": 256, "packing": "ggml", "targets": []}}}
    (path / "config.json").write_text(json.dumps({"model_type": "nemotron_h_omni", "eos_token_id": 11,
                                                  "llm_config": t, "vision_config": {"model_type": "radio"},
                                                  "quantization_config": quant}))
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 11]}))
    tensors: dict[str, torch.Tensor] = {}
    formats: dict[str, dict] = {}
    root = "language_model."

    def rand(*shape, s: float = 0.02) -> torch.Tensor:
        return torch.randn(shape, generator=rng) * s

    def add_q8(name: str, n: int, k: int) -> None:
        tensors[root + name + ".weight"] = q8_0(rand(n, k), rng)
        formats[root + name] = {"quant_algo": "Q8_0", "group_size": 32, "packing": "ggml"}

    def add_iq2(name: str, n: int, k: int) -> None:
        tensors[root + name + ".weight"] = iq2xxs(n, k, rng)
        formats[root + name] = {"quant_algo": "IQ2_XXS", "group_size": 256, "packing": "ggml"}

    def add_fp4(name: str, n: int, k: int) -> None:
        codes, scales, scale2 = nvfp4(n, k, rng)
        tensors[root + name + ".weight"] = codes
        tensors[root + name + ".weight_scale"] = scales
        tensors[root + name + ".weight_scale_2"] = scale2
        formats[root + name] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}

    def norm(n: int) -> torch.Tensor:
        return (1.0 + rand(n, s=0.1)).to(torch.bfloat16)

    heads, kv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    mh, mhd, groups, state = t["mamba_num_heads"], t["mamba_head_dim"], t["n_groups"], t["ssm_state_size"]
    xd = mh * mhd
    conv_dim = xd + 2 * groups * state
    latent, shared = t["moe_latent_size"], t["moe_shared_expert_intermediate_size"]
    tensors[root + "backbone.embeddings.weight"] = rand(vocab, hidden, s=1.0).to(torch.bfloat16)
    tensors[root + "backbone.norm_f.weight"] = norm(hidden)
    add_q8("lm_head", vocab, hidden)
    mi = 0
    for i, kind in enumerate(PATTERN):
        p = f"backbone.layers.{i}."
        tensors[root + p + "norm.weight"] = norm(hidden)
        m = p + "mixer."
        if kind == "mamba":
            add_fp4(m + "in_proj", xd + conv_dim + mh, hidden)
            add_fp4(m + "out_proj", hidden, xd)
            tensors[root + m + "conv1d.weight"] = rand(conv_dim, 1, 4, s=0.3).to(torch.bfloat16)
            tensors[root + m + "conv1d.bias"] = rand(conv_dim, s=0.1).to(torch.bfloat16)
            tensors[root + m + "A_log"] = torch.log(torch.rand(mh, generator=rng) * 7 + 1).to(torch.bfloat16)
            tensors[root + m + "D"] = (1.0 + rand(mh, s=0.1)).to(torch.bfloat16)
            tensors[root + m + "dt_bias"] = rand(mh, s=0.5).to(torch.bfloat16)
            tensors[root + m + "norm.weight"] = norm(xd)
        elif kind == "attention":
            add_q8(m + "q_proj", heads * hd, hidden)
            add_q8(m + "k_proj", kv * hd, hidden)
            add_q8(m + "v_proj", kv * hd, hidden)
            add_q8(m + "o_proj", hidden, heads * hd)
        else:
            e, width = EXPERTS[mi], WIDTHS[mi]
            mi += 1
            tensors[root + m + "gate.weight"] = rand(e, hidden, s=0.05).to(torch.bfloat16)
            tensors[root + m + "gate.e_score_correction_bias"] = rand(e, s=0.01)
            add_q8(m + "fc1_latent_proj", latent, hidden)
            add_q8(m + "fc2_latent_proj", hidden, latent)
            for j in range(e):
                add_iq2(m + f"experts.{j}.up_proj", width, latent)
                add_iq2(m + f"experts.{j}.down_proj", latent, width)
            add_iq2(m + "shared_experts.up_proj", shared, hidden)
            add_iq2(m + "shared_experts.down_proj", hidden, shared)
    tensors["vision_model.radio_model.patch.weight"] = rand(8, 8).to(torch.bfloat16)
    tensors["mlp1.0.weight"] = rand(8).to(torch.bfloat16)
    tensors.update(extra or {})
    names = sorted(tensors)
    half = len(names) // 2
    for part, chunk in enumerate((names[:half], names[half:])):
        save_file({n: tensors[n].contiguous() for n in chunk}, str(path / f"model-0000{part + 1}-of-00002.safetensors"),
                  metadata={"format": "pt"})
    (path / "hf_quant_config.json").write_text(json.dumps({"producer": {"name": "modelopt"}, "quantization": {
        "quant_algo": "MIXED_PRECISION", "quantized_layers": formats}}))
    return path
