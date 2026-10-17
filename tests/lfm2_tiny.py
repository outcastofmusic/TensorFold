"""A tiny d1 (LFM2-VL) checkpoint in the real layout: config.json with a text_config and a vision_config,
model.language_model.* / model.vision_tower.* / model.multi_modal_projector.* bf16 tensors, a byte-level BPE tokenizer
whose yes / no / option letters / digits are single tokens, and a chat template with a ``{% generation %}`` block as
d1's has."""

from __future__ import annotations

import json
from pathlib import Path

import torch

LAYERS = ("conv", "full_attention", "conv", "conv", "full_attention", "conv")
SPECIALS = ["<|pad|>", "<|startoftext|>", "<|im_start|>", "<|im_end|>", "<image>"]
WORDS = ("yes", "Yes", "YES", "no", "No", "NO", "Reply", "with", "only", "Options", "QUESTION", "user", "assistant")
TEMPLATE = ("{{- bos_token -}}{%- for message in messages -%}{{- '<|im_start|>' + message.role + '\\n' -}}"
            "{%- if message.role == 'assistant' -%}{%- generation -%}{{- message.content + '<|im_end|>\\n' -}}"
            "{%- endgeneration -%}{%- else -%}{{- message.content + '<|im_end|>\\n' -}}{%- endif -%}{%- endfor -%}"
            "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}")


def tokenizer():
    """Byte-level BPE: every byte, a merge chain for each word in WORDS and for " A" to " Z", and d1's specials."""

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    merges = []

    def chain(word: str) -> None:
        for end in range(2, len(word) + 1):
            left, right = word[:end - 1], word[end - 1]
            if word[:end] not in vocab:
                vocab[word[:end]] = len(vocab)
                merges.append((left, right))

    for word in WORDS:
        chain(word)
    for letter in range(65, 91):
        chain("Ġ" + chr(letter))                 # "Ġ" is the byte-level space
    tok = Tokenizer(models.BPE(vocab, merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIALS)
    return tok


def text_config(vocab: int) -> dict:
    return {"model_type": "lfm2", "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "intermediate_size": 128, "block_auto_adjust_ff_dim": False, "layer_types": list(LAYERS),
            "num_hidden_layers": len(LAYERS), "conv_L_cache": 3, "conv_bias": False, "vocab_size": vocab,
            "max_position_embeddings": 512, "norm_eps": 1e-5, "block_use_swiglu": True, "dtype": "bfloat16",
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}, "tie_word_embeddings": True}


VISION = {"model_type": "siglip2_vision_model", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
          "num_attention_heads": 2, "num_patches": 16, "patch_size": 16, "num_channels": 3, "layer_norm_eps": 1e-6,
          "hidden_act": "gelu_pytorch_tanh", "vision_use_head": False}


def config(vocab: int, image_token: int, vision: bool) -> dict:
    out = {"model_type": "lfm2_vl", "architectures": ["Lfm2VlForConditionalGeneration"], "dtype": "bfloat16",
           "text_config": text_config(vocab), "image_token_id": image_token, "downsample_factor": 2,
           "projector_hidden_size": 48, "projector_bias": True, "projector_hidden_act": "gelu",
           "projector_use_layernorm": False, "tie_word_embeddings": True}
    if vision:
        out["vision_config"] = dict(VISION)
    return out


def language_tensors(cfg: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    """transformers' Lfm2 names under ``model.language_model.``: random, scaled so activations stay O(1)."""

    rng = torch.Generator().manual_seed(seed)
    t = cfg["text_config"]
    d, ffn, heads, kv = t["hidden_size"], t["intermediate_size"], t["num_attention_heads"], t["num_key_value_heads"]
    hd = d // heads

    def rand(*shape, s: float = 0.1) -> torch.Tensor:
        return (torch.randn(shape, generator=rng) * s).to(torch.bfloat16)

    def norm(n: int) -> torch.Tensor:
        return (1.0 + torch.randn(n, generator=rng) * 0.1).to(torch.bfloat16)

    p = "model.language_model."
    out = {p + "embed_tokens.weight": rand(t["vocab_size"], d, s=1.0), p + "embedding_norm.weight": norm(d)}
    for i, kind in enumerate(t["layer_types"]):
        q = f"{p}layers.{i}."
        out[q + "operator_norm.weight"], out[q + "ffn_norm.weight"] = norm(d), norm(d)
        out[q + "feed_forward.w1.weight"], out[q + "feed_forward.w3.weight"] = rand(ffn, d), rand(ffn, d)
        out[q + "feed_forward.w2.weight"] = rand(d, ffn)
        if kind == "conv":
            out[q + "conv.conv.weight"] = rand(d, 1, t["conv_L_cache"], s=0.5)
            out[q + "conv.in_proj.weight"], out[q + "conv.out_proj.weight"] = rand(3 * d, d), rand(d, d)
        else:
            out[q + "self_attn.q_proj.weight"] = rand(heads * hd, d)
            out[q + "self_attn.k_proj.weight"], out[q + "self_attn.v_proj.weight"] = rand(kv * hd, d), rand(kv * hd, d)
            out[q + "self_attn.out_proj.weight"] = rand(d, heads * hd)
            out[q + "self_attn.q_layernorm.weight"], out[q + "self_attn.k_layernorm.weight"] = norm(hd), norm(hd)
    return out


def vision_tensors(cfg: dict, seed: int = 1) -> dict[str, torch.Tensor]:
    """The tower and projector as transformers' own modules initialize them (seeded), under the checkpoint's names."""

    from transformers import Lfm2VlConfig
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlMultiModalProjector
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel

    torch.manual_seed(seed)
    vl = Lfm2VlConfig(**{k: v for k, v in cfg.items() if k != "architectures"})
    tower, projector = Siglip2VisionModel(vl.vision_config), Lfm2VlMultiModalProjector(vl)
    out = {f"model.vision_tower.{k}": v.to(torch.bfloat16) for k, v in tower.state_dict().items()}
    out.update({f"model.multi_modal_projector.{k}": v.to(torch.bfloat16) for k, v in projector.state_dict().items()})
    return out


def write(path: Path, *, seed: int = 0, vision: bool = False, extra: dict | None = None) -> Path:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    tok.save(str(path / "tokenizer.json"))
    cfg = config(tok.get_vocab_size(), tok.token_to_id("<image>"), vision)
    cfg.update(extra or {})
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<|startoftext|>", "eos_token": "<|im_end|>",
                                                            "pad_token": "<|pad|>"}))
    (path / "chat_template.jinja").write_text(TEMPLATE)
    tensors = language_tensors(cfg, seed)
    if vision:
        tensors.update(vision_tensors(cfg))
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path / "model.safetensors"), metadata={"format": "pt"})
    return path
