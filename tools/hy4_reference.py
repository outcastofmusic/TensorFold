"""transformers' Hy4-flash (PR #50903, ``modeling_hy_v4_flash``) as the reference for the fork's engine.

    python tools/hy4_reference.py fixture OUT.npz      # goldens of the tiny checkpoint (tests/hy4_flash_tiny.py)

Run it where that transformers is installed (``pip install "transformers @ git+https://github.com/huggingface/
transformers.git@b865850d05326cf20e306b31f14b0027de802da3"``) with this checkout's ``src`` and ``tests`` on PYTHONPATH. The model runs
in bf16, as it ships, on the CPU.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import torch

RENAMES = [
    (r"^model\.layers\.", "layers."), (r"^model\.embed_tokens\.", "embed_tokens."), (r"^model\.norm\.", "norm."),
    (r"^model\.hc_head\.hc_head_fn$", "hc_head.hc_fn"), (r"^model\.hc_head\.hc_head_base$", "hc_head.hc_base"),
    (r"^model\.hc_head\.hc_head_scale$", "hc_head.hc_scale"),
    (r"\.self_attn\.learnable_sink_param$", ".self_attn.sinks"), (r"\.self_attn\.linear_gate\.", ".self_attn.gate_proj."),
    (r"\.mlp\.post_fc2_norm\.", ".post_mlp_layernorm."), (r"\.mlp\.routed_expert_down_proj\.", ".mlp.pre_expert_proj."),
    (r"\.mlp\.routed_expert_up_proj\.", ".mlp.post_expert_proj."),
    (r"\.hc_attn_layer\.hc_pre\.hc_(fn|base|scale)$", r".attn_hc.\1"),
    (r"\.hc_mlp_layer\.hc_pre\.hc_(fn|base|scale)$", r".ffn_hc.\1"),
]


def hf_name(name: str) -> str:
    for pattern, repl in RENAMES:
        name = re.sub(pattern, repl, name)
    return name


def tiny_model():
    """transformers' text model and fp32 head holding the tiny checkpoint's values."""

    import hy4_flash_tiny as tiny
    from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashTextConfig
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashTextModel

    text = dict(tiny.config()["text_config"])
    cfg = HYV4FlashTextConfig(**{k: v for k, v in text.items() if k not in ("architectures", "model_type")})
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = HYV4FlashTextModel(cfg).to(torch.bfloat16).eval()
    _, values = tiny.tensors()
    state = model.state_dict()
    loaded = set()
    experts: dict[str, dict[str, np.ndarray]] = {}
    for name, value in values.items():
        if name == "lm_head.weight":
            continue
        m = re.match(r"(.*\.mlp\.experts)\.(gate_proj|up_proj|down_proj)\.weight$", name)
        if m:
            experts.setdefault(hf_name(m.group(1)), {})[m.group(2)] = value
            continue
        key = hf_name(name)
        if key not in state:
            raise KeyError(f"{name} -> {key} is no parameter of transformers' model")
        state[key].copy_(torch.from_numpy(np.asarray(value)).to(state[key].dtype))
        loaded.add(key)
    for prefix, parts in experts.items():
        state[prefix + ".gate_up_proj"].copy_(torch.from_numpy(np.concatenate([parts["gate_proj"], parts["up_proj"]],
                                                                               1)).to(torch.bfloat16))
        state[prefix + ".down_proj"].copy_(torch.from_numpy(parts["down_proj"]).to(torch.bfloat16))
        loaded |= {prefix + ".gate_up_proj", prefix + ".down_proj"}
    for layer in model.layers:                 # the hash draws from the map the checkpoint holds
        e = layer.engram_layer
        if e is not None:
            e.hash_multipliers.copy_(e.draw_hash_multipliers(int(e.lookup_table.max()) + 1))
    missing = [k for k in state if k not in loaded and not k.endswith(("inv_freq", "offsets", "head_vocab_sizes",
                                                                         "hash_multipliers"))]
    if missing:
        raise KeyError(f"parameters the checkpoint does not set: {missing[:8]}")
    head = torch.from_numpy(values["lm_head.weight"]).float()
    return model, head


@torch.no_grad()
def logits_of(model, head, ids: list[int]) -> np.ndarray:
    x = torch.tensor([ids])
    grams = model.build_ngram_tokens(x, None)
    out = model(input_ids=x, ngram_tokens=grams, engram_mask=torch.ones_like(x, dtype=torch.bool), use_cache=False)
    return (out.last_hidden_state[0].float() @ head.t()).numpy()


def fixture(path: str) -> None:
    model, head = tiny_model()
    rng = np.random.default_rng(7)
    prompts = {"long": rng.integers(3, 512, 48).tolist(), "short": rng.integers(3, 512, 9).tolist()}
    out = {}
    picks: dict[int, tuple] = {}
    for k, layer in enumerate(model.layers):          # each MoE layer's picks and its 4th-to-5th score margin
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.register_forward_hook(lambda m, i, o, k=k: picks.__setitem__(k, (o[0], o[2])))
    masks: dict[int, torch.Tensor] = {}
    for k, layer in enumerate(model.layers):          # each sparse layer's mask once its indexer has picked
        ix = layer.self_attn.compressed_indexer
        if ix is not None:
            ix.register_forward_hook(lambda m, i, o, k=k: masks.__setitem__(k, o))
    for name, ids in prompts.items():
        out[f"{name}_ids"] = np.array(ids, np.int64)
        out[f"{name}_logits"] = logits_of(model, head, ids).astype(np.float32)
        for k, (logits, idx) in picks.items():
            gate = model.layers[k].mlp.gate
            ranked = (torch.sigmoid(logits) + gate.e_score_correction_bias).sort(-1, descending=True).values
            out[f"{name}_picks_{k}"] = idx.numpy()
            out[f"{name}_margin_{k}"] = (ranked[:, gate.top_k - 1] - ranked[:, gate.top_k]).float().numpy()
        for k, mask in masks.items():                 # the closed groups a row past the indexer's limit recalled
            ix = model.layers[k].self_attn.compressed_indexer
            keep = ix.group_topk - 1
            groups = np.full((len(ids), keep), -1, np.int64)
            seen = (mask[0, 0] > torch.finfo(mask.dtype).min / 2).numpy()
            for r in range(len(ids)):
                if r // ix.index_kpool >= ix.group_topk:
                    got = sorted({t // ix.index_kpool for t in np.flatnonzero(seen[r]) if t // ix.index_kpool < r // 4})
                    groups[r] = got
            out[f"{name}_groups_{k}"] = groups
    grams = model.build_ngram_tokens(torch.tensor([prompts["long"]]), None)
    for i, layer in enumerate(model.layers):
        if layer.engram_layer is not None:
            out[f"hashes_{i}"] = layer.engram_layer.hash_ngrams(grams)[0].numpy()
    np.savez_compressed(path, **out)
    print(f"wrote {path}: " + ", ".join(f"{k} {v.shape}" for k, v in out.items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fixture")
    f.add_argument("out")
    a = p.parse_args()
    if a.cmd == "fixture":
        fixture(a.out)


if __name__ == "__main__":
    main()
