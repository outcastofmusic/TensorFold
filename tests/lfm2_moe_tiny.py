"""A tiny LFM2.5-24B-A2B checkpoint in the real layout: compressed-tensors 128x128 block FP8 (``weight`` e4m3 and a
bf16 ``weight_scale`` per block) for attention, dense FFN and expert linears; bf16 embeddings, routers, conv layers
and norms; fp32 expert_bias; per-expert w1/w3/w2; a tied head (no lm_head). ``fp8=False`` writes the bf16 export.
The chat template, tokenizer_config.json and LICENSE are the 24B-A2B's own (tests/fixtures/lfm2_moe, revision
f54bf39547fd5b7955abb9fe8d1b151e31fbbddb of LiquidAI/LFM2.5-24B-A2B-FP8). ``Picks`` replays an engine's expert picks
into the reference, so a bf16 engine compares with it where a near-tie pick would otherwise send the two apart."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

FIXTURES = Path(__file__).parent / "fixtures" / "lfm2_moe"
LAYERS = ("conv", "conv", "full_attention", "conv", "full_attention", "conv")
QUANT = {"config_groups": {"FP8_BLOCK": {
    "format": "float-quantized", "targets": ["Linear"],
    "input_activations": {"dynamic": True, "group_size": 128, "num_bits": 8, "strategy": "group", "symmetric": True,
                          "type": "float"},
    "weights": {"block_structure": [128, 128], "dynamic": False, "num_bits": 8, "strategy": "block",
                "symmetric": True, "type": "float"}}},
    "format": "float-quantized", "ignore": ["lm_head", "model.embed_tokens", "re:.*\\.gate$",
                                            "re:.*\\.(?:conv|short_conv)\\..*"],
    "kv_cache_scheme": None, "quant_method": "compressed-tensors", "quantization_status": "compressed",
    "version": "0.19.0"}


def config(*, vocab: int = 512, hidden: int = 256, fp8: bool = True, experts: int = 8, top_k: int = 2) -> dict:
    out = {"architectures": ["Lfm2MoeForCausalLM"], "model_type": "lfm2_moe", "bos_token_id": 1, "eos_token_id": 7,
           "pad_token_id": 0, "conv_L_cache": 3, "conv_bias": False, "hidden_size": hidden, "intermediate_size": 384,
           "layer_types": list(LAYERS), "max_position_embeddings": 4096, "moe_intermediate_size": 256,
           "norm_eps": 1e-5, "norm_topk_prob": True, "num_attention_heads": 4, "num_dense_layers": 2,
           "num_experts": experts, "num_experts_per_tok": top_k, "num_hidden_layers": len(LAYERS),
           "num_key_value_heads": 2, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
           "routed_scaling_factor": 1.0, "tie_word_embeddings": True, "use_expert_bias": True, "vocab_size": vocab,
           "dtype": "bfloat16"}
    if fp8:
        out["quantization_config"] = QUANT
    return out


def fp8_blocks(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 [N, K] -> (e4m3 [N, K], bf16 scales [ceil(N / 128), K / 128]): llm-compressor's absmax / 448 a block."""

    n, k = w.shape
    rows = -(-n // 128) * 128
    padded = torch.zeros(rows, k)
    padded[:n] = w
    blocks = padded.view(rows // 128, 128, k // 128, 128)
    scale = (blocks.abs().amax(dim=(1, 3)) / 448.0).clamp(min=1e-8).to(torch.bfloat16)
    q = (blocks / scale.float()[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(rows, k)[:n].contiguous(), scale.contiguous()


def write(path: Path, *, seed: int = 0, fp8: bool = True, vocab: int = 512, hidden: int = 256, experts: int = 8,
          top_k: int = 2, bias_scale: float = 0.05, router_scale: float = 0.1) -> Path:
    from safetensors.torch import save_file

    rng = torch.Generator().manual_seed(seed)
    path.mkdir(parents=True, exist_ok=True)
    c = config(vocab=vocab, hidden=hidden, fp8=fp8, experts=experts, top_k=top_k)
    (path / "config.json").write_text(json.dumps(c))
    (path / "generation_config.json").write_text(json.dumps({"bos_token_id": 1, "eos_token_id": [7],
                                                             "do_sample": True, "temperature": 0.1, "top_k": 50,
                                                             "repetition_penalty": 1.05}))
    for name in ("chat_template.jinja", "tokenizer_config.json", "LICENSE"):
        shutil.copy(FIXTURES / name, path / name)
    tensors: dict[str, torch.Tensor] = {}

    def rand(*shape, s: float = 0.05) -> torch.Tensor:
        return torch.randn(shape, generator=rng) * s

    def linear(name: str, n: int, k: int, *, quantized: bool = True) -> None:
        w = rand(n, k, s=k ** -0.5)
        if fp8 and quantized:
            tensors[name + ".weight"], tensors[name + ".weight_scale"] = fp8_blocks(w)
        else:
            tensors[name + ".weight"] = w.to(torch.bfloat16)

    def norm(name: str, n: int) -> None:
        tensors[name + ".weight"] = (1.0 + rand(n, s=0.1)).to(torch.bfloat16)

    heads, kv, hd = c["num_attention_heads"], c["num_key_value_heads"], hidden // c["num_attention_heads"]
    tensors["model.embed_tokens.weight"] = rand(vocab, hidden, s=1.0).to(torch.bfloat16)
    norm("model.embedding_norm", hidden)
    for i, kind in enumerate(LAYERS):
        p = f"model.layers.{i}."
        norm(p + "operator_norm", hidden)
        norm(p + "ffn_norm", hidden)
        if kind == "conv":
            linear(p + "conv.in_proj", 3 * hidden, hidden, quantized=False)
            linear(p + "conv.out_proj", hidden, hidden, quantized=False)
            tensors[p + "conv.conv.weight"] = rand(hidden, 1, 3, s=0.5).to(torch.bfloat16)
        else:
            linear(p + "self_attn.q_proj", heads * hd, hidden)
            linear(p + "self_attn.k_proj", kv * hd, hidden)
            linear(p + "self_attn.v_proj", kv * hd, hidden)
            linear(p + "self_attn.out_proj", hidden, heads * hd)
            norm(p + "self_attn.q_layernorm", hd)
            norm(p + "self_attn.k_layernorm", hd)
        f = p + "feed_forward."
        if i < c["num_dense_layers"]:
            linear(f + "w1", c["intermediate_size"], hidden)
            linear(f + "w3", c["intermediate_size"], hidden)
            linear(f + "w2", hidden, c["intermediate_size"])
            continue
        tensors[f + "gate.weight"] = rand(experts, hidden, s=router_scale).to(torch.bfloat16)
        tensors[f + "expert_bias"] = rand(experts, s=bias_scale)
        for e in range(experts):
            linear(f + f"experts.{e}.w1", c["moe_intermediate_size"], hidden)
            linear(f + f"experts.{e}.w3", c["moe_intermediate_size"], hidden)
            linear(f + f"experts.{e}.w2", hidden, c["moe_intermediate_size"])
    names = sorted(tensors)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    for file, chunk in shards.items():
        save_file({n: tensors[n].contiguous() for n in chunk}, str(path / file), metadata={"format": "pt"})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {
        n: file for file, chunk in shards.items() for n in chunk}}))
    return path


def dequantized_copy(src: Path, dst: Path) -> Path:
    """The FP8 checkpoint ``src`` decoded to fp32 (e4m3 x bf16 is exact there, not in bf16): transformers' input."""

    from safetensors.torch import save_file

    from tensorfold.families.lfm2_moe.checkpoint import Tensors, dequantize

    dst.mkdir(parents=True, exist_ok=True)
    t = Tensors(src)
    out = {}
    for name in t.names():
        if name.endswith(".weight_scale"):
            continue
        w = t.get(name)
        if w.dtype == torch.float8_e4m3fn:
            w = dequantize(w.view(torch.uint8), t.get(name.removesuffix(".weight") + ".weight_scale"))
        out[name] = w.float()
    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
    raw = json.loads((src / "config.json").read_text())
    raw.pop("quantization_config", None)
    raw["dtype"] = "float32"
    (dst / "config.json").write_text(json.dumps(raw))
    for name in ("generation_config.json", "chat_template.jinja", "tokenizer_config.json"):
        shutil.copy(src / name, dst / name)
    return dst


def e4m3_codes(rows: int, cols: int, seed: int):
    """Random e4m3 bytes [rows, cols] over every finite code (subnormals and both zeros included, no NaN)."""

    codes = torch.tensor([b for b in range(256) if b & 0x7F != 0x7F], dtype=torch.uint8)
    pick = torch.randint(0, len(codes), (rows, cols), generator=torch.Generator().manual_seed(seed))
    return codes[pick]


def grouped_by_dequantize(x, ids, ex):
    """The grouped path as it was before its one-buffer decode: each expert through ``dequantize``, then the same
    fp32 GEMMs."""

    from tensorfold.families.lfm2_moe.checkpoint import dequantize

    rows, top_k = ids.shape
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    out = torch.empty((rows * top_k, x.shape[1]), dtype=torch.bfloat16, device=x.device)
    start = 0
    for e, n in enumerate(torch.bincount(flat, minlength=ex.count).tolist()):
        if not n:
            continue
        slots = order[start:start + n]
        start += n
        if ex.fp8:
            w13, w2 = dequantize(ex.w13[e], ex.s13[e]), dequantize(ex.w2[e], ex.s2[e])
        else:
            w13, w2 = ex.w13[e].float(), ex.w2[e].float()
        g, u = (x[slots // top_k].float() @ w13.t()).to(torch.bfloat16).float().chunk(2, dim=-1)
        act = (torch.nn.functional.silu(g).to(torch.bfloat16).float() * u).to(torch.bfloat16)
        out[slots] = (act.float() @ w2.t()).to(torch.bfloat16)
    return out


class Picks:
    """Record the engine's picks by MoE layer; replay them into the reference, noting each differing pick's margin."""

    def __init__(self, monkeypatch, layers: int):
        from tensorfold.families.lfm2_moe.cuda import experts as X

        self.layers, self.calls, self.gaps = layers, [], []
        own = X.route

        def record(x, m, top_k, scaling):
            out = own(x, m, top_k, scaling)
            self.calls.append(out[0].cpu())
            return out

        monkeypatch.setattr(X, "route", record)
        self.monkeypatch = monkeypatch

    def replay(self) -> None:
        """The reference's MoE layers take the recorded picks: each layer's calls' rows, in order, in one call."""

        from tensorfold.families.lfm2_moe import reference as R

        by_layer = [torch.cat(self.calls[j::self.layers]) for j in range(self.layers)]
        at = [0]

        def forced(x, router, bias, top_k, scaling):
            scores = torch.sigmoid(x @ router.t())
            picks = by_layer[at[0] % self.layers][:x.shape[0]].long()
            by_layer[at[0] % self.layers] = by_layer[at[0] % self.layers][x.shape[0]:]
            at[0] += 1
            own = torch.topk(scores + bias, top_k, dim=-1)
            differ = (own.indices.sort(-1).values != picks.sort(-1).values).any(-1)
            if differ.any():
                gap = own.values.sum(-1) - (scores + bias).gather(1, picks).sum(-1)
                self.gaps.append(float(gap[differ].max()))
            chosen = scores.gather(1, picks)
            return picks, chosen / (chosen.sum(-1, keepdim=True) + 1e-6) * scaling

        self.monkeypatch.setattr(R, "route", forced)


def reference_logits(tiny: Path, rows: list[int]) -> torch.Tensor:
    """The fp32 reference's logits of every row of ``rows`` fed whole."""

    from tensorfold.families.lfm2_moe.reference import Reference, State

    return Reference(tiny, keep_experts=True).forward(rows, State())
