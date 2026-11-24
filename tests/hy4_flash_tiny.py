"""A tiny random Hy4-flash checkpoint in the release's names and formats, the same bytes from a seed every time.

Six layers (sparse at 0, 1 and 5, sliding between), the dense MLP in layer 0, 16 experts (top 4) on a 128-wide
latent, Engram memories before layers 1 and 3, and the kernels' fixed widths (a 512-wide latent, 64 rotated
channels, 256-wide heads). Attention projections of layers 1-5 are block FP8 and the routed experts NVFP4, both with
power-of-two scales, and each Engram row holds a code of 448, so every stored weight is exact in bf16 and
transformers' FP8 round trip of an Engram row gives the row back: ``tools/hy4_reference.py fixture`` loads the same
values into transformers' model for the goldens the tests compare against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

V, D, LAYERS = 512, 256, 6
Q_RANK, KV_RANK, NOPE, ROPE, VDIM = 128, 512, 192, 64, 256
SPARSE_HEADS, SLIDING_HEADS = 2, 4
INTER, EXPERTS, TOP_K, LATENT, WIDTH = 512, 16, 4, 128, 128
IHEADS, IDIM, ITOPK, WINDOW = 4, 128, 32, 16
ENGRAM = {2: [[1009, 1013], [1019, 1021]], 4: [[1031, 1033], [1039, 1049]]}
ENGRAM_PER_NGRAM, ENGRAM_HEADS = 64, 2
LOOKUP = 500
DFLY_LAYERS, DFLY_INTER = [1, 5], 512          # the drafter's target sparse layers and its MLP width
KINDS = ["deepseek_sparse_attention" if i < 2 or (i - 1) % 4 == 0 else "sliding_attention" for i in range(LAYERS)]


def config() -> dict:
    text = {
        "architectures": ["HYV4FlashForCausalLM"], "model_type": "hy_v4_flash", "vocab_size": V, "hidden_size": D,
        "intermediate_size": INTER, "moe_intermediate_size": WIDTH, "num_hidden_layers": LAYERS,
        "num_attention_heads": SPARSE_HEADS, "num_key_value_heads": SPARSE_HEADS, "swa_num_attention_heads":
        SLIDING_HEADS, "q_lora_rank": Q_RANK, "kv_lora_rank": KV_RANK, "qk_nope_head_dim": NOPE,
        "qk_rope_head_dim": ROPE, "v_head_dim": VDIM, "swa_qk_nope_head_dim": NOPE, "swa_qk_rope_head_dim": ROPE,
        "swa_v_head_dim": VDIM, "head_dim": ROPE, "layer_types": KINDS,
        "mlp_layer_types": ["dense"] + ["sparse"] * (LAYERS - 1), "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K, "routed_expert_hidden_size": LATENT, "routed_scaling_factor": 2.5,
        "n_shared_experts": 1, "n_group": 1, "topk_group": 1, "norm_topk_prob": True, "swiglu_limit": 10.0,
        "hc_mult": 4, "hc_eps": 1e-6, "hc_magnitude": 2.0, "rms_norm_eps": 1e-5, "sliding_window": WINDOW,
        "index_topk": ITOPK, "compressed_indexer_num_attention_heads": IHEADS, "compressed_indexer_head_dim": IDIM,
        "compressed_indexer_compress_ratio": 4, "index_n_heads": IHEADS, "index_head_dim": IDIM,
        "engram_layer_ids": sorted(ENGRAM), "engram_layer_head_vocab_sizes": {str(k): v for k, v in ENGRAM.items()},
        "engram_max_ngram_size": 3, "engram_n_head_per_ngram": ENGRAM_HEADS,
        "engram_n_embed_per_ngram": ENGRAM_PER_NGRAM, "engram_pad_id": 2, "engram_seed": 0,
        "engram_fp8_embedding": True, "engram_lookup_table_size": LOOKUP, "bos_token_id": 0, "eos_token_id": 1,
        "pad_token_id": 2, "max_position_embeddings": 4096, "tie_word_embeddings": False, "attention_bias": False,
        "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0,
                                                  "mrope_section": [12, 10, 10], "mrope_interleaved": True},
                            "deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 1e7,
                                                          "mrope_section": [12, 10, 10], "mrope_interleaved": True}},
        "learnable_sink": True, "gated_mla": True, "use_mla": True, "sandwich_norm": True,
    }
    return {"architectures": ["HYV4FlashForConditionalGeneration"], "model_type": "hy_v4_flash", "text_config": text,
            "dtype": "bfloat16", "dfly_num_hidden_layers": len(DFLY_LAYERS), "dfly_target_kv_layers": DFLY_LAYERS,
            "dfly_target_index_layer": DFLY_LAYERS, "dfly_mask_token_id": 3, "dfly_mtp_fusion": "slot0",
            "dfly_enable_hidden_correction": True, "dfly_context_local_window": None}


def _bf16(x: np.ndarray) -> np.ndarray:
    """fp32 values rounded to bf16 (nearest even), kept as fp32."""

    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def _e4m3_values() -> np.ndarray:
    """Every finite e4m3fn value by code 0..255 (NaN codes as 0)."""

    out = np.zeros(256, np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 0xF, c & 7
        if e == 15 and m == 7:
            continue
        v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
        out[c] = -v if s else v
    return out


E4M3 = _e4m3_values()
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def _nearest_code(x: np.ndarray, table: np.ndarray) -> np.ndarray:
    return np.abs(x[..., None] - table).argmin(-1).astype(np.uint8)


def fp8_block(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N, K] -> (e4m3 codes, fp32 power-of-two scales [ceil(N/128), ceil(K/128)])."""

    n, k = w.shape
    bn, bk = -(-n // 128), -(-k // 128)
    s = np.zeros((bn, bk), np.float32)
    codes = np.zeros((n, k), np.uint8)
    for i in range(bn):
        for j in range(bk):
            blk = w[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
            s[i, j] = 2.0 ** np.ceil(np.log2(max(np.abs(blk).max(), 1e-12) / 448.0))
            codes[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = _nearest_code(blk / s[i, j], E4M3)
    return codes, s


def fp8_block_values(codes: np.ndarray, s: np.ndarray) -> np.ndarray:
    return E4M3[codes] * s.repeat(128, 0)[:codes.shape[0]].repeat(128, 1)[:, :codes.shape[1]]


def nvfp4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[E, N, K] -> (codes [E, N, K/2] low nibble first, e4m3 scale codes [E, N, K/16], fp32 2^-k per expert)."""

    e, n, k = w.shape
    g = 2.0 ** np.ceil(np.log2(np.abs(w).reshape(e, -1).max(1) / (6 * 448)))
    blocks = w.reshape(e, n, k // 16, 16) / g[:, None, None, None]
    sc = _nearest_code(np.abs(blocks).max(-1) / 6.0, E4M3[:127])                     # positive e4m3 codes
    step = E4M3[sc][..., None]
    codes = _nearest_code(np.where(step > 0, blocks / np.maximum(step, 1e-30), 0), E2M1).reshape(e, n, k)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8), sc, g.astype(np.float32)


def nvfp4_values(codes: np.ndarray, sc: np.ndarray, g: np.ndarray) -> np.ndarray:
    full = np.stack([codes & 0xF, codes >> 4], -1).reshape(*codes.shape[:-1], -1)
    return E2M1[full] * E4M3[sc].repeat(16, -1) * g[:, None, None]


def tensors(seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(stored tensors by release name, their values as fp32 for transformers' bf16 parameters)."""

    rng = np.random.default_rng(seed)
    stored: dict[str, np.ndarray] = {}
    values: dict[str, np.ndarray] = {}

    def put(name, arr, dtype="bf16", value=None):
        a = _bf16(arr) if dtype == "bf16" else arr
        stored[name] = (a, dtype)
        values[name] = a.astype(np.float32) if value is None else value

    def normal(*shape, std=0.05):
        return rng.standard_normal(shape).astype(np.float32) * std

    def ones(n, std=0.1):
        return 1.0 + normal(n, std=std)

    def fp8(name, w):
        codes, s = fp8_block(w)
        stored[name + ".weight"] = (codes, "f8")
        stored[name + ".weight_scale_inv"] = (s, "f32")
        values[name + ".weight"] = fp8_block_values(codes, s)

    put("model.embed_tokens.weight", normal(V, D, std=0.5))
    put("lm_head.weight", normal(V, D, std=0.05))
    put("model.norm.weight", ones(D))
    put("model.hc_head.hc_head_fn", normal(4, 4 * D, std=0.02), "f32")
    put("model.hc_head.hc_head_base", np.full(4, -np.log(3), np.float32) + normal(4, std=0.1), "f32")
    put("model.hc_head.hc_head_scale", np.array([0.5], np.float32), "f32")
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        heads = SPARSE_HEADS if KINDS[i] != "sliding_attention" else SLIDING_HEADS
        put(p + "input_layernorm.weight", ones(D))
        put(p + "post_attention_layernorm.weight", ones(D))
        for hc in ("hc_attn_layer", "hc_mlp_layer"):
            put(p + f"{hc}.hc_pre.hc_fn", normal(8, 4 * D, std=0.02), "f32")
            put(p + f"{hc}.hc_pre.hc_base", np.concatenate([np.full(4, -np.log(3)), np.zeros(4)]).astype(np.float32)
                + normal(8, std=0.1), "f32")
            put(p + f"{hc}.hc_pre.hc_scale", np.array([0.5, 0.5], np.float32), "f32")
        a = p + "self_attn."
        put(a + "q_a_proj.weight", normal(Q_RANK, D))
        put(a + "q_a_layernorm.weight", ones(Q_RANK))
        put(a + "kv_a_proj_with_mqa.weight", normal(KV_RANK + ROPE, D))
        put(a + "kv_a_layernorm.weight", ones(KV_RANK))
        put(a + "kv_b_proj.weight", normal(heads * (NOPE + VDIM), KV_RANK, std=0.03))
        put(a + "post_proj_norm.weight", ones(D))
        put(a + "learnable_sink_param", normal(heads, std=1.0), "f32")
        for name, shape in (("q_b_proj", (heads * (NOPE + ROPE), Q_RANK)), ("linear_gate", (heads * VDIM, D)),
                            ("o_proj", (D, heads * VDIM))):
            w = normal(*shape)
            if i == 0:
                put(a + name + ".weight", w)
            else:
                fp8(a + name, w)
        if KINDS[i] != "sliding_attention":
            ci = a + "compressed_indexer."
            put(ci + "query_projection.weight", normal(IHEADS * IDIM, Q_RANK))
            put(ci + "head_weight_projection.weight", normal(IHEADS, D, std=0.2))
            put(ci + "token_key_projection.weight", normal(IDIM, D))
            put(ci + "token_key_norm.weight", ones(IDIM))
            put(ci + "key_value_projection.weight", normal(IDIM, IDIM, std=0.1), "f32")
            put(ci + "key_gate_projection.weight", normal(IDIM, IDIM, std=0.1), "f32")
            put(ci + "intra_group_ape", normal(4, IDIM, std=0.1), "f32")
            put(ci + "key_norm.weight", ones(IDIM), "f32")
        m = p + "mlp."
        if i == 0:
            put(m + "gate_proj.weight", normal(INTER, D))
            put(m + "up_proj.weight", normal(INTER, D))
            put(m + "down_proj.weight", normal(D, INTER))
            put(m + "post_fc2_norm.weight", ones(D))
        else:
            put(m + "gate.weight", normal(EXPERTS, D, std=0.1))
            put(m + "gate.e_score_correction_bias", normal(EXPERTS, std=0.05), "f32")
            put(m + "routed_expert_down_proj.weight", normal(LATENT, D))
            put(m + "routed_expert_up_proj.weight", normal(D, LATENT))
            for name, shape in (("gate_proj", (EXPERTS, WIDTH, LATENT)), ("up_proj", (EXPERTS, WIDTH, LATENT)),
                                ("down_proj", (EXPERTS, LATENT, WIDTH))):
                codes, sc, g = nvfp4(normal(*shape, std=0.08))
                stored[m + f"experts.{name}.weight"] = (codes, "u8")
                stored[m + f"experts.{name}.weight_scale"] = (sc, "f8")
                stored[m + f"experts.{name}.weight_scale_2"] = (g, "f32")
                values[m + f"experts.{name}.weight"] = nvfp4_values(codes, sc, g)
            put(m + "shared_experts.gate_proj.weight", normal(WIDTH, D))
            put(m + "shared_experts.up_proj.weight", normal(WIDTH, D))
            put(m + "shared_experts.down_proj.weight", normal(D, WIDTH))
            put(m + "combine_output_norm.weight", ones(D))
        if i + 1 in ENGRAM:
            e = p + "engram_layer."
            sizes = [s for order in ENGRAM[i + 1] for s in order]
            dim = ENGRAM_PER_NGRAM // ENGRAM_HEADS
            lookup = (np.arange(LOOKUP) * 7 % 433).astype(np.int32)        # a many-to-one map, as the release's
            stored[e + "lookup_table"] = (lookup, "i32")
            values[e + "lookup_table"] = lookup
            rows = sum(sizes)
            table = normal(rows, dim, std=1.0)
            scale = (2.0 ** np.floor(np.log2(np.abs(table).max(1) / 448.0 + 1e-30) + rng.integers(-1, 2, rows))
                     ).astype(np.float32)
            codes = _nearest_code(np.clip(table / scale[:, None], -440, 440), E4M3)
            codes[np.arange(rows), rng.integers(0, dim, rows)] = 0x7E                    # one code of 448 a row
            stored[e + "multi_head_embedding.embedding.weight"] = (codes, "f8")
            stored[e + "multi_head_embedding.embedding.weight_scale"] = (scale, "f32")
            values[e + "multi_head_embedding.embedding.weight"] = E4M3[codes] * scale[:, None]
            for h in range(len(sizes)):
                put(e + f"lookup_per_head_norms.{h}.weight", ones(dim))
            width = len(sizes) * dim
            put(e + "value_proj.weight", normal(D, width))
            put(e + "value_proj.bias", normal(D, std=0.02))
            for s in range(4):
                put(e + f"key_projs.{s}.weight", normal(D, width))
                put(e + f"key_projs.{s}.bias", normal(D, std=0.02))
                put(e + f"norm_key.{s}.weight", ones(D))
                put(e + f"norm_query.{s}.weight", ones(D))
                put(e + f"unified_per_hc_scales.{s}", normal(D, std=0.05))
            put(e + "unified_pre_gamma", ones(D))
    _dfly(rng, stored, put, fp8, normal, ones)
    return stored, values


def _dfly(rng, stored, put, fp8, normal, ones) -> None:
    """DFly2's layers as the NVFP4 pack stores them: FP8-block q_b/o/gate, NVFP4 MLPs and correction, bf16 rest."""

    def nv2(name, w):
        codes, sc, g = nvfp4(w[None])
        stored[name + ".weight"] = (codes[0], "u8")
        stored[name + ".weight_scale"] = (sc[0], "f8")
        stored[name + ".weight_scale_2"] = (g.reshape(()), "f32")

    p = "model.dfly_layers."
    heads = SPARSE_HEADS
    for j in range(len(DFLY_LAYERS)):
        a = f"{p}{j}.self_attn."
        put(f"{p}{j}.input_layernorm.weight", ones(D))
        put(f"{p}{j}.post_attention_layernorm.weight", ones(D))
        put(a + "q_a_proj.weight", normal(Q_RANK, D))
        put(a + "q_a_layernorm.weight", ones(Q_RANK))
        put(a + "kv_a_proj_with_mqa.weight", normal(KV_RANK + ROPE, D))
        put(a + "kv_a_layernorm.weight", ones(KV_RANK))
        put(a + "kv_b_proj.weight", normal(heads * (NOPE + VDIM), KV_RANK, std=0.03))
        put(a + "post_proj_norm.weight", ones(D))
        put(a + "learnable_sink_param", normal(heads, std=1.0), "f32")
        fp8(a + "q_b_proj", normal(heads * (NOPE + ROPE), Q_RANK))
        fp8(a + "linear_gate", normal(heads * VDIM, D))
        fp8(a + "o_proj", normal(D, heads * VDIM))
        nv2(f"{p}{j}.mlp.gate_proj", normal(DFLY_INTER, D))
        nv2(f"{p}{j}.mlp.up_proj", normal(DFLY_INTER, D))
        nv2(f"{p}{j}.mlp.down_proj", normal(D, DFLY_INTER))
    put(p + "final_norm.weight", ones(D))
    put(p + "mtp_fusion.enorm.weight", ones(D))
    put(p + "mtp_fusion.hnorm.weight", ones(D))
    put(p + "mtp_fusion.eh_proj.weight", normal(D, 2 * D))
    put(p + "mtp_fusion.gate", normal(D, std=0.5))
    put(p + "hidden_correction.hidden_norm.weight", ones(D))
    put(p + "hidden_correction.embed_norm.weight", ones(D))
    nv2(p + "hidden_correction.gate_proj", normal(D, 2 * D))
    nv2(p + "hidden_correction.up_proj", normal(D, 2 * D))
    nv2(p + "hidden_correction.down_proj", normal(D, D))
    put(p + "confidence_head.proj.weight", normal(1, D))
    put(p + "confidence_head.proj.bias", normal(1))


def write(folder: str | Path, seed: int = 0) -> Path:
    """The checkpoint in ``folder``: the Engram tables in their own files, as the NVFP4 pack stores them."""

    import torch
    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stored, _ = tensors(seed)
    kinds = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32, "u8": torch.uint8}
    files: dict[str, dict[str, torch.Tensor]] = {}
    for name, (arr, kind) in stored.items():
        if kind == "f8":
            t = torch.from_numpy(np.ascontiguousarray(arr)).view(torch.float8_e4m3fn)
        else:
            t = torch.from_numpy(np.ascontiguousarray(arr)).to(kinds[kind])
        part = f"model-engram-{name.split('.')[2]}.safetensors" if "multi_head_embedding" in name else \
            "model-00001.safetensors"
        files.setdefault(part, {})[name] = t
    index = {}
    for part, ts in files.items():
        save_file(ts, str(folder / part))
        index.update({n: part for n in ts})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}))
    (folder / "config.json").write_text(json.dumps(config(), indent=1))
    (folder / "generation_config.json").write_text(json.dumps({"bos_token_id": 0, "eos_token_id": 1}))
    return folder
