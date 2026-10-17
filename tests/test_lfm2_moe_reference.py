"""LFM2-MoE's config checks and fp32 reference on a tiny checkpoint in the 24B-A2B's FP8 layout: the reference equals
transformers' modeling_lfm2_moe.py, and a prompt fed whole, in chunks or a token at a time gives the same logits."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.torch

FIXTURES = Path(__file__).parent / "fixtures" / "lfm2_moe"
REAL = json.loads((FIXTURES / "config.json").read_text())
IDS = [1, 5, 9, 300, 22, 17, 401, 3, 3, 88, 120, 260, 7, 64, 511, 2]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tests import lfm2_moe_tiny

    return lfm2_moe_tiny.write(tmp_path_factory.mktemp("lfm2_moe") / "fp8")


def test_the_24b_a2b_config_reads_as_its_card_says():
    from tensorfold.families.lfm2_moe.config import Config, refusal

    assert refusal(REAL) is None
    c = Config.parse(REAL)
    assert (c.hidden, c.vocab, len(c.layers), c.heads, c.kv_heads, c.head_dim) == (2048, 128000, 40, 32, 8, 64)
    assert c.attention_layers == tuple(range(2, 40, 4)) and len(c.conv_layers) == 30
    assert (c.experts, c.top_k, c.moe_width, c.dense_layers, c.intermediate) == (64, 4, 1536, 2, 11776)
    assert (c.conv_taps, c.theta, c.context, c.fp8, c.scaling) == (3, 1e7, 131072, True, 1.0)


@pytest.mark.parametrize(("change", "why"), [
    (("model_type", "lfm2"), "model_type"),
    (("tie_word_embeddings", False), "tie_word_embeddings"),
    (("conv_bias", True), "conv_bias"),
    (("use_expert_bias", False), "use_expert_bias"),
    (("rope_parameters", {"rope_type": "yarn", "rope_theta": 1e7}), "RoPE"),
    (("layer_types", ["conv"] * 40), "no attention layer"),
    (("layer_types", ["sliding_attention"] * 40), "layer_types"),
    (("num_key_value_heads", 3), "attention"),
])
def test_a_config_the_engine_does_not_run_is_refused(change, why):
    from tensorfold.families.lfm2_moe.config import refusal

    raw = copy.deepcopy(REAL)
    raw[change[0]] = change[1]
    assert why in (refusal(raw) or "")


@pytest.mark.parametrize(("path", "value", "why"), [
    (("config_groups", "FP8_BLOCK", "weights", "block_structure"), [64, 64], "128x128"),
    (("config_groups", "FP8_BLOCK", "weights", "num_bits"), 4, "4-bit"),
    (("quant_method",), "modelopt", "modelopt"),
    (("ignore",), ["lm_head", "re:.*mlp.*"], "unquantized"),
])
def test_a_quantization_the_engine_does_not_read_is_refused(path, value, why):
    from tensorfold.families.lfm2_moe.config import refusal

    raw = copy.deepcopy(REAL)
    node = raw["quantization_config"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert why in (refusal(raw) or "")


def test_the_family_claims_the_checkpoint_and_reads_it_on_cuda(tiny):
    from tensorfold import families

    family = families.detect(tiny)
    assert family.model_type == "lfm2_moe" and families.backends_of(family) == ("cuda",)
    families.require_readable(family, families.read_config(tiny), "cuda")
    family.package.check(tiny)
    bad = json.loads((tiny / "config.json").read_text())
    bad["conv_bias"] = True
    (tiny.parent / "bad").mkdir(exist_ok=True)
    (tiny.parent / "bad" / "config.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="conv_bias"):
        family.package.check(tiny.parent / "bad")


@pytest.mark.parametrize("options,message", [
    ({"tp": 2}, "one GPU"), ({"master": "10.0.0.2"}, "one GPU"), ({"drafter": "/models/x"}, "--drafter"),
    ({"vision": True}, "text model"), ({"parallel": 4}, "--parallel"), ({"kv_dtype": "int8"}, "kv_dtype"),
    ({"mtp_drafts": 2}, "mtp_drafts"), ({"decode_share": 0.5}, "decode_share"),
])
def test_serve_options_without_a_meaning_are_refused(tiny, options, message):
    from tensorfold.families import lfm2_moe

    with pytest.raises(ValueError, match=message):
        lfm2_moe.cuda_engine(tiny, master_port=29551, no_drafts=False, context=None, context_explicit=False,
                             **options)


def test_block_fp8_decodes_to_e4m3_times_its_block_scale():
    import torch

    from tensorfold.families.lfm2_moe.checkpoint import dequantize
    from tests.lfm2_moe_tiny import fp8_blocks

    w = torch.randn(200, 256, generator=torch.Generator().manual_seed(1))
    q, s = fp8_blocks(w)
    assert tuple(s.shape) == (2, 2) and s.dtype == torch.bfloat16
    back = dequantize(q.view(torch.uint8), s)
    assert back.shape == w.shape
    # e4m3 keeps 3 mantissa bits: half a step is 1/16 of the value, plus the subnormal floor of the block's scale
    assert ((back - w).abs() <= w.abs() / 16 + s.float().max() * 2 ** -9).all()
    # absmax / 448: each block's largest code is the e4m3 maximum, within the bf16 rounding of its scale
    assert q.view(torch.uint8).bitwise_and(0x7F).max() >= 0x7D
    with pytest.raises(ValueError, match="tile"):
        dequantize(q.view(torch.uint8), s[:1])


def test_the_reference_equals_transformers_modeling_lfm2_moe(tiny, tmp_path):
    import torch

    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Lfm2MoeForCausalLM"):
        pytest.skip("this transformers has no LFM2-MoE")
    from tensorfold.families.lfm2_moe.reference import Reference, State
    from tests.lfm2_moe_tiny import dequantized_copy

    ref = Reference(tiny, keep_experts=True).forward(IDS, State())
    model = transformers.Lfm2MoeForCausalLM.from_pretrained(str(dequantized_copy(tiny, tmp_path / "fp32")),
                                                            dtype=torch.float32).eval()
    with torch.no_grad():
        hf = model(torch.tensor([IDS])).logits[0]
    assert (ref - hf).abs().max() <= 1e-5 * ref.abs().max()        # 6.1e-5 of 164 measured


def test_a_prompt_whole_in_chunks_or_a_token_at_a_time_gives_the_same_logits(tiny):
    import torch

    from tensorfold.families.lfm2_moe.reference import Reference, State

    ref = Reference(tiny, keep_experts=True)
    whole = ref.forward(IDS, State())
    s = State()
    chunks = torch.cat([ref.forward(IDS[:5], s), ref.forward(IDS[5:6], s), ref.forward(IDS[6:], s)])
    s = State()
    steps = torch.cat([ref.forward([t], s) for t in IDS])
    scale = whole.abs().max()
    assert (whole - chunks).abs().max() <= 1e-5 * scale and (whole - steps).abs().max() <= 1e-5 * scale
    assert s.pos == len(IDS) and set(s.keys) == {2, 4} and set(s.conv) == {0, 1, 3, 5}


def test_a_conv_layer_keeps_its_last_two_products_of_b_and_x(tiny):
    import torch

    from tensorfold.families.lfm2_moe.reference import Reference, State, rmsnorm

    ref = Reference(tiny, keep_experts=True)
    s = State()
    ref.hidden(IDS, s, upto=1)
    x = rmsnorm(ref.embed(torch.tensor(IDS)), ref.w("model.layers.0.operator_norm"), ref.c.eps)
    b, _, v = (x @ ref.w("model.layers.0.conv.in_proj").t()).chunk(3, dim=-1)
    assert torch.allclose(s.conv[0], (b * v)[-2:])
    assert s.conv[0].shape == (ref.c.conv_taps - 1, ref.c.hidden)


def test_a_checkpoint_with_its_own_head_is_refused(tmp_path):
    import torch
    from safetensors.torch import save_file

    from tensorfold.families.lfm2_moe.reference import Reference
    from tests import lfm2_moe_tiny

    path = lfm2_moe_tiny.write(tmp_path / "head", fp8=False)
    save_file({"lm_head.weight": torch.zeros(4, 4)}, str(path / "model-00003-of-00003.safetensors"))
    index = json.loads((path / "model.safetensors.index.json").read_text())
    index["weight_map"]["lm_head.weight"] = "model-00003-of-00003.safetensors"
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="ties the head"):
        Reference(path)
