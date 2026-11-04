"""The dense Llama family's CPU checks: discovery, refusals from config.json and serve options, the bf16 loader's
layout, the startup estimate, and the fp32 reference against transformers' LlamaForCausalLM."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tensorfold import families

FIXTURES = Path(__file__).parent / "fixtures" / "llama_dense"
NANBEIGE = json.loads((FIXTURES / "config.json").read_text())


def test_llama_checkpoints_find_the_family_and_its_cuda_engine():
    family = families.families()["llama"]
    assert family.module == "tensorfold.families.llama_dense"
    assert families.backends_of(family) == ("cuda",)
    assert families.readable_quants(family, "cuda") == ("exl3", "modelopt", "compressed-tensors", None)


def test_nanbeige_config_is_accepted():
    from tensorfold.families.llama_dense import refusal

    assert refusal(NANBEIGE) is None


@pytest.mark.parametrize(("change", "why"), [
    ({"rope_scaling": {"rope_type": "llama3", "factor": 8.0}}, "rope_scaling"),
    ({"rope_parameters": {"rope_type": "yarn", "rope_theta": 1e6, "factor": 4.0}}, "rope_parameters"),
    ({"attention_bias": True}, "biases"),
    ({"mlp_bias": True}, "biases"),
    ({"hidden_act": "gelu"}, "hidden_act"),
    ({"tie_word_embeddings": True}, "tied"),
    ({"head_dim": 64}, "head_dim 64"),
    ({"num_key_value_heads": 3}, "KV heads"),
    ({"num_key_value_heads": 1, "num_attention_heads": 56}, "at most 16"),
    ({"intermediate_size": 22000}, "multiples of 64"),
    ({"architectures": ["MistralForCausalLM"]}, "architectures"),
])
def test_configs_the_engine_does_not_run_are_refused(change, why):
    from tensorfold.families.llama_dense import refusal

    assert why in (refusal({**copy.deepcopy(NANBEIGE), **change}) or "")


def test_check_refuses_from_config_alone(tmp_path):
    from tensorfold.families.llama_dense import check

    (tmp_path / "config.json").write_text(json.dumps({**NANBEIGE, "rope_scaling": {"type": "dynamic"}}))
    with pytest.raises(ValueError, match="rope_scaling"):
        check(tmp_path)


@pytest.mark.parametrize(("options", "why"), [
    ({"tp": 2}, "one GPU"),
    ({"master": "10.0.0.1"}, "one GPU"),
    ({"drafter": "some/drafter"}, "no config.json"),
    ({"parallel": 2}, "one stream"),
    ({"vision": True}, "text model"),
    ({"mtp_drafts": 3}, "mtp_drafts"),
])
def test_serve_options_it_cannot_honour_are_refused_before_loading(tmp_path, options, why):
    from tensorfold.families.llama_dense import cuda_engine

    (tmp_path / "config.json").write_text(json.dumps(NANBEIGE))
    with pytest.raises(ValueError, match=why):
        cuda_engine(tmp_path, **options)


@pytest.mark.parametrize(("block", "ok"), [
    (None, True),
    ({"quant_method": "exl3", "bits": 4.0, "head_bits": 6}, True),
    ({"quant_method": "modelopt", "quant_algo": "NVFP4"}, True),
    ({"bits": 4, "group_size": 64}, False),                       # MLX affine: no CUDA kernels for it here
])
def test_storage_formats(block, ok):
    family = families.families()["llama"]
    config = {**NANBEIGE, **({"quantization_config": block} if block else {})}
    if ok:
        families.require_readable(family, config, "cuda")
    else:
        with pytest.raises(ValueError, match="does not read"):
            families.require_readable(family, config, "cuda")


@pytest.mark.torch
def test_config_reads_both_eos_sources(tmp_path):
    from tensorfold.families.llama_dense.cuda.weights import Config
    from tests import llama_tiny

    c = Config.read(llama_tiny.write(tmp_path / "bf16"))
    assert c.eos == (2, 3) and (c.hidden, c.heads, c.kv_heads, c.head_dim) == (256, 4, 2, 128)
    (tmp_path / "n").mkdir()
    (tmp_path / "n" / "config.json").write_text(json.dumps(NANBEIGE))
    n = Config.read(tmp_path / "n")
    assert (n.layers, n.hidden, n.intermediate, n.vocab, n.eos, n.rope_theta) == (52, 7168, 22016, 166912, (166101,),
                                                                                  1.2e8)


@pytest.mark.torch
def test_bf16_loader_keeps_the_stored_tensors(tmp_path):
    """Projections are the stored bf16 matrices (``Plain``), norms are gamma as stored (no centred +1)."""

    import torch

    from tensorfold.families.llama_dense.cuda.weights import load
    from tests import llama_tiny

    stored = llama_tiny.tensors()
    w = load(llama_tiny.write(tmp_path / "bf16"), device="cpu")
    layer = w.layers[1]
    assert torch.equal(layer.attn.k.weight, stored["model.layers.1.self_attn.k_proj.weight"])
    assert torch.equal(layer.down.weight, stored["model.layers.1.mlp.down_proj.weight"])
    assert torch.equal(layer.input_norm, stored["model.layers.1.input_layernorm.weight"])
    assert torch.equal(w.norm, stored["model.norm.weight"]) and torch.equal(w.head.weight, stored["lm_head.weight"])
    assert w.quant == "bf16" and w.inv_freq.shape == (64,)


@pytest.mark.torch
def test_bf16_loader_refuses_shapes_config_does_not_imply(tmp_path):
    from tensorfold.families.llama_dense.cuda.weights import load
    from tests import llama_tiny

    with pytest.raises(ValueError, match="config.json implies"):
        load(llama_tiny.write(tmp_path / "bad", intermediate_size=448), device="cpu")


@pytest.mark.torch
def test_startup_estimate_grows_by_the_kv_cache():
    """Each further cache slot costs every layer's keys and values (plus one layer mid-grow) and its share of the
    verify window's attention partials."""

    from tensorfold.families.llama_dense.cuda.engine import geometry

    g = geometry(NANBEIGE, 128, 4096)
    per_slot = 53 * 8 * 128 * 4
    grown = g.bytes_at(1 << 17) - g.bytes_at(1 << 16)
    assert per_slot * (1 << 16) <= grown <= per_slot * (1 << 16) * 1.25
    assert g.needed(1000) == g.bytes_at(1128)


@pytest.mark.torch
def test_the_reference_equals_transformers_llama():
    """The fp32 reference (what the GPU tests check quality against) equals LlamaForCausalLM on the same weights."""

    import torch

    transformers = pytest.importorskip("transformers")
    from tensorfold.families.llama_dense.cuda.weights import Config
    from tensorfold.families.llama_dense.reference import forward
    from tests import llama_tiny

    raw = llama_tiny.text_config()
    stored = {k: v.float() for k, v in llama_tiny.tensors().items()}
    cfg = transformers.LlamaConfig(**{k: v for k, v in raw.items() if k not in ("architectures", "model_type")})
    cfg._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(cfg).float().eval()
    missing, unexpected = model.load_state_dict(stored, strict=False)
    assert not unexpected and all("rotary" in m for m in missing)
    ids = [1, 5, 9, 300, 22, 17, 401, 3, 3, 88, 120, 260, 7, 64, 511, 2] * 3
    with torch.no_grad():
        want = model(torch.tensor([ids])).logits[0]
    c = Config(hidden=256, intermediate=512, layers=2, heads=4, kv_heads=2, head_dim=128, vocab=512, eps=1e-5,
               rope_theta=1.2e8, eos=(2,))
    got = forward(stored, c, ids)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), (got - want).abs().max()


def test_rope_theta_is_read_from_rope_parameters_too(tmp_path):
    """transformers 5 (the ModelOpt NVFP4 export) writes the base under rope_parameters only."""
    from tensorfold.families.llama_dense import refusal
    from tensorfold.families.llama_dense.cuda.weights import Config

    exported = {k: v for k, v in NANBEIGE.items() if k not in ("rope_theta", "rope_scaling")}
    exported["rope_parameters"] = {"rope_theta": 120000000, "rope_type": "default"}
    (tmp_path / "config.json").write_text(json.dumps(exported))
    assert refusal(exported) is None and Config.read(tmp_path).rope_theta == 1.2e8


def test_drafter_estimate_bounds_a_config_without_sliding_window():
    """The Nanbeige drafter's config sets ``sliding_window: null``: its context is bounded as DFlash2 keeps it."""

    from tensorfold.cuda.geometry import DRAFT_CONTEXT, draft_geometry

    drafter = json.loads((FIXTURES / "dflash2_config.json").read_text())
    assert drafter["sliding_window"] is None and drafter["dflash_config"]["target_layer_ids"] == [1, 25, 49]
    g = draft_geometry(drafter, 1, 128, bounded=True, kept=4)
    assert g.bytes_at(DRAFT_CONTEXT) == g.bytes_at(131072) > g.bytes_at(16)
