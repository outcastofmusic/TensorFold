"""NVFP4 and block FP8 trunk projections (``tools/hy4_trunk.py``, ``TF_HY4_TRUNK``) on the tiny checkpoint, CPU."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests import hy4_flash_tiny as tiny

pytestmark = pytest.mark.torch

TOOL = Path(__file__).parents[1] / "tools" / "hy4_trunk.py"
# the tiny checkpoint's bf16 projections the trunk replaces: the dense MLP, shared experts, latent projections, head
NAMES = ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.down_proj", "model.layers.2.mlp.shared_experts.up_proj",
         "model.layers.3.mlp.routed_expert_down_proj", "model.layers.0.self_attn.q_a_proj", "lm_head",
         "model.layers.2.self_attn.kv_b_proj"]


@pytest.fixture(scope="module")
def trunk_tool():
    spec = importlib.util.spec_from_file_location("hy4_trunk", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return tiny.write(tmp_path_factory.mktemp("hy4") / "tiny")


def _write_trunk(tool, checkpoint: Path, folder: Path, fp8: set[str] = frozenset()) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    out, dense = {}, {}
    with safe_open(str(checkpoint / "model-00001.safetensors"), "pt") as f:
        for name in NAMES:
            w = f.get_tensor(name + ".weight").to(torch.bfloat16)
            if name.endswith("kv_b_proj"):
                codes, sc, _ = tool.quantize_rows(w)
                out[name + ".weight"], out[name + ".weight_scale_row"] = codes, sc
                dense[name] = codes.float() * sc[:, None]
            elif name in fp8:
                codes, inv, _ = tool.quantize_fp8(w)
                out[name + ".weight"], out[name + ".weight_scale_inv"] = codes, inv
                dense[name] = codes.float() * inv.repeat_interleave(128, 0)[:w.shape[0]].repeat_interleave(
                    128, 1)[:, :w.shape[1]]
            else:
                codes, scales, scale2, _ = tool.quantize(w, (1.0, 0.9, 0.8))
                out[name + ".weight"], out[name + ".weight_scale"] = codes, scales
                out[name + ".weight_scale_2"] = torch.tensor(scale2)
                from tensorfold.families.hy4_flash.cuda.weights import _nvfp4_dense

                dense[name] = _nvfp4_dense(codes, scales.view(torch.uint8), torch.tensor(scale2))
    folder.mkdir(parents=True, exist_ok=True)
    save_file(out, str(folder / "trunk-test.safetensors"))
    return dense


def test_search_is_never_worse_than_the_amax_rule(trunk_tool):
    import torch

    w = (torch.randn(64, 256, generator=torch.Generator().manual_seed(0)) * 0.02).to(torch.bfloat16)
    plain = trunk_tool.quantize(w, (1.0,))[3]
    searched = trunk_tool.quantize(w, (1.0, 0.95, 0.9, 0.85, 0.8))[3]
    assert searched <= plain
    assert 0.0 < searched < 0.02


def test_codes_decode_to_their_blocks(trunk_tool):
    """Every value decodes to a grid point times its block's scale, and the error the tool reports is the decoded one."""

    import torch

    from tensorfold.families.hy4_flash.cuda.weights import _nvfp4_dense

    w = (torch.randn(32, 128, generator=torch.Generator().manual_seed(1)) * 0.05).to(torch.bfloat16)
    codes, scales, scale2, rel = trunk_tool.quantize(w, (1.0, 0.9))
    back = _nvfp4_dense(codes, scales.view(torch.uint8), torch.tensor(scale2))
    err = float(((back - w.float()) ** 2).sum() / (w.float() ** 2).sum())
    assert abs(err - rel) < 1e-6 * max(rel, 1e-9) + 1e-9


def test_loader_replaces_the_named_projections(trunk_tool, checkpoint, tmp_path):
    """``trunk=`` swaps exactly the named projections for their NVFP4 or FP8 values (the head too) and drops the
    checkpoint's copies, so the unused-tensor check still passes; the rest of the model is untouched."""

    import torch

    from tensorfold.families.hy4_flash.cuda.weights import load

    dense = _write_trunk(trunk_tool, checkpoint, tmp_path / "trunk", fp8={"model.layers.0.mlp.down_proj"})
    base = load(checkpoint, "cpu")
    got = load(checkpoint, "cpu", trunk=str(tmp_path / "trunk"))
    assert torch.equal(got.layers[0].mlp.gate.w, dense["model.layers.0.mlp.gate_proj"])
    assert torch.equal(got.layers[0].mlp.down.w, dense["model.layers.0.mlp.down_proj"])
    assert torch.equal(got.layers[2].mlp.s_up.w, dense["model.layers.2.mlp.shared_experts.up_proj"])
    assert torch.equal(got.layers[3].mlp.pre.w, dense["model.layers.3.mlp.routed_expert_down_proj"])
    assert torch.equal(got.layers[0].attn.q_a.w, dense["model.layers.0.self_attn.q_a_proj"])
    assert torch.equal(got.head.dense(), dense["lm_head"])
    kv = dense["model.layers.2.self_attn.kv_b_proj"].view(tiny.SLIDING_HEADS, tiny.NOPE + tiny.VDIM, tiny.KV_RANK)
    assert torch.equal(got.layers[2].attn.wk.dense(), kv[:, :tiny.NOPE])
    assert torch.equal(got.layers[2].attn.wv.dense(), kv[:, tiny.NOPE:])
    assert torch.equal(got.layers[2].mlp.s_gate.w, base.layers[2].mlp.s_gate.w)
    assert torch.equal(got.layers[1].attn.o.w, base.layers[1].attn.o.w)


def test_quantized_model_stays_near_the_checkpoint(trunk_tool, checkpoint, tmp_path):
    """The tiny model's prompt logits with the trunk against without: a small KL. Its random weights leave many rows'
    top two close, so the top token agrees on most rows only (24/32 at seed 0)."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda import model
    from tensorfold.families.hy4_flash.cuda.weights import load

    _write_trunk(trunk_tool, checkpoint, tmp_path / "trunk")
    ids = list(range(5, 37))
    out = []
    for trunk in (None, str(tmp_path / "trunk")):
        e = model.Engine(load(checkpoint, "cpu", trunk=trunk), max_len=64, prefill_rows=64)
        e.set_sampling(Sampling(seed=0, temperature=0.0))
        out.append(e.forward(ids, prompt=True, every=True).float())
    base, got = out
    lp, lq = base.log_softmax(-1), got.log_softmax(-1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    assert float(kl.mean()) < 0.05
    assert float((base.argmax(-1) == got.argmax(-1)).float().mean()) >= 0.6
