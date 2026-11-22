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


@pytest.mark.parametrize("only, skip", [("self_attn", None), (None, r"lm_head|kv_b_proj"), (r"layers\.2\.", "kv_b")])
def test_filters_take_the_folder_s_projections_by_name(trunk_tool, checkpoint, tmp_path, monkeypatch, only, skip):
    """TF_HY4_TRUNK_ONLY / TF_HY4_TRUNK_SKIP: a projection takes the folder's values when its name passes both,
    else keeps the checkpoint's, and the loader still reads every checkpoint tensor."""

    import re

    import torch

    from tensorfold.families.hy4_flash.cuda.weights import load

    dense = _write_trunk(trunk_tool, checkpoint, tmp_path / "trunk")
    base = load(checkpoint, "cpu")
    for k, v in (("TF_HY4_TRUNK_ONLY", only), ("TF_HY4_TRUNK_SKIP", skip)):
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    got = load(checkpoint, "cpu", trunk=str(tmp_path / "trunk"))

    def taken(name):
        return (only is None or re.search(only, name)) and (skip is None or not re.search(skip, name))

    pairs = {"model.layers.0.mlp.gate_proj": (got.layers[0].mlp.gate.w, base.layers[0].mlp.gate.w),
             "model.layers.2.mlp.shared_experts.up_proj": (got.layers[2].mlp.s_up.w, base.layers[2].mlp.s_up.w),
             "model.layers.0.self_attn.q_a_proj": (got.layers[0].attn.q_a.w, base.layers[0].attn.q_a.w)}
    for name, (g, b) in pairs.items():
        assert torch.equal(g, dense[name] if taken(name) else b), name
    head = got.head.dense() if not isinstance(got.head, torch.Tensor) else got.head.float()
    assert torch.equal(head, dense["lm_head"] if taken("lm_head") else base.head.float())
    kv = "model.layers.2.self_attn.kv_b_proj"
    assert (not isinstance(got.layers[2].attn.wk, torch.Tensor)) == bool(taken(kv))


def _activations(k: int, rows: int = 4096, seed: int = 0):
    """Rows whose channels differ in scale by up to 100x and correlate, as an LLM's projection inputs do."""

    import torch

    g = torch.Generator().manual_seed(seed)
    scale = torch.exp(torch.randn(k, generator=g) * 1.2)
    mix = torch.eye(k) + 0.3 * torch.randn(k, k, generator=g) / k ** 0.5
    return (torch.randn(rows, k, generator=g) @ mix) * scale


def test_gptq_lowers_the_output_error_and_its_codes_decode(trunk_tool):
    """On correlated, unevenly scaled inputs GPTQ's output error sits well below the data-free search's, the
    diagonal-weighted search sits at or below it, and every code decodes to the values the error was measured on."""

    import torch

    from tensorfold.families.hy4_flash.cuda.weights import _nvfp4_dense

    k = 256
    w = (torch.randn(96, k, generator=torch.Generator().manual_seed(3)) * 0.02).to(torch.bfloat16)
    x = _activations(k)
    h = (x.t() @ x).double() / x.shape[0]
    search = (1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75)
    errs = {}
    for kind in ("plain", "weighted", "gptq"):
        if kind == "gptq":
            codes, scales, scale2, rel = trunk_tool.quantize_gptq(w, h, search)
        else:
            codes, scales, scale2, rel = trunk_tool.quantize(w, search, diag=torch.diagonal(h) if kind == "weighted"
                                                             else None)
        back = _nvfp4_dense(codes, scales.view(torch.uint8), torch.tensor(scale2))
        assert torch.equal(back, trunk_tool.dense_of(codes, scales.view(torch.float8_e4m3fn), scale2))
        assert abs(float(((back - w.float()) ** 2).sum() / (w.float() ** 2).sum()) - rel) < 1e-6
        errs[kind] = trunk_tool.output_error(w, back, h)
        direct = float(((x @ (back - w.float()).t()) ** 2).sum() / ((x @ w.float().t()) ** 2).sum())
        assert abs(direct - errs[kind]) < 1e-3 * errs[kind]
    assert errs["weighted"] <= errs["plain"] * 1.001
    assert errs["gptq"] < 0.8 * errs["plain"], errs
    one, chunked = trunk_tool.quantize_gptq(w, h, search), trunk_tool.quantize_gptq(w, h, search, rows=40)
    assert torch.equal(one[0], chunked[0]) and torch.equal(one[1].view(torch.uint8), chunked[1].view(torch.uint8))


def test_stats_folder_drives_the_formats(trunk_tool, checkpoint, tmp_path):
    """``--stats``: a site with a full H gives GPTQ codes, one with a diagonal the weighted search, and quant.json
    records the output error."""

    import json
    import subprocess
    import sys

    import torch
    from safetensors.torch import save_file

    stats = tmp_path / "stats"
    stats.mkdir()
    names = {"model.layers.2.mlp.shared_experts.up_proj": "model.layers.2.mlp.in", "lm_head": "head.in"}
    with open(checkpoint / "config.json") as f:
        hidden = json.load(f)["text_config"]["hidden_size"]
    x = _activations(hidden, 512)
    save_file({"model.layers.2.mlp.in.h": x.t() @ x, "model.layers.2.mlp.in.n": torch.tensor(512)},
              str(stats / "sites-L02.safetensors"))
    save_file({"head.in.diag": (x ** 2).sum(0), "head.in.n": torch.tensor(512)}, str(stats / "sites-head.safetensors"))
    (stats / "sites.json").write_text(json.dumps({"names": names}))
    out = tmp_path / "out"
    done = subprocess.run([sys.executable, str(Path(trunk_tool.__file__)), str(checkpoint), str(out), "--classes",
                    "shared,head", "--stats", str(stats), "--search", "1,0.9"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]

    rep = json.loads((out / "quant.json").read_text())
    assert rep["model.layers.2.mlp.shared_experts.up_proj"]["format"] == "nvfp4-gptq"
    assert rep["lm_head"]["format"] == "nvfp4-weighted"
    assert rep["model.layers.2.mlp.shared_experts.gate_proj"]["format"] == "nvfp4"
    assert 0 < rep["lm_head"]["rel_out"] < 0.05


def test_factors_above_one_lower_the_error(trunk_tool):
    """A block whose largest value lands on 4 or 3 (factor 1.1-1.6) can fit the rest better than the amax rule."""

    import torch

    w = (torch.randn(128, 512, generator=torch.Generator().manual_seed(2)) * 0.02).to(torch.bfloat16)
    low = trunk_tool.quantize(w, (1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7))[3]
    wide = trunk_tool.quantize(w, (1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7))[3]
    assert wide < 0.95 * low
