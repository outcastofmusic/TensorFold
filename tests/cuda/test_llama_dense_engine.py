"""Dense Llama's loaders and engine on CUDA: each format's projections hold the stored tensors in the kernels' layout,
and the engine's replies (prefix reuse on, copies on) equal its serial reference request for request."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import precision
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.llama_dense.cuda.weights import load
from tensorfold.families.qwen3_5.cuda.weights import Exl3, Plain
from tests import llama_tiny


def _stored(folder):
    from safetensors.torch import load_file

    out = {}
    for path in sorted(folder.glob("*.safetensors")):
        out.update(load_file(str(path)))
    return out


def test_exl3_pack_loads_each_group_on_the_shared_linear(tmp_path):
    from tensorfold.cuda.exl3.linear import Exl3Linear

    for head in (True, False):
        folder = llama_tiny.write_exl3(tmp_path / f"exl3-{head}", head=head)
        stored = _stored(folder)
        w = load(folder)
        assert w.quant == "exl3" and isinstance(w.embed, Plain) and w.embed.weight.dtype == torch.bfloat16
        x = torch.randn(5, llama_tiny.HIDDEN, generator=torch.Generator().manual_seed(1)).bfloat16().cuda()
        p = "model.layers.1.self_attn.k_proj"
        own = Exl3Linear.from_tensors(stored[p + ".trellis"], stored[p + ".suh"], stored[p + ".svh"], "mul1",
                                      device="cuda")
        assert isinstance(w.layers[1].attn.k, Exl3) and torch.equal(w.layers[1].attn.k(x), own(x))
        assert torch.equal(w.layers[0].post_norm, stored["model.layers.0.post_attention_layernorm.weight"].cuda())
        assert isinstance(w.head, Exl3 if head else Plain)


@pytest.mark.parametrize("mode", ["full", "checkpoint"])
def test_nvfp4_export_loads_codes_and_scales_as_stored(tmp_path, mode):
    from tensorfold.cuda.nvfp4.linear import Fp4Linear

    folder = llama_tiny.write_nvfp4(tmp_path / "nvfp4")
    stored = _stored(folder)
    with precision.using(mode):
        w = load(folder)
    p = "model.layers.0.mlp.down_proj"
    act = float(stored[p + ".input_scale"]) if mode == "checkpoint" else None
    own = Fp4Linear.from_checkpoint(stored[p + ".weight"].cuda(), stored[p + ".weight_scale"].cuda(),
                                    float(stored[p + ".weight_scale_2"]), act=act)
    got = w.layers[0].down
    assert isinstance(got, Fp4Linear) and got.act == own.act
    assert torch.equal(got.words, own.words) and torch.equal(got.bs, own.bs) and got.scale == own.scale
    assert isinstance(w.head, Plain) and torch.equal(w.head.weight, stored["lm_head.weight"].cuda())
    assert w.precision == mode


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from tensorfold.families.llama_dense import cuda_engine

    eng = cuda_engine(llama_tiny.write(tmp_path_factory.mktemp("bf16")), context=4096, context_explicit=True)
    yield eng
    del eng
    torch.cuda.empty_cache()


def _run(eng, prompt, n, sampling, draft=True):
    out: list[int] = []

    def take(tokens):
        out.extend(tokens)
        return False

    stats = eng.generate(prompt, n, sampling, take, draft=draft, stop_eos=False)
    return out, stats


@pytest.mark.parametrize("sampling", [None, Sampling(77, 0.7, 20, 0.95)], ids=["greedy", "sampled"])
def test_engine_replies_equal_serial_with_prefix_reuse(engine, sampling):
    """A conversation's turns resume from the previous prompt's kept end; each reply equals the serial request's."""

    g = torch.Generator().manual_seed(11)
    system = torch.randint(4, llama_tiny.VOCAB, (700,), generator=g).tolist()
    first = system + torch.randint(4, llama_tiny.VOCAB, (40,), generator=g).tolist()
    reply, stats = _run(engine, first, 64, sampling)
    assert reply == _run(engine, first, 64, sampling, draft=False)[0]
    second = first + reply + torch.randint(4, llama_tiny.VOCAB, (30,), generator=g).tolist()
    again, stats = _run(engine, second, 64, sampling)
    assert stats["cached"] == len(first) - 1                   # the first prompt's entry ends one token early
    assert again == _run(engine, second, 64, sampling, draft=False)[0]
    assert engine.context_window == 4096


def test_engine_refuses_a_prompt_past_its_window(engine):
    with pytest.raises(ValueError, match="safe capacity"):
        engine.generate([5] * 4096, 4, None, lambda tokens: False)
