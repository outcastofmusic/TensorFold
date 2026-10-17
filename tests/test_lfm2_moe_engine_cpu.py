"""LFM2-MoE's CUDA engine code run on the CPU (fp32 matmuls in place of the kernels) against the fp32 reference, on a
tiny checkpoint in the 24B-A2B's FP8 layout: logits, conv states and KV positions across chunks and decode steps,
generation's stops and keyed sampling. The Triton expert kernels run in Triton's interpreter.

The engine rounds activations to bf16, so a pick whose score sits within bf16 noise of the next expert's can differ
from the reference's: the reference is run on the engine's picks, and each pick that differs must be such a near tie.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests import lfm2_moe_tiny

pytestmark = pytest.mark.torch

IDS = [1, 5, 9, 300, 22, 17, 401, 3, 3, 88, 120, 260, 7, 64, 511, 2]
NEAR_TIE = 0.02          # sigmoid score: the engine's bf16 rows moved scores by up to 0.006 on this checkpoint


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return lfm2_moe_tiny.write(tmp_path_factory.mktemp("lfm2_moe") / "fp8")


@pytest.fixture(scope="module")
def weights(tiny):
    from tensorfold.families.lfm2_moe.cuda.weights import load

    return load(tiny, "cpu")


def _engine(weights):
    from tensorfold.families.lfm2_moe.cuda.engine import Engine

    return Engine(weights, max_len=512)


def _close(got, want, rel: float = 0.02) -> bool:
    return float((got.float() - want).abs().max()) <= rel * float(want.abs().max())


def test_the_engine_computes_the_reference_on_its_own_picks(tiny, weights, monkeypatch):
    picks = lfm2_moe_tiny.Picks(monkeypatch, len(weights.config.layers) - weights.config.dense_layers)
    got = _engine(weights).forward(IDS, prompt=True, every=True)
    picks.replay()
    want = lfm2_moe_tiny.reference_logits(tiny, IDS)
    assert _close(got, want, 0.01)                               # 0.54% measured
    assert all(gap < NEAR_TIE for gap in picks.gaps)
    assert (got.argmax(-1) == want.argmax(-1)).all()


def test_prompt_chunks_carry_the_conv_states_and_keys(tiny, weights, monkeypatch):
    import torch

    picks = lfm2_moe_tiny.Picks(monkeypatch, len(weights.config.layers) - weights.config.dense_layers)
    e = _engine(weights)
    parts = torch.cat([e.forward(IDS[:5], prompt=True, every=True), e.forward(IDS[5:6], prompt=True, every=True),
                       e.forward(IDS[6:], prompt=True, every=True)])
    picks.replay()
    assert _close(parts, lfm2_moe_tiny.reference_logits(tiny, IDS), 0.01) and e.pos == len(IDS)
    assert all(gap < NEAR_TIE for gap in picks.gaps)


def test_decode_steps_carry_the_conv_states_and_keys_of_the_prompt(tiny, weights, monkeypatch):
    import torch

    picks = lfm2_moe_tiny.Picks(monkeypatch, len(weights.config.layers) - weights.config.dense_layers)
    e = _engine(weights)
    first = e.forward(IDS[:9], prompt=True, every=True)
    steps = [e.forward([t], prompt=False) for t in IDS[9:]]
    picks.replay()
    want = lfm2_moe_tiny.reference_logits(tiny, IDS)
    assert _close(torch.cat([first, *steps]), want)
    assert all(gap < NEAR_TIE for gap in picks.gaps)
    assert e.pos == len(IDS)


def test_generation_stops_at_an_end_token_at_max_tokens_and_when_asked(weights):
    from dataclasses import replace

    from tensorfold.families.lfm2_moe.cuda.engine import prefill, serial_decode

    e = _engine(weights)
    pending = prefill(e, IDS, None)
    free = serial_decode(e, pending, 12, None, stop_eos=False)
    assert len(free.tokens) == 12 and free.tokens[0] == pending
    e.c = replace(e.c, eos=(free.tokens[3],))
    pending = prefill(e, IDS, None)
    ended = serial_decode(e, pending, 12, None)
    assert ended.tokens == free.tokens[:free.tokens.index(free.tokens[3]) + 1]
    seen: list[int] = []
    pending = prefill(e, IDS, None)
    asked = serial_decode(e, pending, 12, None, stop_eos=False,
                          on_tokens=lambda new: seen.extend(new) or len(seen) == 2)
    assert asked.tokens == free.tokens[:3] and seen == free.tokens[1:3]


def test_keyed_sampling_follows_the_seed_and_the_host_rule(weights):
    import torch

    from tensorfold.engine.exact_sampling import Sampling, choose
    from tensorfold.families.lfm2_moe.cuda.engine import prefill, serial_decode

    e = _engine(weights)

    def reply(seed: int) -> list[int]:
        s = Sampling(seed, temperature=50.0, top_k=50, top_p=1.0)      # the tiny model's logits span 300
        return serial_decode(e, prefill(e, IDS, s), 8, s, stop_eos=False).tokens

    assert reply(3) == reply(3)
    assert len({tuple(reply(seed)) for seed in range(4)}) > 1
    s = Sampling(11, temperature=0.7, top_k=5, top_p=0.9)
    e.reset()
    e.set_sampling(s)
    logits = e.forward(IDS, prompt=True)
    vals, ids = torch.topk(logits[0], 5 + 8)
    assert e.sample(logits) == choose(vals.numpy(), ids.numpy(), len(IDS), s)


def test_the_loader_refuses_tensors_it_does_not_read(tiny, tmp_path):
    import shutil

    import torch
    from safetensors.torch import save_file

    from tensorfold.families.lfm2_moe.cuda.weights import load

    copy = tmp_path / "extra"
    shutil.copytree(tiny, copy)
    save_file({"model.layers.0.conv.conv.bias": torch.zeros(4)}, str(copy / "extra.safetensors"))
    (copy / "model.safetensors.index.json").unlink()            # the shards by glob: the extra file is read too
    (copy / "extra.safetensors").rename(copy / "model-00003-of-00003.safetensors")
    with pytest.raises(ValueError, match="unused checkpoint tensors"):
        load(copy, "cpu")


def test_the_bf16_export_loads_and_runs(tmp_path):
    from tensorfold.families.lfm2_moe.cuda.weights import load

    path = lfm2_moe_tiny.write(tmp_path / "bf16", fp8=False)
    w = load(path, "cpu")
    moe = w.blocks[2].moe.experts
    assert not moe.fp8 and moe.w13.dtype.is_floating_point and w.config.fp8 is False
    assert _engine(w).forward(IDS, prompt=True).shape == (1, w.config.vocab)


def test_stacked_fp8_linears_keep_each_part_s_block_scales(tiny):
    import torch

    from tensorfold.cuda.nvfp4.linear import fragment_index
    from tensorfold.families.lfm2_moe.checkpoint import Tensors, dequantize
    from tensorfold.families.lfm2_moe.cuda.weights import stack_fp8

    t = Tensors(tiny)
    p = "model.layers.2.self_attn."
    parts = [t.stored(p + name) for name in ("q_proj", "k_proj", "v_proj")]
    lin = stack_fp8(parts)
    want = torch.cat([dequantize(w, s) for w, s in parts])
    kk, nn = fragment_index(lin.k, lin.npad, "cpu")                   # the kernels' byte order, undone
    rows = torch.empty((lin.npad, lin.k), dtype=torch.uint8)
    rows.t()[kk, nn] = lin.w8.view(kk.shape)
    got = rows[:lin.n].view(torch.float8_e4m3fn).float() * lin.scale_rows().repeat_interleave(64, dim=1)
    assert lin.n == want.shape[0] and torch.equal(got, want)


def test_an_expert_decodes_to_dequantize_s_bits_in_the_buffer_it_is_given():
    import torch

    from tensorfold.families.lfm2_moe.checkpoint import dequantize
    from tensorfold.families.lfm2_moe.cuda import experts as X

    w = lfm2_moe_tiny.e4m3_codes(384, 256, 1)
    s = torch.rand(3, 2, generator=torch.Generator().manual_seed(2)) * 1e-2
    work = torch.full((384 * 256 + 7,), float("nan"))
    got = X.dense(w, s, work)
    assert got.data_ptr() == work.data_ptr() and got.shape == (384, 256)
    assert torch.equal(got, dequantize(w, s))
    bf16 = torch.randn(256, 128).to(torch.bfloat16)
    assert torch.equal(X.dense(bf16, None, work), bf16.float())


def test_the_grouped_path_keeps_its_bits(weights, tmp_path):
    import torch

    from tensorfold.families.lfm2_moe.cuda import experts as X
    from tensorfold.families.lfm2_moe.cuda.weights import load

    g = torch.Generator().manual_seed(3)
    bf16 = load(lfm2_moe_tiny.write(tmp_path / "bf16", fp8=False), "cpu")
    for ex in (weights.blocks[2].moe.experts, bf16.blocks[2].moe.experts):
        x = torch.randn(40, weights.config.hidden, generator=g).to(torch.bfloat16)
        ids = torch.randint(0, ex.count, (40, 2), generator=g)
        assert torch.equal(X.grouped(x, ids, ex), lfm2_moe_tiny.grouped_by_dequantize(x, ids, ex))


KERNELS = r"""
import torch
from tensorfold.families.lfm2_moe.checkpoint import dequantize
from tensorfold.families.lfm2_moe.cuda import experts as X
from tensorfold.families.lfm2_moe.cuda.weights import Experts
from tests.lfm2_moe_tiny import fp8_blocks

g = torch.Generator().manual_seed(5)
E, D, I = 4, 256, 256
w13 = torch.randn(E, 2 * I, D, generator=g) / 16
w2 = torch.randn(E, D, I, generator=g) / 16
q13, s13 = zip(*(fp8_blocks(w) for w in w13))
q2, s2 = zip(*(fp8_blocks(w) for w in w2))
fp8 = Experts(torch.stack(q13).view(torch.uint8), torch.stack(q2).view(torch.uint8),
              torch.stack(s13).float(), torch.stack(s2).float())
plain = Experts(w13.to(torch.bfloat16), w2.to(torch.bfloat16), None, None)
x = torch.randn(3, D, generator=g).to(torch.bfloat16)
ids = torch.tensor([[0, 2], [3, 1], [2, 0]])
for ex in (fp8, plain):
    exact = []
    for p, e in enumerate(ids.reshape(-1).tolist()):
        up, down = (X.dense(w[e], None if s is None else s[e]).double()
                    for w, s in ((ex.w13, ex.s13), (ex.w2, ex.s2)))
        gate, lin = (x[p // 2].double() @ up.t()).chunk(2)
        exact.append((torch.nn.functional.silu(gate) * lin) @ down.t())
    exact = torch.stack(exact)
    for out, limit in ((X.pairs(x, ids, ex), 0.03), (X.grouped(x, ids, ex), 0.01)):
        err = float((out.double() - exact).abs().max() / exact.abs().max())
        assert err < limit, err
    print("ok")
codes = torch.tensor([b for b in range(256) if b & 0x7F != 0x7F], dtype=torch.uint8)
w = codes[torch.randint(0, len(codes), (256, 384), generator=g)]
s = torch.rand(2, 3, generator=g) * 1e-2
out = torch.full((256, 384), float("nan"))
X.decode_blocks(w, s, out)
assert torch.equal(out, dequantize(w, s))
print("ok")
"""


def test_the_expert_kernels_compute_each_pick_s_swiglu_in_triton_s_interpreter(tmp_path):
    """Against fp64 with no rounding: the interpreter truncates to bf16 where a GPU rounds, so its three roundings
    cost up to 1.8% of the largest output here, the grouped path's 0.5%; a wrong expert or block scale costs 100%.
    The block decode kernel gives ``dequantize``'s bits."""

    pytest.importorskip("triton")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    run = subprocess.run([sys.executable, "-c", KERNELS], check=False, env=env, capture_output=True, text=True,
                         timeout=240, cwd=tmp_path)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.count("ok") == 3


REAL = Path(os.environ.get("TF_LFM2_MOE_MODEL", "/home/agis/models/LFM2.5-24B-A2B-FP8"))


@pytest.mark.skipif(not (REAL / "config.json").is_file(), reason="needs the LFM2.5-24B-A2B-FP8 checkpoint")
def test_the_real_checkpoint_s_first_layers_load_and_compute_the_reference(monkeypatch):
    """The 24B-A2B's own tensors through the loader: two dense layers and the first attention and MoE layer."""

    from tensorfold.families.lfm2_moe.cuda.weights import load
    from tensorfold.families.lfm2_moe.reference import Reference, State, rmsnorm

    w = load(REAL, "cpu", layers=3)
    picks = lfm2_moe_tiny.Picks(monkeypatch, 1)
    ids = [124894, 124899, 892, 1446, 6022, 5025, 1112, 1080, 46562, 124900]
    got = _engine(w).forward(ids, prompt=True, every=True)
    picks.replay()
    ref = Reference(REAL)
    h = rmsnorm(ref.hidden(ids, State(), upto=3), ref.w("model.embedding_norm"), ref.c.eps)
    want = h @ ref.head().t()
    assert _close(got, want) and all(gap < NEAR_TIE for gap in picks.gaps)
    assert w.blocks[2].moe.experts.w13.shape == (64, 3072, 2048) and w.blocks[2].moe.experts.s13.shape == (64, 24, 16)
