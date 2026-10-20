"""TF_LFM2_DECODE_FUSE on the CPU: the switch, the engine's wiring of the fused decode kernels, the kernels in Triton's
interpreter, and their PTX for sm_121 (GB10) compiled without a GPU. tests/cuda/test_lfm2_moe_decode_fuse.py compares
their bytes with the torch chains on the GPU."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests import lfm2_moe_tiny

pytestmark = pytest.mark.torch

IDS = [1, 5, 9, 300, 22, 17, 401, 3, 3, 88, 120, 260, 7, 64, 511, 2]


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    from tensorfold.families.lfm2_moe.cuda.weights import load

    return load(lfm2_moe_tiny.write(tmp_path_factory.mktemp("lfm2_moe") / "fp8"), "cpu")


def test_the_switch_is_read_at_startup_and_a_typo_stops_it(weights, monkeypatch):
    from tensorfold.families.lfm2_moe.cuda.engine import Engine

    monkeypatch.delenv("TF_LFM2_DECODE_FUSE", raising=False)
    assert Engine(weights, max_len=512).fuse is False             # CUDA only: the CPU keeps the torch chains
    monkeypatch.setenv("TF_LFM2_DECODE_FUSE", "1")
    assert Engine(weights, max_len=512).fuse is False
    monkeypatch.setenv("TF_LFM2_DECODE_FUSE", "yes please")
    with pytest.raises(ValueError, match="TF_LFM2_DECODE_FUSE"):
        Engine(weights, max_len=512)


def _torch_conv_step(bcx, state, taps):
    """The engine's unfused conv chain for one row, as ``fused.conv_step`` must compute it."""

    import torch

    d = state.shape[1]
    b, gate, v = bcx.split(d, dim=-1)
    seq = torch.cat([state, b * v])
    acc = seq[0:1].float() * taps[:, 0]
    for t in range(1, taps.shape[1]):
        acc = acc + seq[t:t + 1].float() * taps[:, t]
    state.copy_(seq[1:])
    return gate * acc.to(torch.bfloat16)


def test_fused_decode_steps_keep_the_engine_s_logits_and_states(weights, monkeypatch):
    """The engine with ``fuse`` on (the kernels swapped for their torch chains, which the GPU test pins to the
    kernels' bytes): the conv state moves once a step, every attention layer reads the call's one RoPE and KV rows,
    and prompt chunks keep the unfused path. Decode logits and conv states equal the unfused engine's bit for bit."""

    import torch

    from tensorfold.families.lfm2_moe.cuda import fused
    from tensorfold.families.lfm2_moe.cuda.engine import Engine

    calls = []

    def conv_step(bcx, state, taps):
        calls.append(bcx.shape[0])
        return _torch_conv_step(bcx, state, taps)

    monkeypatch.setattr(fused, "conv_step", conv_step)
    runs = {}
    for fuse in (False, True):
        e = Engine(weights, max_len=512)
        e.fuse = fuse
        first = e.forward(IDS[:6], prompt=True)
        steps = [e.forward([t], prompt=False) for t in IDS[6:]]
        runs[fuse] = (first, steps, e.conv_state.clone(), [k[:e.pos].clone() for k in e.k_cache])
    convs = sum(b.kind == "conv" for b in weights.blocks)
    assert calls == [1] * convs * (len(IDS) - 6)                 # decode rows only, one call a conv layer a step
    (a0, a, sa, ka), (b0, b, sb, kb) = runs[False], runs[True]
    assert torch.equal(a0, b0) and all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(sa, sb) and all(torch.equal(x, y) for x, y in zip(ka, kb))


def test_route_skips_only_a_scaling_of_one_and_keeps_its_bits(weights, monkeypatch):
    import torch

    from tensorfold.families.lfm2_moe.cuda import experts as X

    m = next(b.moe for b in weights.blocks if b.moe is not None)
    x = torch.randn(5, weights.config.hidden, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    ids0, w0 = X.route(x, m, 2, 1.0)
    ids1, w1 = X.route(x, m, 2, None)
    assert torch.equal(ids0, ids1) and torch.equal(w0, w1)
    seen = []
    own = X.route
    monkeypatch.setattr(X, "route", lambda x, m, k, s: (seen.append(s), own(x, m, k, s))[1])
    for fuse in (False, True):
        for scaling in (1.0, 2.5):
            X.moe(x, m, 2, scaling, few=16, fuse=fuse)
    assert seen == [1.0, 2.5, None, 2.5]


def test_conv_fits_one_row_of_three_taps_only():
    import torch

    from tensorfold.families.lfm2_moe.cuda import fused

    state, taps = torch.zeros(2, 512), torch.zeros(512, 3)
    assert fused.conv_fits(1, state, taps)
    assert not fused.conv_fits(2, state, taps)
    assert not fused.conv_fits(1, torch.zeros(3, 512), torch.zeros(512, 4))
    assert not fused.conv_fits(1, torch.zeros(2, 300), torch.zeros(300, 3))


def test_combine_fits_whole_channel_blocks_only():
    from tensorfold.families.lfm2_moe.cuda import fused

    assert fused.combine_fits(2048) and fused.combine_fits(fused.BD)
    assert not fused.combine_fits(300) and not fused.combine_fits(fused.BD + 1)


INTERPRETED = r"""
import torch
from tensorfold.families.lfm2_moe.cuda import experts as X, fused
from tests.test_lfm2_moe_decode_fuse import _torch_conv_step

g = torch.Generator().manual_seed(3)
D = 512
bcx = (torch.randn(1, 3 * D, generator=g) * 2).to(torch.bfloat16)
state = torch.randn(2, D, generator=g).to(torch.bfloat16)
taps = torch.randn(D, 3, generator=g)
want_state = state.clone()
want = _torch_conv_step(bcx, want_state, taps)
got = fused.conv_step(bcx, state, taps)
# the interpreter truncates fp32 to bf16 where the GPU rounds: within two bf16 steps of the largest value
assert (got.float() - want.float()).abs().max() <= 2 ** -6 * want.float().abs().max(), "conv"
assert torch.equal(state[0], want_state[0]), "state shifts"
assert (state[1].float() - want_state[1].float()).abs().max() <= 2 ** -7 * want_state[1].float().abs().max()
print("ok")
for rows in (1, 3):
    outs = torch.randn(rows * 4, D, generator=g).to(torch.bfloat16)
    w = torch.rand(rows, 4, generator=g)
    w = w / w.sum(-1, keepdim=True)
    want = X.combine(outs, w)
    got = fused.combine(outs, w)
    # three truncations (weight, each part, the sum) where the GPU rounds; a wrong pick or row costs about 100%
    assert (got.float() - want.float()).abs().max() <= 2 ** -5 * want.float().abs().max(), "combine"
    print("ok")
"""


def test_the_fused_kernels_follow_the_torch_chains_in_triton_s_interpreter(tmp_path):
    pytest.importorskip("triton")
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    run = subprocess.run([sys.executable, "-c", INTERPRETED], check=False, env=env, capture_output=True, text=True,
                         timeout=240, cwd=tmp_path)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.count("ok") == 3


def _ptx(fn, signature, constexprs, num_warps, fusion):
    """PTX for sm_121 (GB10), with the 16-byte alignment the launches specialize on."""

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile

    names = list(signature)
    attrs = {(names.index(a),): [["tt.divisibility", 16]] for a in names if signature[a].startswith("*")}
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    k = compile(src, target=GPUTarget("cuda", 121, 32), options={"num_warps": num_warps, "enable_fp_fusion": fusion})
    return k.asm["ptx"]


def test_the_fused_kernels_compile_without_fma_for_sm_121():
    """torch rounds each fp32 product and each sum in its own kernel; an fma would round the two once. With Triton's
    default fusion the conv's taps become fmas, so the kernels must launch with it off; a Triton upgrade that
    contracts anyway fails here before it changes a reply."""

    pytest.importorskip("triton")
    from tensorfold.families.lfm2_moe.cuda import fused

    conv_sig = {"BCX": "*bf16", "STATE": "*bf16", "TAPS_W": "*fp32", "Y": "*bf16", "D": "constexpr", "BD": "constexpr"}
    conv_cx = {"D": 2048, "BD": fused.BD}
    assert re.search(r"\bfma\.", _ptx(fused._conv_step, conv_sig, conv_cx, 2, True))      # the hazard is real
    assert not re.search(r"\bfma\.", _ptx(fused._conv_step, conv_sig, conv_cx, 2, False))
    comb_sig = {"OUTS": "*bf16", "WEIGHTS": "*fp32", "Y": "*bf16", "K": "constexpr", "D": "constexpr",
                "BD": "constexpr"}
    assert not re.search(r"\bfma\.", _ptx(fused._combine, comb_sig, {"K": 4, "D": 2048, "BD": fused.BD}, 2, False))
