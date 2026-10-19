"""TF_LFM2_DECODE_FUSE on the GPU: the fused conv and combine kernels give the torch chains' bytes, and the engine's
decode logits and replies are the same bytes with the switch off and on, eager and replayed, on a tiny checkpoint and
on the 24B-A2B (TF_LFM2_MOE_MODEL and TF_LFM2_MOE_GOLDEN, 40 GiB free; skipped without them)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling
from tests.test_lfm2_moe_decode_fuse import _torch_conv_step

REAL = Path(os.environ.get("TF_LFM2_MOE_MODEL", "/home/agis/models/LFM2.5-24B-A2B-FP8"))
GOLDEN = Path(os.environ.get("TF_LFM2_MOE_GOLDEN", "/home/agis/models/LFM2.5-24B-A2B-FP8-golden"))
IDS = [1, 5, 9, 300, 22, 17, 401, 3, 3, 88, 120, 260, 7, 64, 511, 2]


def _bf16(shape, g, scale=1.0, specials=False):
    x = (torch.randn(shape, generator=g) * scale).to(torch.bfloat16)
    if specials:                     # zeros, signed zeros, subnormal-sized values, ties of the bf16 grid (no inf or
        flat = x.view(-1)            # NaN: torch and Triton write different NaN bits, which no reply ever holds)
        flat[:8] = torch.tensor([0.0, -0.0, 1e-38, -1e-38, 1e20, -1e20, 1.00390625, -2.0078125])
    return x.cuda()


@pytest.mark.parametrize("seed", range(6))
def test_conv_step_gives_the_torch_chain_s_bytes(seed):
    from tensorfold.families.lfm2_moe.cuda import fused

    g = torch.Generator().manual_seed(seed)
    d = 2048
    bcx = _bf16((1, 3 * d), g, scale=4.0 if seed % 2 else 0.05, specials=seed == 5)
    state = _bf16((2, d), g, scale=2.0)
    taps = (torch.randn(d, 3, generator=g) * (0.5 if seed < 3 else 30.0)).cuda()
    want_state = state.clone()
    want = _torch_conv_step(bcx, want_state, taps)
    got = fused.conv_step(bcx, state, taps)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert torch.equal(state.view(torch.int16), want_state.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 2, 5, 16])
@pytest.mark.parametrize("top_k", [2, 4])
def test_combine_gives_the_torch_chain_s_bytes(rows, top_k):
    from tensorfold.families.lfm2_moe.cuda import experts as X
    from tensorfold.families.lfm2_moe.cuda import fused

    g = torch.Generator().manual_seed(rows * 10 + top_k)
    outs = _bf16((rows * top_k, 2048), g, scale=3.0, specials=rows == 5)
    w = torch.rand(rows, top_k, generator=g).cuda()
    w = w / (w.sum(-1, keepdim=True) + 1e-6)
    want, got = X.combine(outs, w), fused.combine(outs, w)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def _decode(e, prompt, tokens):
    """Teacher-forced decode logits after a prompt chunk, and the engine's conv states after them."""

    e.reset()
    e.set_sampling(None)
    for start in range(0, len(prompt), e.prefill_rows):
        e.forward(prompt[start:start + e.prefill_rows], prompt=True)
    rows = [e.forward([t], prompt=False) for t in tokens]
    return [r.float().cpu() for r in rows], e.conv_state.clone()


def _switch(e, fuse: bool):
    e.fuse, e.graph = fuse, None            # a graph keeps the kernels it captured: capture again


@pytest.fixture(scope="module")
def tiny_engine(tmp_path_factory):
    from tensorfold.families.lfm2_moe.cuda.engine import Engine
    from tensorfold.families.lfm2_moe.cuda.weights import load
    from tests.lfm2_moe_tiny import write

    return Engine(load(write(tmp_path_factory.mktemp("lfm2_moe") / "fp8"), "cuda"), max_len=1024)


@pytest.mark.parametrize("graphs", [False, True])
def test_the_tiny_engine_decodes_the_same_bytes_with_the_switch_off_and_on(tiny_engine, graphs):
    from tensorfold.families.lfm2_moe.cuda.engine import prefill, serial_decode

    e = tiny_engine
    e.graphs = graphs
    runs = {}
    s = Sampling(5, temperature=50.0, top_k=50, top_p=1.0)
    for fuse in (False, True, False):
        _switch(e, fuse)
        rows, conv = _decode(e, IDS[:5], IDS[5:])
        reply = serial_decode(e, prefill(e, IDS, s), 24, s, stop_eos=False).tokens
        runs.setdefault(fuse, []).append((rows, conv, reply))
    (a_rows, a_conv, a_reply), (b_rows, b_conv, b_reply) = runs[False][0], runs[True][0]
    assert all(torch.equal(x, y) for x, y in zip(a_rows, b_rows))
    assert torch.equal(a_conv, b_conv) and a_reply == b_reply
    assert runs[False][1][2] == a_reply


# -- the 24B-A2B ------------------------------------------------------------------------------------------------------

real = pytest.mark.skipif(not (REAL / "config.json").is_file(), reason="needs the LFM2.5-24B-A2B-FP8 checkpoint")


@pytest.fixture(scope="module")
def big():
    from tensorfold.cuda.capacity import available_bytes

    if available_bytes(torch) < 40 << 30:
        pytest.skip("needs 40 GiB of free memory")
    from tensorfold.families.lfm2_moe.cuda.app import Lfm2MoeEngine

    engine = Lfm2MoeEngine(REAL, context=16384, context_explicit=True)
    yield engine
    del engine
    torch.cuda.empty_cache()


@real
@pytest.mark.parametrize("name", ["chat", "tools", "long"])
def test_the_real_engine_decodes_the_same_bytes_with_the_switch_off_and_on(big, name):
    """Each golden prompt teacher-forced through its reference greedy tokens, eager off, eager on, graphed on, graphed
    off: every step's logits and the final conv states equal bit for bit; then a greedy and a seeded reply both
    ways."""

    path = GOLDEN / f"{name}.json"
    if not path.is_file():
        pytest.skip(f"no golden file {path} (tools/lfm2_moe_golden.py writes it)")
    g = json.loads(path.read_text())
    e = big.e
    runs = []
    for fuse, graphs in ((False, False), (True, False), (True, True), (False, True)):
        _switch(e, fuse)
        e.graphs = graphs
        runs.append(_decode(e, g["prompt_ids"], g["greedy"][:-1]))
    rows0, conv0 = runs[0]
    for rows, conv in runs[1:]:
        assert all(torch.equal(a, b) for a, b in zip(rows0, rows)) and torch.equal(conv0, conv)
    from tensorfold.families.lfm2_moe.cuda.engine import prefill, serial_decode

    for s in (None, Sampling(1234, temperature=1.0, top_k=20, top_p=0.95)):
        toks = []
        for fuse in (False, True):
            _switch(e, fuse)
            toks.append(serial_decode(e, prefill(e, g["prompt_ids"], s), 96, s, stop_eos=False).tokens)
        assert toks[0] == toks[1]
    _switch(e, True)
