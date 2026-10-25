"""TF_EXL3_ROT_FUSE on the GPU: a staged gate/up launch that rotates its rows itself writes rot_in's bytes.

- Routed experts at Flash Next EXL3's shapes (3-bit mul1 experts beside a 5-bit shared expert), at 4 bits in each
  codebook, and at GLM-5.3's (mcg at 2.25 and 2.5 bits): bf16 slots into a buffer, fp32 slots and combined rows,
  windows of 65 to 2,048 rows, with the served switches, with the epilogue kernels (the switch then rides the staged
  EPI_NONE launch), with PDL, and on the spanned layout (which keeps rot_in). Uniform and skewed routing.
- The rows the launch writes for every routed pick are rot_in's bytes, from bf16 and from fp16 inputs.
- Rows with a stride keep the switch; rows off 16 bytes keep rot_in; either way the same bytes.
- A CUDA graph captured with the switch replays the eager bytes; every counter ends at zero.
- Planted control: one input value one ulp off changes the bytes the comparison sees.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

from tensorfold.cuda.exl3 import experts, switches
from tests.cuda.exl3_layers import _same
from tests.cuda.test_exl3_staged_kernels import LAYERS, ROWS, _inputs, _layer, _live, _three

SERVED = switches.Switches()                                        # today's defaults: staged and every fusion on
BASES = {"served": SERVED, "unfused": dataclasses.replace(SERVED, gateup_fuse=False, down_fuse=False),
         "pdl": dataclasses.replace(SERVED, pdl=True), "spanned": dataclasses.replace(SERVED, expert_items=False)}


@pytest.fixture(autouse=True)
def restore():
    before = switches.current()
    yield
    switches.use(before)


def _zeroed(scratch):
    return int(scratch.arrivals.abs().sum()) == 0 and int(scratch.rot_arrivals.abs().sum()) == 0


def _check(name, R, base, hot=False, dtype=torch.bfloat16):
    ex, slots, _, act = _layer(name)
    scratch = experts.Scratch(ex, R, slots)
    x, sel, w = _inputs(name, R, 9500 + R, hot)
    x = x.to(dtype)
    live = _live(sel, ex.count)
    switches.use(base)
    want = _three(x, sel, w, ex, scratch, R, act)
    switches.use(dataclasses.replace(base, rot_fuse=True))
    for t in (scratch.y, scratch.xd, scratch.xg, scratch.xu):
        t.fill_(float("nan"))
    got = _three(x, sel, w, ex, scratch, R, act)
    torch.cuda.synchronize()
    assert torch.isfinite(want[2]).all()
    assert torch.equal(got[0][live].view(torch.int16), want[0][live].view(torch.int16)), (name, R, "bf16 slots")
    assert _same(got[1][live], want[1][live]), (name, R, "fp32 slots")
    assert _same(got[2], want[2]), (name, R, "combined rows")
    assert _zeroed(scratch)


@pytest.mark.parametrize("name", list(LAYERS))
@pytest.mark.parametrize("R", ROWS)
@pytest.mark.parametrize("base", list(BASES))
def test_rotating_inside_the_staged_launch_keeps_the_bytes(name, R, base):
    _check(name, R, BASES[base])


@pytest.mark.parametrize("name", ["flash_next", "glm53_2.25bit"])
@pytest.mark.parametrize("base", ["served", "unfused"])
def test_rotating_inside_the_staged_launch_keeps_the_bytes_on_skewed_routing(name, base):
    _check(name, 2048, BASES[base], hot=True)


@pytest.mark.parametrize("name", ["flash_next", "glm53_2.5bit"])
def test_fp16_rows_keep_the_bytes(name):
    _check(name, 1000, SERVED, dtype=torch.float16)


@pytest.mark.parametrize("name", ["flash_next", "glm53_2.25bit", "4bit_mcg"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_rows_the_launch_writes_are_rot_ins(name, dtype):
    """Direct: every routed pick's gate and up rows in X0, X1 after the fused launch equal rot_in's bytes."""

    ex, slots, _, act = _layer(name)
    R = 1000
    scratch = experts.Scratch(ex, R, slots)
    x, sel, _ = _inputs(name, R, 9600)
    x = x.to(dtype)
    live = _live(sel, ex.count)
    switches.use(dataclasses.replace(SERVED, rot_fuse=True))
    scratch.xg.fill_(float("nan")), scratch.xu.fill_(float("nan"))
    experts.routed(x, sel, None, ex, scratch, None, R, 7.0 if act == experts.ACT_BF16 else math.inf, act)
    xg, xu = torch.empty_like(scratch.xg), torch.empty_like(scratch.xu)
    experts._ext().rot_in(x, x.stride(0), sel, ex.suh_g, ex.suh_u, xg, xu, R, ex.dims, slots, ex.count)
    torch.cuda.synchronize()
    P = R * slots
    assert torch.equal(scratch.xg[:P][live].view(torch.int16), xg[:P][live].view(torch.int16)), (name, "gate")
    assert torch.equal(scratch.xu[:P][live].view(torch.int16), xu[:P][live].view(torch.int16)), (name, "up")


@pytest.mark.parametrize("offset, fused", [(8, True), (1, False)])
def test_rows_with_a_stride_or_off_16_bytes_keep_the_bytes(offset, fused):
    """x as a column slice of a wider buffer: 8 values in, its rows stay on 16 bytes and the switch takes them; 1 value
    in, they do not and rot_in rotates them. The bytes are the served path's either way."""

    name, R = "flash_next", 256
    ex, slots, _, act = _layer(name)
    scratch = experts.Scratch(ex, R, slots)
    x, sel, w = _inputs(name, R, 9700)
    wide = torch.zeros((R, ex.dims + 16), dtype=x.dtype, device="cuda")
    wide[:, offset:offset + ex.dims] = x
    view = wide[:, offset:offset + ex.dims]
    assert experts.rot_fits(view) is fused
    switches.use(SERVED)
    want = _three(x, sel, w, ex, scratch, R, act)
    switches.use(dataclasses.replace(SERVED, rot_fuse=True))
    got = _three(view, sel, w, ex, scratch, R, act)
    torch.cuda.synchronize()
    live = _live(sel, ex.count)
    assert _same(got[1][live], want[1][live]) and _same(got[2], want[2])
    assert _zeroed(scratch)


@pytest.mark.parametrize("name", ["flash_next", "glm53_2.25bit"])
def test_a_graph_with_the_switch_replays_the_eager_bytes(name):
    ex, slots, _, act = _layer(name)
    R = 1024
    scratch = experts.Scratch(ex, R, slots)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    x, sel, _ = _inputs(name, R, 9800)
    into = torch.empty((R * slots, ex.dims), dtype=torch.bfloat16, device="cuda")
    switches.use(dataclasses.replace(BASES["pdl"], rot_fuse=True))
    experts.routed(x, sel, None, ex, scratch, None, R, limit, act, y=into)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        experts.routed(x, sel, None, ex, scratch, None, R, limit, act, y=into)
    nx, nsel, _ = _inputs(name, R, 9801)
    x.copy_(nx), sel.copy_(nsel)
    switches.use(SERVED)
    want = experts.routed(x, sel, None, ex, scratch, None, R, limit, act, y=torch.empty_like(into)).clone()
    into.fill_(float("nan"))
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    live = _live(sel, ex.count)
    assert torch.equal(into[live].view(torch.int16), want[live].view(torch.int16)), name
    assert _zeroed(scratch)


def test_the_control_one_input_ulp_changes_the_bytes():
    """Planted control (playbook 6.2): one bf16 value of one routed row one ulp off changes that row's bytes through
    the fused rotation, so the comparisons above would see a rotation that rounds differently."""

    name, R = "flash_next", 256
    ex, slots, _, _ = _layer(name)
    scratch = experts.Scratch(ex, R, slots)
    x, sel, _ = _inputs(name, R, 9900)
    switches.use(dataclasses.replace(SERVED, rot_fuse=True))
    want = experts.routed(x, sel, None, ex, scratch, None, R).clone()
    bits = x.view(torch.int16)
    bits[0, 1000] ^= 1
    got = experts.routed(x, sel, None, ex, scratch, None, R).clone()
    bits[0, 1000] ^= 1
    torch.cuda.synchronize()
    assert not _same(got, want)
    assert _same(got[slots:], want[slots:])                       # only row 0's slots moved
    again = experts.routed(x, sel, None, ex, scratch, None, R).clone()
    torch.cuda.synchronize()
    assert _same(again, want)
