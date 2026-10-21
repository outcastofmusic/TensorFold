"""TF_EXL3_EXPERT_STAGED on the GPU: a prompt window's routed experts write the same bytes with the staged kernel.

- Routed experts at Flash Next EXL3's shapes (2560 <-> 640, 3-bit mul1 experts beside a 5-bit shared expert, 11
  slots), at 4 bits in each codebook, and at GLM-5.3's (4096 <-> 2048, mcg at 2.25 bits, whose 18-word tiles take
  8-byte copies, and at 2.5 bits, 14-bit window, 8 slots): bf16 slots into a buffer (TF_FLASHNEXT_MOE_DIRECT's
  call), fp32 slots and combined rows, windows of 65 to 2,048 rows, with the served switches, with the epilogue
  kernels (no fusion, so they read the staged path's one total a matrix), on the spanned layout (no item list) and
  with PDL. Uniform and skewed routing (experts holding hundreds of rows, so many member groups an expert).
- A CUDA graph captured on the staged path replays the eager bytes; the arrival counters end at zero.
- A layer whose trellises do not start on 16 bytes keeps grouped_kernel's launches and its bytes.
- Planted control: one flipped trellis word changes the bytes the comparison sees.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

from tensorfold.cuda.exl3 import experts, switches
from tests.cuda.test_exl3_decode_kernels import _flash_next as _fn_layer
from tests.cuda.test_exl3_decode_kernels import _glm, _same, _scale, _trellis

SERVED = switches.Switches()                                        # today's defaults: every fusion on
UNFUSED = dataclasses.replace(SERVED, gateup_fuse=False, down_fuse=False)
SPANNED = dataclasses.replace(SERVED, expert_items=False)
PDL = dataclasses.replace(SERVED, pdl=True)
BASES = {"served": SERVED, "unfused": UNFUSED, "spanned": SPANNED, "pdl": PDL}
ROWS = (65, 128, 1000, 2048)


def _four_bit(cb: str, E=48, seed=4):
    """Flash Next's shapes with every expert at 4 bits (64 int16 a tile): the 8..8 instance."""

    D, I = 2560, 640
    g = torch.Generator().manual_seed(seed)
    mats = {p: [] for p in ("gate", "up", "down")}
    for _ in range(E + 1):
        for p, (k, n) in (("gate", (D, I)), ("up", (D, I)), ("down", (I, D))):
            mats[p].append((_trellis(k, n, 64, g), _scale(k, 1 / math.sqrt(k), g), _scale(n, 0.5, g)))
    return experts.prepare(mats["gate"], mats["up"], mats["down"], cb), 11, E, experts.ACT_F32


LAYERS = {"flash_next": lambda: _fn_layer(), "4bit_3inst": lambda: _four_bit("3inst"),
          "4bit_mcg": lambda: _four_bit("mcg"), "glm53_2.25bit": lambda: _glm(36), "glm53_2.5bit": lambda: _glm(40)}
_built: dict = {}


def _layer(name):
    if name not in _built:
        _built.clear()
        torch.cuda.empty_cache()
        _built[name] = LAYERS[name]()
    return _built[name]


@pytest.fixture(autouse=True)
def restore():
    before = switches.current()
    yield
    switches.use(before)


def _inputs(name, R, seed, hot=False):
    ex, slots, E, _ = _layer(name)
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((R, ex.dims), generator=g) * 0.5).to(torch.bfloat16).cuda()
    shared = ex.count == E + 1                                       # Flash Next's shared expert in the last slot
    routed = slots - 1 if shared else slots
    weights = torch.rand((E,), generator=g) ** (6 if hot else 0)
    sel = torch.stack([torch.multinomial(weights, routed, generator=g) for _ in range(R)]).to(torch.int32)
    if shared:
        sel = torch.cat([sel, torch.full((R, 1), E, dtype=torch.int32)], 1)
    w = torch.rand((R, slots), generator=g) * 0.2 + 0.05
    return x, sel.cuda().contiguous(), w.float().cuda().contiguous()


def _three(x, sel, w, ex, scratch, R, act):
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    into = torch.full((R * scratch.slots, ex.dims), float("nan"), dtype=torch.bfloat16, device="cuda")
    yb = experts.routed(x, sel, None, ex, scratch, None, R, limit, act, y=into).clone()
    y = experts.routed(x, sel, None, ex, scratch, None, R, limit, act).clone()
    out = experts.routed(x, sel, w, ex, scratch, None, R, limit, act).clone()
    return yb, y, out


def _live(sel, E):
    return ((sel >= 0) & (sel < E)).view(-1)


def _check(name, R, base, hot=False, seed=0):
    ex, slots, _, act = _layer(name)
    assert ex.aligned
    scratch = experts.Scratch(ex, R, slots)
    x, sel, w = _inputs(name, R, 9000 + R + seed, hot)
    live = _live(sel, ex.count)
    switches.use(base)
    want = _three(x, sel, w, ex, scratch, R, act)
    switches.use(dataclasses.replace(base, expert_staged=True))
    scratch.y.fill_(float("nan"))
    scratch.xd.fill_(float("nan"))
    got = _three(x, sel, w, ex, scratch, R, act)
    torch.cuda.synchronize()
    assert torch.isfinite(want[2]).all()
    assert torch.equal(got[0][live].view(torch.int16), want[0][live].view(torch.int16)), (name, R, "bf16 slots")
    assert _same(got[1][live], want[1][live]), (name, R, "fp32 slots")
    assert _same(got[2], want[2]), (name, R, "combined rows")
    assert int(scratch.arrivals.abs().sum()) == 0


@pytest.mark.parametrize("name", list(LAYERS))
@pytest.mark.parametrize("R", ROWS)
@pytest.mark.parametrize("base", list(BASES))
def test_the_staged_kernel_keeps_the_bytes(name, R, base):
    _check(name, R, BASES[base])


@pytest.mark.parametrize("name", ["flash_next", "glm53_2.25bit"])
@pytest.mark.parametrize("base", ["served", "unfused"])
def test_the_staged_kernel_keeps_the_bytes_on_skewed_routing(name, base):
    _check(name, 2048, BASES[base], hot=True)


@pytest.mark.parametrize("name", ["flash_next", "glm53_2.25bit"])
def test_a_graph_of_the_staged_path_replays_the_eager_bytes(name):
    ex, slots, _, act = _layer(name)
    R = 1024
    scratch = experts.Scratch(ex, R, slots)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    x, sel, _ = _inputs(name, R, 9300)
    into = torch.empty((R * slots, ex.dims), dtype=torch.bfloat16, device="cuda")
    switches.use(dataclasses.replace(PDL, expert_staged=True))
    experts.routed(x, sel, None, ex, scratch, None, R, limit, act, y=into)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        experts.routed(x, sel, None, ex, scratch, None, R, limit, act, y=into)
    nx, nsel, _ = _inputs(name, R, 9301)
    x.copy_(nx), sel.copy_(nsel)
    switches.use(SERVED)
    want = experts.routed(x, sel, None, ex, scratch, None, R, limit, act, y=torch.empty_like(into)).clone()
    into.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    live = _live(sel, ex.count)
    assert torch.equal(into[live].view(torch.int16), want[live].view(torch.int16)), name
    assert int(scratch.arrivals.abs().sum()) == 0


def test_trellises_off_16_bytes_keep_grouped_kernels_launches():
    """Each trellis 8 bytes into its storage: ``aligned`` is false, so the switch leaves the launches as they were."""

    D, I, E = 2560, 640, 16
    g = torch.Generator().manual_seed(5)
    mats = {p: [] for p in ("gate", "up", "down")}
    for e in range(E + 1):
        for p, (k, n) in (("gate", (D, I)), ("up", (D, I)), ("down", (I, D))):
            w16 = 80 if e == E else 48
            store = _trellis(k, n + 16, w16, g).view(-1)[4:4 + (k // 16) * (n // 16) * w16]
            mats[p].append((store.view(k // 16, n // 16, w16), _scale(k, 1 / math.sqrt(k), g), _scale(n, 0.5, g)))
    ex = experts.prepare(mats["gate"], mats["up"], mats["down"], "mul1")
    assert not ex.aligned
    R = 256
    scratch = experts.Scratch(ex, R, 11)
    gx = torch.Generator().manual_seed(6)
    x = (torch.randn((R, D), generator=gx) * 0.5).to(torch.bfloat16).cuda()
    sel = torch.stack([torch.randperm(E, generator=gx)[:10] for _ in range(R)]).to(torch.int32)
    sel = torch.cat([sel, torch.full((R, 1), E, dtype=torch.int32)], 1).cuda().contiguous()
    switches.use(SERVED)
    want = experts.routed(x, sel, None, ex, scratch, None, R).clone()
    switches.use(dataclasses.replace(SERVED, expert_staged=True))
    got = experts.routed(x, sel, None, ex, scratch, None, R).clone()
    torch.cuda.synchronize()
    assert _same(got, want)


def test_the_control_a_flipped_trellis_word_changes_the_bytes():
    """Planted control (playbook 6.2): the staged path reads the weights it is given, so one word of one routed
    expert's gate trellis flipped changes the rows routed to it, and the comparison above would see it."""

    name, R = "flash_next", 256
    ex, slots, _, _ = _layer(name)
    scratch = experts.Scratch(ex, R, slots)
    x, sel, _ = _inputs(name, R, 9400)
    switches.use(dataclasses.replace(SERVED, expert_staged=True))
    want = experts.routed(x, sel, None, ex, scratch, None, R).clone()
    e = int(sel[0, 0])
    gate = next(t for t in ex.keep if t.data_ptr() == int(ex.gate_ptr[e]))
    gate.view(-1)[1000] ^= 0x5A5A
    got = experts.routed(x, sel, None, ex, scratch, None, R).clone()
    gate.view(-1)[1000] ^= 0x5A5A
    torch.cuda.synchronize()
    assert not _same(got, want)
    again = experts.routed(x, sel, None, ex, scratch, None, R).clone()
    torch.cuda.synchronize()
    assert _same(again, want)
