"""The sweep's kernel switches on the GPU: the same bytes with each switch on or off.

- The item list: routed experts at Flash Next EXL3's shapes (2560 <-> 640, 3-bit mul1 experts beside a 5-bit
  shared expert, 11 slots) and GLM-5.3's (4096 <-> 2048, mcg at 2.25 and 2.5 bits, 14-bit window, 8 slots): per-slot
  outputs and combined rows of windows from 1 to 1,024 rows (decode windows, multi-stream windows of 17-28 rows,
  prompt windows) on the item list alone and as ``routed`` serves them, against the launches of before
  (``tests.exl3_fakes.OLD``), eager and replayed from CUDA graphs; and group_kernel's item list against a Python
  model of it.
- The down epilogue's bf16 slots (the MoE buffer's call) against a copy of its fp32 slots, at both models' shapes.
- The EXL3 linear writing N columns of a wider output at Flash Next's dense shapes and GLM-5.3-sized ones, and Flash
  Next's two stacks (attention: q, k, v, indexer; DeltaNet: qkv, z, fp16 b/a) writing in place against the scratch
  copies an output off 16 bytes takes, 1 to 128 rows.
- ``_f16_mm_ks`` (TF_FLASHNEXT_F16_INNER_K) against ``_f16_mm`` + ``_reduce`` at the pack's fp16 shapes and prompt
  row counts, into contiguous and stack-column outputs.
- An fp16 hyper-connection's prompt read-out with the write-back and norm fused against the separate kernels, in the
  write-back modes the layers use.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

from tensorfold.cuda.exl3 import experts, linear, switches  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import exl3_mm  # noqa: E402

from tests.cuda.exl3_layers import LAYERS, LINEAR, _inputs, _layer, _same, _scale, _trellis  # noqa: E402
from tests.exl3_fakes import OLD, route
from tests.test_exl3_expert_items import _listing  # noqa: E402

ITEMS = dataclasses.replace(OLD, expert_items=True)
# (launches, PDL): the item list alone, and routed's own launches (None)
EXPERT_SETTINGS = [(ITEMS, False), (ITEMS, True), (None, False), (None, True)]
ROWS = (1, 8, 16, 17, 24, 28, 64, 65, 128, 1024)


@pytest.fixture(autouse=True)
def restore():
    before, before_mm = switches.current(), exl3_mm.current()
    yield
    switches.use(before)
    exl3_mm.use(before_mm)


def _run(k, name, R, x, sel, w, scratch):
    ex, slots, E, act = _layer(name)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    y = route(k, x, sel, None, ex, scratch, None, R, limit, act).clone()
    out = route(k, x, sel, w, ex, scratch, None, R, limit, act).clone()
    return y, out


# -- routed experts on the item list ------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(LAYERS))
@pytest.mark.parametrize("R", ROWS)
def test_routed_experts_keep_their_bytes_on_the_item_list(name, R):
    ex, slots, E, _ = _layer(name)
    scratch = experts.Scratch(ex, max(ROWS), slots)
    x, sel, w = _inputs(name, R, 3000 + R)
    want = _run(OLD, name, R, x, sel, w, scratch)
    torch.cuda.synchronize()
    assert torch.isfinite(want[1]).all()
    for k, pdl in EXPERT_SETTINGS:
        switches.use(switches.Switches(pdl=pdl))
        y, out = _run(k, name, R, x, sel, w, scratch)
        torch.cuda.synchronize()
        assert _same(y, want[0]), (name, R, k, pdl)
        assert _same(out, want[1]), (name, R, k, pdl)


@pytest.mark.parametrize("name", list(LAYERS))
@pytest.mark.parametrize("R, tile", [(1, 16), (17, 16), (28, 16), (64, 16), (65, 32), (1024, 32)])
def test_the_item_list_is_group_kernels_listing(name, R, tile):
    ex, slots, E, _ = _layer(name)
    s = experts.Scratch(ex, max(ROWS), slots)
    _, sel, _ = _inputs(name, R, 4000 + R)
    ext = experts._ext()
    ids, members = s.window(R)
    ext.group(sel, ids, s.count, members, R, slots, ex.count)
    plain = (ids.clone(), s.count.clone(), members.clone())
    items = s.items(R, tile)
    items.fill_(-7)
    ext.group(sel, ids, s.count, members, R, slots, ex.count, False, items=items, icount=s.icount, tile=tile)
    torch.cuda.synchronize()
    assert torch.equal(ids, plain[0]) and torch.equal(s.count, plain[1]) and torch.equal(members, plain[2])
    n = int(s.icount)
    want = _listing(sel.cpu(), ex.count, R, tile)
    got = [(int(v) >> 16, int(v) & 0xFFFF) for v in items[:n].cpu()]
    assert got == want
    assert n <= items.numel() and bool((items[n:] == -7).all())


@pytest.mark.parametrize("name", list(LAYERS))
def test_graphs_captured_on_the_item_list_replay_the_old_bytes(name):
    ex, slots, E, act = _layer(name)
    rows = (17, 28, 64, 128)
    scratch = experts.Scratch(ex, max(rows), slots)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    bufs, graphs = {}, {}
    switches.use(switches.Switches(pdl=True))
    for R in rows:
        x, sel, w = _inputs(name, R, 60 + R)
        out = torch.empty((R, ex.dims), dtype=torch.float32, device="cuda")
        experts.routed(x, sel, w, ex, scratch, out, R, limit, act)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            experts.routed(x, sel, w, ex, scratch, out, R, limit, act)
        bufs[R], graphs[R] = (x, sel, w, out), graph
    switches.use(switches.Switches())
    for R in rows:
        x, sel, w, out = bufs[R]
        nx, nsel, nw = _inputs(name, R, 95 + R)                   # new picks: the list is the replay's own
        x.copy_(nx), sel.copy_(nsel), w.copy_(nw)
        eager = route(OLD, x, sel, w, ex, scratch, None, R, limit, act).clone()
        out.fill_(float("nan"))
        graphs[R].replay()
        torch.cuda.synchronize()
        assert _same(out, eager), (name, R)


@pytest.mark.parametrize("name", list(LAYERS))
@pytest.mark.parametrize("R", [17, 64, 1024])
def test_bf16_slots_are_a_copy_of_the_fp32_slots(name, R):
    ex, slots, E, act = _layer(name)
    scratch = experts.Scratch(ex, max(ROWS), slots)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    x, sel, _ = _inputs(name, R, 5000 + R)
    want = experts.routed(x, sel, None, ex, scratch, None, R, limit, act).to(torch.bfloat16)
    into = torch.full((R * slots, ex.dims), float("nan"), dtype=torch.bfloat16, device="cuda")
    got = experts.routed(x, sel, None, ex, scratch, None, R, limit, act, y=into)
    torch.cuda.synchronize()
    assert got is into
    live = ((sel >= 0) & (sel < ex.count)).view(-1)
    assert torch.equal(got[live].view(torch.int16), want[live].view(torch.int16))


# -- the linear into a wider output, and the stacks --------------------------------------------------------------
@pytest.mark.parametrize("k, n, bits", LINEAR)
def test_the_linear_writes_its_columns_of_a_wider_output(k, n, bits):
    g = torch.Generator().manual_seed(7 * k + n + bits)
    tr = _trellis(k, n, 16 * bits, g).cpu()
    lin = linear.Exl3Linear.from_tensors(tr, _scale(k, 1 / math.sqrt(k), g).cpu(), _scale(n, 1.0, g).cpu(), "mul1")
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 16, 28, 128):
        x = torch.randn((m, k), generator=g).to(torch.bfloat16).cuda()
        want = lin(x).clone()
        wide = torch.full((m, n + 384), -3.0, dtype=torch.bfloat16, device="cuda")
        lin(x, out=wide[:, 128:128 + n])
        torch.cuda.synchronize()
        assert _same(wide[:, 128:128 + n].float(), want.float()), (k, n, bits, m)
        assert bool((wide[:, :128] == -3.0).all()) and bool((wide[:, 128 + n:] == -3.0).all()), (k, n, bits, m)


def _scratch(parts, rows=exl3_mm.ROWS):
    """A Scratch sized for ``parts`` as ``allocate`` sizes it (without the experts and n-gram buffers)."""

    sc = parts[0].sc
    xh = z = tmp = part = 1
    for u in parts:
        if isinstance(u, exl3_mm.X3):
            xh = max(xh, rows * u.lin.k)
            if u.lin.split[0] > 1:
                z = max(z, u.lin.split[0] * rows * u.lin.n)
        elif isinstance(u, exl3_mm.F16):
            part = max(part, u.sk * rows * u.n)
    tmp = rows * max(u.n for u in parts)
    sc.xh = torch.empty((xh,), dtype=torch.float16, device="cuda")
    sc.z = torch.empty((z,), dtype=torch.float32, device="cuda")
    sc.tmp = torch.empty((tmp,), dtype=torch.bfloat16, device="cuda")
    sc.part = torch.empty((part,), dtype=torch.float32, device="cuda")
    return sc


def _x3(sc, k, n, bits, g):
    lin = linear.Exl3Linear.from_tensors(_trellis(k, n, 16 * bits, g).cpu(), _scale(k, 1 / math.sqrt(k), g).cpu(),
                                         _scale(n, 1.0, g).cpu(), "mul1")
    return exl3_mm.X3(lin, sc)


def _f16(sc, n, k, g):
    w = (torch.randn((n, k), generator=g) / math.sqrt(k)).half().cuda()
    return exl3_mm.F16(w, n, k, exl3_mm.f16_split(n, k), sc)


STACKS = {"attention": [("x3", 2560, 12288, 5), ("x3", 2560, 512, 5), ("x3", 2560, 512, 5), ("x3", 2560, 640, 3)],
          "deltanet": [("x3", 2560, 10240, 5), ("x3", 2560, 6144, 5), ("f16", 2560, 96, 0)]}


@pytest.mark.parametrize("which", list(STACKS))
def test_a_stack_writes_the_same_bytes_in_place_as_through_the_copies(which):
    g = torch.Generator().manual_seed(len(which))
    sc = exl3_mm.Scratch(11)
    parts = [_x3(sc, k, n, b, g) if kind == "x3" else _f16(sc, n, k, g) for kind, k, n, b in STACKS[which]]
    _scratch(parts)
    st = exl3_mm.Stack(parts, sc)
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 28, 64, 128):
        x = torch.randn((m, 2560), generator=g).to(torch.bfloat16).cuda()
        got = []
        for pad in (1, 0, 1):                        # a row stride 2 bytes off 16 takes the scratch copies
            out = torch.full((m, st.n + pad), float("nan"), dtype=torch.bfloat16, device="cuda")[:, :st.n]
            assert st.aligned(out) is (pad == 0)
            got.append(st(x, out).clone())
        torch.cuda.synchronize()
        assert torch.isfinite(got[0].float()).all(), (which, m)
        assert _same(got[0].float(), got[1].float()) and _same(got[0].float(), got[2].float()), (which, m)


# -- the prompt's fp16 matmuls with their K slices summed in-program ----------------------------------------------
F16_SHAPES = [(324, 10240), (320, 10240), (96, 2560), (2560, 2560)]          # HC down, mixer down, b/a, PLE value
F16_ROWS = (17, 33, 48, 129, 241, 256, 305, 753, 768, 1024, 2048, 2050, 2061)


@pytest.mark.parametrize("n, k", F16_SHAPES)
def test_the_in_program_sum_writes_the_split_sums_bytes(n, k):
    g = torch.Generator().manual_seed(n + k)
    sc = exl3_mm.Scratch(11)
    q = _f16(sc, n, k, g)
    _scratch([q])
    used = 0
    for m in F16_ROWS:
        x = (torch.randn((m, k), generator=g) * 0.3).to(torch.bfloat16).cuda()
        for wide in (False, True):                                    # contiguous, and a stack's columns
            outs = []
            for on in (False, True):
                exl3_mm.use(exl3_mm.Switches(inner_k=on))
                full = torch.full((m, n + (128 if wide else 0)), float("nan"), dtype=torch.bfloat16, device="cuda")
                outs.append(q(x, full[:, :n]).clone())
            torch.cuda.synchronize()
            assert torch.isfinite(outs[0].float()).all(), (n, k, m)
            assert _same(outs[0].float(), outs[1].float()), (n, k, m, wide)
        used += exl3_mm.inner_k_fits(m, n, k, q.sk)
    assert used > 0                                                    # some row counts took _f16_mm_ks


# -- an fp16 hyper-connection's prompt write-back and norm in one kernel -------------------------------------------
def _hc_inputs(R, g, mode):
    from types import SimpleNamespace

    S, D, low = 4, 2560, 320
    wide = S * D
    dev = "cuda"
    b = SimpleNamespace(prefill=True)
    b.pss = torch.empty((R, D // 256, S), dtype=torch.float32, device=dev)
    b.normed = torch.empty((R, wide), dtype=torch.bfloat16, device=dev)
    b.xs_normed = torch.empty((R, wide // 32), dtype=torch.float32, device=dev)
    b.dn = torch.empty((R, low + S), dtype=torch.bfloat16, device=dev)
    b.dn_mix = torch.empty((R, low), dtype=torch.bfloat16, device=dev)
    b.act = torch.empty((R, low), dtype=torch.bfloat16, device=dev)
    b.xs_act = torch.empty((R, low // 32), dtype=torch.float32, device=dev)
    b.up = torch.empty((R, wide), dtype=torch.bfloat16, device=dev)
    b.mixed = torch.empty((R, D), dtype=torch.bfloat16, device=dev)
    b.xs_mixed = torch.empty((R, D // 32), dtype=torch.float32, device=dev)
    h = torch.randn((R, wide), generator=g).to(torch.bfloat16).to(dev)
    extra = {}
    if mode == 1:
        extra["branch"] = torch.randn((R, D), generator=g).to(torch.bfloat16).to(dev)
    elif mode == 2:
        extra["y"] = torch.randn((R, 11, D), generator=g).to(torch.bfloat16).to(dev)
        extra["wts"] = (torch.rand((R, 11), generator=g) * 0.2).to(dev)
    inject_prev = torch.randn((R, S), generator=g).to(torch.bfloat16).to(dev)
    return b, h, extra, inject_prev


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("R", [17, 256, 2048])
def test_an_fp16_hcs_fused_write_back_and_norm_keep_the_read_outs_bytes(mode, R):
    from tensorfold.families.qwen4_exp.cuda import forward
    from tensorfold.families.qwen4_exp.cuda.weight_types import HC

    if forward._hc_fuser(torch.device("cuda")) is None:
        pytest.skip("the write-norm fusion's self-check did not pass on this GPU")
    g = torch.Generator().manual_seed(R + mode)
    sc = exl3_mm.Scratch(11)
    down, up = _f16(sc, 324, 10240, g), _f16(sc, 10240, 320, g)
    _scratch([down, up])
    scale = (torch.rand((10240,), generator=g) + 0.5).float().cuda()
    hc = HC(down, up, scale, True, down, up)
    got = []
    for fuse in (False, True, False):
        b, h, extra, inject_prev = _hc_inputs(R, torch.Generator().manual_seed(R + mode), mode)
        inject_out = torch.empty((R, 4), dtype=torch.bfloat16, device="cuda")
        prev = None if mode == 0 else inject_prev
        if fuse:
            forward.hc_block(hc, b, R, 1e-6, 4, 320, mode, prev, inject_out, h, **extra)
        else:                                       # the separate kernels hc_block ran before the fusion
            forward.glue.hc_writeback(h, h, b.pss, 4, mode, inject=prev, **extra)
            forward._readout(hc, b, h, R, 1e-6, 4, 320, inject_out)
        torch.cuda.synchronize()
        got.append([t.clone() for t in (h, b.pss, b.normed, b.act, b.mixed, inject_out)])
    for a, c in zip(got[0], got[1]):
        assert torch.equal(a.view(torch.uint8), c.view(torch.uint8)), (mode, R)
    for a, c in zip(got[0], got[2]):
        assert torch.equal(a.view(torch.uint8), c.view(torch.uint8)), (mode, R)
