"""TF_EXL3_GROUP_SCAN_DECODE and TF_EXL3_LINEAR_PAIR on the GPU: the same bytes with either switch on or off, the
prompt switches (TF_EXL3_GROUP_SCAN, _DOWN_FUSE, _GATEUP_FUSE) at their defaults.

- group_scan_kernel (TF_EXL3_GROUP_SCAN's, here on decode windows) against group_kernel: the expert ids, their count,
  the member lists and the item list (every int32 of each buffer, filled with a sentinel first, so a write past what
  group_kernel writes shows), at Flash Next's grouping (513 experts with the shared one, 11 slots) and GLM-5.3's
  (8 or 9 slots), windows of 1 to 128 rows and one of 1,024, with repeated picks in a row, picks of -1 and of E, and
  member lists cut short of the count. tests/cuda/test_exl3_prefill_kernels.py covers its prompt windows.
- Routed experts at both models' shapes (tests/cuda/test_exl3_decode_kernels.py's layers): per-slot outputs and
  combined rows with the switch on and off, eager and replayed from CUDA graphs.
- The EXL3 linear with two row tiles a pass against one: Flash Next's dense shapes and GLM-5.3-sized ones, mul1 and
  mcg, 1 to 128 rows (one pass, two, a pass of one tile after full ones), split-K shapes, a wider output, and graphs.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

from tensorfold.cuda.exl3 import experts, linear, switches
from tests.cuda.test_exl3_decode_kernels import LAYERS, LINEAR, _inputs, _layer, _same, _scale, _trellis
from tests.test_exl3_expert_items import _listing

OFF = switches.Switches(group_scan=True, down_fuse=True, gateup_fuse=True, group_scan_decode=False,
                        linear_pair=False)     # the defaults since fba05ef2
SCAN = dataclasses.replace(OFF, group_scan_decode=True)
PAIR = dataclasses.replace(OFF, linear_pair=True)
SENTINEL = -12345


@pytest.fixture(autouse=True)
def restore():
    before = switches.current()
    yield
    switches.use(before)


# -- the grouping ---------------------------------------------------------------------------------------------------
def _picks(R, slots, E, seed, repeats=False, invalid=False):
    """R rows of ``slots`` picks below E (distinct in a row unless ``repeats``), some -1 and E when ``invalid``."""

    g = torch.Generator().manual_seed(seed)
    if repeats:
        sel = torch.randint(0, min(E, 6), (R, slots), generator=g)
    else:
        sel = torch.stack([torch.randperm(E, generator=g)[:slots] for _ in range(R)])
    if invalid:
        hole = torch.rand((R, slots), generator=g)
        sel = torch.where(hole < 0.1, torch.full_like(sel, -1), torch.where(hole > 0.9, torch.full_like(sel, E), sel))
    return sel.to(torch.int32).cuda().contiguous()


def _group(pick, R, slots, E, maxm, scan, tile=0):
    """Every buffer group writes, each filled with SENTINEL first."""

    maxu = min(R * slots, E)
    ids = torch.full((maxu,), SENTINEL, dtype=torch.int32, device="cuda")
    count = torch.full((1,), SENTINEL, dtype=torch.int32, device="cuda")
    members = torch.full((maxu, maxm), SENTINEL, dtype=torch.int32, device="cuda")
    kw = {}
    if tile:
        items = torch.full((experts.item_bound(R * slots, maxu, tile),), SENTINEL, dtype=torch.int32, device="cuda")
        icount = torch.full((1,), SENTINEL, dtype=torch.int32, device="cuda")
        kw = {"items": items, "icount": icount, "tile": tile}
    experts._ext().group(pick, ids, count, members, R, slots, E, False, scan=scan, **kw)
    torch.cuda.synchronize()
    return [ids, count, members] + ([kw["items"], kw["icount"]] if tile else [])


GROUPINGS = {"flash_next": (11, 513), "glm53_9": (9, 257), "glm53_8": (8, 256)}


@pytest.mark.parametrize("model", list(GROUPINGS))
@pytest.mark.parametrize("R", [1, 2, 5, 7, 8, 16, 17, 28, 64, 65, 128, 1024])
@pytest.mark.parametrize("repeats, invalid", [(False, False), (False, True), (True, True)])
def test_group_scan_kernel_writes_group_kernels_integers(model, R, repeats, invalid):
    slots, E = GROUPINGS[model]
    pick = _picks(R, slots, E, 17 * R + slots, repeats, invalid)
    want = _group(pick, R, slots, E, R, False)
    got = _group(pick, R, slots, E, R, True)
    for a, b in zip(want, got):
        assert torch.equal(a, b), (model, R)
    n = int(want[1])
    assert n > 0 or invalid


@pytest.mark.parametrize("R, maxm", [(8, 1), (8, 3), (28, 5), (64, 33)])
def test_member_lists_cut_short_of_the_count_match(R, maxm):
    """A row may pick an expert twice (``repeats``), so an expert's members can pass the list's width."""

    pick = _picks(R, 9, 288, R + maxm, repeats=True)
    want = _group(pick, R, 9, 288, maxm, False)
    got = _group(pick, R, 9, 288, maxm, True)
    for a, b in zip(want, got):
        assert torch.equal(a, b), (R, maxm)


@pytest.mark.parametrize("model", list(GROUPINGS))
@pytest.mark.parametrize("R, tile", [(1, 16), (17, 16), (28, 16), (64, 16), (65, 32), (1024, 32)])
def test_group_scan_kernel_writes_the_same_item_list(model, R, tile):
    slots, E = GROUPINGS[model]
    pick = _picks(R, slots, E, 4000 + R, invalid=True)
    want = _group(pick, R, slots, E, R, False, tile)
    got = _group(pick, R, slots, E, R, True, tile)
    for a, b in zip(want, got):
        assert torch.equal(a, b), (model, R, tile)
    n = int(got[4])
    assert [(int(v) >> 16, int(v) & 0xFFFF) for v in got[3][:n].cpu()] == _listing(pick.cpu(), E, R, tile)


# -- routed experts end to end --------------------------------------------------------------------------------------
def _run(name, R, x, sel, w, scratch):
    ex, _, _, act = _layer(name)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    y = experts.routed(x, sel, None, ex, scratch, None, R, limit, act).clone()
    out = experts.routed(x, sel, w, ex, scratch, None, R, limit, act).clone()
    return y, out


@pytest.mark.parametrize("name", list(LAYERS))
@pytest.mark.parametrize("R", [1, 5, 7, 16, 17, 28, 64, 65, 128])
def test_routed_experts_keep_their_bytes_with_the_scan(name, R):
    ex, slots, _, _ = _layer(name)
    scratch = experts.Scratch(ex, 128, slots)
    x, sel, w = _inputs(name, R, 6000 + R)
    got = []
    for sw in (OFF, SCAN, OFF):
        switches.use(sw)
        got.append(_run(name, R, x, sel, w, scratch))
    torch.cuda.synchronize()
    assert torch.isfinite(got[0][1]).all()
    for y, out in got[1:]:
        assert _same(y, got[0][0]) and _same(out, got[0][1]), (name, R)


@pytest.mark.parametrize("name", list(LAYERS))
def test_graphs_captured_with_the_scan_replay_the_old_bytes(name):
    ex, slots, _, act = _layer(name)
    rows = (1, 5, 7, 8, 28)
    scratch = experts.Scratch(ex, max(rows), slots)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    bufs, graphs = {}, {}
    switches.use(SCAN)
    for R in rows:
        x, sel, w = _inputs(name, R, 70 + R)
        out = torch.empty((R, ex.dims), dtype=torch.float32, device="cuda")
        experts.routed(x, sel, w, ex, scratch, out, R, limit, act)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            experts.routed(x, sel, w, ex, scratch, out, R, limit, act)
        bufs[R], graphs[R] = (x, sel, w, out), graph
    switches.use(OFF)
    for R in rows:
        x, sel, w, out = bufs[R]
        nx, nsel, nw = _inputs(name, R, 110 + R)                  # new picks: the grouping is the replay's own
        x.copy_(nx), sel.copy_(nsel), w.copy_(nw)
        eager = experts.routed(x, sel, w, ex, scratch, None, R, limit, act).clone()
        out.fill_(float("nan"))
        graphs[R].replay()
        torch.cuda.synchronize()
        assert _same(out, eager), (name, R)


# -- the linear -----------------------------------------------------------------------------------------------------
ROWS = (1, 7, 16, 17, 20, 28, 31, 32, 33, 40, 48, 64, 65, 100, 128)


def _lin(k, n, bits, codebook, seed):
    g = torch.Generator().manual_seed(seed)
    tr = _trellis(k, n, 16 * bits, g).cpu()
    return linear.Exl3Linear.from_tensors(tr, _scale(k, 1 / math.sqrt(k), g).cpu(), _scale(n, 1.0, g).cpu(),
                                          codebook), g


@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
@pytest.mark.parametrize("k, n, bits", LINEAR)
def test_the_linear_keeps_its_bytes_with_two_row_tiles(k, n, bits, codebook):
    lin, g = _lin(k, n, bits, codebook, 3 * k + n + bits)
    for m in ROWS:
        x = torch.randn((m, k), generator=g).to(torch.bfloat16).cuda()
        got = []
        for sw in (OFF, PAIR, OFF):
            switches.use(sw)
            got.append(lin(x).clone())
        torch.cuda.synchronize()
        assert torch.isfinite(got[0].float()).all()
        assert _same(got[0].float(), got[1].float()) and _same(got[0].float(), got[2].float()), (k, n, bits, m)


@pytest.mark.parametrize("k, n, bits", [(2560, 10240, 5), (6144, 2560, 5), (2560, 512, 5)])
def test_two_row_tiles_write_their_columns_of_a_wider_output(k, n, bits):
    lin, g = _lin(k, n, bits, "mul1", 11 * k + n)
    for m in (17, 28, 64):
        x = torch.randn((m, k), generator=g).to(torch.bfloat16).cuda()
        switches.use(OFF)
        want = lin(x).clone()
        switches.use(PAIR)
        wide = torch.full((m, n + 384), -3.0, dtype=torch.bfloat16, device="cuda")
        lin(x, out=wide[:, 128:128 + n])
        torch.cuda.synchronize()
        assert _same(wide[:, 128:128 + n].float(), want.float()), (k, n, m)
        assert bool((wide[:, :128] == -3.0).all()) and bool((wide[:, 128 + n:] == -3.0).all()), (k, n, m)


def test_graphs_captured_with_two_row_tiles_replay_the_old_bytes():
    lin, g = _lin(2560, 6144, 5, "mul1", 99)                      # 4 K splits: the counters are reset in the graph
    rows = (17, 28)
    bufs, graphs = {}, {}
    switches.use(PAIR)
    for m in rows:
        x = torch.randn((m, 2560), generator=g).to(torch.bfloat16).cuda()
        out = torch.empty((m, 6144), dtype=torch.bfloat16, device="cuda")
        xh = torch.empty((m, 2560), dtype=torch.float16, device="cuda")
        z = torch.empty((lin.split[0] * m * 6144,), dtype=torch.float32, device="cuda")
        lin(x, out=out, xh=xh, z=z)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            lin(x, out=out, xh=xh, z=z)
        bufs[m], graphs[m] = (x, out), graph
    switches.use(OFF)
    for _ in range(2):                                            # twice: the counters came back to zero
        for m in rows:
            x, out = bufs[m]
            x.copy_(torch.randn((m, 2560), generator=g).to(torch.bfloat16).cuda())
            want = lin(x).clone()
            out.fill_(float("nan"))
            graphs[m].replay()
            torch.cuda.synchronize()
            assert _same(out.float(), want.float()), m
