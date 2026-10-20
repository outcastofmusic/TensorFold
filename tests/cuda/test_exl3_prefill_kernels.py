"""The prompt-window switches on the GPU: the same bytes with each switch on or off.

- TF_EXL3_GROUP_SCAN: group_scan_kernel against group_kernel on the same picks, whole buffers compared (ids, count,
  members, item list and its count, and what neither writes), at Flash Next's grouping (513 experts, 11 slots),
  GLM-5.3's (257, 9), 48 experts and the scan's limit of 1,024, windows of 65 to 2,048 rows, with skewed routing,
  picks of -1 and of E, and PDL on and off.
- TF_EXL3_DOWN_FUSE, TF_EXL3_GATEUP_FUSE, and all three together: routed experts at Flash Next EXL3's shapes (2560 <->
  640, 3-bit mul1 experts beside a 5-bit shared expert, 11 slots) and GLM-5.3's (4096 <-> 2048, mcg at 2.25 and 2.5
  bits, 14-bit window, 8 slots): fp32 slots, bf16 slots into a buffer (TF_FLASHNEXT_MOE_DIRECT's call) and combined
  rows, windows of 65 to 2,048 rows, PDL on and off, eager and replayed from a CUDA graph; the arrival counters are
  zero after every launch. A planted control: counters that never reach the last arrival leave Xd unwritten and the
  bytes differ.
- TF_FLASHNEXT_MOE_CHUNK's call: one routed call of 2,048 rows equals two of 1,024, with the switches off and on.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

from tensorfold.cuda.exl3 import experts, switches  # noqa: E402

from tests.cuda.test_exl3_decode_kernels import LAYERS, _inputs, _layer, _same  # noqa: E402

OFF = switches.Switches(expert_items=True, group_scan=False, down_fuse=False, gateup_fuse=False,
                        group_scan_decode=False, linear_pair=False)                        # today's defaults
SETTINGS = {
    "scan": switches.Switches(expert_items=True, group_scan=True, down_fuse=False, gateup_fuse=False,
                              group_scan_decode=False, linear_pair=False),
    "down": switches.Switches(expert_items=True, down_fuse=True, group_scan=False, gateup_fuse=False,
                              group_scan_decode=False, linear_pair=False),
    "gateup": switches.Switches(expert_items=True, gateup_fuse=True, group_scan=False, down_fuse=False,
                                group_scan_decode=False, linear_pair=False),
    "all": switches.Switches(expert_items=True, group_scan=True, down_fuse=True, gateup_fuse=True,
                             group_scan_decode=False, linear_pair=False),
    "all pdl": switches.Switches(pdl=True, expert_items=True, group_scan=True, down_fuse=True, gateup_fuse=True,
                                 group_scan_decode=False, linear_pair=False),
}
ROWS = (65, 128, 1000, 1024, 2048)


@pytest.fixture(autouse=True)
def restore():
    before = switches.current()
    yield
    switches.use(before)


# -- the grouping ---------------------------------------------------------------------------------------------------
def _picks(R: int, slots: int, E: int, seed: int, hot: bool) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    routed = min(slots - 1, E - 1)
    weights = torch.rand((E - 1,), generator=g) ** (6 if hot else 1)
    pick = torch.stack([torch.multinomial(weights, routed, generator=g) for _ in range(R)]).to(torch.int32)
    pick = torch.cat([pick, torch.full((R, 1), E - 1, dtype=torch.int32)], 1)     # a shared expert every row
    if slots - 1 > routed:
        pick = torch.cat([pick, torch.full((R, slots - 1 - routed), E, dtype=torch.int32)], 1)
    pick[0, 0] = -1
    pick[R // 2, 1] = E
    return pick.cuda().contiguous()


def _group(ext, pick, R, slots, E, tile, scan, pdl):
    maxu = min(R * slots, E)
    ids = torch.full((maxu,), -5, dtype=torch.int32, device="cuda")
    count = torch.full((1,), -5, dtype=torch.int32, device="cuda")
    members = torch.full((maxu, R), -9, dtype=torch.int32, device="cuda")
    kw = {}
    if tile:
        kw = {"items": torch.full((experts.item_bound(R * slots, maxu, tile),), -7, dtype=torch.int32, device="cuda"),
              "icount": torch.full((1,), -5, dtype=torch.int32, device="cuda"), "tile": tile}
    ext.group(pick, ids, count, members, R, slots, E, pdl, **kw, **({"scan": True} if scan else {}))
    torch.cuda.synchronize()
    return [ids, count, members] + ([kw["items"], kw["icount"]] if tile else [])


@pytest.mark.parametrize("E, slots", [(513, 11), (257, 9), (48, 11), (1024, 8)])
@pytest.mark.parametrize("R", [65, 128, 1000, 1024, 2048])
@pytest.mark.parametrize("tile", [0, 16, 32])
@pytest.mark.parametrize("hot", [True, False])
def test_the_scan_grouping_writes_group_kernels_buffers(E, slots, R, tile, hot):
    ext = experts._ext()
    pick = _picks(R, slots, E, 7 * R + E + tile, hot)
    want = _group(ext, pick, R, slots, E, tile, False, False)
    assert int(want[1]) > 0
    for pdl in (False, True):
        got = _group(ext, pick, R, slots, E, tile, True, pdl)
        for a, b in zip(got, want):
            assert torch.equal(a, b), (E, slots, R, tile, hot, pdl)


def _reference(pick: torch.Tensor, R: int, slots: int, E: int, tile: int):
    """group_kernel's listing from torch: experts in id order, members in pick order, -1 after, the item list."""

    flat = pick.flatten().cpu().long()
    valid = (flat >= 0) & (flat < E)
    counts = torch.bincount(flat[valid], minlength=E)
    ids = torch.nonzero(counts).flatten()
    order = torch.argsort(torch.where(valid, flat, E), stable=True)
    starts = torch.cumsum(counts, 0) - counts
    members = torch.full((len(ids), R), -1, dtype=torch.int32)
    items = []
    for place, e in enumerate(ids.tolist()):
        at = order[starts[e]:starts[e] + counts[e]][:R]
        members[place, :len(at)] = ((at // slots) * 32 + at % slots).to(torch.int32)
        items += [(place << 16) | t for t in range(-(-len(at) // tile))]
    return ids.to(torch.int32), members, torch.tensor(items, dtype=torch.int32)


def test_the_scan_grouping_at_its_pick_limit():
    """2,978 rows of 22 slots: 65,516 picks (a 16-bit rank's limit) over 1,024 experts, past group_kernel's shared
    memory, against torch's listing."""

    ext = experts._ext()
    R, slots, E = 2978, 22, 1024
    pick = _picks(R, slots, E, 99, True)
    ids, count, members, items, icount = _group(ext, pick, R, slots, E, 32, True, False)
    want_ids, want_members, want_items = _reference(pick, R, slots, E, 32)
    n, m = int(count), int(icount)
    assert n == len(want_ids) and m == len(want_items)
    assert torch.equal(ids[:n].cpu(), want_ids)
    assert torch.equal(members[:n].cpu(), want_members)
    assert torch.equal(items[:m].cpu(), want_items)
    assert bool((members[n:] == -9).all()) and bool((items[m:] == -7).all())


# -- routed experts with the fused epilogues ------------------------------------------------------------------------
def _three(name, R, x, sel, w, scratch, ex, act):
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    y = experts.routed(x, sel, None, ex, scratch, None, R, limit, act).clone()
    into = torch.full((R * scratch.slots, ex.dims), float("nan"), dtype=torch.bfloat16, device="cuda")
    yb = experts.routed(x, sel, None, ex, scratch, None, R, limit, act, y=into).clone()
    out = experts.routed(x, sel, w, ex, scratch, None, R, limit, act).clone()
    return y, yb, out


def _live(sel, E):
    return ((sel >= 0) & (sel < E)).view(-1)


@pytest.mark.parametrize("name", list(LAYERS))
@pytest.mark.parametrize("R", ROWS)
def test_routed_experts_keep_their_bytes_with_each_prompt_switch(name, R):
    ex, slots, _, act = _layer(name)
    scratch = experts.Scratch(ex, max(ROWS), slots)
    x, sel, w = _inputs(name, R, 7000 + R)
    live = _live(sel, ex.count)
    switches.use(OFF)
    want = _three(name, R, x, sel, w, scratch, ex, act)
    torch.cuda.synchronize()
    assert torch.isfinite(want[2]).all()
    for label, sw in SETTINGS.items():
        switches.use(sw)
        scratch.y.fill_(float("nan"))
        scratch.xd.fill_(float("nan"))
        y, yb, out = _three(name, R, x, sel, w, scratch, ex, act)
        torch.cuda.synchronize()
        assert _same(y[live], want[0][live]), (name, R, label)
        assert torch.equal(yb[live].view(torch.int16), want[1][live].view(torch.int16)), (name, R, label)
        assert _same(out, want[2]), (name, R, label)
        assert int(scratch.arrivals.abs().sum()) == 0, (name, R, label)


@pytest.mark.parametrize("name", list(LAYERS))
def test_a_2048_row_call_equals_two_1024_row_calls(name):
    ex, slots, _, act = _layer(name)
    scratch = experts.Scratch(ex, 2048, slots)
    x, sel, _ = _inputs(name, 2048, 8100)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    live = _live(sel, ex.count)
    got = {}
    for label, sw in (("off", OFF), ("all", SETTINGS["all"])):
        switches.use(sw)
        whole = experts.routed(x, sel, None, ex, scratch, None, 2048, limit, act).clone()
        halves = torch.cat([experts.routed(x[r0:r0 + 1024], sel[r0:r0 + 1024], None, ex, scratch, None, 1024, limit,
                                           act).clone() for r0 in (0, 1024)])
        torch.cuda.synchronize()
        assert _same(whole[live], halves[live]), (name, label)
        got[label] = whole
    assert _same(got["off"][live], got["all"][live]), name


@pytest.mark.parametrize("name", list(LAYERS))
def test_a_graph_of_a_fused_prompt_window_replays_the_eager_bytes(name):
    ex, slots, _, act = _layer(name)
    R = 128
    scratch = experts.Scratch(ex, R, slots)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    x, sel, w = _inputs(name, R, 8200)
    out = torch.empty((R, ex.dims), dtype=torch.float32, device="cuda")
    into = torch.empty((R * slots, ex.dims), dtype=torch.bfloat16, device="cuda")
    switches.use(SETTINGS["all pdl"])
    experts.routed(x, sel, w, ex, scratch, out, R, limit, act)
    experts.routed(x, sel, None, ex, scratch, None, R, limit, act, y=into)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        experts.routed(x, sel, None, ex, scratch, None, R, limit, act, y=into)
    nx, nsel, _ = _inputs(name, R, 8300)
    x.copy_(nx), sel.copy_(nsel)
    switches.use(OFF)
    want = experts.routed(x, sel, None, ex, scratch, None, R, limit, act, y=torch.empty_like(into)).clone()
    into.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    live = _live(sel, ex.count)
    assert torch.equal(into[live].view(torch.int16), want[live].view(torch.int16)), name
    assert int(scratch.arrivals.abs().sum()) == 0


@pytest.mark.parametrize("name", ["flash_next"])
def test_the_control_counters_that_never_finish_leave_other_bytes(name):
    """Planted control (playbook 6.2): with the counters started far from zero no program is the last, Xd keeps
    its NaN fill, and the slots differ from the epilogue kernel's. The test above would go red on a broken count."""

    ex, slots, _, act = _layer(name)
    R = 128
    scratch = experts.Scratch(ex, R, slots)
    x, sel, _ = _inputs(name, R, 8400)
    live = _live(sel, ex.count)
    switches.use(OFF)
    want = experts.routed(x, sel, None, ex, scratch, None, R, math.inf, act).clone()
    switches.use(SETTINGS["gateup"])
    scratch.xd.fill_(float("nan"))
    scratch.arrivals.fill_(1 << 20)
    got = experts.routed(x, sel, None, ex, scratch, None, R, math.inf, act).clone()
    torch.cuda.synchronize()
    scratch.arrivals.zero_()
    assert not _same(got[live], want[live])
    got = experts.routed(x, sel, None, ex, scratch, None, R, math.inf, act).clone()
    torch.cuda.synchronize()
    assert _same(got[live], want[live])
