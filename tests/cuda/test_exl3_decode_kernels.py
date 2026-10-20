"""TF_EXL3_EXPERT_DECODE and TF_EXL3_PDL on the GPU: the same bytes with either switch on or off.

- Routed experts at Flash Next EXL3's shapes (2560 <-> 640, 3-bit mul1 experts beside a 5-bit shared expert, 11
  slots) and GLM-5.3's (4096 <-> 2048, mcg at 2.25 bits and at 2.5 bits, 14-bit window, 8 slots): per-slot outputs
  and the combined rows of decode windows of 1-8 and 16 rows, and the prompt windows (17, 64, 65, 128 rows), equal
  bit for bit across decode_kernel/grouped_kernel and PDL on/off, eager and replayed from CUDA graphs.
- Every decode_kernel tile setting (the default and tools/exl3_expert_sweep.py's) against grouped_kernel.
- The EXL3 linear at Flash Next's dense shapes and GLM-5.3-sized ones, 1-8, 16 and 128 rows, PDL on and off.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

from tensorfold.cuda.exl3 import experts, linear, switches  # noqa: E402

OFF = switches.Switches(expert_decode=False, pdl=False, group_scan=False, down_fuse=False, gateup_fuse=False,
                        group_scan_decode=False, linear_pair=False)
OLD = dict(group_scan=False, down_fuse=False, gateup_fuse=False, group_scan_decode=False, linear_pair=False)
SETTINGS = [OFF, switches.Switches(True, False, **OLD), switches.Switches(False, True, **OLD),
            switches.Switches(True, True, **OLD)]
DECODE_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 16)
PROMPT_ROWS = (17, 64, 65, 128)


@pytest.fixture(autouse=True)
def restore():
    before = switches.current()
    yield
    switches.use(before)


def _trellis(k, n, words16, g):
    v = torch.randint(-32768, 32768, (k // 16, n // 16, words16), dtype=torch.int32, generator=g)
    return v.to(torch.int16).cuda().contiguous()


def _scale(n, mag, g):
    sign = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    return (sign * (torch.rand((n,), generator=g) + 0.5) * mag).half().cuda()


def _flash_next(E=48, seed=1):
    """E routed experts at 3 bits (48 int16 a tile) and the shared expert (index E) at 5 bits (80), mul1."""

    D, I = 2560, 640
    g = torch.Generator().manual_seed(seed)
    mats = {p: [] for p in ("gate", "up", "down")}
    for e in range(E + 1):
        w16 = 80 if e == E else 48
        for p, (k, n) in (("gate", (D, I)), ("up", (D, I)), ("down", (I, D))):
            mats[p].append((_trellis(k, n, w16, g), _scale(k, 1 / math.sqrt(k), g), _scale(n, 0.5, g)))
    return experts.prepare(mats["gate"], mats["up"], mats["down"], "mul1"), 11, E, experts.ACT_F32


def _glm(words16, E=48, seed=2):
    """A stacked GLM-5.3 layer: 2.25 bits (36 int16 a tile) or 2.5 bits (40), mcg, 14-bit window."""

    D, I = 4096, 2048
    g = torch.Generator().manual_seed(seed)
    tr = [_trellis(k, n, words16, g).unsqueeze(0).expand(E, -1, -1, -1).clone()
          for k, n in ((D, I), (D, I), (I, D))]
    # every expert its own words (a stacked copy would hide a wrong expert stride)
    for t in tr:
        t.view(E, -1)[:, :97] = torch.randint(-32768, 32768, (E, 97), dtype=torch.int32, generator=g).to(torch.int16).cuda()

    def sc(n, mag):
        return torch.stack([_scale(n, mag, g) for _ in range(E)])

    ex = experts.prepare_stacked(tr[0], tr[1], tr[2], sc(D, 1 / 64), sc(D, 1 / 64), sc(I, 0.5), sc(I, 0.5),
                                 sc(I, 1 / 45), sc(D, 0.5), "mcg", window=14)
    return ex, 8, E, experts.ACT_BF16


LAYERS = {"flash_next": lambda: _flash_next(), "glm53_2.25bit": lambda: _glm(36), "glm53_2.5bit": lambda: _glm(40)}
_built: dict = {}


def _layer(name):
    if name not in _built:
        _built.clear()
        torch.cuda.empty_cache()
        _built[name] = LAYERS[name]()
    return _built[name]


def _inputs(name, R, seed):
    ex, slots, E, _ = _layer(name)
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((R, ex.dims), generator=g) * 0.5).to(torch.bfloat16).cuda()
    routed = slots - 1 if name == "flash_next" else slots
    sel = torch.stack([torch.randperm(E, generator=g)[:routed] for _ in range(R)]).to(torch.int32)
    w = torch.rand((R, slots), generator=g) * 0.2 + 0.05
    if name == "flash_next":                                  # the shared expert in the last slot
        sel = torch.cat([sel, torch.full((R, 1), E, dtype=torch.int32)], 1)
    return x, sel.cuda().contiguous(), w.float().cuda().contiguous()


def _run(name, R, x, sel, w, scratch):
    ex, slots, E, act = _layer(name)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    y = experts.routed(x, sel, None, ex, scratch, None, R, limit, act).clone()
    out = experts.routed(x, sel, w, ex, scratch, None, R, limit, act).clone()
    return y, out


def _same(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.parametrize("name", list(LAYERS))
@pytest.mark.parametrize("R", DECODE_ROWS + PROMPT_ROWS)
def test_routed_experts_keep_their_bytes_with_either_switch(name, R):
    ex, slots, E, _ = _layer(name)
    scratch = experts.Scratch(ex, max(PROMPT_ROWS), slots)
    x, sel, w = _inputs(name, R, 1000 + R)
    got = []
    for sw in SETTINGS:
        switches.use(sw)
        got.append(_run(name, R, x, sel, w, scratch))
    torch.cuda.synchronize()
    assert torch.isfinite(got[0][1]).all()
    for sw, (y, out) in zip(SETTINGS[1:], got[1:]):
        assert _same(y, got[0][0]), (name, R, sw)
        assert _same(out, got[0][1]), (name, R, sw)


@pytest.mark.parametrize("name", list(LAYERS))
def test_graphs_captured_with_the_switches_on_replay_the_old_bytes(name):
    ex, slots, E, act = _layer(name)
    rows = (1, 2, 4, 7, 8)
    scratch = experts.Scratch(ex, max(rows), slots)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    bufs, graphs = {}, {}
    switches.use(switches.Switches(True, True, group_scan=False, down_fuse=False, gateup_fuse=False,
                                   group_scan_decode=False, linear_pair=False))
    for R in rows:
        x, sel, w = _inputs(name, R, 50 + R)
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
        nx, nsel, nw = _inputs(name, R, 90 + R)
        x.copy_(nx), sel.copy_(nsel), w.copy_(nw)
        eager = experts.routed(x, sel, w, ex, scratch, None, R, limit, act).clone()
        out.fill_(float("nan"))
        graphs[R].replay()
        torch.cuda.synchronize()
        assert _same(out, eager), (name, R)


@pytest.mark.parametrize("name", list(LAYERS))
@pytest.mark.parametrize("nt, pf", [(4, 4), (2, 4), (4, 2), (2, 8)])
def test_every_decode_tile_setting_writes_grouped_kernels_partial_sums(name, nt, pf):
    """The raw split partials Z of both projections, each decode_kernel instance against grouped_kernel."""

    ex, slots, E, _ = _layer(name)
    ext = experts._ext()
    D, I = ex.dims, ex.width
    for R in DECODE_ROWS:
        s = experts.Scratch(ex, R, slots)
        x, sel, _ = _inputs(name, R, 7 * R + nt + pf)
        P = R * slots
        ids, members = s.window(R)
        ext.group(sel, ids, s.count, members, R, slots, E)
        ext.rot_in(x, x.stride(0), sel, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E)
        s.xd.copy_(torch.randn(s.xd.shape, generator=torch.Generator().manual_seed(R)).half().cuda() * 0.1)
        for mats, (k, n), cfg, xs, ptr, k2, rng in (
                (2, (D, I), s.cfg_gu, (s.xg, s.xu), (ex.gate_ptr, ex.up_ptr), (ex.gate_k2, ex.up_k2), ex.k2_gu),
                (1, (I, D), s.cfg_d, (s.xd, s.xd), (ex.down_ptr, ex.down_ptr), (ex.down_k2, ex.down_k2), ex.k2_d)):
            _, w_, sk, pf_old = cfg
            got = []
            for dec, tile in ((False, (cfg[0], pf_old)), (True, (nt, pf))):
                s.z.fill_(float("nan"))
                ext.grouped(*xs, *ptr, *k2, ids, s.count, members, s.z, mats, k, n, P, sk, slots, ex.cb, tile[0], w_,
                            tile[1], rng[0], rng[1], 1, ex.window, dec, True)
                torch.cuda.synchronize()
                got.append(s.z[:mats * sk * P * n].view(mats * sk, P, n).clone())
            routed = (sel.view(-1) < ex.count).cpu()          # the shared expert too (index E of E + 1)
            for a, b in zip(got[0].unbind(0), got[1].unbind(0)):
                assert _same(a[routed], b[routed]), (name, R, mats)


# ------------------------------------------------------------------------------------------------- the linear

# (K, N, bits): Flash Next's dense trellis projections (DeltaNet qkv, z, out; attention q, k/v, o; the shared expert;
# the indexer; the MTP fc), and GLM-5.3-sized ones
LINEAR = [(2560, 10240, 5), (2560, 6144, 5), (6144, 2560, 5), (2560, 12288, 5), (2560, 512, 5), (2560, 640, 5),
          (640, 2560, 5), (2560, 640, 3), (2560, 2560, 4), (4096, 2048, 4), (2048, 4096, 3), (4096, 512, 6)]


@pytest.mark.parametrize("k, n, bits", LINEAR)
def test_the_linear_keeps_its_bytes_with_pdl(k, n, bits):
    g = torch.Generator().manual_seed(k + n + bits)
    tr = _trellis(k, n, 16 * bits, g).cpu()
    lin = linear.Exl3Linear.from_tensors(tr, _scale(k, 1 / math.sqrt(k), g).cpu(), _scale(n, 1.0, g).cpu(), "mul1")
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 16, 128):
        x = torch.randn((m, k), generator=g).to(torch.bfloat16).cuda()
        got = []
        for pdl in (False, True, False):
            switches.use(switches.Switches(pdl=pdl, group_scan=False, down_fuse=False, gateup_fuse=False,
                                           group_scan_decode=False, linear_pair=False))
            got.append(lin(x).clone())
        torch.cuda.synchronize()
        assert _same(got[0].float(), got[1].float()) and _same(got[0].float(), got[2].float()), (k, n, bits, m)
