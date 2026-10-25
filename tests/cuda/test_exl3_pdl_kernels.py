"""TF_EXL3_PDL on the GPU: the same bytes with the switch on or off.

- Routed experts at Flash Next EXL3's shapes (2560 <-> 640, 3-bit mul1 experts beside a 5-bit shared expert, 11
  slots) and GLM-5.3's (4096 <-> 2048, mcg at 2.25 bits and at 2.5 bits, 14-bit window, 8 slots): per-slot outputs
  and the combined rows of decode windows of 1-8 and 16 rows, and the prompt windows (17, 64, 65, 128 rows), equal
  bit for bit with PDL on and off, eager and replayed from CUDA graphs.
- The EXL3 linear at Flash Next's dense shapes and GLM-5.3-sized ones, 1-8, 16 and 128 rows, PDL on and off.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

from tensorfold.cuda.exl3 import experts, linear, switches  # noqa: E402
from tests.cuda.exl3_layers import LAYERS, LINEAR, _inputs, _layer, _same, _scale, _trellis

OLD = dict(group_scan=False, down_fuse=False, gateup_fuse=False, group_scan_decode=False, linear_pair=False)
OFF = switches.Switches(pdl=False, **OLD)
SETTINGS = [OFF, switches.Switches(pdl=True, **OLD)]
DECODE_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 16)
PROMPT_ROWS = (17, 64, 65, 128)


@pytest.fixture(autouse=True)
def restore():
    before = switches.current()
    yield
    switches.use(before)


def _run(name, R, x, sel, w, scratch):
    ex, slots, E, act = _layer(name)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    y = experts.routed(x, sel, None, ex, scratch, None, R, limit, act).clone()
    out = experts.routed(x, sel, w, ex, scratch, None, R, limit, act).clone()
    return y, out


@pytest.mark.parametrize("name", list(LAYERS))
@pytest.mark.parametrize("R", DECODE_ROWS + PROMPT_ROWS)
def test_routed_experts_keep_their_bytes_with_pdl(name, R):
    ex, slots, E, _ = _layer(name)
    scratch = experts.Scratch(ex, max(PROMPT_ROWS), slots)
    x, sel, w = _inputs(name, R, 1000 + R)
    got = []
    for sw in SETTINGS:
        switches.use(sw)
        got.append(_run(name, R, x, sel, w, scratch))
    torch.cuda.synchronize()
    assert torch.isfinite(got[0][1]).all()
    (y0, out0), (y1, out1) = got
    assert _same(y1, y0) and _same(out1, out0), (name, R)


@pytest.mark.parametrize("name", list(LAYERS))
def test_graphs_captured_with_pdl_on_replay_the_old_bytes(name):
    ex, slots, E, act = _layer(name)
    rows = (1, 2, 4, 7, 8)
    scratch = experts.Scratch(ex, max(rows), slots)
    limit = 7.0 if act == experts.ACT_BF16 else math.inf
    bufs, graphs = {}, {}
    switches.use(SETTINGS[1])
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


@pytest.mark.parametrize("k, n, bits", LINEAR)
def test_the_linear_keeps_its_bytes_with_pdl(k, n, bits):
    g = torch.Generator().manual_seed(k + n + bits)
    tr = _trellis(k, n, 16 * bits, g).cpu()
    lin = linear.Exl3Linear.from_tensors(tr, _scale(k, 1 / math.sqrt(k), g).cpu(), _scale(n, 1.0, g).cpu(), "mul1")
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 16, 128):
        x = torch.randn((m, k), generator=g).to(torch.bfloat16).cuda()
        got = []
        for pdl in (False, True, False):
            switches.use(switches.Switches(pdl=pdl, **OLD))
            got.append(lin(x).clone())
        torch.cuda.synchronize()
        assert _same(got[0].float(), got[1].float()) and _same(got[0].float(), got[2].float()), (k, n, bits, m)
