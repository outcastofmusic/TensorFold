"""The EXL3 tests' fakes: a recording extension, a small routed layer, ptxas's verdict on sm_121 instances, and
``routed_with``, the launches ``experts.routed`` made before each of its shape-picked kernels (the GPU byte tests'
reference)."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from tensorfold.cuda.exl3 import experts, linear, switches


class FakeExt:
    """Records each launch's name and its trailing switches (keyword arguments in ``kwargs``, by launch)."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.kwargs: list[dict] = []

    def __getattr__(self, name):
        def call(*args, **kw):
            self.calls.append((name, args))
            self.kwargs.append(kw)
        return call


def _layer(D=256, I=128, E=8):
    t = torch.zeros(E, dtype=torch.int64)
    k2 = torch.full((E,), 6, dtype=torch.int32)
    h = torch.zeros((E, D), dtype=torch.float16)
    return experts.Exl3RoutedExperts(t, t, t, k2, k2, k2, h, h, h[:, :I], h[:, :I], h[:, :I], h, E, D, I,
                                     experts.CB_MUL1, (6, 6), (6, 6), t)


HERE = Path(experts.__file__).parent
NVCC = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)


def _ptxas(kernels: list[str]) -> dict[str, tuple[int, int, bool]]:
    """{instance: (registers, shared bytes, spills)} of the named grouped_kernel instances, built for sm_121."""

    from torch.utils import cpp_extension

    # grouped_kernel's operands, then its item list, the list's count and the epilogue
    src = "#include \"experts_grouped.cuh\"\nnamespace tf_exl3x {\n" + "".join(
        f"template __global__ void {k}(const half*, const half*, const int64_t*, const int64_t*, const int*, "
        "const int*, const int*, const int*, const int*, float*, int, int, int, int, int, int, uint32_t, "
        "const int*, const int*, Epi);\n"
        for k in kernels) + "}\n"
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / "inst.cu"
        cu.write_text(src)
        cmd = [NVCC, "-O3", "-std=c++17", "-gencode=arch=compute_121,code=sm_121", "-Xptxas", "-v", "-c", str(cu),
               "-o", str(Path(tmp) / "inst.o"), f"-I{HERE}"] + [f"-I{p}" for p in cpp_extension.include_paths()]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    got, name = {}, None
    for line in out.stderr.splitlines():
        m = re.search(r"Function properties for (\S+)", line)
        if m:
            name = m.group(1)
            continue
        if name and "spill" in line:
            spills = not re.search(r"0 bytes spill stores, 0 bytes spill loads", line)
            got[name] = [0, 0, spills]
        m = re.search(r"Used (\d+) registers.*?(\d+) bytes smem", line)
        if name and m:
            got[name][0], got[name][1] = int(m.group(1)), int(m.group(2))
            name = None
    return {k: tuple(v) for k, v in got.items()}


@dataclass(frozen=True)
class Launches:
    """The kernels ``experts.routed`` picks from the shape, each of which a TF_EXL3_* switch turned off until 25 Oct.
    The defaults are routed's own choices; a field set False takes that switch's old launch where routed would not."""

    expert_items: bool = True        # the listed layout on prompt windows
    group_scan: bool = True          # group_scan_kernel on prompt windows
    group_scan_decode: bool = True   # and on decode windows
    gateup_fuse: bool = True         # the gate/up epilogue inside grouped_kernel
    down_fuse: bool = True           # the down epilogue inside grouped_kernel
    expert_staged: bool = True       # staged k steps
    rot_fuse: bool = True            # the staged gate/up launch rotates x itself


OLD = Launches(*(False,) * 7)        # the launches before any of them


def routed_with(k: Launches, x, pick, wts, ex, s, out, R, limit=math.inf, act_mode=experts.ACT_F32, group=True,
                prompt=None, y=None):
    """``experts.routed`` with its kernel choices narrowed to ``k``: ``routed_with(Launches(), ...)`` makes routed's
    launches (tests/test_exl3_prefill_switches.py checks it), ``OLD`` the launches before the switches."""

    ext = experts._ext()
    D, I, E = ex.dims, ex.width, ex.count
    slots = s.slots
    P = R * slots
    ids, members = s.window(R)
    pdl = switches.current().pdl
    many = R > experts.PROMPT_ROWS if prompt is None else bool(prompt)
    gu, dn = experts.plan(s.cfg_gu, s.cfg_d, many)
    tile = experts.listed_tile(gu, dn) if group and k.expert_items and many else 0
    listed = {"items": s.items(R, tile), "icount": s.icount} if tile else {}
    if group:
        scan = {"scan": True} if (k.group_scan if many else k.group_scan_decode) and experts.scan_fits(E, P) else {}
        ext.group(pick, ids, s.count, members, R, slots, E, pdl, **listed, **({"tile": tile} if tile else {}),
                  **scan)
    nt, w, sk, pf, rt = gu
    staged = {"staged": True} if k.expert_staged and many and experts.stageable(gu, ex.k2_gu, ex) else {}
    rot = {"rot_x": x, "rot_suh0": ex.suh_g, "rot_suh1": ex.suh_u, "rot_arrivals": s.rot_arrivals} \
        if staged and listed and k.rot_fuse and experts.rot_fits(x) else {}
    if not rot:
        ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E, pdl)
    fuse_gu = k.gateup_fuse and bool(listed) and experts.fusable(gu, ex.k2_gu)
    fused = {"epi": experts.EPI_GATEUP, "s0": ex.svh_g, "s1": ex.svh_u, "s2": ex.suh_d, "out": s.xd,
             "arrivals": s.arrivals, "limit": float(limit), "act_mode": act_mode} if fuse_gu else {}
    ext.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2, D, I,
                P, sk, slots, ex.cb, nt, w, pf, ex.k2_gu[0], ex.k2_gu[1], rt, ex.window, pdl, **listed,
                **fused, **staged, **rot)
    if not fuse_gu:
        ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, 1 if staged else sk, slots, E,
                            float(limit), act_mode, pdl)
    nt, w, sk, pf, rt = dn
    into = s.y if y is None else y
    fuse_dn = k.down_fuse and bool(listed) and wts is None and sk == 1 and experts.fusable(dn, ex.k2_d)
    fused = {"epi": experts.EPI_DOWN, "s0": ex.svh_d, "out": into} if fuse_dn else {}
    staged = {"staged": True} if k.expert_staged and many and experts.stageable(dn, ex.k2_d, ex) else {}
    ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1, I,
                D, P, sk, slots, ex.cb, nt, w, pf, ex.k2_d[0], ex.k2_d[1], rt, ex.window, pdl, **listed,
                **fused, **staged)
    zsk = 1 if staged else sk
    if wts is None:
        if not fuse_dn:
            ext.down_epilogue(s.z, pick, ex.svh_d, into, R, P, D, zsk, slots, E, pdl)
        return s.y[:P] if y is None else y
    if out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    ext.down_combine(s.z, pick, ex.svh_d, s.y, wts, out, R, P, D, zsk, slots, E, pdl)
    return out


def linear_one_tile(lin, x, out=None):
    """``Exl3Linear.__call__`` with one 16-row tile a pass at every row count: the launch before two row tiles."""

    x = x.contiguous()
    m = x.shape[0]
    out = torch.empty((m, lin.n), dtype=x.dtype, device=x.device) if out is None else out
    sk, wk = lin.split
    xh = torch.empty((m, lin.k), dtype=torch.float16, device=x.device)
    z = torch.empty((sk * m * lin.n,), dtype=torch.float32, device=x.device) if sk > 1 else None
    pdl = switches.current().pdl
    ext = linear._ext()
    ext.rot_in(x, lin.suh, xh, pdl)
    ext.linear(xh, lin.words, *lin.strides, lin.svh, lin.bias, out, z, lin.counters, lin.k2,
               linear.CODEBOOK_IDS[lin.codebook], sk, wk, pdl, RT=1)
    return out


def route(k: Launches | None, *args, **kw):
    """``experts.routed`` (``k`` None: the served launches) or ``routed_with(k)``."""

    return experts.routed(*args, **kw) if k is None else routed_with(k, *args, **kw)
