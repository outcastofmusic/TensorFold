"""Decode windows' grouping and the linear's row tiles on the CPU: that every window groups on group_scan_kernel where
it fits, which windows of the linear take two row tiles, and (compiled for sm_121 without a GPU, when nvcc is
present) that the kernel instances fit without spilling. tests/cuda/test_exl3_decode2_kernels.py compares the bytes
on the GPU."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import experts, linear


class FakeExt:
    """Records each launch's name, positional arguments and keyword arguments."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        def call(*args, **kw):
            self.calls.append((name, args, kw))
        return call


def _layer(D=256, I=128, E=8):
    t = torch.zeros(E, dtype=torch.int64)
    k2 = torch.full((E,), 6, dtype=torch.int32)
    h = torch.zeros((E, D), dtype=torch.float16)
    return experts.Exl3RoutedExperts(t, t, t, k2, k2, k2, h, h, h[:, :I], h[:, :I], h[:, :I], h, E, D, I,
                                     experts.CB_MUL1, (6, 6), (6, 6), t)


@pytest.mark.parametrize("rows, items", [(1, False), (7, False), (28, False), (64, False), (65, True), (128, True)])
def test_every_window_groups_on_the_scan(monkeypatch, rows, items):
    fake = FakeExt()
    monkeypatch.setattr(experts, "_ext", lambda: fake)
    ex = _layer()
    s = experts.Scratch(ex, 128, 4, device="cpu")
    experts.routed(torch.zeros((rows, ex.dims), dtype=torch.bfloat16), torch.zeros((rows, 4), dtype=torch.int32), None,
                   ex, s, None, rows)
    group = [c for c in fake.calls if c[0] == "group"]
    assert len(group) == 1 and group[0][2]["scan"] is True
    assert ("items" in group[0][2]) is items                      # the item list stays a prompt window's


def test_no_grouping_launch_without_group():
    fake = FakeExt()
    ex = _layer()
    s = experts.Scratch(ex, 8, 4, device="cpu")
    from unittest import mock
    with mock.patch.object(experts, "_ext", lambda: fake):
        experts.routed(torch.zeros((2, ex.dims), dtype=torch.bfloat16), torch.zeros((2, 4), dtype=torch.int32), None,
                       ex, s, None, 2, group=False)
    assert "group" not in [c[0] for c in fake.calls]


@pytest.mark.parametrize("m, want", [(1, 1), (7, 1), (16, 1), (17, 2), (28, 2), (32, 2), (33, 2), (128, 2)])
def test_only_windows_past_one_row_tile_take_two(m, want):
    assert linear.row_tiles(m) == want


@pytest.mark.parametrize("m", [1, 7, 16, 17, 28, 128])
def test_the_linear_hands_its_launch_the_row_tiles(monkeypatch, m):
    fake = FakeExt()
    monkeypatch.setattr(linear, "_ext", lambda: fake)
    k, n = 256, 128
    lin = linear.Exl3Linear(torch.zeros((n // 128, k // 16, 8, 24), dtype=torch.int32), torch.zeros(k).half(),
                            torch.zeros(n).half(), None, 3.0, "mul1", k, n)
    lin._counters = torch.zeros(8, dtype=torch.int32)
    lin(torch.zeros((m, k), dtype=torch.float16), xh=torch.zeros((m, k), dtype=torch.float16),
        z=torch.zeros(16 * m * n))
    assert [c[0] for c in fake.calls] == ["rot_in", "linear"]
    _, args, kw = fake.calls[1]
    assert kw == {"RT": 2 if m > 16 else 1}
    assert args[-1] is False                                     # the PDL switch keeps its place


# ---------------------------------------------------------------------------------------- compiled for sm_121

HERE = Path(experts.__file__).parent
NVCC = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)


def _ptxas(source: str) -> dict[str, tuple[int, bool]]:
    """{mangled kernel: (registers, spills)} of every kernel in ``source`` (a file next to the kernels), for sm_121."""

    import sysconfig

    from torch.utils import cpp_extension

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [NVCC, "-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_121,code=sm_121",
               "-Xptxas", "-v", "-c", str(HERE / source), "-o", str(Path(tmp) / "out.o"), f"-I{HERE}",
               f"-I{sysconfig.get_paths()['include']}"] + [f"-I{p}" for p in cpp_extension.include_paths()]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    got, name, spills = {}, None, False
    for line in out.stderr.splitlines():
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m:
            name = m.group(1)
            continue
        if name and "spill" in line:
            spills = not re.search(r"0 bytes spill stores, 0 bytes spill loads", line)
        m = re.search(r"Used (\d+) registers", line)
        if name and m:
            got[name] = (int(m.group(1)), spills)
            name = None
    return got


@pytest.mark.skipif(NVCC is None, reason="nvcc builds the sm_121 instances")
def test_the_linear_pair_instances_the_plan_launches_do_not_spill():
    """Every width and codebook at 4 and 8 warps (``plan``'s), one and two row tiles: no spills. Two tiles hold 128
    more accumulators a lane, so Flash Next's 5-bit mul1 instance at 4 warps fits two blocks an SM where one tile fits
    three (166 registers): the price of reading each weight once for 32 rows."""

    got = _ptxas("linear.cu")
    insts = {tuple(int(v) for v in m.groups()): regs_sp
             for k, regs_sp in got.items() if (m := re.search(r"linear_kernelILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E", k))}
    assert len(insts) == 27 * 3 * 2                               # TF_EXL3_ALL x 3 warp counts x 2 row tilings
    for (k2, cb, wk, rt), (regs, spills) in insts.items():
        if wk in (4, 8):
            assert not spills, (k2, cb, wk, rt)
    assert insts[(10, 2, 4, 1)][0] <= 168 and insts[(10, 2, 4, 2)][0] <= 255
