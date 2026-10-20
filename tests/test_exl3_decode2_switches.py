"""TF_EXL3_GROUP_SCAN_DECODE and TF_EXL3_LINEAR_PAIR on the CPU: the flags' values, where the engines read them, that the
expert grouping and the linear hand their launches the switch, which windows take two row tiles, and (compiled for
sm_121 without a GPU, when nvcc is present) that the kernel instances fit without spilling. GROUP_SCAN_DECODE takes
group_scan_kernel (TF_EXL3_GROUP_SCAN's, prompt windows only) on decode windows.
tests/cuda/test_exl3_decode2_kernels.py compares the bytes on the GPU."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import experts, linear, switches

ENVS = {switches.GROUP_SCAN_DECODE_ENV: "group_scan_decode", switches.LINEAR_PAIR_ENV: "linear_pair"}


@pytest.fixture(autouse=True)
def restore():
    before = switches.current()
    yield
    switches.use(before)


@pytest.fixture
def clean_env(monkeypatch):
    for env in switches.ENVS:
        monkeypatch.delenv(env, raising=False)
    return monkeypatch


@pytest.mark.parametrize("env", list(ENVS))
@pytest.mark.parametrize("value, on", [(None, True), ("", True), ("1", True), ("on", True), ("YES", True),
                                       ("0", False), ("off", False), (" no ", False)])
def test_each_flag_is_on_unless_turned_off(clean_env, env, value, on):
    if value is not None:
        clean_env.setenv(env, value)
    got = switches.read()
    assert getattr(got, ENVS[env]) is on
    other = next(name for e, name in ENVS.items() if e != env)
    assert getattr(got, other) is True                                    # on since 20 Oct's GPU A/B
    assert (got.expert_decode, got.pdl, got.expert_items) == (False, False, True)     # the older switches' defaults
    assert (got.group_scan, got.down_fuse, got.gateup_fuse) == (True, True, True)


@pytest.mark.parametrize("env", list(ENVS))
@pytest.mark.parametrize("value", ["2", "scan", "o", "1.0"])
def test_a_flag_typo_stops_both_engines_before_any_read(clean_env, tmp_path, env, value):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    clean_env.setenv(env, value)
    with pytest.raises(ValueError, match=env):
        switches.read()
    with pytest.raises(ValueError, match=env):
        FlashNextEngine(tmp_path / "missing")                      # no checkpoint there: the flag fails first
    with pytest.raises(ValueError, match=env):
        GlmEngine(tmp_path / "missing", rank=0, master="", port=0, world=1)


@pytest.mark.parametrize("scan, pair", [(True, False), (False, True), (True, True)])
def test_both_engines_install_what_they_read(clean_env, tmp_path, scan, pair):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    clean_env.setenv(switches.GROUP_SCAN_DECODE_ENV, "1" if scan else "0")
    clean_env.setenv(switches.LINEAR_PAIR_ENV, "1" if pair else "0")
    want = switches.Switches(group_scan=True, down_fuse=True, gateup_fuse=True, group_scan_decode=scan,
                             linear_pair=pair)
    for build in (lambda: FlashNextEngine(tmp_path / "missing"),
                  lambda: GlmEngine(tmp_path / "missing", rank=0, master="", port=0, world=1)):
        switches.use(switches.Switches(group_scan=False, down_fuse=False, gateup_fuse=False, group_scan_decode=False,
                                       linear_pair=False))
        # fails at the checkpoint (or, without a GPU, at the device), after the switches
        with pytest.raises(Exception):  # noqa: B017
            build()
        assert switches.current() == want


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


@pytest.mark.parametrize("scan", [False, True])
@pytest.mark.parametrize("prompt_scan", [False, True])
@pytest.mark.parametrize("rows, items", [(1, False), (7, False), (28, False), (64, False), (65, True), (128, True)])
def test_decode_windows_take_the_scan_on_their_own_switch(monkeypatch, scan, prompt_scan, rows, items):
    fake = FakeExt()
    monkeypatch.setattr(experts, "_ext", lambda: fake)
    switches.use(switches.Switches(group_scan=prompt_scan, group_scan_decode=scan, down_fuse=False, gateup_fuse=False,
                                   linear_pair=False))
    ex = _layer()
    s = experts.Scratch(ex, 128, 4, device="cpu")
    x = torch.zeros((rows, ex.dims), dtype=torch.bfloat16)
    pick = torch.zeros((rows, 4), dtype=torch.int32)
    experts.routed(x, pick, None, ex, s, None, rows)
    group = [c for c in fake.calls if c[0] == "group"]
    want = prompt_scan if items else scan                         # a prompt window keeps TF_EXL3_GROUP_SCAN's
    assert len(group) == 1 and group[0][2].get("scan", False) is want
    assert ("scan" in group[0][2]) is want                        # off: the launch of before the switch
    assert ("items" in group[0][2]) is items                      # the item list stays a prompt window's
    switches.use(switches.Switches(group_scan=prompt_scan, group_scan_decode=not scan, down_fuse=False,
                                   gateup_fuse=False, linear_pair=False))
    again = FakeExt()
    monkeypatch.setattr(experts, "_ext", lambda: again)
    experts.routed(x, pick, None, ex, s, None, rows)
    assert _launches(again.calls) == _launches(fake.calls)       # the same launches, the switch aside


def _launches(calls):
    """Each launch's name and arguments, tensors by (address, shape), the scan switch left out."""

    def plain(v):
        return (v.data_ptr(), tuple(v.shape)) if isinstance(v, torch.Tensor) else v

    return [(n, [plain(v) for v in a], {k: plain(v) for k, v in kw.items() if k != "scan"}) for n, a, kw in calls]


def test_no_grouping_launch_without_group():
    fake = FakeExt()
    switches.use(switches.Switches(group_scan_decode=True, group_scan=False, down_fuse=False, gateup_fuse=False,
                                   linear_pair=False))
    ex = _layer()
    s = experts.Scratch(ex, 8, 4, device="cpu")
    from unittest import mock
    with mock.patch.object(experts, "_ext", lambda: fake):
        experts.routed(torch.zeros((2, ex.dims), dtype=torch.bfloat16), torch.zeros((2, 4), dtype=torch.int32), None,
                       ex, s, None, 2, group=False)
    assert "group" not in [c[0] for c in fake.calls]


@pytest.mark.parametrize("m, pair, want", [(1, True, 1), (7, True, 1), (16, True, 1), (17, True, 2), (28, True, 2),
                                           (32, True, 2), (33, True, 2), (128, True, 2), (17, False, 1),
                                           (28, False, 1), (128, False, 1)])
def test_only_windows_past_one_row_tile_take_two(m, pair, want):
    assert linear.row_tiles(m, pair) == want


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("m", [1, 7, 16, 17, 28, 128])
def test_the_linear_hands_its_launch_the_row_tiles(monkeypatch, pair, m):
    fake = FakeExt()
    monkeypatch.setattr(linear, "_ext", lambda: fake)
    switches.use(switches.Switches(linear_pair=pair, group_scan=False, down_fuse=False, gateup_fuse=False,
                                   group_scan_decode=False))
    k, n = 256, 128
    lin = linear.Exl3Linear(torch.zeros((n // 128, k // 16, 8, 24), dtype=torch.int32), torch.zeros(k).half(),
                            torch.zeros(n).half(), None, 3.0, "mul1", k, n)
    lin._counters = torch.zeros(8, dtype=torch.int32)
    lin(torch.zeros((m, k), dtype=torch.float16), xh=torch.zeros((m, k), dtype=torch.float16),
        z=torch.zeros(16 * m * n))
    assert [c[0] for c in fake.calls] == ["rot_in", "linear"]
    _, args, kw = fake.calls[1]
    assert kw == {"RT": 2 if pair and m > 16 else 1}
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
