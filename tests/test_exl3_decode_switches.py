"""TF_EXL3_EXPERT_DECODE and TF_EXL3_PDL on the CPU: the flags' values, where the engines read them, which expert
kernel and tile each window takes, that every launch carries the PDL switch, and (compiled for sm_121 without a GPU,
when nvcc is present) that decode_kernel fits more blocks an SM than grouped_kernel without spilling.
tests/cuda/test_exl3_decode_kernels.py compares the bytes on the GPU."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import experts, linear, switches  # noqa: E402

ENVS = (switches.EXPERT_DECODE_ENV, switches.PDL_ENV)
FN_GATEUP, FN_DOWN = experts.GLM_GATEUP, experts.GLM_DOWN       # Flash Next and GLM-5.3 both take GLM's settings
# (K, N) of the gate/up and down projections: Flash Next EXL3 (2560 <-> 640) and GLM-5.3 (4096 <-> 2048)
MODELS = {"flash_next": (2560, 640), "glm53": (4096, 2048)}


@pytest.fixture(autouse=True)
def restore():
    before = switches.current()
    yield
    switches.use(before)


@pytest.mark.parametrize("env", ENVS)
@pytest.mark.parametrize("value, on", [(None, True), ("", True), ("1", True), ("on", True), ("0", False),
                                       ("off", False), ("false", False), (" NO ", False)])
def test_each_flag_is_on_unless_turned_off(monkeypatch, env, value, on):
    for e in ENVS:
        monkeypatch.delenv(e, raising=False)
    if value is not None:
        monkeypatch.setenv(env, value)
    got = switches.read()
    assert (got.expert_decode if env == switches.EXPERT_DECODE_ENV else got.pdl) is on
    assert (got.pdl if env == switches.EXPERT_DECODE_ENV else got.expert_decode) is True


@pytest.mark.parametrize("env", ENVS)
@pytest.mark.parametrize("value", ["2", "decode", "o", "1.0"])
def test_a_flag_typo_stops_both_engines_before_any_read(monkeypatch, tmp_path, env, value):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError, match=env):
        switches.read()
    with pytest.raises(ValueError, match=env):
        FlashNextEngine(tmp_path / "missing")                      # no checkpoint there: the flag fails first
    with pytest.raises(ValueError, match=env):
        GlmEngine(tmp_path / "missing", rank=0, master="", port=0, world=1)


def test_an_engine_installs_what_it_reads(monkeypatch, tmp_path):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setenv(switches.EXPERT_DECODE_ENV, "0")
    monkeypatch.setenv(switches.PDL_ENV, "1")
    switches.use(switches.Switches())
    with pytest.raises(Exception):
        FlashNextEngine(tmp_path / "missing")                      # fails at the checkpoint, after the switches
    assert switches.current() == switches.Switches(expert_decode=False, pdl=True)


def test_use_returns_the_previous_setting():
    off = switches.Switches(False, False)
    first = switches.use(off)
    assert switches.current() is off
    assert switches.use(first) is off


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("rows", [1, 2, 3, 7, 8, 16])
def test_a_decode_window_takes_decode_kernel_with_the_same_warps_and_splits(model, rows):
    D, I = MODELS[model]
    gu, dn = experts.plan(FN_GATEUP, FN_DOWN, I, D, rows, False, True)
    assert gu == ((experts.DECODE_NT, 4, FN_GATEUP[2], experts.DECODE_PF, 1), True)
    assert dn == ((experts.DECODE_NT, 4, FN_DOWN[2], experts.DECODE_PF, 1), True)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("rows, many, decode", [(17, False, True), (64, False, True), (1, False, False),
                                                (8, False, False), (1, True, True), (65, True, True)])
def test_wider_windows_prompts_and_the_switch_off_keep_grouped_kernel(model, rows, many, decode):
    D, I = MODELS[model]
    gu, dn = experts.plan(FN_GATEUP, FN_DOWN, I, D, rows, many, decode)
    want_gu = experts.prompt_config(FN_GATEUP) if many else (*FN_GATEUP, 1)
    want_dn = experts.prompt_config(FN_DOWN) if many else (*FN_DOWN, 1)
    assert gu == (want_gu, False) and dn == (want_dn, False)


def test_a_width_its_n_tiles_do_not_divide_stays_on_grouped_kernel():
    assert experts.decode_config((8, 4, 2, 1), 640 + 32) is None
    assert experts.decode_config((4, 4, 1, 2), 96) is None
    assert experts.decode_config((8, 4, 4, 1), 640) == (experts.DECODE_NT, 4, 4, experts.DECODE_PF)


class FakeExt:
    """Records each launch's name and its trailing switches."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))
        return call


def _layer(D=256, I=128, E=8):
    t = torch.zeros(E, dtype=torch.int64)
    k2 = torch.full((E,), 6, dtype=torch.int32)
    h = torch.zeros((E, D), dtype=torch.float16)
    return experts.Exl3RoutedExperts(t, t, t, k2, k2, k2, h, h, h[:, :I], h[:, :I], h[:, :I], h, E, D, I,
                                     experts.CB_MUL1, (6, 6), (6, 6), t)


@pytest.mark.parametrize("decode, pdl", [(True, True), (False, True), (True, False), (False, False)])
@pytest.mark.parametrize("rows", [1, 8, 16, 17])
def test_routed_hands_each_launch_its_switches(monkeypatch, decode, pdl, rows):
    fake = FakeExt()
    monkeypatch.setattr(experts, "_ext", lambda: fake)
    switches.use(switches.Switches(expert_decode=decode, pdl=pdl))
    ex = _layer()
    s = experts.Scratch(ex, 32, 4, device="cpu")
    x = torch.zeros((rows, ex.dims), dtype=torch.bfloat16)
    pick = torch.zeros((rows, 4), dtype=torch.int32)
    experts.routed(x, pick, None, ex, s, None, rows)
    names = [n for n, _ in fake.calls]
    assert names == ["group", "rot_in", "grouped", "gateup_epilogue", "grouped", "down_epilogue"]
    assert all(args[-1] is pdl for _, args in fake.calls)
    for _, args in (c for c in fake.calls if c[0] == "grouped"):
        dec, nt, pf = args[-2], args[17], args[19]
        assert dec is (decode and rows <= experts.DECODE_ROWS)
        assert (nt, pf) == ((experts.DECODE_NT, experts.DECODE_PF) if dec else (8, 1))
    wts = torch.zeros((rows, 4), dtype=torch.float32)
    fake.calls.clear()
    experts.routed(x, pick, wts, ex, s, torch.zeros((rows, ex.dims)), rows)
    assert fake.calls[-1][0] == "down_combine" and fake.calls[-1][1][-1] is pdl


@pytest.mark.parametrize("pdl", [True, False])
def test_the_linear_hands_its_launches_the_pdl_switch(monkeypatch, pdl):
    fake = FakeExt()
    monkeypatch.setattr(linear, "_ext", lambda: fake)
    switches.use(switches.Switches(pdl=pdl))
    k, n = 256, 128
    lin = linear.Exl3Linear(torch.zeros((n // 128, k // 16, 8, 24), dtype=torch.int32), torch.zeros(k).half(),
                            torch.zeros(n).half(), None, 3.0, "mul1", k, n)
    lin._counters = torch.zeros(8, dtype=torch.int32)
    lin(torch.zeros((2, k), dtype=torch.float16), xh=torch.zeros((2, k), dtype=torch.float16),
        z=torch.zeros(16 * 2 * n))
    assert [c[0] for c in fake.calls] == ["rot_in", "linear"]
    assert all(args[-1] is pdl for _, args in fake.calls)


# ---------------------------------------------------------------------------------------- compiled for sm_121

HERE = Path(experts.__file__).parent
NVCC = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)


def _ptxas(kernels: list[str]) -> dict[str, tuple[int, int, bool]]:
    """{instance: (registers, shared bytes, spills)} of the named template instances, built for sm_121."""

    from torch.utils import cpp_extension

    src = "#include \"experts_grouped.cuh\"\nnamespace tf_exl3x {\n" + "".join(
        f"template __global__ void {k}(const half*, const half*, const int64_t*, const int64_t*, const int*, "
        "const int*, const int*, const int*, const int*, float*, int, int, int, int, int, int, uint32_t);\n"
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


@pytest.mark.skipif(NVCC is None, reason="nvcc builds the sm_121 instances")
def test_decode_kernel_fits_more_blocks_an_sm_than_grouped_kernel_without_spilling():
    """Flash Next 3.05 bpw's instance (mul1, rates 2..10: 3-bit experts, 5-bit shared expert): grouped_kernel at
    (8 n tiles, 1 k step) holds 3 blocks of 4 warps an SM on GB10 (64K registers, 100 KB shared, 1 KB of it reserved a block); decode_kernel at
    its default holds 4, with a third of the shared memory."""

    old = "grouped_kernel<2, 8, 4, 1, 2, 10, 1>"
    new = f"decode_kernel<2, {experts.DECODE_NT}, 4, {experts.DECODE_PF}, 2, 10>"
    got = _ptxas([old, new])
    (r_old, s_old, sp_old), (r_new, s_new, sp_new) = (next(v for k, v in got.items() if tag in k)
                                                      for tag in ("grouped_kernel", "decode_kernel"))

    def blocks(regs, smem):
        return min(65536 // (-(-regs // 8) * 8 * 128), 100 * 1024 // (smem + 1024))

    assert not sp_new
    assert blocks(r_old, s_old) == 3
    assert blocks(r_new, s_new) >= 4
    assert s_new * 2 < s_old
