"""TF_EXL3_ROT_FUSE on the CPU: the flag's values and where the engines read it, which launches ``routed`` hands the
switch (a staged gate/up launch on the item list, with rows of x on 16 bytes; everything else keeps rot_in), and
(compiled for sm_121 without a GPU, when nvcc is present) that the switch's instances do not spill, keep 2 blocks an
SM, and compile rot_in_kernel's multiplies, FMAs and roundings a value. tests/cuda/test_exl3_rot_fuse_kernels.py
compares the bytes on the GPU."""

from __future__ import annotations

import dataclasses
import subprocess
import sysconfig

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import experts, switches
from tests.test_exl3_decode_switches import NVCC, FakeExt, _layer, _ptxas
from tests.test_exl3_expert_staged import _sass_values
from tests.test_exl3_prefill_switches import HERE, _route

ROT = dataclasses.replace(switches.Switches(), rot_fuse=True)


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


# -- the flag -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value, on", [(None, True), ("", True), ("1", True), ("on", True), ("YES", True),
                                       ("0", False), ("off", False), (" no ", False)])
def test_the_flag_is_on_unless_turned_off(clean_env, value, on):
    if value is not None:
        clean_env.setenv(switches.ROT_FUSE_ENV, value)
    got = switches.read()
    assert got.rot_fuse is on
    assert dataclasses.replace(got, rot_fuse=True) == switches.Switches()       # the others keep their defaults


@pytest.mark.parametrize("value", ["2", "rot", "o", "1.0"])
def test_a_flag_typo_stops_both_engines_before_any_read(clean_env, tmp_path, value):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    clean_env.setenv(switches.ROT_FUSE_ENV, value)
    with pytest.raises(ValueError, match=switches.ROT_FUSE_ENV):
        switches.read()
    with pytest.raises(ValueError, match=switches.ROT_FUSE_ENV):
        FlashNextEngine(tmp_path / "missing")
    with pytest.raises(ValueError, match=switches.ROT_FUSE_ENV):
        GlmEngine(tmp_path / "missing", rank=0, master="", port=0, world=1)


def test_both_engines_install_what_they_read(clean_env, tmp_path):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    clean_env.setenv(switches.ROT_FUSE_ENV, "1")
    for build in (lambda: FlashNextEngine(tmp_path / "missing"),
                  lambda: GlmEngine(tmp_path / "missing", rank=0, master="", port=0, world=1)):
        switches.use(switches.Switches())
        with pytest.raises(Exception):  # noqa: B017  # the checkpoint (or, without a GPU, the device) fails after
            build()
        assert switches.current() == ROT


# -- what routed launches -------------------------------------------------------------------------------------------
def _names(fake):
    return [n for n, _ in fake.calls]


def _grouped(fake):
    return [kw for (n, _), kw in zip(fake.calls, fake.kwargs) if n == "grouped"]


@pytest.mark.parametrize("rows", [65, 128, 1024])
def test_a_prompt_window_rotates_inside_the_staged_gate_up_launch(monkeypatch, rows):
    fake, s, ex, _ = _route(monkeypatch, rows, ROT)
    assert _names(fake) == ["group", "grouped", "grouped"]                  # no rot_in
    gu, dn = _grouped(fake)
    x = fake.calls[1][1][0]                                                 # X0 is the scratch's xg, as before
    assert x is s.xg
    assert gu["staged"] is True and gu["epi"] == experts.EPI_GATEUP and "items" in gu
    assert gu["rot_suh0"] is ex.suh_g and gu["rot_suh1"] is ex.suh_u and gu["rot_arrivals"] is s.rot_arrivals
    assert gu["rot_x"].shape == (rows, ex.dims)
    assert not any(k.startswith("rot_") for k in dn)


def test_the_unfused_epilogue_keeps_the_switch_on_the_staged_launch(monkeypatch):
    sw = dataclasses.replace(ROT, gateup_fuse=False, down_fuse=False)
    fake, _, _, _ = _route(monkeypatch, 1024, sw)
    assert _names(fake) == ["group", "grouped", "gateup_epilogue", "grouped", "down_epilogue"]
    assert "rot_x" in _grouped(fake)[0]


def test_the_combined_rows_keep_the_switch(monkeypatch):
    fake, _, _, _ = _route(monkeypatch, 1024, ROT, wts=True)
    assert _names(fake) == ["group", "grouped", "grouped", "down_combine"]


@pytest.mark.parametrize("rows", [1, 8, 16, 17, 28, 64])
def test_decode_windows_keep_rot_in(monkeypatch, rows):
    fake, _, _, _ = _route(monkeypatch, rows, ROT)
    assert "rot_in" in _names(fake)
    assert all(not any(k.startswith("rot_") for k in kw) for kw in _grouped(fake))


@pytest.mark.parametrize("sw", [dataclasses.replace(ROT, expert_staged=False),
                                dataclasses.replace(ROT, expert_items=False),
                                switches.Switches(rot_fuse=False)], ids=["staged off", "spanned", "switch off"])
def test_without_a_staged_listed_launch_or_the_switch_rot_in_stays(monkeypatch, sw):
    fake, _, _, _ = _route(monkeypatch, 1024, sw)
    assert _names(fake)[:2] == ["group", "rot_in"]
    assert all(not any(k.startswith("rot_") for k in kw) for kw in fake.kwargs)


def test_layers_the_staged_path_leaves_keep_rot_in(monkeypatch):
    wide = _layer()
    wide.k2_gu, wide.k2_d = (6, 12), (12, 12)
    off16 = dataclasses.replace(_layer(), aligned=False)
    for ex in (wide, off16):
        fake, _, _, _ = _route(monkeypatch, 1024, ROT, ex=ex)
        assert "rot_in" in _names(fake)


def _x_route(monkeypatch, x):
    fake = FakeExt()
    monkeypatch.setattr(experts, "_ext", lambda: fake)
    switches.use(ROT)
    ex = _layer()
    s = experts.Scratch(ex, 1024, 4, device="cpu")
    experts.routed(x, torch.zeros((x.shape[0], 4), dtype=torch.int32), None, ex, s, None, x.shape[0])
    return fake


@pytest.mark.parametrize("make, fused", [
    (lambda: torch.zeros((128, 256), dtype=torch.bfloat16), True),
    (lambda: torch.zeros((128, 256), dtype=torch.float16), True),
    (lambda: torch.zeros((128, 272), dtype=torch.bfloat16)[:, 8:264], True),     # a stride, rows on 16 bytes
    (lambda: torch.zeros((128, 272), dtype=torch.bfloat16)[:, 1:257], False),    # rows 2 bytes off 16
    (lambda: torch.zeros((128, 260), dtype=torch.bfloat16)[:, :256], False),     # a stride of 520 bytes
    (lambda: torch.zeros((128, 256), dtype=torch.float32), False),
], ids=["bf16", "fp16", "stride on 16", "off 16", "stride off 16", "fp32"])
def test_rows_the_16_byte_loads_cannot_take_keep_rot_in(monkeypatch, make, fused):
    x = make()
    assert experts.rot_fits(x) is fused
    names = _names(_x_route(monkeypatch, x))
    assert ("rot_in" not in names) is fused


def test_the_scratch_counts_an_item_and_matrix_from_zero():
    s = experts.Scratch(_layer(), 1024, 4, device="cpu")
    assert s.rot_arrivals.dtype == torch.int32 and s.rot_arrivals.numel() == 2 * s.items_buf.numel()
    assert int(s.rot_arrivals.abs().sum()) == 0


# -- compiled for sm_121 --------------------------------------------------------------------------------------------
SETTINGS = [(cb, epi, lo, hi) for cb in (0, 1, 2) for epi in (0, 1) for lo, hi in ((2, 10), (118, 118), (8, 8))]


@pytest.mark.skipif(NVCC is None, reason="nvcc builds the sm_121 instances")
def test_the_switch_instances_spill_only_where_their_twins_do_and_keep_two_blocks_an_sm():
    """Every ROT instance against its twin without the switch. The 3inst and mcg twins at rates 2..10 spill on
    fork/dev 1d2deb4e already (255 registers); Flash Next's (mul1, 2..10) and GLM-5.3 2.25 bits' (mcg, 118) spill in
    neither."""

    got = _ptxas([f"grouped_kernel<{cb}, 8, 4, 1, {lo}, {hi}, 2, {epi}, 8, {rot}>" for cb, epi, lo, hi in SETTINGS
                  for rot in ("false", "true")])

    def find(cb, epi, lo, hi, rot):
        tag = f"grouped_kernelILi{cb}ELi8ELi4ELi1ELi{lo}ELi{hi}ELi2ELi{epi}ELi8ELb{rot}E"
        (v,) = [v for k, v in got.items() if tag in k]
        return v

    for cb, epi, lo, hi in SETTINGS:
        regs, smem, spills = find(cb, epi, lo, hi, 1)
        assert spills <= find(cb, epi, lo, hi, 0)[2], (cb, epi, lo)
        if (cb, lo) in ((2, 2), (1, 118)):
            assert not spills, (cb, epi, lo)
        assert smem <= 48 * 1024 and 2 * (smem + 1024) <= 102400, (cb, epi, lo, smem)
        assert 65536 // (-(-regs // 8) * 8 * 128) >= 2, (cb, epi, lo, regs)


@pytest.mark.skipif(NVCC is None, reason="nvcc builds the sm_121 instances")
@pytest.mark.parametrize("epi", [0, 1])
def test_the_rotation_compiles_rot_ins_multiplies_fmas_and_roundings_a_value(tmp_path, epi):
    """Each ROT instance, less its twin without the switch, holds rot_in_kernel's FMULs, FFMAs and fp16 roundings
    (F2FP values) times the values a lane rotates in one pass of the unrolled loop over rot_in's 4: the same
    contraction, fma(x0, s0, x1 * s1), and no multiply contracted into an add rot_in rounds apart."""

    from torch.utils import cpp_extension

    sig = ("(const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, "
           "const int*, const int*, float*, int, int, int, int, int, int, uint32_t, const int*, const int*, Epi);\n")
    src = tmp_path / "inst.cu"
    src.write_text((HERE / "experts.cu").read_text() + "\nnamespace tf_exl3x {\n" + "".join(
        f"template __global__ void grouped_kernel<2, 8, 4, 1, 2, 10, 2, {epi}, 8, {rot}>" + sig
        for rot in ("false", "true")) + "}\n")
    obj = tmp_path / "inst.o"
    cmd = [NVCC, "-O3", "-std=c++17", "-gencode=arch=compute_121,code=sm_121", "-c", str(src), "-o", str(obj),
           f"-I{HERE}", f"-I{sysconfig.get_paths()['include']}"] + [f"-I{p}" for p in cpp_extension.include_paths()]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    ops = _sass_values(obj)

    def one(*tags):
        found = [v for k, v in ops.items() if all(t in k for t in tags)]
        assert len(found) == 1, tags
        return found[0]

    rot_in = one("rot_in_kernelI13__nv_bfloat16")
    twin = one(f"ELi2ELi10ELi2ELi{epi}ELi8ELb0E")
    fused = one(f"ELi2ELi10ELi2ELi{epi}ELi8ELb1E")
    per_pass = 4 * 4 * 8 // 4                     # ROT_UNITS units of 4 rows, 8 values a lane, over rot_in's 4
    added = fused - twin
    for op in ("FMUL", "FFMA", "F2FP.F16.F32.PACK_AB"):
        assert added[op] == per_pass * rot_in[op], (op, added[op], rot_in[op])
    # the rest: fp16 to fp32 conversions (suh, fp16 rows) and the integer division of a unit's index
    assert set(added) <= {"FMUL", "FFMA", "F2FP.F16.F32.PACK_AB", "HADD2.F32", "MUFU.RCP"}, added
