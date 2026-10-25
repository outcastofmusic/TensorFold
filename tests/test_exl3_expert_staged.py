"""TF_EXL3_EXPERT_STAGED on the CPU: the flag's values and where the engines read it, which launches ``routed`` hands
the switch (prompt windows at the fused epilogues' setting and rates, on trellises that start on 16 bytes), the one
split the epilogue kernels then read, and (compiled for sm_121 without a GPU, when nvcc is present) that the staged
instances do not spill, keep 2 blocks an SM in static shared memory, and compile the fused epilogues' multiply, FMA,
exponent and rounding instructions in the epilogue kernels' counts. tests/cuda/test_exl3_staged_kernels.py compares
the bytes on the GPU."""

from __future__ import annotations

import dataclasses
import re
import subprocess
import sysconfig
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import experts, switches
from tests.exl3_fakes import NVCC, _layer, _ptxas
from tests.test_exl3_prefill_switches import FLOAT_OPS, HERE, _route

STAGED = dataclasses.replace(switches.Switches(), expert_staged=True, rot_fuse=False)      # rot_in's launches


@pytest.fixture(autouse=True)
def restore():
    from tensorfold.families.qwen4_exp.cuda import exl3_mm

    before, before_mm = switches.current(), exl3_mm.current()
    yield
    switches.use(before)
    exl3_mm.use(before_mm)


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
        clean_env.setenv(switches.EXPERT_STAGED_ENV, value)
    got = switches.read()
    assert got.expert_staged is on
    assert dataclasses.replace(got, expert_staged=True) == switches.Switches()      # the others keep their defaults


@pytest.mark.parametrize("value", ["2", "staged", "o", "1.0"])
def test_a_flag_typo_stops_both_engines_before_any_read(clean_env, tmp_path, value):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    clean_env.setenv(switches.EXPERT_STAGED_ENV, value)
    with pytest.raises(ValueError, match=switches.EXPERT_STAGED_ENV):
        switches.read()
    with pytest.raises(ValueError, match=switches.EXPERT_STAGED_ENV):
        FlashNextEngine(tmp_path / "missing")
    with pytest.raises(ValueError, match=switches.EXPERT_STAGED_ENV):
        GlmEngine(tmp_path / "missing", rank=0, master="", port=0, world=1)


def test_both_engines_install_what_they_read(clean_env, tmp_path):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    clean_env.setenv(switches.EXPERT_STAGED_ENV, "1")
    for build in (lambda: FlashNextEngine(tmp_path / "missing"),
                  lambda: GlmEngine(tmp_path / "missing", rank=0, master="", port=0, world=1)):
        switches.use(switches.Switches())
        with pytest.raises(Exception):  # noqa: B017  # the checkpoint (or, without a GPU, the device) fails after
            build()
        assert switches.current() == dataclasses.replace(switches.Switches(), expert_staged=True)


# -- what routed launches -------------------------------------------------------------------------------------------
def _grouped(fake):
    return [kw for (n, _), kw in zip(fake.calls, fake.kwargs) if n == "grouped"]


def _args(fake, name):
    return [a for n, a in fake.calls if n == name]


@pytest.mark.parametrize("rows", [65, 128, 1024])
def test_a_prompt_window_stages_both_projections(monkeypatch, rows):
    fake, _, _, _ = _route(monkeypatch, rows, STAGED)
    assert [n for n, _ in fake.calls] == ["group", "rot_in", "grouped", "grouped"]
    assert all(kw.get("staged") is True for kw in _grouped(fake))


@pytest.mark.parametrize("rows", [1, 8, 16, 17, 28, 64])
def test_decode_windows_keep_their_launches(monkeypatch, rows):
    fake, _, _, _ = _route(monkeypatch, rows, STAGED)
    assert all("staged" not in kw for kw in _grouped(fake))


def test_switched_off_no_launch_names_it(monkeypatch):
    fake, _, _, _ = _route(monkeypatch, 1024, switches.Switches(expert_staged=False))
    assert all("staged" not in kw for kw in fake.kwargs)


def test_the_epilogue_kernels_read_one_split_after_a_staged_launch(monkeypatch):
    """Without the fusions the gate/up epilogue kernel reads the staged program's one total a matrix (SK 1), where
    the old launch left its K splits; down has one split either way."""

    unfused = dataclasses.replace(switches.Switches(), gateup_fuse=False, down_fuse=False, expert_staged=False,
                                  rot_fuse=False)
    for sw, sk in ((unfused, 4), (dataclasses.replace(unfused, expert_staged=True), 1)):
        fake, s, _, _ = _route(monkeypatch, 1024, sw)
        assert [n for n, _ in fake.calls] == ["group", "rot_in", "grouped", "gateup_epilogue", "grouped",
                                              "down_epilogue"]
        assert s.cfg_gu[2] == 4                                    # the launch itself keeps the shape's 4 splits
        assert _grouped(fake)[0].get("staged", False) is (sk == 1)
        assert _args(fake, "gateup_epilogue")[0][9] == sk
        assert _args(fake, "down_epilogue")[0][7] == 1


def test_the_down_combine_reads_one_split(monkeypatch):
    fake, _, _, _ = _route(monkeypatch, 1024, STAGED, wts=True)
    assert [n for n, _ in fake.calls] == ["group", "rot_in", "grouped", "grouped", "down_combine"]
    assert _args(fake, "down_combine")[0][9] == 1


def test_rates_past_five_bits_and_unaligned_trellises_keep_grouped_kernels_launches(monkeypatch):
    wide = _layer()
    wide.k2_gu, wide.k2_d = (6, 12), (12, 12)                      # a 2..16 instance outgrows static shared memory
    off16 = dataclasses.replace(_layer(), aligned=False)
    for ex in (wide, off16):
        fake, _, _, _ = _route(monkeypatch, 1024, STAGED, ex=ex)
        assert all("staged" not in kw for kw in _grouped(fake))


def test_only_aligned_layers_at_the_fused_setting_stage():
    ex = _layer()
    prompt = (8, 4, 4, 1, 2)
    assert experts.stageable(prompt, (6, 10), ex)
    assert experts.stageable(prompt, (118, 118), ex) and experts.stageable(prompt, (8, 8), ex)
    assert not experts.stageable(prompt, (6, 11), ex)
    assert not experts.stageable((8, 4, 4, 1, 1), (6, 6), ex)
    assert not experts.stageable(prompt, (6, 6), dataclasses.replace(ex, aligned=False))


def test_prepare_records_whether_every_trellis_starts_on_16_bytes():
    if not torch.cuda.is_available():
        pytest.skip("prepare takes CUDA trellises")
    base = torch.zeros(16 * 8 * 24 + 8, dtype=torch.int16, device="cuda")
    one = base[:16 * 8 * 24].view(16, 8, 24)
    off = base[4:4 + 16 * 8 * 24].view(16, 8, 24)
    s = (torch.ones(256, dtype=torch.float16, device="cuda"), torch.ones(128, dtype=torch.float16, device="cuda"))
    sd = (torch.ones(128, dtype=torch.float16, device="cuda"), torch.ones(256, dtype=torch.float16, device="cuda"))
    down = torch.zeros((8, 16, 24), dtype=torch.int16, device="cuda")
    assert experts.prepare([(one, *s)], [(one, *s)], [(down, *sd)], "mul1").aligned
    assert not experts.prepare([(one, *s)], [(off, *s)], [(down, *sd)], "mul1").aligned


# -- compiled for sm_121 --------------------------------------------------------------------------------------------
@pytest.mark.skipif(NVCC is None, reason="nvcc builds the sm_121 instances")
def test_the_staged_instances_do_not_spill_and_keep_two_blocks_an_sm():
    names = [f"grouped_kernel<2, 8, 4, 1, {lo}, {hi}, 2, {epi}, 8>" for epi in (0, 1, 2) for lo, hi in
             ((2, 10), (118, 118), (8, 8))]
    got = {k: v for k, v in _ptxas(names).items() if "grouped_kernel" in k}
    assert len(got) == len(names)
    for name, (regs, smem, spills) in got.items():
        assert not spills, name
        assert smem <= 48 * 1024 and 2 * (smem + 1024) <= 102400, (name, smem)
        assert 65536 // (-(-regs // 8) * 8 * 128) >= 2, (name, regs)


def _sass_values(obj: Path) -> dict[str, Counter]:
    """tests/test_exl3_prefill_switches.py's float instruction counts, with each F2FP counted by the values it
    rounds: the staged epilogues store a lane's 4 outputs at once, so one F2FP.PACK_AB rounds two (RN either way)."""

    out, name = {}, None
    text = subprocess.run([str(Path(NVCC).parent / "cuobjdump"), "-sass", str(obj)], capture_output=True, text=True,
                          check=True).stdout
    for line in text.splitlines():
        m = re.match(r"\s+Function : (\S+)", line)
        if m:
            name = m.group(1)
            out[name] = Counter()
            continue
        m = re.match(r"\s+/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;?\s*/\*", line)
        if not (name and m and (o := FLOAT_OPS.match(m.group(1)))):
            continue
        op = o.group(1)
        if op.startswith("F2FP") and op.endswith("PACK_AB"):
            sources = [x.strip() for x in m.group(1).split(",")[1:3]]
            out[name][op] += sum(x != "RZ" for x in sources)
        else:
            out[name][op] += 1
    return out


@pytest.mark.skipif(NVCC is None, reason="nvcc builds the sm_121 instances")
def test_the_staged_fused_epilogues_compile_the_epilogue_kernels_rounding_and_fma_instructions(tmp_path):
    """As tests/test_exl3_prefill_switches.py checks the fused instances: each staged fused instance, less its EPI 0
    twin, holds the standalone epilogue kernel's multiplies, FMAs, exponents and min/max in the same counts, and rounds
    as many values, so no multiply was contracted into an add the epilogue kernel rounds apart."""

    from torch.utils import cpp_extension

    src = tmp_path / "inst.cu"
    src.write_text((HERE / "experts.cu").read_text() + "\nnamespace tf_exl3x {\n" + "".join(
        f"template __global__ void grouped_kernel<2, 8, 4, 1, 2, 10, 2, {epi}, 8>(const half*, const half*, "
        "const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, const int*, float*, int, "
        "int, int, int, int, int, uint32_t, const int*, const int*, Epi);\n" for epi in (0, 1, 2)) + "}\n")
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

    gateup = one("gateup_epilogue_kernel")
    down = one("down_epilogue_kernelIf") + one("down_epilogue_kernelI13__nv_bfloat16")
    plain = one("grouped_kernelILi2ELi8ELi4ELi1ELi2ELi10ELi2ELi0ELi8E")
    assert one("grouped_kernelILi2ELi8ELi4ELi1ELi2ELi10ELi2ELi1ELi8E") - plain == gateup
    assert one("grouped_kernelILi2ELi8ELi4ELi1ELi2ELi10ELi2ELi2ELi8E") - plain == down + down


# -- the bench tool -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("on", [set(), {"staged"}])
def test_the_bench_sweeps_the_staged_switch_alone(on):
    pytest.importorskip("triton")
    from tensorfold.families.qwen4_exp.cuda import exl3_mm
    from tools.exl3_prefill_bench import _set

    base_sw = switches.Switches(pdl=True, group_scan=False, down_fuse=True, gateup_fuse=False)
    base_mm = exl3_mm.Switches(moe_chunk=False)
    _set(base_sw, base_mm, on, ("staged",))
    assert switches.current() == dataclasses.replace(base_sw, expert_staged="staged" in on)
    assert exl3_mm.current() == base_mm
