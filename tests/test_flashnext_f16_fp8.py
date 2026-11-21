"""TF_FLASHNEXT_F16_FP8: an EXL3 pack's fp16 matrices as FP8 e4m3 with a scale a row. The switch, the conversion,
the loader's choice, and (in the Triton interpreter) the matmul against torch at both of its launch shapes."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda import exl3_mm


@pytest.fixture
def switches():
    previous = exl3_mm.current()
    yield
    exl3_mm.use(previous)


def test_the_switch_is_off_by_default_and_read_from_the_environment(monkeypatch):
    monkeypatch.delenv("TF_FLASHNEXT_F16_FP8", raising=False)
    assert exl3_mm.read().f16_fp8 is False
    monkeypatch.setenv("TF_FLASHNEXT_F16_FP8", "1")
    assert exl3_mm.read().f16_fp8 is True
    monkeypatch.setenv("TF_FLASHNEXT_F16_FP8", "fp8")
    with pytest.raises(ValueError, match="TF_FLASHNEXT_F16_FP8"):
        exl3_mm.read()


def test_a_row_converts_with_its_largest_value_on_the_fp8_maximum():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(64, 512, generator=g).half()
    w[3] *= 1000
    w[5] = 0
    values, scale = exl3_mm.to_fp8(w)
    assert values.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert values.float().abs().amax(1)[[0, 3]].tolist() == [exl3_mm.FP8_MAX] * 2
    assert scale[5].item() == 1.0 and not values[5].float().any()
    back = values.float() * scale[:, None]
    for row in (0, 3):
        assert ((back[row] - w[row].float()).norm() / w[row].float().norm()).item() < 0.035


def test_the_loader_stores_fp8_only_with_the_switch(switches):
    rows = [torch.randn(48, 256).half(), torch.randn(8, 256).half()]
    exl3_mm.use(exl3_mm.Switches())
    plain = exl3_mm.f16(exl3_mm.Scratch(2), rows, "cpu")
    assert plain.w.dtype == torch.float16 and plain.scale is None and plain.nbytes() == 56 * 256 * 2
    exl3_mm.use(dataclasses.replace(exl3_mm.Switches(), f16_fp8=True))
    fp8 = exl3_mm.f16(exl3_mm.Scratch(2), rows, "cpu")
    assert (fp8.n, fp8.k, fp8.sk) == (plain.n, plain.k, plain.sk)
    assert fp8.w.dtype == torch.float8_e4m3fn and fp8.scale.shape == (56,)
    assert fp8.nbytes() == 56 * 256 + 56 * 4


INTERPRETED = r"""
import dataclasses, torch
from tensorfold.families.qwen4_exp.cuda import exl3_mm
exl3_mm.use(dataclasses.replace(exl3_mm.Switches(), f16_fp8=True))
g = torch.Generator().manual_seed(2)
worst = 0.0
for n, k in ((48, 256), (320, 10240 // 8)):          # one K slice (direct store), then split K (slices and reduce)
    sc = exl3_mm.Scratch(2)
    m = exl3_mm.f16(sc, [torch.randn(n, k, generator=g).half()], "cpu")
    sc.part = torch.empty((m.sk * exl3_mm.ROWS * n,), dtype=torch.float32)
    x = torch.randn(5, k, generator=g).to(torch.bfloat16)
    ref = x.half().float() @ (m.w.float() * m.scale[:, None]).T
    out = m(x, torch.empty((5, n), dtype=torch.float32))
    worst = max(worst, ((out - ref).abs().max() / ref.abs().max()).item())
    one = m(x[2:3], torch.empty((1, n), dtype=torch.float32))      # a row's bits do not depend on the window
    print("sk", m.sk, "row alone equal", torch.equal(one[0], out[2]))
print("worst", worst)
"""


def test_the_fp8_matmul_matches_torch_in_the_triton_interpreter(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    run = subprocess.run([sys.executable, "-c", INTERPRETED], check=False, env=env, capture_output=True, text=True,
                         timeout=600, cwd=tmp_path)
    assert run.returncode == 0, (run.stdout[-2000:], run.stderr[-2000:])
    lines = run.stdout.split("\n")
    assert [line.split()[1] != "1" for line in lines if line.startswith("sk")] == [False, True]   # both shapes ran
    assert all(line.endswith("True") for line in lines if line.startswith("sk"))
    worst = float(next(line for line in lines if line.startswith("worst")).split()[1])
    assert worst < 1e-3, run.stdout
