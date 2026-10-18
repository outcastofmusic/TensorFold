"""TF_FLASHNEXT_HC_FUSE on the CPU: the flag's values, where the engine and loader read it, which kernels a bf16
hyper-connection's read-out launches with it on or off, and (in the Triton interpreter) that the fused kernels write
the old kernels' bytes. tests/cuda/test_hc_fusions.py checks the bytes on the GPU."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda import bf16, forward, hc_b16  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weight_types import HC  # noqa: E402

ENV = hc_b16.FUSE_ENV


@pytest.mark.parametrize("value, on", [(None, True), ("", True), ("1", True), ("on", True), ("0", False),
                                       ("off", False), ("false", False), (" NO ", False)])
def test_the_flag_is_on_unless_turned_off(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert hc_b16.fuse() is on


@pytest.mark.parametrize("value", ["2", "fused", "o", "1.0"])
def test_a_flag_typo_stops_the_engine_and_the_loader_before_any_read(monkeypatch, tmp_path, value):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.weights import load

    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=ENV):
        hc_b16.fuse()
    with pytest.raises(ValueError, match=ENV):
        FlashNextEngine(tmp_path / "missing")                      # no checkpoint there: the flag fails first
    with pytest.raises(ValueError, match=ENV):
        load(tmp_path / "missing", "cpu")


def test_an_hc_fuses_only_when_the_loader_says_so():
    assert HC(None, None, None, True).fuse is False                 # MLX and EXL3 hyper-connections


ROWS = 2048


def _hc(fuse: bool):
    return SimpleNamespace(down=SimpleNamespace(kernel="b16", n=324, b="down"),
                           up=SimpleNamespace(kernel="b16", n=10240, b="up"), scale="scale", inject=True, fuse=fuse)


def _buffers(prefill: bool):
    names = ("pss", "normed", "xs_normed", "act", "xs_act", "up", "mixed", "xs_mixed")
    return SimpleNamespace(prefill=prefill, **{n: torch.zeros((ROWS, 1)) for n in names})


@pytest.fixture
def launches(monkeypatch):
    """The read-out's kernels as names, in launch order; ``fused`` sets what ``hc_check.fuser`` returns."""

    calls: list[str] = []
    state = SimpleNamespace(calls=calls, fused=lambda *a, **k: calls.append("write_norm"))

    def rec(name):
        return lambda *a, **k: calls.append(name)

    for name in ("hc_writeback", "hc_normed", "hc_act", "hc_mix"):
        monkeypatch.setattr(forward.glue, name, rec(name))

    def matmul(x, b, *, out=None, f32=False, inner_k=False, **kw):
        calls.append(f"down inner_k={inner_k}")
        return out

    monkeypatch.setattr(forward.bf16, "matmul", matmul)
    monkeypatch.setattr(forward.hc_b16, "up_mix", rec("up_mix"))
    monkeypatch.setattr(forward, "_mm", rec("up"))
    monkeypatch.setattr(forward, "_hc_fuser", lambda device, **kw: state.fused)
    return state


def _block(hc, b, rows):
    h = torch.zeros((ROWS, 1))
    forward.hc_block(hc, b, rows, 1e-6, 4, 320, 1, torch.zeros((ROWS, 4)), torch.zeros((ROWS, 4)), h,
                     branch=torch.zeros((ROWS, 1)))


OLD = ["hc_writeback", "hc_normed", "down inner_k=False", "hc_act", "up", "hc_mix"]


@pytest.mark.parametrize("rows", [17, 128, 129, 2048])
def test_a_prompt_chunk_takes_the_three_fusions(launches, rows):
    _block(_hc(True), _buffers(True), rows)
    assert launches.calls == ["write_norm", "down inner_k=True", "hc_act", "up_mix"]


def test_without_the_checked_write_norm_the_other_two_fusions_stay(launches):
    launches.fused = None                       # another GPU, an older Triton, or bytes that differed
    _block(_hc(True), _buffers(True), ROWS)
    assert launches.calls == ["hc_writeback", "hc_normed", "down inner_k=True", "hc_act", "up_mix"]


@pytest.mark.parametrize("fuse, prefill, rows", [(False, True, ROWS), (False, True, 17), (True, True, 16),
                                                 (True, True, 1), (True, False, 17), (True, False, 2048)])
def test_the_switch_off_decode_windows_and_short_chunks_keep_the_separate_kernels(launches, fuse, prefill, rows):
    launches.fused = lambda *a, **k: pytest.fail("the write-norm fusion ran")
    _block(_hc(fuse), _buffers(prefill), rows)
    assert launches.calls == OLD


def test_the_final_read_out_keeps_the_group_sums_the_draft_head_reads(launches):
    """``finish`` reads the mixer out through ``_readout`` with no ``fuse``: b.xs_mixed is written."""

    b = _buffers(True)
    forward._readout(_hc(True), b, torch.zeros((ROWS, 1)), ROWS, 1e-6, 4, 320, None)
    assert launches.calls == ["hc_normed", "down inner_k=False", "hc_act", "up", "hc_mix"]


class _Launch:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def __getitem__(self, grid):
        return lambda *args, **kw: self.calls.append((self.name, tuple(grid), kw))


@pytest.fixture
def kernels(monkeypatch):
    calls: list = []
    for name in ("_b16mm", "_b16mm_ks", "_reduce"):
        monkeypatch.setattr(bf16, name, _Launch(name, calls), raising=False)
    monkeypatch.setattr(bf16, "HAS_TRITON", True)
    return calls


def _down(rows, inner_k, sk=None):
    x = torch.zeros((rows, 10240), dtype=torch.bfloat16)
    b = bf16.B16(torch.zeros((324, 10240), dtype=torch.bfloat16), 324, 10240)
    return bf16.matmul(x, b, f32=True, inner_k=inner_k, sk=sk)


@pytest.mark.parametrize("rows", [129, 300, 2048])
def test_the_down_projection_sums_its_slices_in_one_launch_on_the_old_tiles(kernels, rows):
    _down(rows, True)
    (name, grid, kw), = kernels
    _down(rows, False)
    (old, old_grid, old_kw), (reduce, _, _) = kernels[1:]
    assert (name, old, reduce) == ("_b16mm_ks", "_b16mm", "_reduce")
    assert grid == old_grid[:2] == (-(-rows // 128), 6) and old_grid[2] == kw["SK"] == 32
    assert kw == old_kw                                           # BM 128, BLOCK_N 64, BK 64, warps and stages


@pytest.mark.parametrize("rows, sk", [(128, None), (17, None), (2048, 1)])
def test_up_to_128_rows_or_one_slice_the_down_projection_keeps_its_launches(kernels, rows, sk):
    _down(rows, True, sk)
    assert [name for name, _, _ in kernels] == (["_b16mm", "_reduce"] if sk is None else ["_b16mm"])


@pytest.fixture
def up_mix_launches(monkeypatch):
    launched: list = []
    monkeypatch.setattr(hc_b16, "_hc_up_mix", _Launch("_hc_up_mix", launched), raising=False)
    monkeypatch.setattr(hc_b16, "HAS_TRITON", True)
    return launched


def _up_mix(act, n, k, normed, mixed):
    hc_b16.up_mix(torch.zeros(act, dtype=torch.bfloat16), SimpleNamespace(weight=None, n=n, k=k),
                  torch.zeros(normed, dtype=torch.bfloat16), torch.zeros(mixed, dtype=torch.bfloat16), 4)


def test_the_up_mix_launches_a_program_per_64_rows_and_64_dims(up_mix_launches):
    _up_mix((200, 320), 10240, 320, (200, 10240), (200, 2560))
    assert [(name, grid) for name, grid, _ in up_mix_launches] == [("_hc_up_mix", (4, 40))]


@pytest.mark.parametrize("act, n, k, normed, mixed", [
    ((32, 300), 10240, 300, (32, 10240), (32, 2560)),        # K not whole 64 blocks
    ((32, 320), 4 * 2528, 320, (32, 4 * 2528), (32, 2528)),  # D not whole 64 tiles
    ((32, 320), 10240, 320, (31, 10240), (32, 2560)),
    ((32, 320), 10240, 320, (32, 10240), (32, 2559)),
])
def test_the_up_mix_refuses_shapes_it_does_not_tile(up_mix_launches, act, n, k, normed, mixed):
    with pytest.raises(ValueError, match="HC up-mix"):
        _up_mix(act, n, k, normed, mixed)
    assert not up_mix_launches


INTERPRETED = r"""
import sys
import torch
from tensorfold.families.qwen4_exp.cuda import bf16, glue, hc_b16

def same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))

g = torch.Generator().manual_seed(int(sys.argv[1]))
def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

out = []
# the down projection: FP4-Local's [324, 10240] at SK 32 (5 blocks a slice), and a [68, 512] one at SK 2, 4 and 8
for n, k, sks in ((324, 10240, (None,)), (68, 512, (2, 4, None))):
    b = bf16.B16(rnd(n, k, scale=0.05), n, k)
    for rows in (129, 200):
        x = rnd(rows, k)
        x[3] = 0.0
        x[5, : k // 2] = -x[5, k // 2:]                       # slices that cancel
        for sk in sks:
            for f32 in (True, False):
                old = bf16.matmul(x, b, f32=f32, sk=sk)
                new = bf16.matmul(x, b, f32=f32, sk=sk, inner_k=True)
                out.append(("down", n, rows, sk, f32, same(old, new)))
# the up projection and the mix: FP4-Local's [10240, 320] and four streams of 2560, on the old matmul's 128-row tile
for d, low in ((2560, 320), (256, 64)):
    up = bf16.B16(rnd(4 * d, low, scale=0.1), 4 * d, low)
    for rows in (129, 200):
        act, normed = rnd(rows, low), rnd(rows, 4 * d)
        up_rows = bf16.matmul(act, up)
        old = torch.empty((rows, d), dtype=torch.bfloat16)
        glue.hc_mix(up_rows, normed, old, torch.empty((rows, d // 32)), 4)
        new = torch.full((rows, d), float("nan"), dtype=torch.bfloat16)
        hc_b16.up_mix(act, up, normed, new, 4, bm=128)
        out.append(("upmix", d, rows, None, None, same(old, new)))
bad = [o for o in out if not o[-1]]
print(len(out), "cases;", "differ:", bad)
sys.exit(1 if bad else 0)
"""


def test_the_fused_kernels_write_the_old_bytes_in_the_triton_interpreter(tmp_path):
    """The interpreter runs each dot as a numpy matmul, so this checks the K slices' order and adds, the bf16
    roundings and the mix's expression; the GPU test checks the tensor cores' bytes and other tiles."""

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    run = subprocess.run([sys.executable, "-c", INTERPRETED, "11"], check=False, env=env, capture_output=True,
                         text=True, timeout=600, cwd=tmp_path)
    assert run.returncode == 0, (run.stdout[-2000:], run.stderr[-2000:])
    assert "differ: []" in run.stdout
