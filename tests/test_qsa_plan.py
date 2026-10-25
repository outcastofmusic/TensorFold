"""The prompt indexer's CPU side: which scratches take qsa_scores.cu, and which select runs at which block counts.
tests/cuda/test_qsa_scores.py checks the kernels' bytes and lists on the GPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda import attention as att  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attn_multi, qsa  # noqa: E402


def test_the_kernel_covers_the_shipped_indexer_only():
    assert qsa.fits(4, 128)                            # FP4-Local and EXL3 3.05bpw: indexer_n_heads 4, head_dim 128
    assert not qsa.fits(8, 128) and not qsa.fits(4, 64)


@pytest.mark.parametrize("prefill, heads, dim, fast", [
    (True, 4, 128, True),
    (False, 4, 128, False),                            # decode windows keep the Triton kernels
    (True, 8, 128, False),                             # an indexer the kernel does not cover
])
def test_prompt_buffers_take_the_cuda_kernels(prefill, heads, dim, fast):
    assert qsa.fast(prefill, heads, dim) is fast


@pytest.mark.parametrize("blocks, fast, tiled", [
    (1, False, False), (1, True, False),
    (8191, True, False), (8192, True, True), (8192, False, False),               # the fast scratch's threshold
    (16384, False, False), (32768, False, False), (32768, True, True),
    (32769, False, True), (32769, True, True), (65538, False, True),             # past the register width: always
])
def test_the_tiled_select_runs_from_8192_blocks_on_a_fast_scratch_and_past_the_registers_on_any(blocks, fast, tiled):
    assert qsa.tiled(blocks, att.triton.next_power_of_2(blocks), fast, att.SELECT_REGS) is tiled


@pytest.mark.parametrize("indexer, built", [
    ((4, 128), True),
    ((8, 128), False), ((4, 64), False),                   # an indexer qsa_scores.cu does not cover keeps Triton
])
def test_the_qsa_extension_builds_only_for_an_indexer_it_covers(monkeypatch, indexer, built):
    from tensorfold.cuda import experts
    from tensorfold.cuda.kernels import gdn as shared_gdn
    from tensorfold.cuda.kernels import qmm
    from tensorfold.families.qwen4_exp.cuda import engine, gdn, gdn_io

    loaded = []
    for module in (experts, shared_gdn, qmm, gdn, gdn_io, qsa):
        monkeypatch.setattr(module, "_ext", lambda module=module: loaded.append(module))
    engine.build_kernels(indexer=indexer)
    assert (qsa in loaded) is built and len(loaded) == 5 + built


class _Launch:
    """A Triton kernel stand-in: ``k[grid](*args, **kw)`` records (grid, kw)."""

    def __init__(self, calls: list, name: str) -> None:
        self.calls, self.name = calls, name

    def __getitem__(self, grid):
        return lambda *args, **kw: self.calls.append((self.name, grid, kw))


@pytest.fixture
def calls(monkeypatch):
    seen: list = []
    for name in ("_scores", "_select", "_select_tiles"):
        monkeypatch.setattr(att, name, _Launch(seen, name))
    monkeypatch.setattr(qsa, "scores", lambda *args: seen.append(("qsa_scores", args[4:], {})))
    return seen


def _scratch(rows: int, capacity: int, fast: bool) -> att.AttnScratch:
    return att.AttnScratch(rows, 4, 64, capacity, "cpu", budget=2048, ratio=4, fast=fast)


@pytest.mark.parametrize("context, fast, want", [
    (4096, False, ["_scores", "_select"]),
    (4096, True, ["qsa_scores", "_select"]),
    (32764, True, ["qsa_scores", "_select"]),                 # 8,191 blocks
    (32765, True, ["qsa_scores", "_select_tiles"]),           # 8,192 blocks: cdiv(32765, 4)
    (32769, False, ["_scores", "_select"]),
    (131072, False, ["_scores", "_select"]),                  # 32,768 blocks still fit the registers
    (131076, False, ["_scores", "_select_tiles"]),
    (131076, True, ["qsa_scores", "_select_tiles"]),
])
def test_rows_score_and_select_on_the_scratch_s_kernels(calls, context, fast, want):
    sc = _scratch(256, 262144, fast)
    iq = torch.zeros((256, 4, 128), dtype=torch.bfloat16)
    pooled = torch.zeros((sc.nb, 128), dtype=torch.bfloat16)
    pos = torch.zeros((1,), dtype=torch.int32)
    att.qsa_rows(iq, pooled, pos, sc, 256, context=context)
    assert [c[0] for c in calls] == want
    blocks = -(-context // 4)
    if fast:
        assert calls[0][1] == (256, blocks, 4, 512)           # rows, blocks, ratio, top
    else:
        assert calls[0][1] == (256, -(-blocks // 64))         # _scores' grid: a row by 64-block tiles
    if want[1] == "_select_tiles":
        assert calls[1][2]["TB"] == 4096                      # a prompt's row block


def test_a_window_s_row_view_keeps_the_scratch_s_kernels():
    for fast in (False, True):
        assert attn_multi._rows_from(_scratch(16, 65536, fast), 3).fast is fast


def test_the_scratch_is_plain_unless_asked():
    assert _scratch(8, 65536, False).fast is False
    assert att.AttnScratch(8, 4, 64, 65536, "cpu").fast is False
