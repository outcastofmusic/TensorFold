"""A shared round wider than one EXL3 fp16 call: the hyper-connection's down projection on decode rows.

Six streams of 32-row copy windows verify 192 rows in one round, past the 128 rows (``exl3_mm.ROWS``) that
``F16.partials`` takes (tf-runs/parallel6, 2026-10-21). The kernels are swapped for torch references, so this runs
without CUDA; tests/cuda/test_flashnext_tandem_checks.py runs the real ones."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import exl3_mm, forward

S, LOW, K = 4, 8, 64


class _Down(exl3_mm.F16):
    """``F16`` with its Triton launch replaced by fp32 slices in torch, a row at a time as the kernel's rows are (CPU
    BLAS blocks by row count): ``partials`` keeps its row limit and scratch."""

    def _launch(self, x, out, f32):
        ks = self.k // self.sk
        for s in range(self.sk):
            w = self.w[:, s * ks:(s + 1) * ks].float().t()
            for r in range(x.shape[0]):
                out[s, r] = x[r:r + 1, s * ks:(s + 1) * ks].float() @ w


def _reduce_act(part, act, xs, inj, streams, low):
    """``glue.hc_reduce_act``'s shape: each row from its own slices, in slice order."""

    v = part[0].clone()
    for s in range(1, part.shape[0]):
        v += part[s]
    act.copy_((v[:, :low] / streams).to(act.dtype))
    xs.copy_(v[:, :low].reshape(v.shape[0], -1, low // 2).sum(-1))
    if inj is not None:
        inj.copy_((v[:, low:] / streams).to(inj.dtype))


def _run(down, x, inject: bool):
    rows = x.shape[0]
    b = SimpleNamespace(prefill=False, normed=x, dn=torch.empty((rows, LOW + S)), dn_mix=torch.empty((rows, LOW)),
                        act=torch.zeros((rows, LOW)), xs_act=torch.zeros((rows, 2)))
    inj = torch.zeros((rows, S)) if inject else None
    forward._down_act(SimpleNamespace(down=down), b, rows, S, LOW, inj)
    return b.act, b.xs_act, inj


@pytest.mark.parametrize("inject", [True, False])
@pytest.mark.parametrize("rows", [129, 192, 300])
def test_a_round_past_one_calls_rows_gives_each_row_its_own_bits(monkeypatch, rows, inject):
    monkeypatch.setattr(forward.glue, "hc_reduce_act", _reduce_act)
    g = torch.Generator().manual_seed(rows)
    n = LOW + (S if inject else 0)
    sc = exl3_mm.Scratch(11, False)
    down = _Down(torch.randn((n, K), generator=g).half(), n, K, 2, sc)
    sc.part = torch.empty((down.sk * exl3_mm.ROWS * n,), dtype=torch.float32)
    x = torch.randn((rows, K), generator=g).to(torch.bfloat16)
    act, xs, inj = _run(down, x, inject)
    for r in (0, exl3_mm.ROWS - 1, exl3_mm.ROWS, rows - 1):
        one = _run(down, x[r:r + 1].contiguous(), inject)
        assert torch.equal(act[r:r + 1], one[0]) and torch.equal(xs[r:r + 1], one[1]), r
        if inject:
            assert torch.equal(inj[r:r + 1], one[2]), r
