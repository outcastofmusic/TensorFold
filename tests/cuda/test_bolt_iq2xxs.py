"""Bolt's IQ2_XXS kernels against a numpy port of llama.cpp's dequantize_row_iq2_xxs, decode and prefill shapes."""

import hashlib
import struct

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.nemotron_bolt.cuda import iq2xxs

DEV = "cuda"


def dequant(packed: np.ndarray) -> np.ndarray:
    """uint8 [..., K / 256 * 66] -> float32 [..., K]: ggml-quants.c dequantize_row_iq2_xxs, block by block."""

    grid = np.array(iq2xxs.GRID, dtype="<u8").view(np.uint8).reshape(256, 8).astype(np.float32)
    signs = np.array([iq2xxs.ksigns(i) for i in range(128)], dtype=np.uint8)
    lead = packed.shape[:-1]
    blocks = packed.reshape(-1, iq2xxs.BLOCK_BYTES)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)[:, 0]                       # [B]
    qs = blocks[:, 2:].copy().view("<u4").reshape(-1, 8, 2)                               # [B, ib32, aux32]
    db = d[:, None] * (np.float32(0.5) + (qs[:, :, 1] >> 28).astype(np.float32)) * np.float32(0.25)
    gi = qs[:, :, 0:1].view(np.uint8).reshape(-1, 8, 4)                                   # aux8[l]
    si = (qs[:, :, 1:2] >> (7 * np.arange(4, dtype=np.uint32))) & 127                     # [B, 8, 4]
    sg = signs[si]
    neg = (sg[..., None] >> np.arange(8, dtype=np.uint8)) & 1                             # kmask_iq2xs[j]
    y = db[:, :, None, None] * grid[gi] * np.where(neg == 1, np.float32(-1), np.float32(1))
    return y.astype(np.float32).reshape(*lead, -1)


def random_packed(shape: tuple, k: int, seed: int) -> np.ndarray:
    """Random qs bytes; d a finite fp16 of either sign, 1e-3 .. 2e-2 like real experts'."""

    rng = np.random.default_rng(seed)
    nb = k // 256
    raw = rng.integers(0, 256, size=(*shape, nb, iq2xxs.BLOCK_BYTES), dtype=np.uint8)
    d = (rng.uniform(1e-3, 2e-2, size=(*shape, nb)) * rng.choice([-1, 1], size=(*shape, nb))).astype("<f2")
    raw[..., :2] = d[..., None].view(np.uint8)
    return raw.reshape(*shape, nb * iq2xxs.BLOCK_BYTES)


def bf16(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).to(torch.bfloat16)


def check(out: torch.Tensor, x: torch.Tensor, wq: torch.Tensor) -> None:
    """out within bf16 rounding of x @ wq.T in float64, plus fp32 accumulation slack; most outputs exact."""

    ref = x.double() @ wq.double().T
    mag = x.double().abs() @ wq.double().abs().T
    err = (out.double().cpu() - ref).abs()
    tol = ref.abs() * 2.0 ** -8 + mag * 2.0 ** -18 + 1e-30
    assert torch.all(err <= tol), f"max excess {float((err / tol).max()):.3f} of the bound"
    exact = (out.cpu() == ref.to(torch.bfloat16)).double().mean()
    assert exact > 0.97, f"{float(exact):.4f} of outputs equal bf16(ref)"


def test_grid_is_llama_cpp_table():
    blob = struct.pack("<256Q", *iq2xxs.GRID)
    assert hashlib.sha256(blob).hexdigest() == iq2xxs.GRID_SHA256
    assert set(blob) == set(iq2xxs.MAGNITUDES)
    assert iq2xxs.ksigns(1) == 129 and iq2xxs.ksigns(3) == 3 and iq2xxs.ksigns(127) == 255   # ggml-common.h:513


def test_dequant_matches_hand_decoded_block():
    """One block by hand: d 1, sub-block 0 grid entry 1 (bytes 43, then 8 x 7), sign index 1 (weights 0 and 7 flip)."""

    blk = np.zeros(66, dtype=np.uint8)
    blk[:2] = np.array([1.0], dtype="<f2").view(np.uint8)
    blk[2:6] = [1, 0, 0, 0]                                      # aux32[0]: entry 1, then entry 0 three times
    aux1 = 1 | 3 << 28                                           # sign index 1 for entry 0, scale 3
    blk[6:10] = np.array([aux1], dtype="<u4").view(np.uint8)
    y = dequant(blk)
    db = 1.0 * (0.5 + 3) * 0.25
    want = np.array([43] + [8] * 7, dtype=np.float32) * db
    want[0] = -want[0]
    want[7] = -want[7]                                           # ksigns[1] = 129: bits 0 and 7
    assert np.array_equal(y[:8], want)
    assert np.array_equal(y[8:32], np.full(24, 8 * db, dtype=np.float32))


@pytest.mark.parametrize("rows", [1, 5, 16, 17, 33, 64, 65, 200])
@pytest.mark.parametrize("n,k", [(40, 256), (64, 768), (128, 1024)])
def test_linear(rows, n, k):
    packed = random_packed((n,), k, seed=rows * 7 + n)
    wq = bf16(dequant(packed))
    x = torch.randn((rows, k), generator=torch.Generator().manual_seed(rows)).to(torch.bfloat16)
    out = iq2xxs.iq2xxs_linear(x.to(DEV), torch.from_numpy(packed).to(DEV))
    torch.cuda.synchronize()
    assert out.shape == (rows, n) and out.dtype == torch.bfloat16
    check(out, x, wq)


def test_linear_non_multiple_of_tile_columns():
    """N = 264: a decode tile of 8 columns at the end, a prefill block with one warp's columns past N."""

    packed = random_packed((264,), 512, seed=3)
    wq = bf16(dequant(packed))
    for rows in (3, 130):
        x = torch.randn((rows, 512), generator=torch.Generator().manual_seed(rows)).to(torch.bfloat16)
        check(iq2xxs.iq2xxs_linear(x.to(DEV), torch.from_numpy(packed).to(DEV)), x, wq)


@pytest.mark.parametrize("rows,experts,skew", [(1, 3, False), (9, 9, False), (9, 4, False), (64, 5, False),
                                               (64, 30, True), (65, 7, False), (300, 11, False), (700, 9, True)])
def test_grouped(rows, experts, skew):
    n, k = 48, 512
    packed = random_packed((experts, n), k, seed=rows + experts)
    wq = bf16(dequant(packed))
    g = torch.Generator().manual_seed(rows * 31 + experts)
    x = torch.randn((rows, k), generator=g).to(torch.bfloat16)
    if skew:                                                     # most rows on expert 0, some experts unused
        ids = torch.where(torch.rand(rows, generator=g) < 0.7, 0, torch.randint(0, experts // 2 + 1, (rows,),
                                                                                generator=g))
    else:
        ids = torch.randint(0, experts, (rows,), generator=g)
    ids = ids.to(torch.int32)
    out = iq2xxs.iq2xxs_grouped(x.to(DEV), torch.from_numpy(packed).to(DEV), ids.to(DEV)).cpu()
    ref = torch.stack([(x[r].double() @ wq[int(ids[r])].double().T) for r in range(rows)])
    mag = torch.stack([(x[r].double().abs() @ wq[int(ids[r])].double().abs().T) for r in range(rows)])
    err = (out.double() - ref).abs()
    assert torch.all(err <= ref.abs() * 2.0 ** -8 + mag * 2.0 ** -18 + 1e-30)
    assert (out == ref.to(torch.bfloat16)).double().mean() > 0.97


@pytest.mark.parametrize("n,k", [(2560, 1024), (1024, 2560)])
@pytest.mark.parametrize("rows", [9, 36, 300])
def test_grouped_bolt_shapes(n, k, rows):
    """Bolt's widest expert (up: N 2560, K 1024; down: N 1024, K 2560), one token, a verify window and a prompt."""

    experts = 12
    packed = random_packed((experts, n), k, seed=n + rows)
    wq = bf16(dequant(packed))
    g = torch.Generator().manual_seed(rows)
    x = torch.randn((rows, k), generator=g).to(torch.bfloat16)
    ids = torch.cat([torch.randperm(experts, generator=g)[:9] for _ in range(-(-rows // 9))])[:rows].to(torch.int32)
    out = iq2xxs.iq2xxs_grouped(x.to(DEV), torch.from_numpy(packed).to(DEV), ids.to(DEV)).cpu()
    for e in range(experts):
        sel = (ids == e).nonzero().flatten()
        if sel.numel():
            check(out[sel], x[sel], wq[e])


def test_decode_and_prefill_agree():
    """Rows 0..63 of a 64-row (decode) and a 65-row (prefill) call differ by at most one bf16 rounding."""

    packed = torch.from_numpy(random_packed((128,), 1024, seed=11)).to(DEV)
    x = torch.randn((65, 1024), generator=torch.Generator().manual_seed(2)).to(torch.bfloat16).to(DEV)
    a = iq2xxs.iq2xxs_linear(x[:64], packed).float()
    b = iq2xxs.iq2xxs_linear(x, packed)[:64].float()
    assert torch.all((a - b).abs() <= a.abs() * 2.0 ** -7 + 1e-6)


def test_rows_bitwise_independent_of_the_call():
    """A row's output bits do not depend on the other rows of a grouped decode call."""

    packed = torch.from_numpy(random_packed((6, 32), 256, seed=5)).to(DEV)
    x = torch.randn((20, 256), generator=torch.Generator().manual_seed(9)).to(torch.bfloat16).to(DEV)
    ids = torch.randint(0, 6, (20,), generator=torch.Generator().manual_seed(4)).to(torch.int32).to(DEV)
    whole = iq2xxs.iq2xxs_grouped(x, packed, ids)
    for r in (0, 7, 19):
        alone = iq2xxs.iq2xxs_grouped(x[r:r + 1], packed, ids[r:r + 1])
        assert torch.equal(whole[r:r + 1], alone)


def test_empty_and_bad_shapes():
    packed = torch.from_numpy(random_packed((16,), 256, seed=1)).to(DEV)
    assert iq2xxs.iq2xxs_linear(torch.empty((0, 256), dtype=torch.bfloat16, device=DEV), packed).shape == (0, 16)
    with pytest.raises(ValueError):
        iq2xxs.iq2xxs_linear(torch.zeros((2, 512), dtype=torch.bfloat16, device=DEV), packed)
