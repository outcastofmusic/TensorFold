"""The 6-bit affine tensor-core matmul (``tensorfold/cuda/affine6``) on synthetic MLX group-128 weights.

- ``pack`` keeps every code: reading the lane words back the way the kernel does gives the checkpoint's codes.
- The output agrees with a float64 reference and with the generic affine kernel.
- A row gives the same bits alone and in windows up to 128 rows, from a strided input too.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

SHAPES = [(256, 128), (512, 1536), (2048, 4096), (4096, 2048), (1040, 384)]     # (n, k); the last: one warp, odd tiles


def _weight(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(0, 64, (n, k), generator=g, dtype=torch.int64)
    bit = torch.arange(k) * 6
    words = torch.zeros((n, k * 6 // 32 + 1), dtype=torch.int64)
    words.scatter_add_(1, (bit // 32).expand(n, k), (c << (bit % 32)) & 0xFFFFFFFF)
    words.scatter_add_(1, (bit // 32 + 1).expand(n, k), (c << (bit % 32)) >> 32)
    words = words[:, :k * 6 // 32]
    stored = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    scales = ((torch.rand((n, k // 128), generator=g) + 0.5) * 2e-3).to(torch.bfloat16)
    biases = (-(torch.rand((n, k // 128), generator=g) + 0.5) * 6e-2).to(torch.bfloat16)
    return c, stored.cuda(), scales.cuda(), biases.cuda()


@pytest.mark.parametrize("n,k", SHAPES)
def test_pack_keeps_every_code(n, k):
    from tensorfold.cuda.affine6 import affine6

    c, w, s, b = _weight(n, k, 1)
    assert torch.equal(affine6.codes(w, k).cpu().long(), c)
    q = affine6.pack(w, s, b, rows=64)
    words = q.words.cpu().to(torch.int64) & 0xFFFFFFFF                   # [n / 16, k / 32, 32, 3]
    stream = [words[..., 0] | (words[..., 1] << 32), words[..., 2]]
    lane = torch.arange(32)
    for code in range(16):
        bit = 6 * code
        got = ((stream[0] >> bit) & 63 if bit <= 54 else
               ((stream[0] >> bit) & ((1 << (64 - bit)) - 1)) | ((stream[1] << (64 - bit)) & 63) if bit < 64 else
               (stream[1] >> (bit - 64)) & 63)
        kt, half, reg, e = code >> 3, (code >> 2) & 1, (code >> 1) & 1, code & 1
        rows = lane // 4 + 8 * half
        cols = 16 * kt + 2 * (lane % 4) + 8 * reg + e
        want = c.view(n // 16, 16, k // 32, 32).permute(0, 2, 1, 3)[:, :, rows, cols]
        assert torch.equal(got, want), code
    assert torch.equal(q.scales.t(), s) and torch.equal(q.biases.t(), b)


@pytest.mark.parametrize("n,k", SHAPES)
def test_the_matmul_matches_float64_and_the_generic_kernel_and_keeps_rows_independent(n, k):
    from tensorfold.cuda.affine6 import affine6
    from tensorfold.cuda.kernels import affine

    c, w, s, b = _weight(n, k, 2)
    q = affine6.pack(w, s, b)
    g = torch.Generator().manual_seed(3)
    x = torch.randn((128, k), generator=g).to(torch.bfloat16).cuda()
    full = affine6.matmul(x, q, f32=True)
    weight = (c.cuda().double().view(n, k // 128, 128) * s.double()[:, :, None] + b.double()[:, :, None]).view(n, k)
    want = x[:8].double() @ weight.t()
    assert (full[:8].double() - want).abs().max() <= 2e-5 * want.abs().max()
    generic = affine.matmul(x[:8], SimpleNamespace(layout="mlx", weight=w, scales=s, biases=b, bits=6, gs=128), f32=True)
    assert (full[:8] - generic).abs().max() <= 2e-5 * want.abs().max()
    for rows in (1, 2, 15, 16, 17, 64):
        for start in (0, 128 - rows):
            assert torch.equal(affine6.matmul(x[start:start + rows], q, f32=True), full[start:start + rows]), (rows, start)
    wide = torch.zeros((128, k + 64), dtype=torch.bfloat16, device="cuda")
    wide[:, :k] = x
    assert torch.equal(affine6.matmul(wide[:, :k], q, f32=True), full)
    assert torch.equal(affine6.matmul(x, q), full.to(torch.bfloat16))
