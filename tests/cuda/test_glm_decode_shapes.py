"""GLM's decode kernels on the GPU: the served kernels write the old kernels' bytes.

``prefill._plain_narrow`` against ``prefill._plain`` on every plain projection GLM-5.3 Flash runs (f_a and g_a
[128, 4096], b_proj [64, 4096], the router gate [288, 4096], the indexer's wk [128, 4096], weights_proj [32, 4096] and
kpool gate [128, 4096]) and a 512-wide K [16384, 512], at decode rows 1 to 6 and 8 and at a prompt window's 128, from
contiguous, column-sliced and misaligned rows, into bf16, fp32 and column slices. ``glue.hc_pre`` on
``_hc_partial_split`` against ``_hc_partial`` on the fp32 carrier (packed TP1) and the bf16 one, at the model's
4 x 4,096 streams and at 4 x 512 (one 128-wide step a K block), at rows 1 to 6, 8 and prompt chunks of 128, 129,
2,047 and 2,048. Rows include a zero row (every product +-0) and a row whose halves cancel. The real checkpoint's
served replies are in tests/cuda/test_glm_sweep_switches.py."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import glue, prefill

DEV = "cuda"
DECODE_ROWS = [1, 2, 3, 4, 5, 6, 8]
PLAIN = [(128, 4096), (64, 4096), (288, 4096), (32, 4096), (16384, 512)]
HC_ROWS = DECODE_ROWS + [128, 129, 2047, 2048]


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _rows(rows: int, k: int, g: torch.Generator, dtype=torch.bfloat16, scale: float = 1.0) -> torch.Tensor:
    x = torch.randn((rows, k), generator=g, device=DEV) * scale
    if rows > 2:
        x[1] = 0.0                                     # every product +-0.0
        x[2, k // 2:] = -x[2, :k // 2]
    return x.to(dtype)


def _plain_old(w, x, out):
    m, k = x.shape
    prefill._plain[(m, -(-w.shape[0] // 16))](x, w, out, m, x.stride(0), out.stride(0), N=w.shape[0], K=k, BK=128,
                                               BN=16, num_warps=4)


def _plain_new(w, x, out):
    m, k = x.shape
    u = next(u for u in (8, 4, 2, 1) if (k // 128) % u == 0)
    prefill._plain_narrow[(m, -(-w.shape[0] // 2))](x, w, out, m, x.stride(0), out.stride(0), N=w.shape[0], K=k,
                                                     BK=128, BN=2, U=u, num_warps=1)


@pytest.mark.parametrize("n,k", PLAIN)
@pytest.mark.parametrize("w_dtype", [torch.float16, torch.bfloat16])
def test_the_narrow_plain_kernel_writes_the_old_bytes(n, k, w_dtype):
    g = torch.Generator(device=DEV).manual_seed(n + k)
    w = (torch.randn((n, k), generator=g, device=DEV) * 0.02).to(w_dtype)
    w[0] = -w[0].abs()                                   # a zero row's products are -0.0 on this output
    checked = 0
    for rows in DECODE_ROWS + [128]:
        wide = _rows(rows, k + 128, g)
        for x_name, x in (("contiguous", wide[:, :k].contiguous()), ("column slice", wide[:, 128:]),
                          ("misaligned", wide[:, 1:k + 1]), ("fp32", wide[:, :k].float())):
            for out_name, dtype, pad in (("bf16", torch.bfloat16, 0), ("fp32", torch.float32, 0),
                                         ("column slice", torch.bfloat16, 192), ("misaligned", torch.float32, 2)):
                got = []
                for kernel in (_plain_old, _plain_new):
                    parent = torch.empty((rows, n + pad), dtype=dtype, device=DEV)
                    parent.view(torch.uint8).fill_(0x5A)
                    out = parent[:, pad // 2:pad // 2 + n] if pad else parent
                    kernel(w, x, out)
                    torch.cuda.synchronize()
                    got.append(_bytes(parent))
                assert got[0] == got[1], (rows, x_name, out_name)
                checked += 1
    assert checked == 8 * 4 * 4


@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_plain_matmul_writes_the_old_kernels_bytes_on_decode_rows(rows):
    g = torch.Generator(device=DEV).manual_seed(rows)
    for n, k in PLAIN + [(8192, 128)]:
        w = (torch.randn((n, k), generator=g, device=DEV) * 0.02).half()
        x = _rows(rows, k, g)
        a = torch.empty((rows, n), device=DEV)
        _plain_old(w, x, a)
        b = prefill.plain_matmul(w, x, torch.empty((rows, n), device=DEV))
        assert _bytes(a) == _bytes(b), (n, k)


def _hc_case(rows: int, d: int, x_dtype, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    wide = 4 * d
    x = _rows(rows, wide, g, x_dtype)
    fn = torch.randn((24, wide), generator=g, device=DEV) * 0.02
    base = torch.randn(24, generator=g, device=DEV) * 0.1
    scale = torch.rand(3, generator=g, device=DEV) + 0.5
    norm_w = (torch.randn(d, generator=g, device=DEV) * 0.05 + 1.0).to(torch.bfloat16)
    return x, fn, base, scale, norm_w


@pytest.mark.parametrize("d", [4096, 512])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", HC_ROWS)
def test_split_hc_partials_give_hc_pre_the_old_bytes(monkeypatch, d, x_dtype, rows):
    x, fn, base, scale, norm_w = _hc_case(rows, d, x_dtype, rows * 31 + d)
    assert glue.splits(fn)
    splits = glue.splits
    got = []
    for split in (False, True):             # False: _hc_partial, the old kernel, as for a matrix that does not split
        monkeypatch.setattr(glue, "splits", splits if split else lambda fn: False)
        out = torch.empty((rows, d), dtype=torch.bfloat16, device=DEV)
        xs = torch.empty((rows, d // 64), device=DEV)
        post = torch.empty((rows, 4), device=DEV)
        comb = torch.empty((rows, 16), device=DEV)
        part = torch.full((rows, glue.HC_BLOCKS, 32), 7.0, device=DEV)
        glue.hc_pre(x, fn, base, scale, norm_w, out, xs, post, comb, part, 1e-5, 1e-6, 20)
        torch.cuda.synchronize()
        got.append([_bytes(t) for t in (out, xs, post, comb, part)])
    for name, a, b in zip(("normed", "xs", "post", "comb", "partials"), *got):
        assert a == b, name
