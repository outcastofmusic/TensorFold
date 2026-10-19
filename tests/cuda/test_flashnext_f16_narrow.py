"""TF_FLASHNEXT_F16_NARROW on the GPU: the fp16 matrices of a Flash Next EXL3 pack write the same bytes on the narrow
decode tiles as on BN 64.

Every fp16 shape of Qwen3.8 Flash Next EXL3 3.05 bpw (HC down with and without the inject rows, HC up, DeltaNet's
b/a, the PLE key and value projections) at rows 1-8 and 16, through ``F16.__call__`` (bf16 and fp32 outputs, the
split-K reduce where the matrix has slices) and ``F16.partials``, on random weights and on the pack's own layer-0
weights when MODELS names the models directory. Rows 17 and 128 check that a wider call keeps BN 64 either way."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("Flash Next CUDA kernels run on sm_12x (GB10, RTX 50, RTX PRO 6000) only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import exl3_mm

DEV = "cuda"
SHAPES = [(324, 10240), (320, 10240), (10240, 320), (96, 2560), (10240, 2560), (2560, 2560)]
ROWS = [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 128]
PACK = Path(os.environ.get("MODELS", "/home/agis/models")) / "Qwen3.8-Flash-Next-exl3-3.05bpw"


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _matrix(w: torch.Tensor) -> exl3_mm.F16:
    sc = exl3_mm.Scratch(11)
    n, k = w.shape
    sk = exl3_mm.f16_split(n, k)
    sc.part = torch.empty((sk * exl3_mm.ROWS * n,), dtype=torch.float32, device=DEV)
    return exl3_mm.F16(w.to(DEV, torch.float16).contiguous(), n, k, sk, sc)


def _inputs(rows: int, k: int, dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn((rows, k), generator=g, device=DEV).to(dtype)
    x[rows // 2] = 0.0                                   # a zero row: every slice -0.0 or 0.0
    return x


def _both(q: exl3_mm.F16, x: torch.Tensor, run) -> tuple[torch.Tensor, torch.Tensor]:
    got = []
    for narrow in (False, True):
        q.sc.narrow = narrow
        got.append(run(q, x).clone())
    return got[0], got[1]


def _check(q: exl3_mm.F16, rows: int, seed: int) -> None:
    for x_dtype in (torch.bfloat16, torch.float16):
        x = _inputs(rows, q.k, x_dtype, seed)
        for out_dtype in (torch.bfloat16, torch.float32):
            old, new = _both(q, x, lambda m, v, d=out_dtype: m(v, torch.empty((rows, m.n), dtype=d, device=DEV)))
            assert _same(old, new), (q.n, q.k, rows, x_dtype, out_dtype)
        old, new = _both(q, x, lambda m, v: m.partials(v))
        assert _same(old, new), (q.n, q.k, rows, x_dtype, "partials")


@pytest.mark.parametrize("n, k", SHAPES)
@pytest.mark.parametrize("rows", ROWS)
def test_random_weights_keep_their_bytes_on_the_narrow_tiles(n, k, rows):
    g = torch.Generator(device=DEV).manual_seed(n * 31 + k)
    q = _matrix(torch.randn((n, k), generator=g, device=DEV) * 0.02)
    _check(q, rows, seed=rows * 7 + n + k)


def _pack_matrices() -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    T = "model.language_model.layers."
    want = {"hc down": [f"{T}0.attn_hyper_connection.input_mix_weight_down.weight",
                        f"{T}0.attn_hyper_connection.block_inject_weight.weight"],
            "hc up": [f"{T}0.attn_hyper_connection.input_mix_weight_up.weight"],
            "gdn b/a": [f"{T}0.linear_attn.in_proj_b.weight", f"{T}0.linear_attn.in_proj_a.weight"]}
    tensors: dict[str, torch.Tensor] = {}
    for part in sorted(PACK.glob("*.safetensors")):
        with safe_open(str(part), framework="pt") as f:
            for name in set(f.keys()) & {r for rows in want.values() for r in rows}:
                tensors[name] = f.get_tensor(name)
    return {key: torch.cat([tensors[r] for r in rows]) for key, rows in want.items()}


@pytest.mark.skipif(not PACK.is_dir(), reason="the Flash Next EXL3 pack is not under MODELS")
@pytest.mark.parametrize("rows", ROWS)
def test_the_packs_own_matrices_keep_their_bytes_on_the_narrow_tiles(rows):
    found = _pack_matrices()
    assert set(found) == {"hc down", "hc up", "gdn b/a"}
    for i, w in enumerate(found.values()):
        _check(_matrix(w), rows, seed=rows * 13 + i)
