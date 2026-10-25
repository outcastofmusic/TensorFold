"""The bf16 hyper-connection's prompt fusions on the GPU: they write the old kernels' bytes.

Each fusion against the kernels it replaces, on every HC shape FP4-Local's prompt chunks use (hidden 2560, 4 streams,
low rank 320; the down projection with the inject rows, 324, and without, 320) at chunk rows 1, 128, 129, 2,047,
2,048 and 4,096: the down projection with its K slices summed in each program (``bf16._b16mm_ks``), the up projection
with the mix (``hc_b16.up_mix``), the write-back with the norm (``hc_check.fuser``), and ``forward.hc_block`` whole
against the separate kernels. The first three also run at the tiny checkpoint's shapes (hidden 512, low rank 64) at
its chunk rows 188 and 512. Then a prompt and 48 greedy tokens as served, drafted equal serial: on the tiny ModelOpt
checkpoint in FP4-Local's layout, and on FP4-Local itself when TF_FP4_LOCAL names its directory."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("Flash Next CUDA kernels run on sm_12x (GB10, RTX 50, RTX PRO 6000) only", allow_module_level=True)

from nvfp4_tiny import write  # noqa: E402

from tensorfold.families.qwen4_exp.cuda import bf16, forward, glue, hc_b16, hc_check  # noqa: E402

DEV = "cuda"
D, S, LOW, SLOTS = 2560, 4, 320, 11                 # FP4-Local: hidden, hc_count, hc_lowrank, top 10 + the shared
ROWS = [1, 128, 129, 2047, 2048, 4096]


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _rand(shape, g, scale=1.0, dtype=torch.bfloat16):
    return (torch.randn(shape, generator=g, device=DEV) * scale).to(dtype)


def _down_bytes(n: int, k: int, rows: int, f32: bool, sk=None):
    g = torch.Generator(device=DEV).manual_seed(rows * 7 + n + k)
    b = bf16.B16(_rand((n, k), g, 0.02), n, k)
    x = _rand((rows, k), g)
    x[rows // 2] = 0.0                                   # a zero row: every slice -0.0 or 0.0
    x[rows - 1, : k // 2] = -x[rows - 1, k // 2:]
    return bf16.matmul(x, b, f32=f32, sk=sk), bf16.matmul(x, b, f32=f32, sk=sk, inner_k=True)


@pytest.mark.parametrize("n", [LOW + S, LOW])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("f32", [True, False])
def test_the_down_projection_sums_its_slices_in_each_program_with_the_same_bytes(n, rows, f32):
    assert bf16.split_k(n, S * D) == 32 and bf16.inner_k_fits(bf16.INNER_K_ROWS, S * D, 32)    # 5 blocks a slice
    old, new = _down_bytes(n, S * D, rows, f32)
    assert _same(old, new), (n, rows, f32)


# The tiny checkpoint (hidden 512, 4 streams, low rank 64): its down projections [64 or 128, 2,048] split K 32 ways
# at one 64-wide block a slice, where ``_b16mm_ks`` would fold each slice's add into the tensor-core dot. ``matmul``
# keeps the old launches there (``bf16.inner_k_fits``); from two blocks a slice it fuses, with the same bytes.
TINY_D, TINY_LOW = 512, 64


@pytest.mark.parametrize("n, k, sk, fused", [
    (TINY_LOW, S * TINY_D, None, False), (2 * TINY_LOW, S * TINY_D, None, False),     # the tiny checkpoint: SK 32
    (2 * TINY_LOW, S * TINY_D, 16, True), (TINY_LOW, 2 * S * TINY_D, 32, True), (TINY_LOW + S, 3 * S * TINY_D, 32, True)])
@pytest.mark.parametrize("rows", [188, 512, 700])
@pytest.mark.parametrize("f32", [True, False])
def test_the_down_projection_keeps_the_old_bytes_at_one_block_a_slice_and_fuses_from_two(n, k, sk, fused, rows, f32):
    assert bf16.inner_k_fits(rows, k, sk or bf16.split_k(n, k)) is fused
    old, new = _down_bytes(n, k, rows, f32, sk)
    assert _same(old, new), (n, k, sk, rows, f32)


@pytest.mark.parametrize("d, low, rows", [(D, LOW, r) for r in ROWS] + [(TINY_D, TINY_LOW, r) for r in (188, 512)])
@pytest.mark.parametrize("bm", [64, 128])
def test_the_up_projection_and_the_mix_in_one_kernel_write_the_old_bytes(d, low, rows, bm):
    g = torch.Generator(device=DEV).manual_seed(rows * 11 + bm + d)
    up = bf16.B16(_rand((S * d, low), g, 0.05), S * d, low)
    act, normed = _rand((rows, low), g), _rand((rows, S * d), g)
    normed[0, :64] = -0.0
    old_up = bf16.matmul(act, up)
    old = torch.empty((rows, d), dtype=torch.bfloat16, device=DEV)
    glue.hc_mix(old_up, normed, old, torch.empty((rows, d // 32), device=DEV), S)
    new = torch.full_like(old, float("nan"))
    hc_b16.up_mix(act, up, normed, new, S, bm=bm)
    assert _same(old, new), (d, rows, bm)
    if rows > 1:                                         # a row's bytes do not depend on its chunk
        one = torch.empty((1, d), dtype=torch.bfloat16, device=DEV)
        hc_b16.up_mix(act[rows - 1:].contiguous(), up, normed[rows - 1:].contiguous(), one, S, bm=bm)
        assert _same(one, new[rows - 1:]), (d, rows)


def _fuser():
    fused = hc_check.fuser(torch.device(DEV, torch.cuda.current_device()))
    if fused is None:
        pytest.skip("the checked write-norm kernel is not enabled on this device")
    return fused


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("d, rows", [(D, r) for r in ROWS] + [(TINY_D, r) for r in (188, 512)])
def test_the_write_back_and_norm_in_one_kernel_write_the_old_bytes(d, rows, mode):
    fused = _fuser()
    g = torch.Generator(device=DEV).manual_seed(rows * 13 + mode + d)
    h, scale = _rand((rows, S * d), g), _rand((S * d,), g, dtype=torch.float32)
    inject, branch = _rand((rows, S), g), _rand((rows, d), g)
    y, wts = _rand((rows, SLOTS, d), g), _rand((rows, SLOTS), g, dtype=torch.float32)

    def buffers():
        return (h.clone(), torch.empty((rows, d // 256, S), device=DEV), torch.empty_like(h),
                torch.empty((rows, S * d // 32), device=DEV))

    old, new = buffers(), buffers()
    glue.hc_writeback(old[0], old[0], old[1], S, mode, branch=branch, inject=inject, y=y, wts=wts)
    glue.hc_normed(old[0], old[1], scale, old[2], old[3], S, 1e-6)
    fused(new[0], new[1], scale, new[2], new[3], S, 1e-6, mode, branch=branch, inject=inject, y=y, wts=wts)
    for i, (a, b) in enumerate(zip(old, new)):
        assert _same(a, b), (d, rows, mode, i)


def _hc(g, inject: bool):
    from tensorfold.families.qwen4_exp.cuda.bf16 import b16_from_rows
    from tensorfold.families.qwen4_exp.cuda.weight_types import HC

    down = b16_from_rows(_rand((LOW + (S if inject else 0), S * D), g, 0.02))
    up = b16_from_rows(_rand((S * D, LOW), g, 0.05))
    return HC(down, up, 1.0 + _rand((S * D,), g, 0.1, torch.float32), inject)


def _buffers(rows: int):
    f32 = torch.float32
    return SimpleNamespace(prefill=True, pss=torch.empty((rows, D // 256, S), device=DEV),
                           normed=torch.empty((rows, S * D), dtype=torch.bfloat16, device=DEV),
                           xs_normed=torch.empty((rows, S * D // 32), dtype=f32, device=DEV),
                           act=torch.empty((rows, LOW), dtype=torch.bfloat16, device=DEV),
                           xs_act=torch.empty((rows, LOW // 32), dtype=f32, device=DEV),
                           up=torch.empty((rows, S * D), dtype=torch.bfloat16, device=DEV),
                           mixed=torch.empty((rows, D), dtype=torch.bfloat16, device=DEV),
                           xs_mixed=torch.empty((rows, D // 32), dtype=f32, device=DEV))


@pytest.mark.parametrize("inject", [True, False])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("rows", ROWS)
def test_a_hyper_connection_block_writes_the_separate_kernels_bytes(rows, mode, inject):
    g = torch.Generator(device=DEV).manual_seed(rows * 17 + mode + inject)
    h0 = _rand((rows, S * D), g)
    prev, branch = _rand((rows, S), g), _rand((rows, D), g)
    y, wts = _rand((rows, SLOTS, D), g), _rand((rows, SLOTS), g, dtype=torch.float32)
    seed = rows * 19 + inject
    got = {}
    for fuse in (False, True):
        hc = _hc(torch.Generator(device=DEV).manual_seed(seed), inject)
        b, h, inj = _buffers(rows), h0.clone(), torch.zeros((rows, S), dtype=torch.bfloat16, device=DEV)
        kw = {"y": y, "wts": wts} if mode == 2 else {"branch": branch} if mode == 1 else {}
        if fuse:
            forward.hc_block(hc, b, rows, 1e-6, S, LOW, mode, None if mode == 0 else prev, inj, h, **kw)
        else:                                            # the separate kernels hc_block ran before the fusions
            glue.hc_writeback(h, h, b.pss, S, mode, inject=None if mode == 0 else prev, **kw)
            forward._readout(hc, b, h, rows, 1e-6, S, LOW, inj if inject else None)
        got[fuse] = (h, b.normed, b.act, b.xs_act, b.mixed, inj)
    for i, (a, b) in enumerate(zip(got[False], got[True])):
        assert _same(a, b), (rows, mode, inject, i)


# -- whole prompts and replies ----------------------------------------------------------------------------------


def _hcs(w):
    layers = list(w.layers) + ([w.mtp.layer] if w.mtp is not None else [])
    return [hc for layer in layers for hc in (layer.attn_hc, layer.mlp_hc)]


def _reply(w, prompt, prefill_rows: int, capacity: int):
    """First token, the last row's streams after the prompt, 48 greedy tokens serial, then drafted (with counts)."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    e = Engine(w, capacity=capacity, max_rows=8, prefill_rows=prefill_rows)
    first = prefill(e, prompt, None)
    streams = e.last_streams.clone()
    serial = serial_decode(e, first, 48, None).tokens
    assert prefill(e, prompt, None) == first
    drafted = mtp_decode(e, first, 48, None, depth=3, confidence=0.0)
    out = (first, streams, serial, drafted.tokens, drafted.drafted, drafted.accepted)
    del e
    torch.cuda.empty_cache()
    return out


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    import json

    path = write(tmp_path_factory.mktemp("hcfuse") / "mx", mxfp8=True, ple_nvfp4=True, hidden=512)
    config = json.loads((path / "config.json").read_text())
    config["quantization"]["quantized_layers"] = {
        "model.layers.1.ple.ple_embedding.ngram_embedding": {"quant_algo": "NVFP4", "group_size": 16}}
    (path / "config.json").write_text(json.dumps(config))
    return path


def test_the_tiny_checkpoints_drafted_reply_equals_serial(tiny):
    from tensorfold.families.qwen4_exp.cuda.weights import load

    prompt = torch.randint(1, 256, (700,), generator=torch.Generator().manual_seed(3)).tolist()
    w = load(tiny, mtp=True, draft_vocab=None)
    _, _, serial, drafted, rounds, _ = _reply(w, prompt, 512, 1024)          # chunks of 512 and 188 rows
    assert drafted == serial and rounds > 0


FP4_LOCAL = os.environ.get("TF_FP4_LOCAL")


@pytest.mark.skipif(not FP4_LOCAL, reason="TF_FP4_LOCAL names FP4-Local's directory")
def test_fp4_locals_drafted_reply_equals_serial():
    from tensorfold.families.qwen4_exp.cuda.weights import load

    w = load(Path(FP4_LOCAL), mtp=True, draft_vocab=None)
    hcs = _hcs(w)
    assert len(hcs) == 98 and all(hc.down.n == LOW + S and hc.up.n == S * D for hc in hcs)
    prompt = torch.randint(10, 150_000, (4500,), generator=torch.Generator().manual_seed(8)).tolist()
    _, _, serial, drafted, rounds, _ = _reply(w, prompt, 2048, 8192)         # chunks of 2,048, 2,048 and 404 rows
    assert drafted == serial and rounds > 0
