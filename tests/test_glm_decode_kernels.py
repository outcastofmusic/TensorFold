"""TF_GLM_PLAIN_NARROW: the switch, which calls each kernel takes, and the thread layouts that keep the old bytes. CPU only: the kernels are spies that record their launches, and the layout checks compile the
Triton kernels for sm_121 without a GPU (tests/cuda/test_glm_decode_shapes.py compares the bytes on the GPU)."""

from __future__ import annotations

import re
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.torch

from tensorfold.families.glm5_next.cuda import engine, forward, packed, prefill


class Spy:
    """A Triton kernel stand-in: ``kernel[grid](*args, **kw)`` records the launch."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def call(*args, **kw):
            self.calls.append((tuple(grid), kw))
        return call


# -- the switches -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("env,read", [("TF_GLM_PLAIN_NARROW", packed.plain_narrow)])
def test_each_switch_is_on_unless_set_off_and_a_typo_raises(monkeypatch, env, read):
    monkeypatch.delenv(env, raising=False)
    assert read() is True
    for value, want in (("0", False), ("off", False), ("1", True), ("on", True), ("", True)):
        monkeypatch.setenv(env, value)
        assert read() is want, value
    monkeypatch.setenv(env, "maybe")
    with pytest.raises(ValueError, match=env):
        read()


def test_the_engine_reads_the_switch_into_the_weights_meta(monkeypatch):
    monkeypatch.setenv("TF_GLM_PLAIN_NARROW", "0")
    assert engine.kernel_switches() == {"plain_narrow": False}
    monkeypatch.delenv("TF_GLM_PLAIN_NARROW")
    assert engine.kernel_switches() == {"plain_narrow": True}


def _weights(meta):
    c = NS(hidden=128, streams=4, heads=2, lin_heads=2, kv_lora=128, q_lora=128, qk_dim=128, v_dim=128,
           index_dim=128, index_heads=2, dense_width=256, top_k=1, moe_width=128, shared_width=128, experts=2,
           quant="exl3", mtp_layers=0)
    head = packed.PlainLinear(torch.ones(256, 128, dtype=torch.float16))
    return NS(cfg=c, device="cpu", world=1, meta={"packed_exl3": True, **meta}, layers=[], head=head, mtp=None)


@pytest.mark.parametrize("meta,narrow", [({}, True), ({"plain_narrow": False}, False), ({"plain_narrow": True}, True)])
def test_buffers_take_the_switch_from_the_meta_and_default_on(monkeypatch, meta, narrow):
    monkeypatch.setattr(packed, "projections", lambda w: [w.head])
    monkeypatch.setattr(forward.latent, "ENABLED", False)
    for prefill_rows in (False, True):
        b = forward.Buffers(_weights(meta), 4, capacity=16, prefill=prefill_rows)
        assert b.projection_scratch.narrow is narrow


# -- plain projections ---------------------------------------------------------------------------------------------
# GLM-5.3 Flash's plain fp16 projections [N, K]: f_a and g_a, b_proj, the router gate, the indexer's wk, weights_proj
# and kpool gate, f_b and g_b, and kv_b's key and value blocks (TF_GLM_LATENT=0)
PLAIN = [(128, 4096), (64, 4096), (288, 4096), (32, 4096), (8192, 128), (16384, 512)]


@pytest.mark.parametrize("n,k", PLAIN)
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 8, 9, 128])
@pytest.mark.parametrize("on", [True, False])
def test_decode_rows_of_a_multi_block_k_take_the_narrow_kernel(monkeypatch, n, k, rows, on):
    old, new = Spy(), Spy()
    monkeypatch.setattr(prefill, "_plain", old)
    monkeypatch.setattr(prefill, "_plain_narrow", new)
    w = torch.empty((n, k), dtype=torch.float16)
    prefill.plain_matmul(w, torch.empty((rows, k), dtype=torch.bfloat16), torch.empty((rows, n)), on)
    narrow = on and rows <= 8 and k >= 256
    assert (len(new.calls), len(old.calls)) == ((1, 0) if narrow else (0, 1))
    if narrow:
        grid, kw = new.calls[0]
        assert grid == (rows, n // 2) and kw["BN"] == 2 and kw["BK"] == 128 and kw["num_warps"] == 1
        assert kw["U"] == {4096: 8, 512: 4}[k] and (k // 128) % kw["U"] == 0
    else:
        grid, kw = old.calls[0]
        assert grid == (rows, -(-n // 16)) and kw["BN"] == 16 and kw["num_warps"] == 4


def test_k_blocks_a_trip_divide_the_k_blocks_and_other_weights_stay_on_the_old_kernel():
    assert [prefill.narrow_blocks(1, 128 * nk, torch.float16) for nk in (2, 3, 4, 6, 12, 32, 40)] == [2, 1, 4, 2, 4, 8,
                                                                                                      8]
    assert prefill.narrow_blocks(1, 4096, torch.bfloat16) == 8
    assert prefill.narrow_blocks(1, 4096, torch.float32) == 0          # 4 values a thread there, not 8
    assert prefill.narrow_blocks(1, 128, torch.float16) == 0
    assert prefill.narrow_blocks(prefill.NARROW_ROWS + 1, 4096, torch.float16) == 0


@pytest.mark.parametrize("narrow", [True, False])
def test_a_projection_passes_the_scratch_switch_to_the_kernel(monkeypatch, narrow):
    seen = []
    monkeypatch.setattr(packed, "_plain_cuda", lambda q, x, out, n: seen.append(n))
    q = packed.PlainLinear(torch.zeros(64, 256, dtype=torch.float16))
    s = packed.ProjectionScratch([q], 4, "cpu", narrow=narrow)
    packed.matmul(q, _Cudaish(torch.zeros((2, 256), dtype=torch.bfloat16)), torch.empty((2, 64)), s)
    assert seen == [narrow]


class _Cudaish(torch.Tensor):
    """A CPU tensor that reports a CUDA device, so ``packed.matmul`` takes its GPU branch (the kernel is a spy)."""

    @staticmethod
    def __new__(cls, t):
        return torch.Tensor._make_subclass(cls, t)

    @property
    def device(self):
        return torch.device("cuda")


# -- the thread layouts behind the bytes ----------------------------------------------------------------------------
def _compile(fn, signature, constexprs, num_warps):
    """TTGIR and PTX for sm_121 (GB10), with the 16-byte alignment the launches specialize on."""

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile

    names = list(signature)
    attrs = {(names.index(a),): [["tt.divisibility", 16]] for a in names if signature[a].startswith("*")}
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    k = compile(src, target=GPUTarget("cuda", 121, 32), options={"num_warps": num_warps})
    return k.asm["ttgir"], k.asm["ptx"]


def _butterflies(ptx):
    return {int(x) for x in re.findall(r"shfl\.sync\.bfly\.b32\s+%r\d+, %r\d+, (\d+),", ptx)}


@pytest.mark.parametrize("w_dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("k,u", [(4096, 8), (512, 4)])
def test_the_narrow_kernel_lays_k_out_as_the_old_one_does(w_dtype, k, u):
    """Both kernels reduce a K block of 128 as 8 values a thread and 16 threads, never across warps, so its sum keeps
    its tree; a Triton upgrade that changes either layout fails here before it changes a reply."""

    sig = {"X": "*bf16", "W": "*" + w_dtype, "Y": "*fp32", "M": "i32", "xs": "i32", "ys": "i32", "N": "constexpr",
           "K": "constexpr", "BK": "constexpr", "BN": "constexpr"}
    old, _ = _compile(prefill._plain, sig, {"N": 288, "K": k, "BK": 128, "BN": 16}, 4)
    new, ptx = _compile(prefill._plain_narrow, {**sig, "U": "constexpr"},
                        {"N": 288, "K": k, "BK": 128, "BN": 2, "U": u}, 1)
    assert "sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1]" in old
    assert "sizePerThread = [1, 1, 8], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 1, 1]" in new
    assert {8, 4, 2, 1} <= _butterflies(ptx)
