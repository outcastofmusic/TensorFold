"""GLM packed projections read and write rows where they lie when the kernel takes the layout, and decode skips the
group sums packed projections never read. CPU only: the EXL3 and affine kernels are fakes that record what they were
handed (tests/cuda/test_glm_packed_in_place.py compares the real kernels' bytes)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.cuda.affine6 import affine6  # noqa: E402
from tensorfold.cuda.affine6.affine6 import Affine6  # noqa: E402
from tensorfold.cuda.exl3.linear import Exl3Linear  # noqa: E402
from tensorfold.families.glm5_next.cuda import forward, packed  # noqa: E402
from tensorfold.families.glm5_next.cuda.packed import ComposedLinear, PlainLinear, ProjectionScratch, matmul  # noqa: E402

SENTINEL = -7.0


def _exl3(k: int, n: int) -> Exl3Linear:
    tr = torch.zeros(k // 16, n // 16, 64, dtype=torch.int16)
    return Exl3Linear.from_tensors(tr, torch.ones(k, dtype=torch.float16), torch.ones(n, dtype=torch.float16), "mul1",
                                   device="cpu", layout="stored")


def _weight(k: int, n: int, seed: int) -> torch.Tensor:
    return torch.randn((k, n), generator=torch.Generator().manual_seed(seed))


class Kernels:
    """Fake EXL3 and affine kernels: y = x @ W in fp32, rounded to y's dtype, and the tensors they were handed."""

    def __init__(self, monkeypatch):
        self.seen: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        self.weights: dict[int, torch.Tensor] = {}
        kernels = self

        def exl3(q, x, out=None, xh=None, z=None, out_dtype=None):
            assert x.is_contiguous() and out.is_contiguous()     # what the real kernel checks
            kernels.seen.append(("exl3", x, out))
            out.copy_(x.float() @ kernels.weights[id(q)])
            return out

        def a6(x, q, out=None, f32=False, xs=None):
            assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and out.is_contiguous()
            assert out.dtype in (torch.bfloat16, torch.float32)
            kernels.seen.append(("affine6", x, out))
            out.copy_(x.float() @ kernels.weights[id(q)])
            return out

        monkeypatch.setattr(Exl3Linear, "__call__", exl3)
        monkeypatch.setattr(affine6, "matmul", a6)

    def exl3(self, k: int, n: int, seed: int) -> Exl3Linear:
        q = _exl3(k, n)
        self.weights[id(q)] = _weight(k, n, seed)
        return q

    def affine(self, k: int, n: int, seed: int) -> Affine6:
        q = Affine6(torch.zeros(1, dtype=torch.int32), torch.zeros(k // 128, n, dtype=torch.bfloat16),
                    torch.zeros(k // 128, n, dtype=torch.bfloat16), n, k)
        self.weights[id(q)] = _weight(k, n, seed)
        return q


def _scratch(qs, rows=4) -> ProjectionScratch:
    s = ProjectionScratch(qs, rows, "cpu")
    for t in (*s.inputs.values(), *s.outputs.values()):
        t.fill_(SENTINEL)
    return s


def _untouched(s: ProjectionScratch) -> bool:
    return all(bool((t == SENTINEL).all()) for t in (*s.inputs.values(), *s.outputs.values()))


def _inside(t: torch.Tensor, owners) -> bool:
    return any(o.data_ptr() <= t.data_ptr() < o.data_ptr() + o.numel() * o.element_size() for o in owners)


def _both(monkeypatch, run):
    """run() under the in-place path and under TF_GLM_PACKED_COPIES=1: its two results."""

    monkeypatch.setattr(packed, "COPIES", False)
    new = run()
    monkeypatch.setattr(packed, "COPIES", True)
    old = run()
    monkeypatch.setattr(packed, "COPIES", False)
    return new, old


def test_a_decode_row_of_a_composed_projection_touches_no_scratch(monkeypatch):
    """One row: the composed output's column slices are contiguous, so every part reads x and writes out in place."""

    k = Kernels(monkeypatch)
    q = ComposedLinear([k.exl3(256, 384, 1), PlainLinear(torch.randn(128, 256)), k.exl3(256, 128, 2)])
    s = _scratch([q])
    x = torch.randn(1, 256).to(torch.bfloat16)
    out = torch.full((1, q.n), float("nan"), dtype=torch.bfloat16)
    matmul(q, x, out, s)
    assert _untouched(s)
    assert [(kind, a.data_ptr(), b.data_ptr()) for kind, a, b in k.seen] == [
        ("exl3", x.data_ptr(), out.data_ptr()), ("exl3", x.data_ptr(), out[:, 512:].data_ptr())]
    assert not bool(out.isnan().any())


def test_layouts_the_exl3_kernel_cannot_take_go_through_the_scratch(monkeypatch):
    """Several rows of a composed slice, a column-sliced input and a misaligned one are staged; values agree."""

    k = Kernels(monkeypatch)
    q = ComposedLinear([k.exl3(256, 384, 1), PlainLinear(torch.randn(128, 256))])
    s = _scratch([q], rows=8)
    wide = torch.randn(3, 512).to(torch.bfloat16)
    cases = {
        "composed rows": (torch.randn(3, 256).to(torch.bfloat16), (False, True)),
        "column slice": (wide[:, 128:384], (True, True)),
        "misaligned": (torch.randn(3 * 256 + 1).to(torch.bfloat16)[1:].view(3, 256), (True, True)),
    }
    for name, (x, (staged_in, staged_out)) in cases.items():
        k.seen.clear()
        new, old = _both(monkeypatch, lambda x=x: matmul(q, x, torch.full((3, q.n), SENTINEL, dtype=torch.float32), s))
        assert torch.equal(new, old), name
        _, xi, yi = k.seen[0]                 # the in-place run's EXL3 call
        assert _inside(xi, s.inputs.values()) == staged_in, name
        assert _inside(yi, s.outputs.values()) == staged_out, name


def test_plain_reads_strided_rows_and_writes_slices_in_place(monkeypatch):
    """The plain kernel takes any row stride: a column-sliced input into a column slice of the output, no scratch."""

    q = ComposedLinear([PlainLinear(torch.randn(64, 128, dtype=torch.float16)), PlainLinear(torch.randn(32, 128))])
    s = _scratch([q])
    x = torch.randn(3, 256).to(torch.bfloat16)[:, 128:]
    monkeypatch.setattr(packed, "COPIES", False)
    new = matmul(q, x, torch.empty((3, q.n), dtype=torch.float32), s)
    assert _untouched(s)
    monkeypatch.setattr(packed, "COPIES", True)
    old = matmul(q, x, torch.empty((3, q.n), dtype=torch.float32), s)
    assert not _untouched(s)                  # the copies path staged its rows
    assert torch.equal(new, old)
    want = torch.cat([x.float() @ q.parts[0].weight.float().T, x.float() @ q.parts[1].weight.T], dim=1)
    assert torch.equal(new, want)


def test_affine_converts_other_dtypes_once_and_writes_in_place(monkeypatch):
    """An fp16 input converts straight into the bf16 scratch (its own-dtype scratch untouched), an fp16 output straight
    from the fp32 scratch; bf16 rows with an even stride are read where they lie."""

    k = Kernels(monkeypatch)
    q = k.affine(256, 128, 3)
    s = _scratch([q])
    x16 = torch.randn(2, 256).to(torch.float16)
    out16 = torch.empty((2, 128), dtype=torch.float16)
    matmul(q, x16, out16, s)
    assert bool((s.inputs[torch.float16] == SENTINEL).all()) and bool((s.outputs[torch.float16] == SENTINEL).all())
    _, xb, yo = k.seen[-1]
    assert _inside(xb, [s.inputs[torch.bfloat16]]) and _inside(yo, [s.outputs[torch.float32]])
    monkeypatch.setattr(packed, "COPIES", True)
    assert torch.equal(matmul(q, x16, torch.empty_like(out16), s), out16)
    monkeypatch.setattr(packed, "COPIES", False)

    s = _scratch([q])
    x = torch.randn(2, 512).to(torch.bfloat16)[:, 256:]
    out = torch.empty((2, 128), dtype=torch.float32)
    matmul(q, x, out, s)
    assert _untouched(s)
    assert (k.seen[-1][1].data_ptr(), k.seen[-1][2].data_ptr()) == (x.data_ptr(), out.data_ptr())


def test_an_output_overlapping_the_input_is_staged(monkeypatch):
    """When out shares bytes with x, every chunk reads its rows before any is written: the copies path."""

    k = Kernels(monkeypatch)
    q = k.exl3(128, 128, 4)
    s = _scratch([q], rows=256)
    buf = torch.randn(200, 128)
    want = (buf.float() @ k.weights[id(q)])
    matmul(q, buf, buf, s)
    assert torch.equal(buf, want)
    assert not _untouched(s)


def test_chunks_past_128_rows_run_in_place(monkeypatch):
    k = Kernels(monkeypatch)
    q = k.exl3(128, 256, 5)
    s = _scratch([q], rows=300)
    x = torch.randn(300, 128).to(torch.bfloat16)
    new, old = _both(monkeypatch, lambda: matmul(q, x, torch.empty((300, 256), dtype=torch.bfloat16), s))
    assert torch.equal(new, old)
    rows = [(a.shape[0], a.data_ptr() == x[i * 128:].data_ptr()) for i, (_, a, _) in enumerate(k.seen[:3])]
    assert rows == [(128, True), (128, True), (44, True)]


# -- group sums ----------------------------------------------------------------------------------------------------
class Sums:
    def __init__(self, monkeypatch):
        self.calls = 0

        def count(x, out=None):
            self.calls += 1
            return out

        monkeypatch.setattr(forward.qmm, "group_sums", count)


def _kda_fixture(R=2):
    proj = PlainLinear(torch.zeros(3 * 128 + 128 + 128 + 64, 64))
    kda = NS(proj=proj, fb=PlainLinear(torch.zeros(256, 128)), gb=PlainLinear(torch.zeros(256, 128)),
             o=PlainLinear(torch.zeros(64, 256)), fa_off=384, ga_off=512, b_off=640, conv=None, a_log=None,
             dt_bias=None, norm=None)
    layer = NS(kda=kda, index=0)
    w = NS(cfg=NS(eps=1e-5, lower=-5.0), comm=None)
    st = NS(kda_index={0: 0}, proj=torch.zeros(1, R, proj.n), cur=[0], conv=[None], rec=torch.zeros(2, 1, 1),
            scratch=[None])
    b = NS(prefill=False, normed=torch.zeros(R, 64), xs=torch.zeros(R, 1), xs_fa=torch.zeros(R, 2),
           xs_ga=torch.zeros(R, 2), ka=torch.zeros(R, 256), kg=torch.zeros(R, 256), kxs=torch.zeros(R, 4),
           part=torch.zeros(R, 64), world=1)
    return layer, w, st, b


@pytest.mark.parametrize("copies", [False, True], ids=["in-place", "copies"])
def test_kda_decode_skips_the_group_sums_packed_projections_ignore(monkeypatch, copies):
    sums = Sums(monkeypatch)
    handed = []
    monkeypatch.setattr(forward, "mm", lambda b, x, q, xs, out, f32=False: handed.append((q, xs)) or out)
    monkeypatch.setattr(forward.kda_mod, "chain", lambda p, *a: torch.zeros(p.shape[0], 256))
    monkeypatch.setattr(packed, "COPIES", copies)
    layer, w, st, b = _kda_fixture()
    forward.kda_block(layer, w, st, b, 2)
    assert sums.calls == (3 if copies else 0)
    k = layer.kda
    assert [q for q, _ in handed] == [k.proj, k.fb, k.gb, k.o]
    assert [xs is None for _, xs in handed[1:]] == [not copies] * 3


def test_latent_attention_skips_the_out_projection_group_sums(monkeypatch):
    sums = Sums(monkeypatch)
    handed = []
    monkeypatch.setattr(packed, "COPIES", False)
    monkeypatch.setattr(forward, "mm", lambda b, x, q, xs, out, f32=False: handed.append((q, xs)) or out)
    monkeypatch.setattr(forward.latent, "latent_write", lambda *a: None)
    monkeypatch.setattr(forward.latent, "absorb_q", lambda q, absorb, out: out)
    monkeypatch.setattr(forward, "dense_attention", lambda *a, **kw: None)
    monkeypatch.setattr(forward.latent, "expand_v", lambda ol, absorb, out: out)
    monkeypatch.setattr(forward.glue, "rmsnorm", lambda *a: None)
    R, o = 2, PlainLinear(torch.zeros(64, 128))
    a = NS(heads=2, o=o, absorb=None, proj=object(), q_b=object(), q_norm=None, kv_norm=None)
    w = NS(cfg=NS(dense_limit=2048, qk_dim=64, v_dim=64, q_lora=32, eps=1e-5), comm=None)
    b = NS(lat=torch.zeros(R, 32), lat_s=NS(qa=torch.zeros(R, 2, 32), ol=torch.zeros(R, 2, 32), nch=1),
           q=torch.zeros(R, 2, 64), vn=torch.zeros(R, 2, 64), xs_ao=torch.zeros(R, 2), part=torch.zeros(R, 64),
           normed=torch.zeros(R, 64), xs=torch.zeros(R, 1), dp=torch.zeros(R, 64), qr=torch.zeros(R, 32),
           xs_qr=torch.zeros(R, 1), xs_lat=torch.zeros(R, 1), world=1, prefill=False)
    forward.dsa_block(NS(dsa=a), w, None, None, b, R, 1, None, 0)
    assert sums.calls == 0
    assert handed[-1] == (o, None)


def test_unpacked_projections_keep_their_group_sums(monkeypatch):
    sums = Sums(monkeypatch)
    monkeypatch.setattr(packed, "COPIES", False)
    out = torch.zeros(2, 2)
    assert forward.group_sums(object(), torch.zeros(2, 128), out) is out and sums.calls == 1
    assert forward.group_sums(PlainLinear(torch.zeros(4, 128)), torch.zeros(2, 128), out) is None and sums.calls == 1
