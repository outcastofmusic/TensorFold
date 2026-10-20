"""TF_FLASHNEXT_F16_NARROW on the CPU: the flag's values, where the engine and loader read it, which tile each fp16
matrix of an EXL3 pack takes at which rows, and (compiled for sm_121 without a GPU) that the narrow tiles keep each
output's chain of MMA steps. tests/cuda/test_flashnext_f16_narrow.py compares the bytes on the GPU."""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda import exl3_mm

ENV = exl3_mm.NARROW_ENV

# Flash Next EXL3's fp16 matrices [n, k]: HC down with the inject rows, HC up, the mixer's down, DeltaNet's b/a, and
# the PLE key and value projections
SHAPES = [(324, 10240), (10240, 320), (320, 10240), (96, 2560), (10240, 2560), (2560, 2560)]


@pytest.mark.parametrize("value, on", [(None, False), ("", False), ("1", True), ("on", True), ("0", False),
                                       ("off", False), ("false", False), (" NO ", False)])
def test_the_flag_is_off_unless_turned_on(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert exl3_mm.f16_narrow() is on


@pytest.mark.parametrize("value", ["2", "narrow", "o", "1.0"])
def test_a_flag_typo_stops_the_engine_and_the_loaders_before_any_read(monkeypatch, tmp_path, value):
    from tensorfold.families.qwen4_exp.cuda import exl3
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.weights import load

    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=ENV):
        FlashNextEngine(tmp_path / "missing")                      # no checkpoint there: the flag fails first
    with pytest.raises(ValueError, match=ENV):
        load(tmp_path / "missing", "cpu")
    with pytest.raises(ValueError, match=ENV):
        exl3.load(tmp_path / "missing", "cpu")


def test_the_scratch_keeps_what_it_is_given():
    """No default: the loader passes the flag it read (unset: off), so no caller gets narrow tiles by omission."""

    assert exl3_mm.Scratch(11, True).narrow is True
    assert exl3_mm.Scratch(11, False).narrow is False


@pytest.mark.parametrize("n, k", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 4, 7, 8, 16, 17, 128])
def test_one_m_tile_takes_a_narrow_tile_of_sixteen_columns_a_warp(n, k, m):
    sk = exl3_mm.f16_split(n, k)
    bn, warps, stages = exl3_mm.narrow_tile(m, n, sk)
    if m > exl3_mm.NARROW_ROWS:
        assert (bn, warps, stages) == exl3_mm.WIDE_TILE
        return
    assert bn // warps == 16 and bn < 64
    assert -(-n // bn) * sk > -(-n // 64) * sk                        # more programs than BN 64


def test_the_narrow_programs_of_each_matrix():
    got = {(n, k): (exl3_mm.f16_split(n, k), exl3_mm.narrow_tile(1, n, exl3_mm.f16_split(n, k))[0]) for n, k in SHAPES}
    assert got == {(324, 10240): (16, 32), (10240, 320): (1, 32), (320, 10240): (32, 32), (96, 2560): (8, 16),
                   (10240, 2560): (1, 32), (2560, 2560): (4, 32)}


class Spy:
    """A Triton kernel stand-in: ``kernel[grid](*args, **kw)`` records the launch."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def call(*args, **kw):
            self.calls.append((tuple(grid), kw))
        return call


def _f16(n, k, narrow):
    sc = exl3_mm.Scratch(11, narrow)
    sc.part = torch.empty((exl3_mm.f16_split(n, k) * exl3_mm.ROWS * n,), dtype=torch.float32)
    return exl3_mm.F16(torch.empty((n, k), dtype=torch.float16), n, k, exl3_mm.f16_split(n, k), sc)


@pytest.mark.parametrize("n, k", SHAPES)
@pytest.mark.parametrize("m", [1, 7, 16, 17, 128])
@pytest.mark.parametrize("narrow", [True, False])
@pytest.mark.parametrize("path", ["call", "partials"])
def test_a_call_launches_the_tile_the_scratch_switch_picks(monkeypatch, n, k, m, narrow, path):
    mm, red = Spy(), Spy()
    # the split path's tiles; TF_FLASHNEXT_F16_INNER_K's one launch is tests/test_flashnext_exl3_sweep2.py's
    monkeypatch.setattr(exl3_mm, "_current", exl3_mm.Switches(inner_k=False, moe_chunk=False))
    monkeypatch.setattr(exl3_mm, "_f16_mm", mm)
    monkeypatch.setattr(exl3_mm, "_reduce", red)
    q = _f16(n, k, narrow)
    x = torch.empty((m, k), dtype=torch.bfloat16)
    if path == "call":
        q(x, torch.empty((m, n), dtype=torch.bfloat16))
    else:
        q.partials(x)
    bn, warps, stages = exl3_mm.narrow_tile(m, n, q.sk) if narrow else exl3_mm.WIDE_TILE
    assert len(mm.calls) == 1
    grid, kw = mm.calls[0]
    assert grid == (-(-m // 16), -(-n // bn), q.sk)
    assert (kw["BN"], kw["num_warps"], kw["num_stages"]) == (bn, warps, stages)
    assert (kw["BM"], kw["BK"], kw["KS"]) == (16, 64, k // q.sk)       # the K slices and steps of every tile
    assert kw["F32"] is (path == "partials" or q.sk > 1)
    assert len(red.calls) == (1 if path == "call" and q.sk > 1 else 0)


# -- the MMA chains behind the bytes ----------------------------------------------------------------------------------
SIG = {"X": "*bf16", "W": "*fp16", "OUT": "*fp32", "M": "i32", "N": "i32", "x_stride": "i32", "o_stride": "i32",
       "K": "constexpr", "KS": "constexpr", "BM": "constexpr", "BN": "constexpr", "BK": "constexpr",
       "F32": "constexpr"}
MMA = re.compile(r"mma\.sync\.aligned\.(\S+) \{ ([^}]*) \}, \{ ([^}]*) \}, \{ ([^}]*) \}, \{ ([^}]*) \};")


def _compile(k, ks, bn, warps, stages, f32=True):
    """TTGIR and PTX of ``_f16_mm`` for sm_121 (GB10), with the 16-byte alignment the launches specialize on."""

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile

    names = list(SIG)
    attrs = {(names.index(a),): [["tt.divisibility", 16]] for a in names if SIG[a].startswith("*")}
    src = ASTSource(fn=exl3_mm._f16_mm, signature={**SIG, "OUT": "*fp32" if f32 else "*bf16"},
                    constexprs={"K": k, "KS": ks, "BM": 16, "BN": bn, "BK": 64, "F32": f32}, attrs=attrs)
    got = compile(src, target=GPUTarget("cuda", 121, 32), options={"num_warps": warps, "num_stages": stages})
    return got.asm["ttgir"], got.asm["ptx"]


def _chain(ptx):
    """A warp's MMAs of one BK step as (accumulator, A fragment) in issue order, each accumulating in place."""

    accs, frags, out = {}, {}, []
    for kind, d, a, _b, c in MMA.findall(ptx):
        assert kind == "m16n8k16.row.col.f32.f16.f16.f32" and d == c
        out.append((accs.setdefault(d, len(accs)), frags.setdefault(a, len(frags))))
    return out


@pytest.mark.parametrize("n, k", SHAPES)
@pytest.mark.parametrize("f32", [True, False])
def test_the_narrow_tile_keeps_each_outputs_mma_chain(n, k, f32):
    """BN 64 on four warps and the narrow tiles both give a warp 16 columns as two n8 accumulators, each taking the
    BK step's four k16 fragments in order in place: an output's sum is the same chain of m16n8k16 steps, so a Triton
    upgrade that changes the layout fails here before it changes a reply."""

    sk = exl3_mm.f16_split(n, k)
    bn, warps, stages = exl3_mm.narrow_tile(1, n, sk)
    old_ir, old = _compile(k, k // sk, *exl3_mm.WIDE_TILE, f32=f32)
    new_ir, new = _compile(k, k // sk, bn, warps, stages, f32=f32)
    assert "warpsPerCTA = [1, 4], instrShape = [16, 8]" in old_ir
    assert f"warpsPerCTA = [1, {warps}], instrShape = [16, 8]" in new_ir
    want = [(acc, frag) for frag in range(4) for acc in range(2)]
    assert _chain(old) == want and _chain(new) == want
