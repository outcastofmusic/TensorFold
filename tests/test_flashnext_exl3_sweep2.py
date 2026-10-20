"""The EXL3 pack's matmul switches on the CPU: TF_FLASHNEXT_STACK_DIRECT, TF_FLASHNEXT_F16_INNER_K and
TF_FLASHNEXT_MOE_DIRECT (their values, where the engine reads them, what each call launches with them on or off),
TF_FLASHNEXT_HC_FUSE on the pack's fp16 hyper-connections, and (compiled for sm_121 without a GPU) that
``_f16_mm_ks`` keeps ``_f16_mm``'s chain of MMA steps and adds its slices after their loops.
tests/cuda/test_exl3_sweep2_kernels.py and test_exl3_sweep2_replies.py compare the bytes on the GPU."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda import exl3_mm, forward  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weight_types import HC  # noqa: E402

from tests.test_flashnext_f16_narrow import SHAPES, Spy, _chain  # noqa: E402

ENVS = (exl3_mm.STACK_DIRECT_ENV, exl3_mm.INNER_K_ENV, exl3_mm.MOE_DIRECT_ENV)
FIELDS = dict(zip(ENVS, ("stack_direct", "inner_k", "moe_direct")))
DEFAULTS = dict(zip(ENVS, (True, False, True)))      # inner_k off: 0.994x prefill served (19 Oct)


@pytest.fixture(autouse=True)
def restore():
    before = exl3_mm.current()
    yield
    exl3_mm.use(before)


# -- the flags ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("env", ENVS)
@pytest.mark.parametrize("value, on", [(None, True), ("", True), ("1", True), ("on", True), ("0", False),
                                       ("off", False), ("false", False), (" NO ", False)])
def test_each_flag_takes_its_default_unless_set(monkeypatch, env, value, on):
    for e in ENVS:
        monkeypatch.delenv(e, raising=False)
    if value is not None:
        monkeypatch.setenv(env, value)
    got = exl3_mm.read()
    assert getattr(got, FIELDS[env]) is (DEFAULTS[env] if value in (None, "") else on)
    assert all(getattr(got, f) is DEFAULTS[e] for e, f in FIELDS.items() if e != env)


@pytest.mark.parametrize("env", ENVS)
@pytest.mark.parametrize("value", ["2", "direct", "o", "1.0"])
def test_a_flag_typo_stops_the_engine_before_any_read(monkeypatch, tmp_path, env, value):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError, match=env):
        FlashNextEngine(tmp_path / "missing")


def test_the_engine_installs_what_it_reads(monkeypatch, tmp_path):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    for e in exl3_mm.ENVS:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv(exl3_mm.STACK_DIRECT_ENV, "0")
    monkeypatch.setenv(exl3_mm.MOE_DIRECT_ENV, "off")
    exl3_mm.use(exl3_mm.Switches(moe_chunk=False))
    with pytest.raises(Exception):
        FlashNextEngine(tmp_path / "missing")                      # fails at the checkpoint, after the switches
    assert exl3_mm.current() == exl3_mm.Switches(stack_direct=False, inner_k=False, moe_direct=False, moe_chunk=True)


# -- a stack's parts write their columns in place ------------------------------------------------------------------
class Part:
    """A stack part that records where it wrote: the output's data pointer, shape and row stride."""

    def __init__(self, n, log):
        self.n, self.log = n, log

    def __call__(self, x, out):
        self.log.append((out.data_ptr(), tuple(out.shape), out.stride(0)))
        out.fill_(float(self.n))
        return out


def _stack(widths):
    log: list = []
    sc = exl3_mm.Scratch(11)
    sc.tmp = torch.zeros((exl3_mm.ROWS * max(widths),), dtype=torch.bfloat16)
    return exl3_mm.Stack([Part(n, log) for n in widths], sc), log


@pytest.mark.parametrize("widths", [(12288, 512, 512, 640), (10240, 6144, 96)])   # attention, DeltaNet
@pytest.mark.parametrize("m", [2, 3, 8, 16, 28, 129])
def test_with_the_switch_each_part_writes_its_columns_in_place(widths, m):
    s, log = _stack(widths)
    out = torch.zeros((m, sum(widths)), dtype=torch.bfloat16)
    exl3_mm.use(exl3_mm.Switches(stack_direct=True, moe_chunk=False))
    s(torch.zeros((m, 8)), out)
    at = 0
    for (ptr, shape, stride), n in zip(log, widths):
        assert (ptr, shape, stride) == (out.data_ptr() + at * 2, (m, n), sum(widths))
        assert torch.all(out[:, at:at + n] == float(n))
        at += n


@pytest.mark.parametrize("m", [2, 8, 28])
def test_without_the_switch_rows_go_to_scratch_and_are_copied(m):
    widths = (12288, 512, 512, 640)
    s, log = _stack(widths)
    out = torch.zeros((m, sum(widths)), dtype=torch.bfloat16)
    exl3_mm.use(exl3_mm.Switches(stack_direct=False, moe_chunk=False))
    s(torch.zeros((m, 8)), out)
    assert {ptr for ptr, _, _ in log} == {s.sc.tmp.data_ptr()}
    at = 0
    for n in widths:
        assert torch.all(out[:, at:at + n] == float(n))
        at += n


def test_one_row_writes_in_place_either_way():
    widths = (10240, 6144, 96)
    for on in (True, False):
        s, log = _stack(widths)
        out = torch.zeros((1, sum(widths)), dtype=torch.bfloat16)
        exl3_mm.use(exl3_mm.Switches(stack_direct=on, moe_chunk=False))
        s(torch.zeros((1, 8)), out)
        assert log[1][0] == out.data_ptr() + 10240 * 2


@pytest.mark.parametrize("widths, ok", [((10240, 6144, 96), True), ((12288, 512, 512, 640), True),
                                        ((96, 10240), True), ((100, 10240), False), ((10240, 6143, 96), False)])
def test_a_part_that_would_start_off_16_bytes_falls_back_to_the_copies(widths, ok):
    s, _ = _stack(widths)
    width = sum(widths) + (-sum(widths)) % 8                      # a row stride of whole 16-byte words
    out = torch.zeros((4, width), dtype=torch.bfloat16)[:, :sum(widths)]
    assert s.aligned(out) is ok
    assert s.aligned(torch.zeros((4, width + 1), dtype=torch.bfloat16)[:, :sum(widths)]) is False


# -- the prompt's fp16 matmuls sum their K slices in-program -----------------------------------------------------------
@pytest.mark.parametrize("n, k", SHAPES)
@pytest.mark.parametrize("m", [1, 16, 17, 47, 48, 255, 256, 767, 768, 2048, 2050])
def test_which_calls_fit_the_in_program_sum(n, k, m):
    sk = exl3_mm.f16_split(n, k)
    want = sk > 1 and m > 16 and -(-m // 16) * -(-n // 64) >= 96
    assert exl3_mm.inner_k_fits(m, n, k, sk) is want
    if want:
        assert k // sk // 64 >= 2                                 # every slice two K blocks or more (playbook 6.9)


def test_the_programs_each_matrix_needs():
    first = {}
    for n, k in SHAPES:
        sk = exl3_mm.f16_split(n, k)
        first[(n, k)] = next((m for m in range(17, 4096) if exl3_mm.inner_k_fits(m, n, k, sk)), None)
    assert first == {(324, 10240): 241, (10240, 320): None, (320, 10240): 305, (96, 2560): 753,
                     (10240, 2560): None, (2560, 2560): 33}


def _f16(n, k):
    sc = exl3_mm.Scratch(11, True)
    sc.part = torch.empty((exl3_mm.f16_split(n, k) * exl3_mm.ROWS * n,), dtype=torch.float32)
    return exl3_mm.F16(torch.empty((n, k), dtype=torch.float16), n, k, exl3_mm.f16_split(n, k), sc)


@pytest.mark.parametrize("n, k", SHAPES)
@pytest.mark.parametrize("m", [8, 17, 256, 2048, 2050])
@pytest.mark.parametrize("on", [True, False])
def test_a_call_launches_the_in_program_sum_where_it_fits(monkeypatch, n, k, m, on):
    mm, ks, red = Spy(), Spy(), Spy()
    monkeypatch.setattr(exl3_mm, "_f16_mm", mm)
    monkeypatch.setattr(exl3_mm, "_f16_mm_ks", ks)
    monkeypatch.setattr(exl3_mm, "_reduce", red)
    exl3_mm.use(exl3_mm.Switches(inner_k=on, moe_chunk=False))
    q = _f16(n, k)
    out = torch.empty((m, n + 64), dtype=torch.bfloat16)[:, :n]      # a stack's columns: any row stride
    q(torch.empty((m, k), dtype=torch.bfloat16), out)
    if on and exl3_mm.inner_k_fits(m, n, k, q.sk):
        assert mm.calls == [] and red.calls == []
        (grid, kw), = ks.calls
        assert grid == (-(-m // 16), -(-n // 64))
        bn, warps, stages = exl3_mm.WIDE_TILE
        assert (kw["BM"], kw["BN"], kw["BK"], kw["num_warps"], kw["num_stages"]) == (16, bn, 64, warps, stages)
        assert kw["SK"] == q.sk and kw["K"] == k
        return
    assert ks.calls == []
    blocks = 1 if q.sk == 1 else -(-m // exl3_mm.ROWS)
    assert len(mm.calls) == blocks and len(red.calls) == (blocks if q.sk > 1 else 0)


def test_decode_partials_never_take_the_in_program_sum(monkeypatch):
    mm, ks = Spy(), Spy()
    monkeypatch.setattr(exl3_mm, "_f16_mm", mm)
    monkeypatch.setattr(exl3_mm, "_f16_mm_ks", ks)
    exl3_mm.use(exl3_mm.Switches(inner_k=True, moe_chunk=False))
    _f16(324, 10240).partials(torch.empty((8, 10240), dtype=torch.bfloat16))
    assert ks.calls == [] and len(mm.calls) == 1


SIG_KS = {"X": "*bf16", "W": "*fp16", "OUT": "*bf16", "M": "i32", "N": "i32", "x_stride": "i32", "o_stride": "i32",
          "K": "constexpr", "SK": "constexpr", "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"}


def _compile_ks(k, sk, bn, warps, stages, out="*bf16"):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile

    names = list(SIG_KS)
    attrs = {(names.index(a),): [["tt.divisibility", 16]] for a in names if SIG_KS[a].startswith("*")}
    src = ASTSource(fn=exl3_mm._f16_mm_ks, signature={**SIG_KS, "OUT": out},
                    constexprs={"K": k, "SK": sk, "BM": 16, "BN": bn, "BK": 64}, attrs=attrs)
    got = compile(src, target=GPUTarget("cuda", 121, 32), options={"num_warps": warps, "num_stages": stages})
    return got.asm["ttgir"], got.asm["ptx"]


@pytest.mark.parametrize("n, k", [(324, 10240), (320, 10240), (96, 2560), (2560, 2560)])
def test_the_in_program_sum_keeps_each_slices_mma_chain_and_adds_after_the_loop(n, k):
    """``_f16_mm_ks`` on WIDE_TILE: the same layout and per-step chain of m16n8k16 steps as ``_f16_mm`` (one BK step:
    two n8 accumulators a warp, each taking the step's four k16 fragments in order), and the slice's sum added to the
    total by an fp32 add after the slice's loop, not folded into the MMA chain (playbook 6.9)."""

    sk = exl3_mm.f16_split(n, k)
    ir, ptx = _compile_ks(k, sk, *exl3_mm.WIDE_TILE)
    assert "warpsPerCTA = [1, 4], instrShape = [16, 8]" in ir
    want = [(acc, frag) for frag in range(4) for acc in range(2)]
    assert _chain(ptx) == want
    # the slice loop's result reaches the total through an add on the MMA layout
    assert re.search(r"arith\.addf %\S+, %\S+ : tensor<16x64xf32, #mma", ir), "no slice add on the accumulators"


# -- the pack's fp16 hyper-connections take the write-back and norm fusion -------------------------------------------
ROWS = 2048


def _x3_hc(fuse: bool):
    return HC(SimpleNamespace(n=324), SimpleNamespace(n=10240), "scale", True, fuse=fuse)    # no kernel tag: fp16


def _buffers(prefill: bool):
    names = ("pss", "normed", "xs_normed", "act", "xs_act", "up", "mixed", "xs_mixed", "dn", "dn_mix")
    return SimpleNamespace(prefill=prefill, **{n: torch.zeros((ROWS, 324 if n == "dn" else 1)) for n in names})


@pytest.fixture
def launches(monkeypatch):
    calls: list[str] = []
    state = SimpleNamespace(calls=calls, fused=lambda *a, **k: calls.append("write_norm"))

    def rec(name):
        return lambda *a, **k: calls.append(name)

    for name in ("hc_writeback", "hc_normed", "hc_act", "hc_mix"):
        monkeypatch.setattr(forward.glue, name, rec(name))
    monkeypatch.setattr(forward, "_down_act", rec("down_act"))
    monkeypatch.setattr(forward, "_mm", rec("up"))
    monkeypatch.setattr(forward, "_readout_b16", lambda *a, **k: pytest.fail("the bf16 read-out ran"))
    monkeypatch.setattr(forward, "_hc_fuser", lambda device, **kw: state.fused)
    return state


def _block(hc, b, rows):
    forward.hc_block(hc, b, rows, 1e-6, 4, 320, 2, torch.zeros((ROWS, 4)), torch.zeros((ROWS, 4)),
                     torch.zeros((ROWS, 1)), y=torch.zeros((ROWS, 11, 1)), wts=torch.zeros((ROWS, 11)))


OLD = ["hc_writeback", "hc_normed", "down_act", "up", "hc_mix"]


@pytest.mark.parametrize("rows", [17, 128, 2048])
def test_a_prompt_chunk_writes_back_and_normalizes_in_one_kernel(launches, rows):
    _block(_x3_hc(True), _buffers(True), rows)
    assert launches.calls == ["write_norm", "down_act", "up", "hc_mix"]


@pytest.mark.parametrize("fuse, prefill, rows", [(False, True, ROWS), (True, True, 16), (True, False, 17),
                                                 (True, False, 8)])
def test_the_switch_off_decode_windows_and_short_chunks_keep_the_separate_kernels(launches, fuse, prefill, rows):
    launches.fused = lambda *a, **k: pytest.fail("the write-norm fusion ran")
    _block(_x3_hc(fuse), _buffers(prefill), rows)
    assert launches.calls == OLD


def test_without_the_checked_fusion_the_separate_kernels_run(launches):
    launches.fused = None
    _block(_x3_hc(True), _buffers(True), ROWS)
    assert launches.calls == OLD


def test_the_loaders_hand_the_pack_the_hc_switch(monkeypatch, tmp_path):
    from tensorfold.families.qwen4_exp.cuda import exl3, weights

    got = {}
    monkeypatch.setattr(exl3, "is_exl3", lambda path: True)
    monkeypatch.setattr(exl3, "load", lambda *a, **kw: got.update(kw))
    weights.load(tmp_path, "cpu", hc_fuse=False, f16_narrow=True)
    assert got["hc_fuse"] is False and got["f16_narrow"] is True


# -- a prompt window's expert slots round straight into the MoE buffer ---------------------------------------------
@pytest.mark.parametrize("on", [True, False])
@pytest.mark.parametrize("rows", [17, 1024, 1500, 2048])
def test_a_prompt_windows_slots_go_straight_into_the_buffer(monkeypatch, on, rows):
    from tensorfold.cuda.exl3 import experts

    slots, dims, window = 11, 8, 1024
    calls = []

    def routed(x, pick, wts, ex, s, out, n, *a, y=None, **kw):
        calls.append((n, None if y is None else y.data_ptr()))
        got = y if y is not None else torch.full((n * slots, dims), 0.5, dtype=torch.float32)
        got.fill_(0.5)
        return got

    monkeypatch.setattr(experts, "routed", routed)
    monkeypatch.setattr(forward.moe_mod, "router", lambda *a, **k: None)
    monkeypatch.setattr(forward.moe_mod, "select_rows", lambda *a, **k: None)
    exl3_mm.use(exl3_mm.Switches(moe_direct=on, moe_chunk=False))
    buf = SimpleNamespace(logits=torch.zeros((rows, 1)), pick=torch.zeros((rows, slots), dtype=torch.int32),
                          wts=torch.zeros((rows, slots)), slots=slots,
                          y=torch.zeros((rows, slots, dims), dtype=torch.bfloat16))
    b = SimpleNamespace(prefill=True, moe=buf, mixed=torch.zeros((rows, dims)))
    w = SimpleNamespace(cfg=SimpleNamespace(top_k=10, experts=512), x3=SimpleNamespace(moe=None))
    mode, y, _ = forward._exl3_moe(SimpleNamespace(router=None, experts=None), w, b, rows)
    assert mode == 2 and y.data_ptr() == buf.y.data_ptr()
    assert torch.all(buf.y.float() == 0.5)
    starts = list(range(0, rows, window))
    assert [n for n, _ in calls] == [min(window, rows - r0) for r0 in starts]
    if on:
        assert [p for _, p in calls] == [buf.y[r0].data_ptr() for r0 in starts]
    else:
        assert all(p is None for _, p in calls)
