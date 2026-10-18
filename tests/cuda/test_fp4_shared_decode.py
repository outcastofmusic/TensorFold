"""TF_FP4_SHARED_DECODE: a decode window's MXFP8 shared expert on the lane matmul (qmmf) or the prompt GEMM (prompt).

The prompt GEMM keeps a row's bits in its few-row tile. Prompt rows take the prompt GEMM under either value, so a
prompt's bits never move. Decode rows take the flag's kernel, and both values keep drafted == serial, streams
together == alone, and resumed == fresh."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("Flash Next CUDA kernels run on sm_12x (GB10, RTX 50, RTX PRO 6000) only",
                allow_module_level=True)

from nvfp4_tiny import write  # noqa: E402
from test_nvfp4_linear import _fp4, _fp8, _mx8  # noqa: E402

from tensorfold.cuda.nvfp4 import linear  # noqa: E402
from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear, Mx8Linear  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import nvfp4_moe  # noqa: E402

ENV = nvfp4_moe.SHARED_DECODE_ENV
SHAPES = [(2560, 1280), (1280, 1280), (320, 1024)]     # FP4-Local's shared gate/up and down, and a padded one


def _linears(n: int, k: int):
    """An NVFP4, an FP8 and an MXFP8 linear of [n, k] as (name, mode, weight bytes, block scales, scale, linear)."""

    packed, scale, g = _fp4(n, k, n + k)
    fp4 = Fp4Linear.from_checkpoint(torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda(), g)
    w8, s8 = _fp8(n, k, n + k + 1)
    fp8 = Fp8Linear.from_checkpoint(torch.from_numpy(w8).cuda().view(torch.float8_e4m3fn), s8)
    wm, sm = _mx8(n, k, n + k + 2)
    mx8 = Mx8Linear.from_checkpoint(torch.from_numpy(wm).cuda().view(torch.float8_e4m3fn), torch.from_numpy(sm).cuda())
    return [("fp4", linear.FP4, fp4.words, fp4.bs, fp4.scale, fp4), ("fp8", linear.FP8, fp8.w8, None, fp8.scale, fp8),
            ("mxfp8", linear.MXFP8, mx8.w8, mx8.bs, 1.0, mx8)]


def _prompt16(tile: int, mode: int, w, bs, scale: float, n: int, npad: int, x: torch.Tensor) -> torch.Tensor:
    y = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    linear._prompt_ext().prompt16(x, w, bs, float(scale), y, mode, n, npad, tile, False)
    return y


@pytest.mark.parametrize("n,k", SHAPES)
def test_the_few_row_tile_keeps_the_prompt_gemms_bits(n, k):
    """Tile 6 (16x64 blocks) gives every row the bits of tile 4 and of the row inside a 300-row prompt."""

    x = (torch.randn((300, k), generator=torch.Generator().manual_seed(n)) * 0.5).to(torch.bfloat16).cuda()
    for name, mode, w, bs, scale, lin in _linears(n, k):
        big = _prompt16(linear.PROMPT_TILE, mode, w, bs, scale, n, lin.npad, x)
        for rows in (1, 2, 3, 7, 8, 16, 17, 40):
            part = x[:rows].contiguous()
            few = _prompt16(linear.FEW_TILE, mode, w, bs, scale, n, lin.npad, part)
            assert torch.equal(few, big[:rows]), (name, rows)
            assert torch.equal(few, _prompt16(linear.PROMPT_TILE, mode, w, bs, scale, n, lin.npad, part)), (name, rows)
        one = x[123:124].contiguous()
        assert torch.equal(lin.prefill(one), big[123:124]), name          # ``_prompt`` picks tile 6 for one row


@pytest.mark.parametrize("n,k", SHAPES[:2])
def test_qmmf_rows_keep_their_bits_in_any_window_and_track_the_prompt_gemm(n, k):
    """The lane matmul's decode rows: a row's bits alone equal its bits in any window up to 28 rows (a verify round
    of four streams at depth 6). They round differently from the prompt GEMM, within fp32 accumulation."""

    wm, sm = _mx8(n, k, 11)
    lin = Mx8Linear.from_checkpoint(torch.from_numpy(wm).cuda().view(torch.float8_e4m3fn), torch.from_numpy(sm).cuda())
    x = (torch.randn((28, k), generator=torch.Generator().manual_seed(4)) * 0.5).to(torch.bfloat16).cuda()
    full = lin(x)
    for rows in (1, 2, 4, 7, 16, 17, 28):
        assert torch.equal(lin(x[:rows].contiguous()), full[:rows]), rows
    assert torch.equal(lin(x[5:6].contiguous()), full[5:6])
    prompt = lin.prefill(x).double()
    err = ((full.double() - prompt).abs() / (prompt.abs() + prompt.abs().mean())).max().item()
    assert err < 1e-2, err


def _moe4(dev: str = "cuda"):
    """A three-expert NVFP4 layer whose shared expert is MXFP8 (FP4-Local's layout), with its router and config."""

    torch.manual_seed(9)
    e, d, ni = 3, 256, 128

    def proj(n, k):
        return (torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev),
                torch.randint(40, 60, (n, k // 16), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn), 0.01)

    def mx(n, k, seed):
        w, s = _mx8(n, k, seed)
        return Mx8Linear.from_checkpoint(torch.from_numpy(w).to(dev).view(torch.float8_e4m3fn),
                                         torch.from_numpy(s).to(dev))

    shared = nvfp4_moe.Expert4(mx(2 * ni, d, 1), mx(d, ni, 2))
    ex = nvfp4_moe.moe4_from_experts([proj(ni, d) for _ in range(e)], [proj(ni, d) for _ in range(e)],
                                     [proj(d, ni) for _ in range(e)], shared)
    cfg = SimpleNamespace(num_experts_per_tok=1, num_experts=e, moe_intermediate_size=ni, hidden_size=d)
    router = (torch.randn(e + 1, d, device=dev) * 0.1).to(torch.bfloat16)
    return ex, cfg, router


@pytest.mark.parametrize("mode", ["qmmf", "prompt"])
def test_decode_rows_take_the_flags_kernel_and_prompt_rows_the_prompt_gemm(monkeypatch, mode):
    from tensorfold.cuda import moe as moe_mod

    monkeypatch.setenv(ENV, mode)
    ex, cfg, router = _moe4()
    assert ex.shared_decode == mode
    x = (torch.randn(8, cfg.hidden_size, device="cuda") * 0.5).to(torch.bfloat16)
    ni = ex.width

    def run(rows, decode):
        buf = moe_mod.MoEBuffers(8, cfg, "cuda", prefill=True)              # bf16 slots, as ``moe_prefill`` gives
        nvfp4_moe.moe(rows, None, router, ex, buf, cfg, decode=decode)
        return buf.act[:rows.shape[0]].clone(), buf.y[:rows.shape[0]].clone()

    got = {decode: run(x, decode) for decode in (False, True)}
    for decode, (act, y) in got.items():
        lane = decode and mode == "qmmf"
        mm = (lambda lin, v: lin(v)) if lane else (lambda lin, v: lin.prefill(v))
        g = mm(ex.shared.gu, x).float()
        gate, up = g[:, :ni], g[:, ni:]
        want = ((gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16).float() * up).to(torch.bfloat16)
        assert torch.equal(act[:, 1], want), decode
        assert torch.equal(y[:, 1], mm(ex.shared.down, want)), decode
    assert torch.equal(got[False][1][:, 0], got[True][1][:, 0])            # routed slots: the same kernels
    assert torch.equal(got[False][0][:, 0], got[True][0][:, 0])
    for r in (0, 5):                                                       # a decode row alone: its bits in the window
        act, y = run(x[r:r + 1].contiguous(), True)
        assert torch.equal(y[0], got[True][1][r]) and torch.equal(act[0], got[True][0][r]), r


# A tiny FP4-Local layout: NVFP4 routed experts, MXFP8 DeltaNet, attention and shared expert, NVFP4 n-gram table.

PROMPT = [5, 17, 99, 250, 7, 64, 30, 11, 12, 13, 81, 3, 140, 9, 200, 41, 66, 18, 2, 77]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    path = write(tmp_path_factory.mktemp("fp4local") / "mx", mxfp8=True, ple_nvfp4=True, hidden=512)
    import json

    config = json.loads((path / "config.json").read_text())
    config["quantization"]["quantized_layers"] = {
        "model.layers.1.ple.ple_embedding.ngram_embedding": {"quant_algo": "NVFP4", "group_size": 16}}
    (path / "config.json").write_text(json.dumps(config))
    return path


_LOADED: dict[str, object] = {}


def _weights(tiny: Path, mode: str, monkeypatch):
    """The tiny checkpoint loaded under TF_FP4_SHARED_DECODE=mode (the flag is read when a layer loads)."""

    from tensorfold.families.qwen4_exp.cuda.weights import load

    if mode not in _LOADED:
        monkeypatch.setenv(ENV, mode)
        _LOADED[mode] = load(tiny, mtp=True, draft_vocab=None)
    w = _LOADED[mode]
    assert all(layer.moe.experts.shared_decode == mode for layer in w.layers)
    assert isinstance(w.layers[0].moe.experts.shared.gu, Mx8Linear)
    return w


SAMPLINGS = [None, "seeded"]


def _sampling(kind):
    from tensorfold.engine.exact_sampling import Sampling

    return None if kind is None else Sampling(seed=1234, top_k=20, top_p=0.95)


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("kind", SAMPLINGS)
@pytest.mark.parametrize("mode", ["qmmf", "prompt"])
def test_drafted_equals_serial(tiny, monkeypatch, mode, kind, graphs):
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    w, sampling = _weights(tiny, mode, monkeypatch), _sampling(kind)
    e = Engine(w, capacity=512, max_rows=8, prefill_rows=16, graphs=graphs)
    first = prefill(e, PROMPT, sampling)
    ref = serial_decode(e, first, 24, sampling).tokens
    for depth, confidence in ((3, 0.0), (6, 0.3)):
        assert prefill(e, PROMPT, sampling) == first
        assert mtp_decode(e, first, 24, sampling, depth=depth, confidence=confidence).tokens == ref, depth


@pytest.mark.parametrize("mode", ["qmmf", "prompt"])
def test_streams_decoded_together_equal_each_alone(tiny, monkeypatch, mode):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    w = _weights(tiny, mode, monkeypatch)
    prompts = [PROMPT[:9], PROMPT[3:17], PROMPT[:4], PROMPT]
    samplings = [_sampling(None), _sampling("seeded"), _sampling(None), _sampling("seeded")]
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=512, max_rows=8, prefill_rows=16, kv_dtype="int8")
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 16, sampling).tokens)
    dec = MultiDecoder(w, slots=4, capacity=512, depth=3, confidence=0.3, kv_dtype="int8")
    streams = [Stream(p, 16, s, draft=i != 2) for i, (p, s) in enumerate(zip(prompts, samplings))]
    for s in streams:
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in streams] == refs


@pytest.mark.parametrize("kind", SAMPLINGS)
@pytest.mark.parametrize("mode", ["qmmf", "prompt"])
def test_a_resumed_prompt_equals_a_fresh_one(tiny, monkeypatch, mode, kind):
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode

    w, sampling = _weights(tiny, mode, monkeypatch), _sampling(kind)
    head, full = PROMPT[:13], PROMPT + [90, 91, 92]
    warm = Engine(w, capacity=512, max_rows=8, prefill_rows=16)
    prefill(warm, head, sampling, keep_at=len(head) - 1)
    kept = warm.kept
    serial_decode(warm, warm.first, 8, sampling)                     # decode rows run between the keep and the resume
    got = prefill(warm, full, sampling, resume=kept)
    fresh = Engine(w, capacity=512, max_rows=8, prefill_rows=16)
    want = prefill(fresh, full, sampling)
    assert got == want and torch.equal(warm.last_streams, fresh.last_streams)
    assert serial_decode(warm, got, 12, sampling).tokens == serial_decode(fresh, want, 12, sampling).tokens


def test_prompt_rows_keep_their_bits_under_either_value(tiny, monkeypatch):
    """The flag reaches decode windows only: a prompt's last streams and first token match bit for bit."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill

    got = {}
    for mode in ("qmmf", "prompt"):
        e = Engine(_weights(tiny, mode, monkeypatch), capacity=512, max_rows=8, prefill_rows=16)
        got[mode] = (prefill(e, PROMPT, None), e.last_streams.clone())
    assert got["qmmf"][0] == got["prompt"][0] and torch.equal(got["qmmf"][1], got["prompt"][1])
