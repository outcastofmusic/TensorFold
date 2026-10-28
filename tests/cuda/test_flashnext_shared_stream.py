"""TF_FLASHNEXT_SHARED_STREAM on the GPU: the NVFP4 shared expert on a second stream writes the bytes one stream
writes, waits for rows written before it, holds the write-back behind it, captures into a graph with no stream left
outside, and keeps memory flat. Then the tiny FP4-Local layout end to end: prefill streams, verify logits, drafted
tokens and streams decoded together are the same bytes and tokens in modes 1 and 2 as in mode 0."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("Flash Next CUDA kernels run on sm_12x (GB10, RTX 50, RTX PRO 6000) only",
                allow_module_level=True)

from nvfp4_tiny import write  # noqa: E402
from test_fp4_shared_decode import _moe4  # noqa: E402

from tensorfold.cuda import moe as moe_mod  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import nvfp4_moe, shared_stream  # noqa: E402


@pytest.fixture
def mode(monkeypatch):
    def set_(value: str) -> None:
        monkeypatch.setenv(shared_stream.ENV, value)
        shared_stream.mode.cache_clear()

    yield set_
    shared_stream.mode.cache_clear()


def _moe4_bf16():
    """The MTP layer's layout: the bf16 shared expert on identity-scaled FP4 tables (the Triton matmul)."""

    ex, cfg, router = _moe4()
    torch.manual_seed(10)
    d, ni = cfg.hidden_size, cfg.moe_intermediate_size
    bf = [(torch.randn(*s, device="cuda") * 0.05).to(torch.bfloat16) for s in ((ni, d), (ni, d), (d, ni))]
    return nvfp4_moe.MoE4(ex.routed_experts, nvfp4_moe.expert4_from_bf16(*bf)), cfg, router


LAYERS = {"mxfp8": _moe4, "bf16": _moe4_bf16}


def _layer(kind: str, slots: str, rows: int = 32):
    ex, cfg, router = LAYERS[kind]()
    buf = moe_mod.MoEBuffers(rows, cfg, "cuda", prefill=slots == "bf16")
    return ex, cfg, router, buf


def _x(rows: int, d: int, seed: int) -> torch.Tensor:
    return (torch.randn((rows, d), generator=torch.Generator().manual_seed(seed)) * 0.5).to(torch.bfloat16).cuda()


def _poison(buf) -> None:
    buf.act.fill_(float("nan"))
    buf.y.fill_(float("nan"))


def _ref(mode, ex, cfg, router, buf, x, prompt_pass):
    """Mode 0's act and y (the one-stream schedule), poison first so an unwritten byte shows."""

    mode("0")
    _poison(buf)
    nvfp4_moe.moe(x, None, router, ex, buf, cfg, prompt_pass=prompt_pass)
    rows = x.shape[0]
    return buf.act[:rows].clone(), buf.y[:rows].clone()


def _same(got, want) -> bool:
    # NaN poison in an unwritten byte fails equal_nan=False; every byte both modes write must match exactly
    return all(torch.equal(g.view(torch.int16) if g.dtype == torch.bfloat16 else g.view(torch.int32),
                           w.view(torch.int16) if w.dtype == torch.bfloat16 else w.view(torch.int32))
               for g, w in zip(got, want))


@pytest.mark.parametrize("kind", sorted(LAYERS))
@pytest.mark.parametrize("slots", ["fp32", "bf16"])
@pytest.mark.parametrize("m, prompt_pass", [("1", False), ("2", False), ("2", True)])
def test_the_side_stream_writes_the_one_stream_bytes(mode, kind, slots, m, prompt_pass):
    ex, cfg, router, buf = _layer(kind, slots)
    for rows in (1, 7, 24, 32):
        x = _x(rows, cfg.hidden_size, rows)
        want = _ref(mode, ex, cfg, router, buf, x, prompt_pass)
        mode(m)
        _poison(buf)
        nvfp4_moe.moe(x, None, router, ex, buf, cfg, prompt_pass=prompt_pass)
        assert _same((buf.act[:rows], buf.y[:rows]), want), (rows, kind, slots, m)


def test_the_fork_waits_for_rows_written_before_it(mode):
    """The rows land late on the compute stream; a side stream that did not wait would read the zeros before."""

    ex, cfg, router, buf = _layer("mxfp8", "fp32")
    x = _x(16, cfg.hidden_size, 3)
    want = _ref(mode, ex, cfg, router, buf, x, False)
    mode("1")
    rows = torch.zeros_like(x)
    torch.cuda.synchronize()
    torch.cuda._sleep(50_000_000)                        # ~25 ms of the compute stream before the rows are written
    rows.copy_(x)
    _poison(buf)
    nvfp4_moe.moe(rows, None, router, ex, buf, cfg)
    assert _same((buf.act[:16], buf.y[:16]), want)


def test_the_join_holds_the_next_read_behind_the_shared_expert(mode, monkeypatch):
    """The shared expert starts late on the side stream; a read queued on the compute stream after ``moe`` must still
    see its slot."""

    ex, cfg, router, buf = _layer("mxfp8", "fp32")
    x = _x(16, cfg.hidden_size, 4)
    want = _ref(mode, ex, cfg, router, buf, x, False)
    mode("1")
    act = ex.shared_act

    def late(x, prompt=False):
        torch.cuda._sleep(50_000_000)                    # on the current stream: the side stream here
        return act(x, prompt)

    monkeypatch.setattr(ex, "shared_act", late)
    torch.cuda.synchronize()
    _poison(buf)
    nvfp4_moe.moe(x, None, router, ex, buf, cfg)
    got = (buf.act[:16].clone(), buf.y[:16].clone())    # queued on the compute stream right behind the join
    assert _same(got, want)


@pytest.mark.parametrize("kind", sorted(LAYERS))
def test_a_captured_layer_replays_the_one_stream_bytes(mode, kind):
    """The capture takes the side stream in and back out (capture_begin/end would raise on an unjoined stream), and
    replays on new rows give mode 0's eager bytes."""

    ex, cfg, router, buf = _layer(kind, "fp32")
    rows = 12
    xs = [_x(rows, cfg.hidden_size, 20 + i) for i in range(3)]
    wants = [_ref(mode, ex, cfg, router, buf, x, False) for x in xs]
    mode("1")
    static = xs[0].clone()
    nvfp4_moe.moe(static, None, router, ex, buf, cfg)    # eager warm-up compiles the launch shapes
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        nvfp4_moe.moe(static, None, router, ex, buf, cfg)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(2):
        for x, want in zip(xs, wants):
            static.copy_(x)
            _poison(buf)
            g.replay()
            assert _same((buf.act[:rows], buf.y[:rows]), want)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


def test_memory_and_streams_stay_flat_over_many_layers(mode):
    """One side stream a device however many calls; the side pool reaches its peak on the first call and stays."""

    ex, cfg, router, buf = _layer("mxfp8", "bf16")
    mode("2")
    x = _x(32, cfg.hidden_size, 5)
    nvfp4_moe.moe(x, None, router, ex, buf, cfg, prompt_pass=True)
    torch.cuda.synchronize()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    for i in range(200):
        nvfp4_moe.moe(x, None, router, ex, buf, cfg, prompt_pass=bool(i % 2))
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated and torch.cuda.memory_reserved() == reserved
    assert list(shared_stream._SIDES) == [torch.cuda.current_device()]


# ---- the tiny FP4-Local layout end to end ------------------------------------------------------------------------

PROMPT = [5, 17, 99, 250, 7, 64, 30, 11, 12, 13, 81, 3, 140, 9, 200, 41, 66, 18, 2, 77]


@pytest.fixture(scope="module")
def tiny_w(tmp_path_factory):
    import json

    from tensorfold.families.qwen4_exp.cuda.weights import load

    path: Path = write(tmp_path_factory.mktemp("fp4local") / "mx", mxfp8=True, ple_nvfp4=True, hidden=512)
    config = json.loads((path / "config.json").read_text())
    config["quantization"]["quantized_layers"] = {
        "model.layers.1.ple.ple_embedding.ngram_embedding": {"quant_algo": "NVFP4", "group_size": 16}}
    (path / "config.json").write_text(json.dumps(config))
    return load(path, mtp=True, draft_vocab=None)


def _sampling(kind):
    from tensorfold.engine.exact_sampling import Sampling

    return None if kind is None else Sampling(seed=1234, top_k=20, top_p=0.95)


def _engine_run(w, graphs: bool, kind):
    """Prefill streams, a 4-row verify window's logits, and drafted tokens, on a fresh engine."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill

    sampling = _sampling(kind)
    e = Engine(w, capacity=512, max_rows=8, prefill_rows=16, graphs=graphs)
    first = prefill(e, PROMPT, sampling)
    streams = e.last_streams.clone()
    logits = e.forward([first, 3, 4, 5]).clone()
    first = prefill(e, PROMPT, sampling)
    tokens = mtp_decode(e, first, 24, sampling, depth=3, confidence=0.0).tokens
    return streams, logits, tokens


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("kind", [None, "seeded"])
@pytest.mark.parametrize("m", ["1", "2"])
def test_an_engine_with_the_side_stream_gives_the_one_stream_bytes(tiny_w, mode, graphs, kind, m):
    mode("0")
    want = _engine_run(tiny_w, graphs, kind)
    mode(m)
    got = _engine_run(tiny_w, graphs, kind)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]) and got[2] == want[2]


def test_streams_decoded_together_are_the_one_stream_tokens(tiny_w, mode):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    prompts = [PROMPT[:9], PROMPT[3:17], PROMPT[:4], PROMPT]
    samplings = [_sampling(None), _sampling("seeded"), _sampling(None), _sampling("seeded")]

    def run():
        dec = MultiDecoder(tiny_w, slots=4, capacity=512, depth=3, confidence=0.3, kv_dtype="int8")
        streams = [Stream(p, 16, s, draft=i != 2) for i, (p, s) in enumerate(zip(prompts, samplings))]
        for s in streams:
            dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        return [s.out for s in streams]

    mode("0")
    want = run()
    for m in ("1", "2"):
        mode(m)
        assert run() == want, m
