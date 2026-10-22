"""TENSORFOLD_PREFILL_ROWS=4096 on Flash Next EXL3: 4,096-row prompt chunks keep the bytes of 2,048-row ones.

The pack's real weights cut to TENSORFOLD_EXL3_FLASHNEXT_LAYERS layers (default 4: three DeltaNet, one attention,
the n-gram layer, the head and the MTP head), loaded once with the scratch for 4,096-row chunks. The 2,048-row runs
take the served routed window (2,048 rows a call); the 4,096-row runs take the switch's (4,096). A 9,500-token prompt
(two whole 4,096-row chunks and a partial one), with a kept point inside a chunk at either size: the first token,
greedy and seeded, MTP on and off; the committed state, KV caches, kept state and last row's streams; the last row's
logits; 12 serial tokens after it; and the kept point's resume. Needs the Flash Next pack under $MODELS (about 15 GiB
free), so it runs beside a served model."""

from __future__ import annotations

import dataclasses
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import exl3_mm  # noqa: E402
from tests.cuda.test_exl3_prefill_replies import FN_PACK, _free, _prompt  # noqa: E402
from tests.cuda.test_flashnext_prompt_cache import _assert_same_state, _same_bits, _same_snap  # noqa: E402

LAYERS = int(os.environ.get("TENSORFOLD_EXL3_FLASHNEXT_LAYERS", "4"))
SAMPLINGS = [None, Sampling(seed=31, top_k=20, top_p=0.95)]


def _rows(rows: int) -> None:
    exl3_mm.use(dataclasses.replace(exl3_mm.Switches(), prompt_rows=rows))


@pytest.fixture(scope="module")
def cut():
    if not (FN_PACK / "config.json").exists():
        pytest.skip("the Flash Next EXL3 pack is not here")
    from tensorfold.families.qwen4_exp.cuda import exl3
    from tensorfold.families.qwen4_exp.cuda import weights as W

    before, real = exl3_mm.current(), W.Config.read

    def config(d):
        c = real(d)
        c.layers = LAYERS
        c.ple_layers = [i for i in c.ple_layers if i < LAYERS]
        return c

    _rows(4096)
    W.Config.read = staticmethod(config)
    try:
        w = exl3.load(FN_PACK, "cuda", mtp=True, draft_vocab=None)
    finally:
        W.Config.read = real
    assert w.x3.moe.rows == 4096 and w.x3.ple_emb.shape[0] == 4096
    assert any(lay.ple is not None for lay in w.layers) and any(not lay.linear for lay in w.layers)
    yield w
    exl3_mm.use(before)
    del w
    _free()


def _run(w, rows: int, prompt, sampling, mtp: bool, keep_at: int):
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, prefill_begin, prefill_chunk

    _rows(rows)
    assert exl3_mm.moe_window() == rows
    e = Engine(w, capacity=12288, max_rows=8, prefill_rows=rows, graphs=False)
    first = prefill(e, prompt, sampling, mtp=mtp, keep_at=keep_at)
    start, last = prefill_begin(e, prompt, mtp=mtp), None
    while start < len(prompt):
        last, _ = prefill_chunk(e, prompt, start, mtp=mtp)
        start = min(start + e.prefill_rows, len(prompt))
    assert e.sample(last, [len(prompt)], sampling)[0] == first
    return e, first, last.clone()


@pytest.mark.parametrize("mtp", [False, True], ids=["mtp-off", "mtp-on"])
@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "seeded"])
def test_4096_row_chunks_keep_the_2048_row_bytes(cut, sampling, mtp):
    from tensorfold.families.qwen4_exp.cuda.decode import prefill, serial_decode

    prompt, keep_at = _prompt(9500, 7, 150_000), 5001
    small, first, logits = _run(cut, 2048, prompt, sampling, mtp, keep_at)
    large, got, got_logits = _run(cut, 4096, prompt, sampling, mtp, keep_at)
    assert got == first
    assert _same_bits(got_logits, logits)
    _assert_same_state(small.st, large.st)
    assert _same_snap(small.kept["state"], large.kept["state"])
    assert _same_bits(small.last_streams, large.last_streams)
    if mtp:
        assert _same_bits(small.kept["tail"], large.kept["tail"])
    _rows(2048)
    want = serial_decode(small, first, 12, sampling, stop_eos=False).tokens
    _rows(4096)
    assert serial_decode(large, got, 12, sampling, stop_eos=False).tokens == want
    extended = prompt[:keep_at] + _prompt(300, 9, 150_000)
    _rows(2048)
    fresh = prefill(small, extended, sampling, mtp=mtp)
    _rows(4096)
    assert prefill(large, extended, sampling, mtp=mtp, resume=large.kept) == fresh
    _assert_same_state(small.st, large.st)
    del small, large
    _free()
