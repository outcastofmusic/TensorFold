"""Flash Next EXL3 replies as served, on the kernels ``routed`` picks from the shape (the listed layout, the grouping
scan, the fused epilogues, the staged and rotating launches, one routed call a 4,096-row chunk). Their bytes against
the launches of before each kernel are in the kernel tests (test_exl3_prefill_kernels.py, test_exl3_staged_kernels.py,
test_exl3_rot_fuse_kernels.py, test_exl3_decode2_kernels.py, test_exl3_sweep2_kernels.py).

- Flash Next EXL3 3.05 bpw: a 4,700-token prompt's last logits and last row's streams, the same in 2,048-row chunks
  as in the served 4,096-row ones (rows never depend on the chunk); 64 serial and 64 drafted tokens (MTP at the
  default depth, confidence 0) from CUDA graphs, greedy and seeded, drafted equal serial. Then four streams decoded
  together equal each alone (their prompts take prompt windows; their decode windows of 4 to 28 rows take two row
  tiles past 16).

GLM-5.3's replies are in test_glm_exl3_replies.py, which runs in its own pytest process.
Needs the Flash Next pack under $MODELS (about 60 GiB free)."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import exl3_mm  # noqa: E402

MODELS = Path(os.environ.get("MODELS", "/home/agis/models"))
FN_PACK = MODELS / "Qwen3.8-Flash-Next-exl3-3.05bpw"
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95)]


def _prompt(n: int, seed: int, top: int) -> list[int]:
    return torch.randint(10, top, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _free():
    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------- Flash Next


@pytest.fixture(scope="module")
def flash_next():
    if not (FN_PACK / "config.json").exists():
        pytest.skip("the Flash Next EXL3 pack is not here")
    from tensorfold.families.qwen4_exp.cuda.weights import load

    exl3_mm.use(exl3_mm.Switches())                               # served: the 4,096-row routed window
    w = load(FN_PACK, mtp=True, draft_vocab=None, hc_fuse=True)
    assert w.x3.moe.rows == exl3_mm.PROMPT_ROWS
    yield w
    del w
    _free()


def _fn_reply(w, prompt, sampling):
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    e = Engine(w, capacity=8192, max_rows=8, prefill_rows=2048, graphs=True)
    first = prefill(e, prompt, sampling)
    serial = serial_decode(e, first, 64, sampling).tokens
    assert prefill(e, prompt, sampling) == first
    drafted = mtp_decode(e, first, 64, sampling, confidence=0.0)
    out = (first, serial, drafted.tokens, drafted.drafted)
    del e
    _free()
    return out


def _fn_logits(w, prompt, rows=2048):
    """The prompt's last-row logits and its last row's streams, from a fresh prefill in ``rows``-row chunks."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill_begin, prefill_chunk

    e = Engine(w, capacity=8192, max_rows=8, prefill_rows=rows, graphs=False)
    start, last = prefill_begin(e, prompt), None
    while start < len(prompt):
        last, _ = prefill_chunk(e, prompt, start)
        start = min(start + e.prefill_rows, len(prompt))
    out = (last.clone(), e.last_streams.clone())
    del e
    _free()
    return out


def test_flash_next_prompt_logits_keep_their_bytes_in_either_chunk_size(flash_next):
    prompt = _prompt(4700, 3, 150_000)
    served, halves = _fn_logits(flash_next, prompt, exl3_mm.PROMPT_ROWS), _fn_logits(flash_next, prompt, 2048)
    assert torch.isfinite(served[0].float()).all()
    for a, b in zip(served, halves):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "seeded"])
def test_flash_next_drafted_replies_equal_serial(flash_next, sampling):
    _, serial, drafted, rounds = _fn_reply(flash_next, _prompt(4700, 5, 150_000), sampling)
    assert drafted == serial
    assert rounds > 0                                  # the drafts ran


def test_flash_next_four_streams_equal_each_alone(flash_next):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    w = flash_next
    prompts = [_prompt(n, 20 + n, 150_000) for n in (700, 1300, 90, 2100)]
    samplings = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(7, 1.0, 20, 0.95), None]
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=4096, max_rows=8, prefill_rows=2048, kv_dtype="int8")
        first = prefill(e, prompt, sampling)
        refs.append(serial_decode(e, first, 48, sampling).tokens)
        del e
    dec = MultiDecoder(w, slots=4, capacity=4096, depth=6, confidence=0.0, kv_dtype="int8", graphs=False)
    outs = []
    for prompt, sampling in zip(prompts, samplings):
        got: list[int] = []
        dec.admit(Stream(prompt, 48, sampling, stop_eos=False, emit=lambda new, got=got: got.extend(new)))
        outs.append(got)
    while dec.live():
        dec.finish(dec.round())
    assert outs == refs
    del dec
    _free()
