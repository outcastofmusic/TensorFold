"""Flash Next EXL3 replies with TF_EXL3_GROUP_SCAN_DECODE and TF_EXL3_LINEAR_PAIR off, on, off on one load (two loads of
some checkpoints already differ, so comparing loads would test that).

- Flash Next EXL3 3.05 bpw: a 4,700-token prompt in 2,048-row chunks (its 1,024-row expert windows take the scan),
  its last logits and streams; the first token, 64 serial tokens and 64 drafted tokens (MTP at the default depth,
  confidence 0 so every verify window is full) from CUDA graphs, greedy and seeded, drafted equal serial in every
  setting; then four streams decoded together (eager windows of up to 28 rows, past one row tile, which the pair
  switch takes) equal each alone. Every setting equal to the first.

GLM-5.3's replies are in test_glm_exl3_decode2_replies.py, which runs in its own pytest process. Needs the Flash
Next pack under $MODELS (about 60 GiB free)."""

from __future__ import annotations

import dataclasses
import gc
import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.exl3 import switches
from tensorfold.engine.exact_sampling import Sampling

MODELS = Path(os.environ.get("MODELS", "/home/agis/models"))
FN_PACK = MODELS / "Qwen3.8-Flash-Next-exl3-3.05bpw"
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95)]
DEFAULTS = switches.Switches(group_scan=True, down_fuse=True, gateup_fuse=True, group_scan_decode=False,
                             linear_pair=False)
ORDER = (False, True, False)                    # off twice: a difference is the switches', not noise


@pytest.fixture(autouse=True)
def restore():
    from tensorfold.families.qwen4_exp.cuda import exl3_mm

    before, before_mm = switches.current(), exl3_mm.current()
    exl3_mm.use(exl3_mm.Switches(moe_chunk=True))    # the pack's switches at their defaults (MOE_CHUNK on since fba05ef2)
    yield
    switches.use(before)
    exl3_mm.use(before_mm)


def _prompt(n: int, seed: int, top: int) -> list[int]:
    return torch.randint(10, top, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def _set(on: bool) -> None:
    """The two switches on or off, every other EXL3 switch at its default (fba05ef2: the prompt switches on)."""

    switches.use(dataclasses.replace(DEFAULTS, group_scan_decode=on, linear_pair=on))


@pytest.fixture(scope="module")
def flash_next():
    if not (FN_PACK / "config.json").exists():
        pytest.skip("the Flash Next EXL3 pack is not here")
    from tensorfold.families.qwen4_exp.cuda import exl3_mm
    from tensorfold.families.qwen4_exp.cuda.weights import load

    exl3_mm.use(exl3_mm.Switches(moe_chunk=True))                 # the routed scratch for 2,048-row calls (default)
    w = load(FN_PACK, mtp=True, draft_vocab=None)
    assert w.x3.moe.rows == exl3_mm.MOE_CHUNK_WINDOW
    yield w
    del w
    _free()


def _logits(w, prompt):
    """The prompt's last-row logits and its last row's streams, from a fresh prefill in 2,048-row chunks."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill_begin, prefill_chunk

    e = Engine(w, capacity=8192, max_rows=8, prefill_rows=2048, graphs=False)
    start, last = prefill_begin(e, prompt), None
    while start < len(prompt):
        last, _ = prefill_chunk(e, prompt, start)
        start = min(start + e.prefill_rows, len(prompt))
    out = (last.clone(), e.last_streams.clone())
    del e
    _free()
    return out


def _reply(w, prompt, sampling):
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


def test_prompt_logits_keep_their_bytes_with_the_switches_off_on_off(flash_next):
    prompt = _prompt(4700, 3, 150_000)
    got = []
    for on in ORDER:
        _set(on)
        got.append(_logits(flash_next, prompt))
    for other in got[1:]:
        for a, b in zip(got[0], other):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "seeded"])
def test_replies_keep_their_bytes_with_the_switches_off_on_off(flash_next, sampling):
    prompt = _prompt(4700, 5, 150_000)
    got = []
    for on in ORDER:
        _set(on)
        got.append(_reply(flash_next, prompt, sampling))
    for setting, g in zip(ORDER, got):
        assert g[2] == g[1], setting                   # drafted equal serial
        assert g[3] > 0, setting                       # the drafts ran
    assert got[0][:3] == got[2][:3]
    assert got[0][:3] == got[1][:3]


def test_four_streams_equal_each_alone_with_the_switches_off_on_off(flash_next):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    w = flash_next
    prompts = [_prompt(n, 40 + n, 150_000) for n in (700, 1300, 90, 2100)]
    samplings = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(7, 1.0, 20, 0.95), None]
    got_all = []
    for on in ORDER:
        _set(on)
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
        assert outs == refs, on
        got_all.append(outs)
        del dec
        _free()
    assert got_all[0] == got_all[1] == got_all[2]
