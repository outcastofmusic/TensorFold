"""Flash Next EXL3 replies with TF_EXL3_EXPERT_STAGED off, on, off on one load, on top of today's defaults.

- Flash Next EXL3 3.05 bpw: a 4,700-token prompt in 2,048-row chunks (two whole chunks and a partial one), its last
  logits and last row's streams; 64 serial and 64 drafted tokens (MTP at the default depth, confidence 0) from CUDA
  graphs, greedy and seeded, drafted equal serial in every setting and every setting equal to the first; four streams
  decoded together equal each alone (their prompts take prompt windows, so the staged kernel).

Needs the Flash Next pack under $MODELS (about 60 GiB free): run it with production stopped
(notes/tf-runs/exl3-prefill-1021/run.sh)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.exl3 import switches
from tensorfold.families.qwen4_exp.cuda import exl3_mm
from tests.cuda.test_exl3_prefill_replies import FN_PACK, SAMPLINGS, _fn_logits, _fn_reply, _free, _prompt

ORDER = (False, True, False)                    # off twice: a difference is the switch's, not noise


@pytest.fixture(autouse=True)
def restore():
    before, before_mm = switches.current(), exl3_mm.current()
    yield
    switches.use(before)
    exl3_mm.use(before_mm)


def _set(on: bool) -> None:
    switches.use(dataclasses.replace(switches.Switches(), expert_staged=on))
    exl3_mm.use(exl3_mm.Switches())


@pytest.fixture(scope="module")
def flash_next():
    if not (FN_PACK / "config.json").exists():
        pytest.skip("the Flash Next EXL3 pack is not here")
    from tensorfold.families.qwen4_exp.cuda.weights import load

    exl3_mm.use(exl3_mm.Switches())                               # served: the 4,096-row routed window
    w = load(FN_PACK, mtp=True, draft_vocab=None, f16_narrow=False, hc_fuse=True)
    assert w.x3.moe.rows == exl3_mm.PROMPT_ROWS
    assert all(lay.moe.experts.aligned for lay in w.layers)      # every layer takes the staged kernel
    yield w
    del w
    _free()


def test_flash_next_prompt_logits_keep_their_bytes_staged_off_on_off(flash_next):
    prompt = _prompt(4700, 3, 150_000)
    got = []
    for on in ORDER:
        _set(on)
        got.append(_fn_logits(flash_next, prompt))
    for on, g in zip(ORDER[1:], got[1:]):
        for a, b in zip(got[0], g):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), on


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "seeded"])
def test_flash_next_replies_keep_their_bytes_staged_off_on_off(flash_next, sampling):
    prompt = _prompt(4700, 5, 150_000)
    got = []
    for on in ORDER:
        _set(on)
        got.append(_fn_reply(flash_next, prompt, sampling))
    for on, g in zip(ORDER, got):
        assert g[2] == g[1], on                       # drafted equal serial
        assert g[3] > 0, on                           # the drafts ran
    assert got[0][:3] == got[1][:3] == got[2][:3]


def test_flash_next_four_streams_equal_each_alone_staged_off_on_off(flash_next):
    from tensorfold.cuda.streams import Stream
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    prompts = [_prompt(n, 20 + n, 150_000) for n in (700, 1300, 90, 2100)]
    samplings = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(7, 1.0, 20, 0.95), None]
    got_all = []
    for on in ORDER:
        _set(on)
        refs = []
        for prompt, sampling in zip(prompts, samplings):
            e = Engine(flash_next, capacity=4096, max_rows=8, prefill_rows=2048, kv_dtype="int8")
            first = prefill(e, prompt, sampling)
            refs.append(serial_decode(e, first, 48, sampling).tokens)
            del e
        dec = MultiDecoder(flash_next, slots=4, capacity=4096, depth=6, confidence=0.0, kv_dtype="int8",
                           graphs=False)
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
