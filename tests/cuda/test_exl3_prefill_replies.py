"""Flash Next EXL3 replies with the prompt-window switches off, on, off on one load (two loads of some checkpoints
already differ, so comparing loads would test that): TF_EXL3_GROUP_SCAN, TF_EXL3_DOWN_FUSE, TF_EXL3_GATEUP_FUSE and
TF_FLASHNEXT_MOE_CHUNK, on top of today's defaults (TF_EXL3_EXPERT_ITEMS, TF_FLASHNEXT_STACK_DIRECT and
TF_FLASHNEXT_MOE_DIRECT on). The pack loads with the chunk window on, so its routed scratch holds 2,048 rows and either
window runs on the one load.

- Flash Next EXL3 3.05 bpw: a 4,700-token prompt in 2,048-row chunks (two whole chunks and a partial one, which every
  switch touches), its last logits and last row's streams with all four off, on, off and then each alone; 64 serial
  and 64 drafted tokens (MTP at the default depth, confidence 0) from CUDA graphs, greedy and seeded, drafted equal
  serial in every setting and every setting equal to the first. Then four streams decoded together equal each alone,
  in every setting (their prompts take prompt windows; their decode windows do not change).

GLM-5.3's replies are in test_glm_exl3_prefill_replies.py, which runs in its own pytest process.
Needs the Flash Next pack under $MODELS (about 60 GiB free)."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.exl3 import switches  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import exl3_mm  # noqa: E402

MODELS = Path(os.environ.get("MODELS", "/home/agis/models"))
FN_PACK = MODELS / "Qwen3.8-Flash-Next-exl3-3.05bpw"
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95)]
ORDER = (False, True, False)                    # off twice: a difference is the switches', not noise


@pytest.fixture(autouse=True)
def restore():
    before, before_mm = switches.current(), exl3_mm.current()
    yield
    switches.use(before)
    exl3_mm.use(before_mm)


def _prompt(n: int, seed: int, top: int) -> list[int]:
    return torch.randint(10, top, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _free():
    gc.collect()
    torch.cuda.empty_cache()


NAMES = ("scan", "down", "gateup", "chunk")


def _set(w, on: bool | set[str]) -> None:
    """Today's defaults, plus every new switch (True), none (False), or the named ones."""

    names = set(NAMES) if on is True else (set() if on is False else on)
    switches.use(switches.Switches(expert_items=True, group_scan="scan" in names, down_fuse="down" in names,
                                   gateup_fuse="gateup" in names))
    exl3_mm.use(exl3_mm.Switches(moe_chunk="chunk" in names))


# ------------------------------------------------------------------------------------------------- Flash Next


@pytest.fixture(scope="module")
def flash_next():
    if not (FN_PACK / "config.json").exists():
        pytest.skip("the Flash Next EXL3 pack is not here")
    from tensorfold.families.qwen4_exp.cuda.weights import load

    exl3_mm.use(exl3_mm.Switches(moe_chunk=True))                 # the routed scratch for 2,048-row calls
    w = load(FN_PACK, mtp=True, draft_vocab=None, f16_narrow=True, hc_fuse=True)
    assert w.x3.moe.rows == exl3_mm.MOE_CHUNK_WINDOW
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


def _fn_logits(w, prompt):
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


def test_flash_next_prompt_logits_keep_their_bytes_with_the_switches_off_on_off_and_each_alone(flash_next):
    w = flash_next
    prompt = _prompt(4700, 3, 150_000)
    settings = [*ORDER, *({n} for n in NAMES)]
    got = []
    for on in settings:
        _set(w, on)
        got.append(_fn_logits(w, prompt))
    for setting, g in zip(settings[1:], got[1:]):
        for a, b in zip(got[0], g):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), setting


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "seeded"])
def test_flash_next_replies_keep_their_bytes_with_the_switches_off_on_off(flash_next, sampling):
    w = flash_next
    prompt = _prompt(4700, 5, 150_000)
    got = []
    for on in ORDER:
        _set(w, on)
        got.append(_fn_reply(w, prompt, sampling))
    for setting, g in zip(ORDER, got):
        assert g[2] == g[1], setting                   # drafted equal serial
        assert g[3] > 0, setting                       # the drafts ran
    assert got[0][:3] == got[2][:3]
    assert got[0][:3] == got[1][:3]


def test_flash_next_four_streams_equal_each_alone_with_the_switches_off_on_off(flash_next):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    w = flash_next
    prompts = [_prompt(n, 20 + n, 150_000) for n in (700, 1300, 90, 2100)]
    samplings = [None, Sampling(1234, 1.0, 20, 0.95), Sampling(7, 1.0, 20, 0.95), None]
    got_all = []
    for on in ORDER:
        _set(w, on)
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
