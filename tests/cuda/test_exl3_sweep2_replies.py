"""Replies on the real checkpoints with the sweep's switches off, on, off on one load (two loads of some checkpoints
already differ, so comparing loads would test that): TF_EXL3_EXPERT_ITEMS, and on Flash Next also
TF_FLASHNEXT_STACK_DIRECT, TF_FLASHNEXT_F16_INNER_K, TF_FLASHNEXT_MOE_DIRECT and TF_FLASHNEXT_HC_FUSE on the pack's
fp16 hyper-connections (the loader's flag, flipped here on every HC of the load).

- Flash Next EXL3 3.05 bpw: a 4,700-token prompt in 2,048-row chunks (two whole chunks and a partial one, which the
  prompt switches all touch), its last logits, 64 serial tokens and 64 drafted tokens (MTP at the default depth,
  confidence 0 so every verify window is full) from CUDA graphs, greedy and seeded; drafted equal serial in every
  setting, and every setting equal to the first. Then four streams decoded together (windows of 4 to 16 rows, past
  decode_kernel's 16 once the drafts fill) equal each alone, in every setting.
- GLM-5.3 (the sushi-experts pack): eager replies serial and drafted, greedy and seeded.

Needs the checkpoints under $MODELS (about 60 GiB free for Flash Next, 100 GiB for GLM-5.3)."""

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
GLM = MODELS / "glm53-flash-sushi-experts-mosaic-dense"
GLM_DRAFTER = MODELS / "GLM-5.3-Flash-DFlash2-TR3-v3"
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


def _set(w, on: bool) -> None:
    switches.use(switches.Switches(expert_items=on))
    exl3_mm.use(exl3_mm.Switches(stack_direct=on, inner_k=on, moe_direct=on))
    if w is None:
        return
    hcs = [w.mixer] + [hc for layer in w.layers for hc in (layer.attn_hc, layer.mlp_hc)]
    if w.mtp is not None:
        hcs += [w.mtp.mixer, w.mtp.layer.attn_hc, w.mtp.layer.mlp_hc]
    for hc in hcs:
        hc.fuse = on


# ------------------------------------------------------------------------------------------------- Flash Next


@pytest.fixture(scope="module")
def flash_next():
    if not (FN_PACK / "config.json").exists():
        pytest.skip("the Flash Next EXL3 pack is not here")
    from tensorfold.families.qwen4_exp.cuda.weights import load

    w = load(FN_PACK, mtp=True, draft_vocab=None, f16_narrow=True, hc_fuse=True)
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


def test_flash_next_prompt_logits_keep_their_bytes_with_the_switches_off_on_off(flash_next):
    w = flash_next
    prompt = _prompt(4700, 3, 150_000)
    got = []
    for on in ORDER:
        _set(w, on)
        got.append(_fn_logits(w, prompt))
    for a, b in zip(got[0], got[1]):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    for a, b in zip(got[0], got[2]):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


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
            dec.admit(Stream(prompt, 48, sampling, emit=lambda new, got=got: got.extend(new)))
            outs.append(got)
        while dec.live():
            dec.finish(dec.round())
        assert outs == refs, on
        got_all.append(outs)
        del dec
        _free()
    assert got_all[0] == got_all[1] == got_all[2]


# ------------------------------------------------------------------------------------------------- GLM-5.3


@pytest.fixture(scope="module")
def glm():
    if not (GLM / "config.json").exists() or not (GLM_DRAFTER / "config.json").exists():
        pytest.skip("the GLM-5.3 pack or its drafter is not here")
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    engine = GlmEngine(GLM, rank=0, master="", port=0, world=1, drafter=GLM_DRAFTER, context=8192, eager=True,
                       calibrate=False)
    yield engine
    del engine
    _free()


def _glm_reply(engine, prompt, sampling, draft, tokens=48):
    out: list[int] = []
    engine.cache.clear()
    engine.live = []
    engine.request.policy = None
    engine.request.stop_eos = False
    stats = engine.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=draft)
    assert len(out) == tokens
    if draft:
        assert stats["rounds"] >= 1, stats
    return out


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "seeded"])
def test_glm_replies_keep_their_bytes_with_the_item_list_off_on_off(glm, sampling):
    prompt = _prompt(2600, 11, 150_000)
    got = []
    for on in ORDER:
        _set(None, on)
        got.append((_glm_reply(glm, prompt, sampling, False), _glm_reply(glm, prompt, sampling, True)))
    for setting, (serial, drafted) in zip(ORDER, got):
        assert drafted == serial, setting
    assert got[0] == got[2]
    assert got[0] == got[1]
