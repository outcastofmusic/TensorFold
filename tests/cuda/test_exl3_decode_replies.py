"""Replies on the real checkpoints with the EXL3 decode switches off, on, off on one load (two loads of some
checkpoints already differ, so comparing loads would test that): TF_EXL3_EXPERT_DECODE and TF_EXL3_PDL, and on Flash
Next also TF_FLASHNEXT_F16_NARROW.

- Flash Next EXL3 3.05 bpw: the first token, 64 serial tokens and 64 drafted tokens (MTP at the default depth,
  confidence 0 so every verify window is full) from CUDA graphs, greedy and seeded; drafted equal serial in every
  setting, and every setting equal to the first.
- GLM-5.3 (the sushi-experts pack and its DFlash2 drafter): eager replies serial and drafted, greedy and seeded, then
  serial replies from CUDA graphs captured in each setting.

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

MODELS = Path(os.environ.get("MODELS", "/home/agis/models"))
FN_PACK = MODELS / "Qwen3.8-Flash-Next-exl3-3.05bpw"
GLM = MODELS / "glm53-flash-sushi-experts-mosaic-dense"
GLM_DRAFTER = MODELS / "GLM-5.3-Flash-DFlash2-TR3-v3"
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95)]
ORDER = (False, True, False)                    # off twice: a difference is the switches', not noise


@pytest.fixture(autouse=True)
def restore():
    before = switches.current()
    yield
    switches.use(before)


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

    w = load(FN_PACK, mtp=True, draft_vocab=None, f16_narrow=True)
    yield w
    del w
    _free()


def _fn_reply(w, prompt, sampling):
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    e = Engine(w, capacity=8192, max_rows=8, prefill_rows=512, graphs=True)
    first = prefill(e, prompt, sampling)
    serial = serial_decode(e, first, 64, sampling).tokens
    assert prefill(e, prompt, sampling) == first
    drafted = mtp_decode(e, first, 64, sampling, confidence=0.0)
    out = (first, serial, drafted.tokens, drafted.drafted)
    del e
    _free()
    return out


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "seeded"])
def test_flash_next_replies_keep_their_bytes_with_the_switches_off_on_off(flash_next, sampling):
    w = flash_next
    prompt = _prompt(1500, 5, 150_000)
    got = []
    for on in ORDER:
        switches.use(switches.Switches(expert_decode=on, pdl=on))
        w.x3.narrow = on
        got.append(_fn_reply(w, prompt, sampling))
    for setting, g in zip(ORDER, got):
        assert g[2] == g[1], setting                   # drafted equal serial
        assert g[3] > 0, setting                       # the drafts ran
    assert got[0][:3] == got[2][:3]
    assert got[0][:3] == got[1][:3]


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
def test_glm_eager_replies_keep_their_bytes_with_the_switches_off_on_off(glm, sampling):
    prompt = _prompt(600, 11, 150_000)
    got = []
    for on in ORDER:
        switches.use(switches.Switches(expert_decode=on, pdl=on))
        got.append((_glm_reply(glm, prompt, sampling, False), _glm_reply(glm, prompt, sampling, True)))
    for setting, (serial, drafted) in zip(ORDER, got):
        assert drafted == serial, setting
    assert got[0] == got[2]
    assert got[0] == got[1]


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "seeded"])
def test_glm_graph_replies_keep_their_bytes_with_the_switches_off_on_off(glm, sampling):
    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill, serial_decode

    prompt = _prompt(600, 13, 150_000)
    got = []
    for on in ORDER:
        switches.use(switches.Switches(expert_decode=on, pdl=on))
        e = Engine(glm.w, capacity=8192, max_rows=8, prefill_rows=128, graphs=True, long_context=True)
        first = prefill(e, prompt, sampling, mtp=False)
        got.append(serial_decode(e, first, 48, sampling).tokens)
        assert e.graphs is not None
        del e
        _free()
    assert got[0] == got[2]
    assert got[0] == got[1]
