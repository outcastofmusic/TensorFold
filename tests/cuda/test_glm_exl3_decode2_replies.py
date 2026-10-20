"""GLM-5.3 (the sushi-experts pack) replies with TF_EXL3_GROUP_SCAN_DECODE and TF_EXL3_LINEAR_PAIR off, on, off on one
load: GLM shares the EXL3 grouping and linear. Eager replies serial and drafted (the DFlash2 drafter) after a
2,600-token prompt, greedy and seeded, then serial replies from CUDA graphs captured in each setting.

Run it in its own pytest process, not after test_exl3_decode2_replies.py: the GLM engine's memory admission needs
about 100 GiB free, which a process that held the Flash Next pack does not give back."""

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
GLM = MODELS / "glm53-flash-sushi-experts-mosaic-dense"
GLM_DRAFTER = MODELS / "GLM-5.3-Flash-DFlash2-TR3-v3"
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95)]
DEFAULTS = switches.Switches(group_scan=True, down_fuse=True, gateup_fuse=True, group_scan_decode=False,
                             linear_pair=False)
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


def _set(on: bool) -> None:
    """The two switches on or off, every other EXL3 switch at its default (fba05ef2: the prompt switches on)."""

    switches.use(dataclasses.replace(DEFAULTS, group_scan_decode=on, linear_pair=on))


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
    prompt = _prompt(2600, 11, 150_000)
    got = []
    for on in ORDER:
        _set(on)
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
        _set(on)
        e = Engine(glm.w, capacity=8192, max_rows=8, prefill_rows=128, graphs=True, long_context=True)
        first = prefill(e, prompt, sampling, mtp=False)
        got.append(serial_decode(e, first, 48, sampling).tokens)
        assert e.graphs is not None
        del e
        _free()
    assert got[0] == got[2]
    assert got[0] == got[1]
