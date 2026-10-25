"""GLM-5.3 (the sushi-experts pack) replies as served, on the EXL3 kernels ``routed`` picks from the shape: eager
replies drafted (the DFlash2 drafter) equal serial after a 2,600-token prompt, whose prompt windows of up to 1,024
rows take the listed layout, the grouping scan and the fused epilogues, greedy and seeded; and serial replies from
CUDA graphs equal eager ones. Their bytes against the launches of before each kernel are in the kernel tests
(test_exl3_prefill_kernels.py, test_exl3_decode2_kernels.py, test_exl3_sweep2_kernels.py).

Run it in its own pytest process, not after test_exl3_prefill_replies.py: the GLM engine's memory admission needs
about 100 GiB free, which a process that held the Flash Next pack does not give back."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

MODELS = Path(os.environ.get("MODELS", "/home/agis/models"))
GLM = MODELS / "glm53-flash-sushi-experts-mosaic-dense"
GLM_DRAFTER = MODELS / "GLM-5.3-Flash-DFlash2-TR3-v3"
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95)]


def _prompt(n: int, seed: int, top: int) -> list[int]:
    return torch.randint(10, top, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _free():
    gc.collect()
    torch.cuda.empty_cache()


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
def test_glm_drafted_replies_equal_serial(glm, sampling):
    prompt = _prompt(2600, 11, 150_000)
    assert _glm_reply(glm, prompt, sampling, True) == _glm_reply(glm, prompt, sampling, False)


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "seeded"])
def test_glm_graph_replies_equal_eager(glm, sampling):
    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill, serial_decode

    prompt = _prompt(600, 13, 150_000)
    got = []
    for graphs in (True, False):
        e = Engine(glm.w, capacity=8192, max_rows=8, prefill_rows=128, graphs=graphs, long_context=True)
        first = prefill(e, prompt, sampling, mtp=False)
        got.append(serial_decode(e, first, 48, sampling).tokens)
        assert (e.graphs is not None) is graphs
        del e
        _free()
    assert got[0] == got[1]
