"""TF_GLM_SHARED_STREAM, TF_GLM_KDA_WIDE and TF_GLM_EARLY_LAUNCH on the GPU: every switch keeps the bytes, and the
served token-list kernel writes its torch reference's integers.

Kernels first: ``sparse.token_rows`` (one kernel) against ``sparse.token_rows_ops`` (the torch ops) at decode rows 1 to
8 and a prompt chunk's 2,048, around the dense limit and deep in a 98,304-token window; the KDA chain on the
three-kernel path with the caller's q and read-out rows against the fused kernel at GLM-5.3 Flash's 64 heads and decode
rows 1 to 8 (outputs, saved replay rows, final state).

Then the real checkpoint end to end (TF_GLM_PACKED_MODEL and TF_GLM_PACKED_DRAFTER, about 100 GiB free): replies with
every switch off, on and off again on one eager engine, and from CUDA graphs captured with the switches off and on,
greedy and seeded, serial and drafted, short, past the dense limit (sparse steps, the token list) and over two prompt
chunks (two routed windows of 1,024 rows each), all equal."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import kda, sparse

DEV = "cuda"


# -- the token list ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pos,R", [(0, 1), (2046, 3), (2047, 1), (2048, 6), (2049, 2), (2050, 8), (5000, 6),
                                   (70001, 8), (98299, 5), (2047, 2048), (40000, 2048)])
def test_the_token_list_kernel_writes_the_torch_ops_integers(pos, R):
    g = torch.Generator(device=DEV).manual_seed(pos + R)
    pools = torch.sort(torch.randint(0, max(1, (pos + R) // 4 + 1), (R, sparse.TOPK_POOLS), generator=g, device=DEV),
                       1).values
    pos_dev = torch.tensor([pos], dtype=torch.int32, device=DEV)
    want = sparse.token_rows_ops(pools, pos_dev)
    got = sparse.token_rows(pools, pos_dev)
    for x, y in zip(got, want):
        assert x.dtype == y.dtype == torch.int32 and torch.equal(x, y)


# -- the KDA chain ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 8])
def test_decode_rows_on_the_wide_chain_with_their_own_rows_give_the_fused_bits(rows):
    heads = 64
    g = torch.Generator().manual_seed(rows)
    C = 3 * heads * kda.DK
    b_off = C + 256
    p = (torch.randn((rows, b_off + heads + 32), generator=g) * 0.5).to(torch.bfloat16).cuda()
    a = torch.randn((rows, heads * kda.DK), generator=g).to(torch.bfloat16).cuda()
    gate = torch.randn((rows, heads * kda.DV), generator=g).to(torch.bfloat16).cuda()
    cs = (torch.randn((3, C), generator=g) * 0.5).to(torch.bfloat16).cuda()
    cw = (torch.randn((C, 4), generator=g) * 0.5).to(torch.bfloat16).cuda()
    state = (torch.randn((heads, kda.DV, kda.DK), generator=g) * 0.1).cuda()
    a_log = (torch.rand(heads, generator=g) * 2 - 1).cuda()
    dt_bias = (torch.randn(heads * kda.DK, generator=g) * 0.1).cuda()
    norm_w = (torch.rand(kda.DV, generator=g) + 0.5).to(torch.bfloat16).cuda()
    got = []
    for wide in (False, True):
        sc = kda.KDAScratch(8, heads, DEV)
        tmp = (torch.full((8, heads, kda.DK), float("nan"), device=DEV),
               torch.full((8, heads, kda.DV), float("nan"), dtype=torch.bfloat16, device=DEV)) if wide else None
        out_state = torch.empty_like(state)
        out = kda.chain(p, b_off, a, gate, cs, cw, state, a_log, dt_bias, norm_w, 1e-5, -5.0, rows, sc, out_state,
                        wide=wide, tmp=tmp).clone()
        got.append((out, out_state, sc.k[:rows], sc.v[:rows], sc.g[:rows], sc.b[:rows]))
    for name, x, y in zip(("out", "state", "k", "v", "g", "beta"), *got):
        assert torch.equal(x, y), name


# -- the real checkpoint ------------------------------------------------------------------------------------------
MODEL = Path(os.environ.get("TF_GLM_PACKED_MODEL", "/home/agis/models/glm53-flash-sushi-experts-mosaic-dense"))
DRAFTER = Path(os.environ.get("TF_GLM_PACKED_DRAFTER", "/home/agis/models/GLM-5.3-Flash-DFlash2-TR3-v3"))
real = pytest.mark.skipif(not (MODEL / "config.json").exists() or not (DRAFTER / "config.json").exists(),
                          reason="the packed GLM checkpoint or its drafter is not here")
CONTEXT = 8192            # past the dense limit: sparse steps run the indexer and the token list
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95)]
SWITCHES = ("TF_GLM_SHARED_STREAM", "TF_GLM_KDA_WIDE", "TF_GLM_EARLY_LAUNCH")
_held: dict[str, object] = {}


def _engine(kind: str):
    """One engine at a time, as each holds the whole model: "eager", "graphs-off" or "graphs-on" (captured with every
    switch set that way at startup)."""

    if kind not in _held:
        _held.clear()
        gc.collect()
        torch.cuda.empty_cache()
        from tensorfold.families.glm5_next.cuda.engine import GlmEngine

        old = {k: os.environ.get(k) for k in SWITCHES}
        try:
            for k in SWITCHES:
                os.environ[k] = "0" if kind == "graphs-off" else "1"
            _held[kind] = GlmEngine(MODEL, rank=0, master="", port=0, world=1, drafter=DRAFTER, context=CONTEXT,
                                    eager=kind == "eager", calibrate=False)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return _held[kind]


def _switch(engine, on: bool) -> None:
    """Every switch on every buffer set of an eager engine (graphs keep the path they were captured on)."""

    engine.w.meta["early_launch"] = on
    for b in (engine.e.buf, engine.e.pbuf, engine.e.mbuf):
        if b is None or b.prefill:
            continue
        b.side = torch.cuda.Stream() if on else None
        LL = engine.w.cfg.lin_heads // engine.w.world
        b.kda_tmp = ((torch.empty((b.rows, LL, kda.DK), device=DEV),
                      torch.empty((b.rows, LL, kda.DV), dtype=torch.bfloat16, device=DEV)) if on else None)


def _prompt(n: int, seed: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(100, 100000, size=n)]


def _replies(engine, prompt, sampling, draft, tokens=48):
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


PROMPTS = {"short": _prompt(40, 11), "past the dense limit": _prompt(2100, 13), "two chunks": _prompt(2600, 17)}


@real
@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "sampled"])
def test_eager_replies_are_the_same_with_the_switches_off_on_off_and_drafted_equal_serial(sampling):
    engine = _engine("eager")
    meta = engine.w.meta
    assert meta["shared_stream"] and meta["kda_wide"] and meta["early_launch"]
    for name, prompt in PROMPTS.items():
        got = []
        for on in (False, True, False):        # off twice: a difference is the switches', not the engine's
            _switch(engine, on)
            got.append((_replies(engine, prompt, sampling, False), _replies(engine, prompt, sampling, True)))
        _switch(engine, True)
        assert got[0] == got[2], name
        assert got[0][0] == got[1][0], name
        for serial, drafted in got:
            assert drafted == serial, name


@real
def test_a_stop_from_the_callback_after_an_early_launch_resumes_like_a_fresh_prompt():
    """An early launch's uncommitted step must not leak into the next request's state."""

    engine = _engine("eager")
    _switch(engine, True)
    prompt = PROMPTS["short"]
    want = _replies(engine, prompt, None, False)
    for draft in (False, True):
        seen: list[int] = []
        engine.cache.clear()
        engine.live = []
        engine.generate(list(prompt), 48, None, lambda new, seen=seen: seen.extend(new) or len(seen) >= 5, draft=draft)
        assert _replies(engine, prompt, None, draft) == want, draft


_graph_replies: dict[str, dict] = {}


@real
@pytest.mark.parametrize("kind", ["graphs-off", "graphs-on"])
def test_graph_replies(kind):
    """Graphs captured with the switches off and on: drafted equal serial, recorded for the comparison below."""

    engine = _engine(kind)
    on = kind == "graphs-on"
    meta = engine.w.meta
    assert engine.e.graphs is not None
    assert [meta[k] for k in ("shared_stream", "kda_wide", "early_launch")] == [on] * 3
    assert (engine.e.buf.side is not None) is on and (engine.e.buf.kda_tmp is not None) is on
    got = {}
    for name, prompt in PROMPTS.items():
        for i, sampling in enumerate(SAMPLINGS):
            serial = _replies(engine, prompt, sampling, False)
            assert _replies(engine, prompt, sampling, True) == serial, (name, i)
            got[(name, i)] = serial
    assert engine.e.replays["main"] > 0
    _graph_replies[kind] = got


@real
def test_graphs_captured_with_the_switches_off_or_on_give_the_same_replies():
    if len(_graph_replies) != 2:
        pytest.skip("runs after both test_graph_replies cases")
    assert _graph_replies["graphs-off"] == _graph_replies["graphs-on"]
