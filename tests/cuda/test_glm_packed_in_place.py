"""GLM-5.3-Flash's packed projections read and write rows in place, against TF_GLM_PACKED_COPIES=1 (every projection's
rows staged in the scratch and copied back, and the group sums packed projections ignore computed): the same bytes on
the real checkpoint for every projection shape at the engine's layouts, for prompt states and decode logits, and for
replies, eager and from CUDA graphs captured on each path, drafted equal to serial.

Needs the checkpoint (TF_GLM_PACKED_MODEL) and its DFlash2 drafter (TF_GLM_PACKED_DRAFTER), about 100 GiB free."""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.affine6.affine6 import Affine6  # noqa: E402
from tensorfold.cuda.exl3.linear import Exl3Linear  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import packed  # noqa: E402
from tensorfold.families.glm5_next.cuda.packed import ComposedLinear, ProjectionScratch  # noqa: E402

MODEL = Path(os.environ.get("TF_GLM_PACKED_MODEL", "/home/agis/models/glm53-flash-sushi-experts-mosaic-dense"))
DRAFTER = Path(os.environ.get("TF_GLM_PACKED_DRAFTER", "/home/agis/models/GLM-5.3-Flash-DFlash2-TR3-v3"))
pytestmark = pytest.mark.skipif(not (MODEL / "config.json").exists() or not (DRAFTER / "config.json").exists(),
                                reason="the packed GLM checkpoint or its drafter is not here")

CONTEXT = 8192            # past the dense limit: the indexer's projections run every step, as in serving
ROWS = (1, 2, 3, 4, 6, 8, 129, 256)
SAMPLINGS = [None, Sampling(1234, 1.0, 20, 0.95)]


@contextmanager
def copies(on: bool):
    old = packed.COPIES
    packed.COPIES = on
    try:
        yield
    finally:
        packed.COPIES = old


_held: dict[str, object] = {}


def _engine(kind: str):
    """One engine at a time, as each holds the whole model: "eager", or "graphs-old" / "graphs-new" (captured on that
    path)."""

    if kind not in _held:
        _held.clear()
        gc.collect()
        torch.cuda.empty_cache()
        from tensorfold.families.glm5_next.cuda.engine import GlmEngine

        with copies(kind == "graphs-old"):
            _held[kind] = GlmEngine(MODEL, rank=0, master="", port=0, world=1, drafter=DRAFTER, context=CONTEXT,
                                    eager=kind == "eager", calibrate=False)
    return _held[kind]


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _signature(q):
    if isinstance(q, ComposedLinear):
        return ("composed",) + tuple(_signature(p) for p in q.parts)
    if isinstance(q, Exl3Linear):
        return ("exl3", q.k, q.n, q.bits, q.codebook)
    if isinstance(q, Affine6):
        return ("affine6", q.k, q.n)
    return ("plain", q.k, q.n, q.weight.dtype)


def _projections(w) -> list:
    """One projection of each shape the engine runs (packed.projections, and the draft head)."""

    seen: dict = {}
    for q in [*packed.projections(w), *([w.draft_head] if w.draft_head is not None else [])]:
        seen.setdefault(_signature(q), q)
    return list(seen.values())


def _prompt(n: int, seed: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(100, 100000, size=n)]


def test_every_projection_shape_keeps_its_bytes_in_place():
    """Each shape at 1 to 256 rows (two 128-row chunks), from contiguous, column-sliced and 2-byte-misaligned bf16
    rows, into contiguous bf16 and fp32 outputs and into aligned and misaligned column slices of a wider one: the
    in-place path writes the same bytes as the copies path, and nothing outside the output."""

    w = _engine("eager").w
    qs = _projections(w)
    assert {s[0] for s in map(_signature, qs)} >= {"composed", "exl3", "plain"}
    s = ProjectionScratch(qs, max(ROWS), w.device)
    gen = torch.Generator(device=w.device).manual_seed(5)
    checked = 0
    for q in qs:
        for R in ROWS:
            wide = torch.randn((R, q.k + 128), generator=gen, device=w.device).to(torch.bfloat16)
            for x_name, x in (("contiguous", wide[:, :q.k].contiguous()), ("column slice", wide[:, 128:]),
                              ("misaligned", wide[:, 1:q.k + 1])):
                for out_name, dtype, pad in (("bf16", torch.bfloat16, 0), ("fp32", torch.float32, 0),
                                             ("column slice", torch.bfloat16, 192), ("misaligned", torch.bfloat16, 2)):
                    got = []
                    for on in (True, False):
                        parent = torch.empty((R, q.n + pad), dtype=dtype, device=w.device)
                        parent.view(torch.uint8).fill_(0x5A)
                        out = parent[:, pad // 2:pad // 2 + q.n] if pad else parent
                        with copies(on):
                            packed.matmul(q, x, out, s)
                        torch.cuda.synchronize()
                        got.append(_bytes(parent))
                    assert got[0] == got[1], (_signature(q), R, x_name, out_name)
                    checked += 1
    print(f"\n{len(qs)} projection shapes, {checked} layouts byte-identical")


def test_prompt_states_and_decode_logits_keep_their_bytes():
    """A prompt chunked at 128 rows leaves the same state and first token on both paths, and decode windows of 1 to 6
    rows on it give the same logits, carrier rows and uncommitted KDA projections."""

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill
    from tensorfold.families.glm5_next.cuda.forward import forward

    w = _engine("eager").w
    prompt = _prompt(300, 7)
    seen = []
    for on in (True, False):
        with copies(on):
            e = Engine(w, capacity=CONTEXT, max_rows=8, prefill_rows=128, long_context=True)
            first = prefill(e, prompt, None, mtp=False)
            st = e.st
            state = [st.rec[st.cur[0]], st.conv] + [x[:st.pos] for x in st.kc if x is not None]
            state += [t for trio in st.index or () for t in trio]
            steps = []
            for R in range(1, 7):
                logits = forward(w, st, e.buf, _prompt(R, 100 + R))
                steps.append((_bytes(logits), _bytes(e.buf.x[:R]), _bytes(st.proj[:, :R])))
            seen.append((first, [_bytes(t) for t in state], steps))
            del e
            torch.cuda.empty_cache()
    assert seen[0][0] == seen[1][0]
    assert seen[0][1] == seen[1][1]
    for R, (a, b) in enumerate(zip(seen[0][2], seen[1][2]), start=1):
        assert a == b, R


def _replies(engine, prompt, sampling, draft, tokens=32):
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


PROMPTS = {"short": _prompt(40, 11), "past the dense limit": _prompt(2100, 13)}


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "sampled"])
def test_eager_replies_match_and_drafted_equal_serial(sampling):
    engine = _engine("eager")
    for name, prompt in PROMPTS.items():
        got = {}
        for on in (True, False):
            with copies(on):
                got[on] = (_replies(engine, prompt, sampling, False), _replies(engine, prompt, sampling, True))
        assert got[True][0] == got[False][0], name
        assert got[False][1] == got[False][0], name           # drafted equal serial on the in-place path
        assert got[True][1] == got[True][0], name


def test_a_decode_step_launches_fewer_kernels():
    """Device work (kernels and copies) of one eager decode window, counted by the profiler: printed for the record."""

    from torch.profiler import ProfilerActivity, profile

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill
    from tensorfold.families.glm5_next.cuda.forward import forward

    w = _engine("eager").w
    e = Engine(w, capacity=CONTEXT, max_rows=8, prefill_rows=128, long_context=True)
    prefill(e, _prompt(40, 17), None, mtp=False)
    counts = {}
    for R in (1, 4):
        for on in (True, False):
            with copies(on):
                forward(w, e.st, e.buf, _prompt(R, 3))           # Triton specializations compile outside the count
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as p:
                    forward(w, e.st, e.buf, _prompt(R, 3))
                    torch.cuda.synchronize()
                counts[(R, on)] = sum(1 for ev in p.events() if ev.device_type == torch.autograd.DeviceType.CUDA)
    for R in (1, 4):
        print(f"\nR={R}: {counts[(R, True)]} device ops with copies, {counts[(R, False)]} in place, "
              f"{counts[(R, True)] - counts[(R, False)]} fewer")
        assert counts[(R, False)] < counts[(R, True)]


_graph_replies: dict[str, dict] = {}


@pytest.mark.parametrize("kind", ["graphs-old", "graphs-new"])
def test_graph_replies(kind):
    """Graphs captured on each path: drafted equal serial (both replay graphs), recorded for the comparison below."""

    engine = _engine(kind)
    assert engine.e.graphs is not None
    got = {}
    with copies(kind == "graphs-old"):
        for name, prompt in PROMPTS.items():
            for i, sampling in enumerate(SAMPLINGS):
                serial = _replies(engine, prompt, sampling, False)
                assert _replies(engine, prompt, sampling, True) == serial, (name, i)
                got[(name, i)] = serial
    assert engine.e.replays["main"] > 0
    _graph_replies[kind] = got


def test_graphs_captured_on_either_path_give_the_same_replies():
    if len(_graph_replies) != 2:
        pytest.skip("runs after both test_graph_replies cases")
    assert _graph_replies["graphs-old"] == _graph_replies["graphs-new"]
