"""Nemotron 3.5 Bolt's prompt reuse on CPU: states kept inside prompt chunks, and a resumed prompt's reply against a
fresh prefill's, through ``engine.prefill`` and ``BoltEngine.generate`` on the fake engine and a real tiny LiLiCorr
drafter. The store and ``keep_points`` are tested in ``test_kept_states``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bolt_lilicorr_fakes import FakeEngine, random_tensors, tiny_config

from tensorfold.cuda.kept_states import Store
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.nemotron_bolt.cuda.lilicorr import (
    DraftConfig,
    Drafter,
    expected_shapes,
    lilicorr_decode,
)

SAMPLINGS = [None, Sampling(seed=77, temperature=1.0, top_k=20, top_p=0.95)]
OPENER = 1                              # the fake chat template's message start


def drafter(target: FakeEngine) -> Drafter:
    cfg = DraftConfig.from_json(tiny_config(window=6))
    return Drafter((cfg, random_tensors(expected_shapes(cfg), 0)), embed=target.embed, lm_head=target.lm_head,
                   device="cpu", dtype=torch.float32)


def text(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return [int(t) for t in torch.randint(10, 300, (n,), generator=g)]


def marks(ids) -> list[int]:
    return [i for i, t in enumerate(ids) if t == OPENER]


def prefill(e: FakeEngine, d: Drafter | None, prompt, sampling, **kw) -> int:
    from tensorfold.families.nemotron_bolt.cuda.engine import prefill as run

    return run(e, prompt, sampling, on_taps=d.add_taps if d is not None else None, **kw)


def reference(prompt: list[int], count: int, sampling) -> list[int]:
    """A fresh prefill, then one verify row a token."""

    e = FakeEngine()
    out = [prefill(e, None, prompt, sampling)]
    while len(out) < count:
        logits = e.verify([out[-1]])
        out.append(e.sample(logits, [e.pos + 1], sampling)[0])
        e.commit(1)
    return out


# -- prefill ------------------------------------------------------------------------------------------------------
def kept_prefill(prompt, points, sampling=None, rows: int = 64):
    e = FakeEngine(prefill_rows=rows)
    d = drafter(e)
    got = {}

    def keep(n, st):
        got[n] = (st, d.snapshot())

    pending = prefill(e, d, prompt, sampling, keeps=points, keep=keep)
    return e, d, pending, got


def test_prefill_keeps_each_state_as_a_fresh_pass_reaches_it():
    prompt = text(300, 1)
    e, d, _, got = kept_prefill(prompt, [100, 128, 299, 400])
    assert e.chunks == [64, 64, 64, 64, 44]                          # kept states never add chunks
    assert sorted(got) == [100, 128, 299]
    for n, (st, ctx) in got.items():
        assert st == {"ids": prompt[:n], "pos": n} and ctx["context_end"] == n
    assert d.context_end == e.pos == 300


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "topk"])
@pytest.mark.parametrize("at", [100, 128, 299])
def test_a_resumed_prompt_replies_as_a_fresh_one(sampling, at):
    first = text(300, 1)
    _, _, _, got = kept_prefill(first, [at])
    st, ctx = got[at]
    prompt = first + text(150, 2)
    fresh = FakeEngine()
    df = drafter(fresh)
    want = prefill(fresh, df, prompt, sampling)
    e = FakeEngine()
    d = drafter(e)
    d.restore(ctx)
    assert prefill(e, d, prompt, sampling, resume=st) == want
    assert e.ids == fresh.ids and d.context_end == df.context_end == len(prompt)
    assert e.chunks[0] == min(64, len(prompt) - at)                  # only the new rows were prefilled
    reply = lilicorr_decode(e, d, want, 40, sampling).tokens
    assert reply == lilicorr_decode(fresh, df, want, 40, sampling).tokens == reference(prompt, 40, sampling)


def test_a_resume_must_leave_a_token_to_prefill():
    e = FakeEngine()
    with pytest.raises(ValueError, match="extend the kept positions"):
        prefill(e, None, [3, 4], None, resume={"ids": [3, 4], "pos": 2})
    with pytest.raises(ValueError, match="need a callback"):
        prefill(e, None, [3, 4, 5], None, keeps=[2])


# -- the server engine ---------------------------------------------------------------------------------------------
def bolt(*, keep: bool = True):
    from tensorfold.families.nemotron_bolt.cuda.app import BoltEngine

    b = object.__new__(BoltEngine)
    b.e = FakeEngine(prefill_rows=64)
    b.drafter = drafter(b.e)
    b.store = Store(8, 1 << 30 if keep else 0)
    b.marks, b.eos, b.max_len, b.drafts, b.adaptive = marks, (), 8192, 7, False
    return b


def reply(b, prompt, count, sampling=None, **kw):
    out: list[int] = []
    stats = b.generate(prompt, count, sampling, lambda new: out.extend(new), stop_eos=False, **kw)
    return out, stats


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "topk"])
def test_a_conversation_resumes_each_turn(sampling):
    warm, cold = bolt(), bolt(keep=False)
    system = text(400, 3) + [OPENER]
    prompt, resumes = system + text(300, 4), 0
    for turn in range(3):
        got, stats = reply(warm, prompt, 30, sampling)
        want, base = reply(cold, prompt, 30, sampling)
        assert got == want and base["cached"] == 0
        # turn 1 starts afresh; later turns resume from the previous prompt's end less one token
        assert stats["cached"] == resumes and stats["drafted"] > 0
        resumes = len(prompt) - 1
        prompt = prompt + got + [OPENER] + text(280, 5 + turn)
    # another conversation with the same system block resumes from its end, a message start
    other = system + text(200, 9)
    got, stats = reply(warm, other, 20, sampling)
    assert stats["cached"] == len(system) - 1 and got == reply(cold, other, 20, sampling)[0]


def test_a_grammar_request_keeps_states_no_drafting_request_resumes():
    b = bolt()
    prompt = text(500, 6)
    b.store.add(prompt[:300], {"ids": prompt[:300], "pos": 300}, None)    # as a grammar request keeps it
    got, stats = reply(b, prompt, 10)
    assert stats["cached"] == 0 and got == reply(bolt(keep=False), prompt, 10)[0]
