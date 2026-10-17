"""Nemotron 3.5 Bolt's prompt reuse on CPU: the kept-state store, where a prefill keeps states, and a resumed prompt's
reply against a fresh prefill's, through ``engine.prefill`` and ``BoltEngine.generate`` on the fake engine and a real
tiny LiLiCorr drafter."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bolt_lilicorr_fakes import FakeEngine, random_tensors, tiny_config

from tensorfold.cuda.markers import MIN_GAP
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.nemotron_bolt.cuda.lilicorr import (
    DraftConfig,
    Drafter,
    expected_shapes,
    lilicorr_decode,
)
from tensorfold.families.nemotron_bolt.cuda.reuse import Store, keep_points

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


# -- where states are kept ----------------------------------------------------------------------------------------
def test_keep_points_take_message_starts_and_the_prompt_end():
    p = [5] * 2000
    assert keep_points(p, 0, None) == [1999]
    p[300] = p[900] = p[1990] = OPENER
    # the end gives way to a message start within MIN_GAP before it
    assert keep_points(p, 0, marks) == [300, 900, 1990]
    # nothing within MIN_GAP of the resume point, and the end still counts
    assert keep_points(p[:1500], 100, marks) == [900, 1499]
    assert keep_points(p[:1500], 300 - MIN_GAP + 1, marks) == [900, 1499]
    assert keep_points(p[:1500], 1499, marks) == []                  # resumed at the end: kept already
    assert keep_points([7], 0, None) == [1]


# -- the store ----------------------------------------------------------------------------------------------------
def snap(n: int) -> dict:
    return {"x": torch.zeros(n, dtype=torch.uint8)}


def test_store_resumes_the_longest_prefix_with_a_drafter_context():
    s = Store(8, 1 << 20)
    s.add([1, 2], snap(10), None)
    s.add([1, 2, 3], snap(10), snap(5))
    s.add([1, 2, 3, 4], snap(10), None)
    s.add([9, 9], snap(10), snap(5))
    assert s.longest([1, 2, 3, 4, 5], False).ids == (1, 2, 3, 4)
    assert s.longest([1, 2, 3, 4, 5], True).ids == (1, 2, 3)          # a drafting prompt needs the drafter's context
    assert s.kept[-1].ids == (1, 2, 3)                               # a hit is the newest entry now
    assert s.longest([1, 2, 3], False).ids == (1, 2)                 # one token is always left to prefill
    assert s.longest([8], False) is None
    assert s.held == 4 * 10 + 2 * 5


def test_store_gives_up_a_conversations_earlier_turns_first():
    s = Store(3, 1 << 20)
    system = [7] * 4
    s.add(system, snap(1), None)                                     # a system block's end
    s.add(system + [1], snap(1), None)                               # turn 1
    s.add(system + [1, 2], snap(1), None)                            # turn 2 supersedes turn 1
    s.add([8, 8], snap(1), None)                                     # another conversation
    assert [k.ids for k in s.kept] == [tuple(system), (7, 7, 7, 7, 1, 2), (8, 8)]
    s.add([8, 8, 8], snap(1), None)                                  # nothing between two entries: the oldest goes
    assert [k.ids for k in s.kept] == [(7, 7, 7, 7, 1, 2), (8, 8), (8, 8, 8)]


def test_store_keeps_within_its_bytes():
    s = Store(8, 100)
    s.add([1], snap(40), None)
    s.add([2], snap(40), None)
    s.add([3], snap(40), None)
    assert [k.ids for k in s.kept] == [(2,), (3,)] and s.held == 80
    s.add([4], snap(101), None)                                      # larger than the budget: not kept
    assert [k.ids for k in s.kept] == [(2,), (3,)]
    s.add([3], snap(30), snap(10))                                   # the same ids replace their entry
    assert [k.ids for k in s.kept] == [(2,), (3,)] and s.held == 80
    assert not Store(0, 100).on and not Store(8, 0).on


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
