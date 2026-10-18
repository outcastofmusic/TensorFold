"""LFM2-MoE's prompt reuse on the CPU: states kept inside and at the end of prompt chunks, and a resumed prompt against
a fresh prefill, through ``engine.prefill`` and ``Lfm2MoeEngine.generate`` on a tiny checkpoint.

A resumed prompt starts its chunks at the kept length instead of at 0, so its bf16 rows can round apart from a fresh
prefill's: the two must agree within the engine's tolerance against the reference and give the same greedy reply.
"""

from __future__ import annotations

import pytest

from tests import lfm2_moe_tiny

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    from tensorfold.families.lfm2_moe.cuda.weights import load

    return load(lfm2_moe_tiny.write(tmp_path_factory.mktemp("lfm2_moe") / "fp8"), "cpu")


def text(n: int, seed: int) -> list[int]:
    return torch.randint(10, 500, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def engine(weights, prefill_rows: int = 2048):
    from tensorfold.families.lfm2_moe.cuda.engine import Engine

    return Engine(weights, max_len=1024, prefill_rows=prefill_rows)


def close(got: torch.Tensor, want: torch.Tensor, rel: float = 0.01) -> bool:
    return float((got.float() - want.float()).abs().max()) <= rel * float(want.float().abs().max())


def test_a_state_kept_inside_a_chunk_is_the_state_a_prefill_of_its_prefix_reaches(weights):
    from tensorfold.families.lfm2_moe.cuda.engine import prefill

    ids = text(40, 1)
    kept: dict[int, dict] = {}
    whole = engine(weights)
    prefill(whole, ids, None, keeps=[13, 40], keep=lambda n, st: kept.setdefault(n, st))
    assert sorted(kept) == [13, 40] and kept[40]["pos"] == 40
    for n in (13, 40):
        alone = engine(weights)
        prefill(alone, ids[:n], None)
        want = alone.state(n)
        assert close(kept[n]["conv"], want["conv"])
        assert all(k.shape[0] == n and close(k, w) for k, w in zip(kept[n]["k"] + kept[n]["v"], want["k"] + want["v"]))


def test_a_resumed_prompt_continues_as_a_fresh_one(weights):
    """Chunks of 16 rows: the resumed prompt's chunks start at 21, the fresh one's at 0."""

    from tensorfold.families.lfm2_moe.cuda.engine import prefill

    ids = text(60, 2)
    kept: dict[int, dict] = {}
    first = engine(weights, prefill_rows=16)
    prefill(first, ids[:30], None, keeps=[21], keep=lambda n, st: kept.setdefault(n, st))
    resumed, fresh = engine(weights, prefill_rows=16), engine(weights, prefill_rows=16)
    a = prefill(resumed, ids, None, resume=kept[21])
    b = prefill(fresh, ids, None)
    assert a == b and resumed.pos == fresh.pos == len(ids)
    assert close(resumed.forward([a], prompt=False), fresh.forward([b], prompt=False))


def test_a_resume_must_leave_a_token_to_prefill(weights):
    from tensorfold.families.lfm2_moe.cuda.engine import prefill

    ids = text(20, 3)
    e = engine(weights)
    prefill(e, ids, None)
    with pytest.raises(ValueError, match="extend the kept positions"):
        prefill(engine(weights), ids, None, resume=e.state(20))


def app_engine(weights, entries: int = 8, budget: int = 1 << 30):
    """``Lfm2MoeEngine`` around a CPU engine: its constructor loads onto the GPU, its ``generate`` does not."""

    from tensorfold.families.lfm2_moe.cuda.app import Lfm2MoeEngine
    from tensorfold.families.nemotron_bolt.cuda.reuse import Store

    be = Lfm2MoeEngine.__new__(Lfm2MoeEngine)
    be.e, be.max_len, be.eos = engine(weights, prefill_rows=128), 1024, ()
    be.marks = lambda ids: [i for i, t in enumerate(ids) if t == 1]      # the message starts: token 1
    be.store = Store(entries, budget)
    return be


def reply(be, prompt: list[int], count: int = 6, draft: bool = True) -> tuple[list[int], dict]:
    out: list[int] = []
    stats = be.generate(prompt, count, None, lambda new: out.extend(new) and False, draft=draft, stop_eos=False)
    return out, stats


def test_a_conversation_resumes_each_turn_and_replies_as_a_fresh_prefill(weights):
    from tensorfold.cuda.markers import MIN_GAP

    be = app_engine(weights)
    system = text(MIN_GAP + 20, 4)
    turn1 = system + [1] + text(MIN_GAP + 30, 5)
    out1, stats1 = reply(be, turn1)
    assert stats1["cached"] == 0
    # kept: the second message's start and the prompt end less one token
    assert sorted(len(k.ids) for k in be.store.kept) == [len(system), len(turn1) - 1]
    turn2 = turn1 + out1 + [1] + text(40, 6)
    out2, stats2 = reply(be, turn2)
    assert stats2["cached"] == len(turn1) - 1
    assert out2 == reply(app_engine(weights), turn2, draft=False)[0]
    other = system + [1] + text(50, 7)                  # another conversation on the same system block
    out3, stats3 = reply(be, other)
    assert stats3["cached"] == len(system) and out3 == reply(app_engine(weights), other, draft=False)[0]


def test_draft_false_neither_resumes_nor_keeps(weights):
    be = app_engine(weights)
    prompt = text(300, 8)
    reply(be, prompt, draft=False)
    assert not be.store.kept
    reply(be, prompt)
    _, stats = reply(be, prompt + [9, 9], draft=False)
    assert stats["cached"] == 0 and len(be.store.kept) == 1
