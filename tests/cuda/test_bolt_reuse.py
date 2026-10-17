"""Nemotron 3.5 Bolt's prompt reuse on the GPU: prompt rows ignore the chunk they come in, so a prompt resumed from a
kept state ends in a fresh prefill's state bit for bit and replies as a fresh prompt does. The tiny checkpoint always;
the real model and its LiLiCorr drafter where they are present (TF_BOLT_TARGET, TF_BOLT_DRAFT)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling

SAMPLINGS = [None, Sampling(seed=7, temperature=1.0, top_k=20, top_p=0.95)]
FIRST = [int(t) for t in torch.randint(3, 1024, (300,), generator=torch.Generator().manual_seed(1))]
MORE = [int(t) for t in torch.randint(3, 1024, (140,), generator=torch.Generator().manual_seed(2))]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tests.bolt_tiny import write

    return write(tmp_path_factory.mktemp("bolt"))


@pytest.fixture(scope="module")
def weights(tiny):
    from tensorfold.families.nemotron_bolt.cuda.weights import load

    return load(tiny, "cuda")


def test_prompt_experts_ignore_the_row_count():
    """A prompt call's row gets the same bits alone, in a few rows (the decode kernel's range) or in many."""

    from tensorfold.families.nemotron_bolt.cuda import iq2xxs
    from tests.bolt_tiny import iq2xxs as packed

    g = torch.Generator().manual_seed(3)
    lin = packed(256, 1024, g).cuda()
    experts = torch.stack([packed(256, 1024, g) for _ in range(12)]).cuda()
    x = torch.randn((1500, 1024), generator=g).to(torch.bfloat16).cuda()
    ids = torch.randint(0, 12, (1500,), generator=g).to(torch.int32).cuda()
    whole_lin = iq2xxs.iq2xxs_linear(x, lin, prompt=True)
    whole_grouped = iq2xxs.iq2xxs_grouped(x, experts, ids, prompt=True)
    for a, b in ((0, 1), (1, 10), (10, 300), (300, 900)):
        assert torch.equal(iq2xxs.iq2xxs_linear(x[a:b], lin, prompt=True), whole_lin[a:b]), (a, b)
        assert torch.equal(iq2xxs.iq2xxs_grouped(x[a:b], experts, ids[a:b], prompt=True), whole_grouped[a:b]), (a, b)


def _engine(weights, **kw):
    from tensorfold.families.nemotron_bolt.cuda.engine import Engine

    return Engine(weights, max_len=1024, prefill_rows=64, taps=(1, 3), **kw)


def _decode(e, pending: int, steps: int) -> tuple[list[torch.Tensor], list[int]]:
    logits, toks = [], [pending]
    for _ in range(steps):
        logits.append(e.verify([toks[-1]]).clone())
        toks.append(e.tokens()[0])
        e.commit(1)
    return logits, toks


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "keyed"])
@pytest.mark.parametrize("at", [100, 128, 299])        # inside a chunk, a chunk's end, the prompt end less one
def test_a_resumed_prefill_ends_in_the_fresh_state(weights, sampling, at):
    from tensorfold.families.nemotron_bolt.cuda.engine import prefill

    kept = {}
    e = _engine(weights)
    prefill(e, FIRST, sampling, keeps=[at], keep=lambda n, st: kept.setdefault(n, st))
    prompt = FIRST + MORE
    fresh = _engine(weights)
    want = prefill(fresh, prompt, sampling)
    assert prefill(e, prompt, sampling, resume=kept[at]) == want
    n = len(prompt)
    assert torch.equal(e.ssm, fresh.ssm) and torch.equal(e.conv_base, fresh.conv_base)
    for a, b in zip(e.k_cache + e.v_cache, fresh.k_cache + fresh.v_cache):
        assert torch.equal(a[:n], b[:n])
    got, got_toks = _decode(e, want, 8)
    ref, ref_toks = _decode(fresh, want, 8)
    assert got_toks == ref_toks and all(torch.equal(a, b) for a, b in zip(got, ref))


def _cached(turn: int, sampling, prompt: list[int], prev: list[int]) -> int:
    """What a turn resumes from: nothing at first, the previous prompt's end less one, the same prompt's when repeated."""

    return len(prompt) - 1 if sampling is not None else len(prev) - 1 if turn else 0


def _both(eng, prompt: list[int], count: int, sampling) -> tuple[list[list[int]], list[dict]]:
    """The reply drafted from kept states, then the reference's (``draft: false``: a fresh prefill), with their stats."""

    got: list[list[int]] = [[], []]
    stats = [eng.generate(prompt, count, sampling, lambda new, i=i: got[i].extend(new) and False, draft=draft,
                          stop_eos=False) for i, draft in enumerate((True, False))]
    return got, stats


def test_bolt_engine_resumes_a_conversation(tiny):
    from tensorfold.families.nemotron_bolt.cuda.app import BoltEngine

    eng = BoltEngine(tiny, context=1024, context_explicit=True)
    prompt, prev = list(FIRST), []
    for turn in range(3):
        for sampling in SAMPLINGS:                       # greedy first, then the same prompt sampled
            got, stats = _both(eng, prompt, 12, sampling)
            assert got[0] == got[1] and len(got[0]) == 12
            assert stats[1]["cached"] == 0 and stats[0]["cached"] == _cached(turn, sampling, prompt, prev)
        prompt, prev = prompt + got[0] + MORE[:40 + 20 * turn], prompt


TARGET = Path(os.environ.get("TF_BOLT_TARGET", "/home/agis/models/Nemotron-3.5-Bolt-74B-A9B-24GB"))
DRAFT = Path(os.environ.get("TF_BOLT_DRAFT", "/home/agis/models/Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"))
TEXT = ("The lighthouse keeper wrote in his log every evening, noting the wind, the color of the sea and the ships "
        "that passed the point. Over thirty years the entries grew longer, and his daughter, reading them later, "
        "found that the weather was only ever an excuse to describe the people he missed. ")


@pytest.mark.skipif(not (TARGET / "model.safetensors.index.json").exists() or not (DRAFT / "config.json").exists(),
                    reason="needs the Bolt checkpoint and its LiLiCorr drafter")
def test_the_model_resumes_a_conversation_as_a_fresh_prompt_replies():
    """Three turns on the real model and drafter: each resumed turn replies as the fresh reference (draft: false)."""

    from tokenizers import Tokenizer

    from tensorfold.families.nemotron_bolt.cuda.app import BoltEngine

    tok = Tokenizer.from_file(str(TARGET / "tokenizer.json"))
    eng = BoltEngine(TARGET, drafter=DRAFT, context=16384, context_explicit=True)
    prompt, prev = tok.encode(TEXT * 12, add_special_tokens=False).ids, []
    for turn in range(3):
        for sampling in SAMPLINGS:
            got, stats = _both(eng, prompt, 64, sampling)
            assert got[0] == got[1], (turn, sampling)
            assert stats[0]["cached"] == _cached(turn, sampling, prompt, prev), stats[0]
            print(f"turn {turn} {'keyed' if sampling else 'greedy'}: {stats[0]}", flush=True)
        question = tok.encode(" Then she asked: what did he miss most?" * (turn + 1), add_special_tokens=False).ids
        prompt, prev = prompt + got[0] + question, prompt
