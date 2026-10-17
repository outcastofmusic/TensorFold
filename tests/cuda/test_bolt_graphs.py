"""Nemotron 3.5 Bolt's window graphs on the GPU: a replayed window gives the eager window's logits, samples and taps bit
for bit at 1, 2, 8 and 16 rows, greedy, keyed and masked, across partial commits; drafted replies equal serial ones with
graphs on; and what graphs do to speed and memory on the real model (TF_BOLT_TARGET, TF_BOLT_DRAFT)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling

PROMPT = [5, 77, 300, 2, 999, 14, 15, 600, 41, 42, 43, 7, 8, 9, 512, 513, 100, 101, 102, 103, 104]
# (rows, kept): partial commits make the next window's Mamba states replay kept rows from the device's prev_keep
SCHEDULE = [(1, 1), (2, 1), (8, 5), (16, 16), (1, 1), (8, 8), (2, 2), (16, 3), (8, 1)]
SAMPLINGS = [None, Sampling(seed=7, temperature=1.0, top_k=20, top_p=0.95),
             Sampling(seed=5, temperature=0.8, top_k=0, top_p=0.9)]
IDS = ["greedy", "keyed", "nucleus"]

TARGET = Path(os.environ.get("TF_BOLT_TARGET", "/home/agis/models/Nemotron-3.5-Bolt-74B-A9B-24GB"))
DRAFT = Path(os.environ.get("TF_BOLT_DRAFT", "/home/agis/models/Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"))
full_model = pytest.mark.skipif(not (TARGET / "model.safetensors.index.json").exists()
                                or not (DRAFT / "config.json").exists(),
                                reason="needs the Bolt checkpoint and its LiLiCorr drafter")


class Grammar:
    """A constraint that allows a fixed random half of the vocabulary at every row of a window."""

    def window(self, tokens, parents):
        return SimpleNamespace(rows=list(range(len(tokens))))

    def allowed(self, window, width, device, offset):
        g = torch.Generator().manual_seed(len(window.rows))
        return (torch.rand((len(window.rows), width), generator=g) < 0.5).to(device)


def _windows(e, prompt, sampling, *, graphs: bool, grammar=None, schedule=SCHEDULE) -> list[tuple]:
    """Prefill ``prompt``, then the schedule's windows (the pending token, then made-up drafts), each kept in part:
    every window's logits, samples, taps and whether it captured its graph."""

    from tensorfold.families.nemotron_bolt.cuda.engine import prefill

    kept, e.graphs = e.graphs, (e.graphs if graphs else None)
    try:
        tok = prefill(e, prompt, sampling)
        out = []
        for i, (rows, keep) in enumerate(schedule):
            window = [tok] + [(31 * tok + 17 * j + i) % 997 + 3 for j in range(1, rows)]
            if grammar is not None:
                e.mask(grammar, grammar.window(window, None))
            logits = e.verify(window).clone()
            sampled = e.tokens()
            e.mask(None, None)
            out.append((logits, sampled, e.tap_rows(rows).clone(), e.captured))
            e.commit(keep)
            tok = sampled[keep - 1]
        return out
    finally:
        e.graphs = kept


def _same(a: list[tuple], b: list[tuple]) -> None:
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x[0].view(torch.int16), y[0].view(torch.int16)), f"window {i}: logits"
        assert x[1] == y[1], f"window {i}: samples"
        assert torch.equal(x[2].view(torch.int16), y[2].view(torch.int16)), f"window {i}: taps"


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    from tensorfold.families.nemotron_bolt.cuda.weights import load
    from tests.bolt_tiny import write

    return load(write(tmp_path_factory.mktemp("bolt")), "cuda")


@pytest.mark.parametrize("masked", [False, True], ids=["plain", "masked"])
@pytest.mark.parametrize("sampling", SAMPLINGS, ids=IDS)
def test_replayed_windows_equal_eager_ones_bit_for_bit(weights, sampling, masked):
    from tensorfold.families.nemotron_bolt.cuda.engine import Engine

    e = Engine(weights, max_len=1024, taps=(1, 3))
    grammar = Grammar() if masked else None
    eager = _windows(e, PROMPT, sampling, graphs=False, grammar=grammar)
    assert len(e.graphs) == 0
    first = _windows(e, PROMPT, sampling, graphs=True, grammar=grammar)     # each size's first window captures
    assert [w[3] for w in first] == [True, True, True, True, False, False, False, False, False]
    again = _windows(e, PROMPT, sampling, graphs=True, grammar=grammar)     # every window replays
    assert not any(w[3] for w in again) and len(e.graphs) == 4
    _same(eager, first)
    _same(eager, again)


def test_warm_captures_on_a_scratch_state(weights):
    from tensorfold.families.nemotron_bolt.cuda.engine import Engine

    e = Engine(weights, max_len=1024, taps=(1, 3))
    eager = _windows(e, PROMPT, None, graphs=False)
    assert e.warm([1, 2, 8, 16]) == 4 and e.pos == 0
    warmed = _windows(e, PROMPT, None, graphs=True)
    assert not any(w[3] for w in warmed)
    _same(eager, warmed)


def test_the_tiny_engine_replies_the_same_with_and_without_graphs(tmp_path, monkeypatch):
    from tensorfold.families.nemotron_bolt.cuda.app import BoltEngine
    from tests.bolt_tiny import write

    tiny = write(tmp_path / "bolt")
    replies = []
    for flag in ("1", "0"):
        monkeypatch.setenv("TF_BOLT_GRAPHS", flag)
        eng = BoltEngine(tiny, context=512, context_explicit=True)
        assert (eng.e.graphs is None) == (flag == "0")
        for sampling in SAMPLINGS[:2]:
            got: list[int] = []
            eng.generate(PROMPT, 24, sampling, lambda new, got=got: got.extend(new) and False, stop_eos=False)
            replies.append(got)
        del eng
    assert replies[:2] == replies[2:]


# -- the real model -------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def eng():
    from tensorfold.families.nemotron_bolt.cuda.app import BoltEngine

    return BoltEngine(TARGET, drafter=DRAFT, context=16384, context_explicit=True)


def _prompt(text: str) -> list[int]:
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    tok = Tokenizer.from_file(str(TARGET / "tokenizer.json"))
    t = ChatTemplate(TARGET)
    rendered = t.template.render(**t.specials, messages=[{"role": "user", "content": text}],
                                 add_generation_prompt=True)
    return tok.encode(rendered, add_special_tokens=False).ids


PROSE = "Explain how a Mamba-2 state space layer differs from attention, in four paragraphs."
CODE = "Write a Python module with a binary search tree class: insert, delete, search, in-order traversal and tests."


@full_model
@pytest.mark.parametrize("sampling", SAMPLINGS[:2], ids=IDS[:2])
def test_full_model_replays_equal_eager_windows(eng, sampling):
    prompt = _prompt(PROSE)
    eng.store.kept.clear()
    eager = _windows(eng.e, prompt, sampling, graphs=False)
    _windows(eng.e, prompt, sampling, graphs=True)                         # captures what start-up did not
    replay = _windows(eng.e, prompt, sampling, graphs=True)
    assert not any(w[3] for w in replay)
    _same(eager, replay)


@full_model
def test_full_model_graph_memory(eng):
    from tensorfold.families.nemotron_bolt.cuda.graphs import graph_bytes

    e = eng.e
    e.graphs.by_set.clear()                                                # start from no graphs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    reserved, free = torch.cuda.memory_reserved(), torch.cuda.mem_get_info()[0]
    captured = e.warm(range(1, 17), Sampling(seed=1, temperature=0.8, top_k=0, top_p=0.9))     # the widest sampler
    torch.cuda.synchronize()
    pool, total = torch.cuda.memory_reserved() - reserved, free - torch.cuda.mem_get_info()[0]
    budget = graph_bytes(e.c, e.max_len, 16, len(e.tap_layers))
    mib = 1 << 20
    print(f"\n{captured} graphs: pool {pool / mib:.0f} MiB, device {total / mib:.0f} MiB "
          f"({(total - pool) / captured / mib:.2f} MiB a graph beside the pool); admission charges "
          f"{budget / mib:.0f} MiB for 4 sets", flush=True)
    assert captured == 16 and pool < budget


@full_model
@pytest.mark.parametrize("sampling", [None, Sampling(seed=11, temperature=0.6, top_k=20, top_p=0.95)],
                         ids=["greedy", "sampled"])
@pytest.mark.parametrize("text", [PROSE, CODE], ids=["prose", "code"])
def test_full_model_drafted_equals_serial_with_graphs(eng, text, sampling):
    prompt = _prompt(text)
    e = eng.e
    graphs = e.graphs
    runs = {}
    try:
        for name in ("warm", "graphs", "eager"):
            e.graphs = None if name == "eager" else graphs
            for draft in (False, True):
                eng.store.kept.clear()
                out: list[int] = []
                stats = eng.generate(prompt, 300, sampling, lambda new, out=out: out.extend(new) and False, draft=draft,
                                     stop_eos=False)
                runs[name, draft] = (out, stats)
    finally:
        e.graphs = graphs
    replies = {key: out for key, (out, _) in runs.items()}
    assert len({tuple(r) for r in replies.values()}) == 1                 # one reply, drafted or not, graphs or not
    s = {key: stats for key, (_, stats) in runs.items()}
    print(f"\n{'prose' if text is PROSE else 'code'} {'greedy' if sampling is None else 'sampled'}: "
          f"serial {s['eager', False]['decode_tps']} -> {s['graphs', False]['decode_tps']} tok/s with graphs; "
          f"drafted {s['eager', True]['decode_tps']} -> {s['graphs', True]['decode_tps']} tok/s "
          f"({s['graphs', True]['accepted']}/{s['graphs', True]['drafted']} accepted, "
          f"{s['graphs', True]['serial_rounds']}/{s['graphs', True]['rounds']} serial rounds)", flush=True)
    assert s["graphs", False]["decode_tps"] > s["eager", False]["decode_tps"]
