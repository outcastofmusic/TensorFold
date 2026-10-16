"""Nemotron 3.5 Bolt's CUDA engine on a tiny checkpoint in the real layout: the kernels against the fp32 reference,
and the property drafting rests on, that a verify window's rows equal serial steps bit for bit."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling

PROMPT = [5, 77, 300, 2, 999, 14, 15, 600, 41, 42, 43, 7, 8, 9, 512, 513, 100, 101, 102, 103, 104]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tests.bolt_tiny import write

    return write(tmp_path_factory.mktemp("bolt"))


@pytest.fixture(scope="module")
def weights(tiny):
    from tensorfold.families.nemotron_bolt.cuda.weights import load

    return load(tiny, "cuda", raw=True)


def _engine(weights, **kw):
    from tensorfold.families.nemotron_bolt.cuda.engine import Engine

    return Engine(weights, max_len=1024, **kw)


def test_the_dequant_kernel_is_the_torch_definition():
    from tensorfold.families.nemotron_bolt.cuda.fallback import dequantize_iq2xxs, dequantize_iq2xxs_torch
    from tests.bolt_tiny import iq2xxs

    packed = iq2xxs(300, 1024, torch.Generator().manual_seed(4)).reshape(300, -1).cuda()
    assert torch.equal(dequantize_iq2xxs(packed), dequantize_iq2xxs_torch(packed))


def test_prefill_and_decode_follow_the_reference(weights):
    from tensorfold.families.nemotron_bolt.cuda import reference
    from tensorfold.families.nemotron_bolt.cuda.engine import prefill

    e = _engine(weights, prefill_rows=8, taps=(1, 3))         # three chunks: the states carry across them
    pending = prefill(e, PROMPT, None)
    taps = e.tap_rows(len(PROMPT) - 16)
    rows = [pending]
    logits = []
    for _ in range(6):
        logits.append(e.verify([rows[-1]])[0].float().clone())
        rows.append(e.tokens()[0])
        e.commit(1)
    want, streams = reference.forward(weights, torch.tensor(PROMPT + rows[:-1], device="cuda"), taps=(1, 3))
    assert pending == int(want[len(PROMPT) - 1].argmax())
    got = torch.stack(logits)
    ref = want[len(PROMPT):]
    scale = ref.abs().max()
    assert (got - ref).abs().max() / scale < 0.05
    assert (got.argmax(-1) == ref.argmax(-1)).float().mean() >= 5 / 6
    tail = torch.cat([s[16:len(PROMPT)] for s in streams], dim=1)      # the last chunk's rows
    assert taps.shape == (len(PROMPT) - 16, 2 * 512)
    assert ((taps.float() - tail).abs().max() / tail.abs().max()) < 0.05


@pytest.mark.parametrize("sampling", [None, Sampling(seed=7, temperature=1.0, top_k=20, top_p=0.95)],
                         ids=["greedy", "keyed"])
def test_window_rows_equal_serial_steps(weights, sampling):
    from tensorfold.families.nemotron_bolt.cuda.engine import prefill

    e = _engine(weights)
    window = [11, 12, 13, 14, 15, 16, 17]
    pending = prefill(e, PROMPT, sampling)
    serial_logits, serial_tokens = [], []
    for tok in [pending] + window:
        serial_logits.append(e.verify([tok])[0].clone())
        serial_tokens.append(e.tokens()[0])
        e.commit(1)
    # the same positions as windows: keep 3 of the first window, so the Mamba states replay kept rows
    assert prefill(e, PROMPT, sampling) == pending
    first = e.verify([pending] + window[:4]).clone()
    assert e.tokens() == serial_tokens[:5]
    e.commit(3)
    second = e.verify(window[2:]).clone()
    assert e.tokens() == serial_tokens[3:]
    for r in range(5):
        assert torch.equal(first[r], serial_logits[r]), r
    for r in range(5):
        assert torch.equal(second[r], serial_logits[3 + r]), r


def test_taps_of_kept_rows_equal_serial_taps(weights):
    from tensorfold.families.nemotron_bolt.cuda.engine import prefill

    e = _engine(weights, taps=(0, 5))
    pending = prefill(e, PROMPT, None)
    serial = []
    for tok in [pending, 3, 4]:
        e.verify([tok])
        serial.append(e.tap_rows(1).clone())
        e.commit(1)
    prefill(e, PROMPT, None)
    e.verify([pending, 3, 4])
    assert torch.equal(e.tap_rows(3), torch.cat(serial))


def test_engine_generates_the_same_reply_drafted_or_not(tiny):
    from tensorfold.families.nemotron_bolt.cuda.app import BoltEngine

    eng = BoltEngine(tiny, context=512, context_explicit=True)
    got: list[list[int]] = [[], []]
    for i, draft in enumerate((True, False)):
        stats = eng.generate(PROMPT, 12, None, lambda new, i=i: got[i].extend(new) and False, draft=draft,
                             stop_eos=False)
        assert stats["rounds"] == 11
    assert len(got[0]) == 12 and got[0] == got[1]
    stopped: list[int] = []
    eng.generate(PROMPT, 12, None, lambda new: stopped.extend(new) or len(stopped) >= 4, stop_eos=False)
    assert stopped == got[0][:4]
