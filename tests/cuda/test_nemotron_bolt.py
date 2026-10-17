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


def _torch_weighted_sum(y: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
    """The MoE weighted sum as torch ops, one kernel a step: fl(y0 w0), then fl(acc + fl(yj wj)), then bf16."""

    acc = y[:, 0].float() * wts[:, 0:1]
    for j in range(1, y.shape[1]):
        acc = acc + y[:, j].float() * wts[:, j:j + 1]
    return acc.to(torch.bfloat16)


def _awkward_bf16(shape, g: torch.Generator) -> torch.Tensor:
    """Random bf16 with -0, +0, subnormals (bit patterns 0x0001..0x007f, either sign) and large values mixed in."""

    x = (torch.randn(shape, generator=g) * 4).to(torch.bfloat16)
    bits = x.view(torch.int16)
    pick = torch.randint(0, 8, shape, generator=g)
    sub = torch.randint(1, 0x80, shape, generator=g, dtype=torch.int16)
    sign = torch.randint(0, 2, shape, generator=g, dtype=torch.int16) * -0x8000
    bits = torch.where(pick == 0, sub | sign, bits)                                  # subnormals
    bits = torch.where(pick == 1, torch.full_like(bits, -0x8000), bits)              # -0
    bits = torch.where(pick == 2, torch.zeros_like(bits), bits)                      # +0
    big = (torch.randn(shape, generator=g) * 1e30).to(torch.bfloat16).view(torch.int16)
    return torch.where(pick == 3, big, bits).view(torch.bfloat16)


@pytest.mark.parametrize("rows,k,width", [(1, 9, 1024), (16, 9, 1024), (2048, 9, 1024), (7, 3, 256), (5, 1, 100),
                                          (3, 9, 1000)])
def test_weighted_sum_is_the_torch_steps_bit_for_bit(rows, k, width):
    from tensorfold.families.nemotron_bolt.cuda.engine import weighted_sum

    g = torch.Generator().manual_seed(rows * 13 + k)
    y = _awkward_bf16((rows, k, width), g).cuda()
    wide = torch.rand((rows, k + 2), generator=g) * 2.5                              # the router's [R, NS] buffer
    if rows > 1:
        wide[-1, 0] = -0.0                                                           # a -0 weight, off row 0
    wide[-1, -3] = 1e-39                                                             # an fp32 subnormal weight
    wts = wide.cuda()[:, :k]
    y[0, :, :8] = -0.0                                                               # row 0: -0 products sum to -0
    got = weighted_sum(y, wts)
    want = _torch_weighted_sum(y, wts)
    assert got.shape == (rows, width) and got.dtype == torch.bfloat16
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))              # bits: -0 differs from +0
    assert (got[0, :8].view(torch.int16) == -0x8000).all()


@pytest.mark.parametrize("shape", [(1, 2560), (9, 2560), (4096, 2560), (3, 1000)])
def test_relu2_is_relu_then_square_bit_for_bit(shape):
    from tensorfold.families.nemotron_bolt.cuda.engine import relu2

    x = _awkward_bf16(shape, torch.Generator().manual_seed(shape[0])).cuda()
    r = torch.relu(x)
    assert torch.equal(relu2(x).view(torch.int16), (r * r).view(torch.int16))


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
