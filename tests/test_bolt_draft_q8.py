"""Nemotron 3.5 Bolt's drafter-only Q8_0 lm_head on CPU: the loader keeps the stored bytes, admission charges them, the
head's values are the bf16 head's, and drafted replies with it equal serial ones."""

from __future__ import annotations

from functools import partial

import pytest

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tests.bolt_tiny import write

    return write(tmp_path_factory.mktemp("bolt"))


def test_the_loader_keeps_the_stored_q8_0_head(tiny):
    import torch
    from safetensors import safe_open

    from tensorfold.families.nemotron_bolt.cuda.weights import dequantize_q8_0_rows, load

    w = load(tiny, "cpu", q8_head=True)
    stored = None
    for f in tiny.glob("model-*.safetensors"):
        with safe_open(str(f), "pt") as st:
            if "language_model.lm_head.weight" in st.keys():   # noqa: SIM118  (safe_open has no __contains__)
                stored = st.get_tensor("language_model.lm_head.weight")
    assert w.head_q8.dtype == torch.uint8 and torch.equal(w.head_q8, stored.reshape(stored.shape[0], -1))
    assert torch.equal(w.head, dequantize_q8_0_rows(w.head_q8))
    plain = load(tiny, "cpu")
    assert plain.head_q8 is None and torch.equal(plain.head, w.head)


def test_admission_charges_the_draft_head_only_when_drafting_with_it(tiny, tmp_path):
    from tensorfold.families.nemotron_bolt.cuda.app import head_is_q8, need_bytes
    from tensorfold.families.nemotron_bolt.cuda.q8_head import head_bytes
    from tensorfold.families.nemotron_bolt.cuda.weights import Config
    from tests.bolt_lilicorr_fakes import write_draft

    c = Config.read(tiny)
    draft = write_draft(tmp_path / "draft")
    assert head_is_q8(tiny) and head_bytes(c) == 1024 * 512 // 32 * 34
    with_q8 = need_bytes(tiny, c, 1024, drafter=draft, draft_q8=True)
    assert with_q8 - need_bytes(tiny, c, 1024, drafter=draft) == head_bytes(c)
    assert need_bytes(tiny, c, 1024, draft_q8=True) == need_bytes(tiny, c, 1024)       # no drafter, no draft head


def test_q8_0_logits_are_the_bf16_heads_rows():
    import torch

    from tensorfold.families.nemotron_bolt.cuda.q8_head import logits
    from tensorfold.families.nemotron_bolt.quant import dequantize_q8_0
    from tests.bolt_tiny import q8_0

    g = torch.Generator().manual_seed(0)
    packed = q8_0(torch.randn((200, 128), generator=g) * 0.02, g).flatten(1)
    h = torch.randn((5, 128), generator=g).to(torch.bfloat16)
    got = logits(h, packed)
    assert got.dtype == torch.float32 and got.shape == (5, 200)
    assert torch.equal(got, h.float() @ dequantize_q8_0(packed).float().T)
    with pytest.raises(ValueError):
        logits(h[:, :96], packed)


@pytest.mark.parametrize("sampling", [None, "keyed"], ids=["greedy", "keyed"])
def test_drafts_from_a_q8_0_head_keep_the_serial_reply(sampling):
    import torch

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.nemotron_bolt.cuda.lilicorr import DraftConfig, Drafter, expected_shapes, lilicorr_decode
    from tensorfold.families.nemotron_bolt.cuda.q8_head import logits
    from tests.bolt_lilicorr_fakes import FakeTarget, random_tensors, serial, tiny_config
    from tests.bolt_tiny import q8_0

    sampling = Sampling(seed=77, temperature=1.0, top_k=20, top_p=0.95) if sampling else None
    prompt = [4, 9, 2, 6, 8, 1, 3]
    want = serial(prompt, 60, sampling)
    cfg = DraftConfig.from_json(tiny_config(window=6))
    for q8 in (False, True):
        t = FakeTarget(prompt)
        head = partial(logits, packed=q8_0(t.head, torch.Generator().manual_seed(1)).flatten(1)) if q8 else t.lm_head
        d = Drafter((cfg, random_tensors(expected_shapes(cfg), 0)), embed=t.embed, lm_head=head, device="cpu",
                    dtype=torch.float32)
        d.add_taps(t.taps_of(0, prompt))
        first = t.sample(t.prompt_logits, [t.pos], sampling)[0]
        res = lilicorr_decode(t, d, first, 60, sampling, depth=7)
        assert res.tokens == want and res.drafted > 0
